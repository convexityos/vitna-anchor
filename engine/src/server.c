/**
 * server.c - The engine's HTTP server: the network half of api.c.
 *
 * Each connection gets a thread of its own, up to MAX_CONNECTIONS at once,
 * which reads one HTTP/1.1 request (headers, then a body of Content-Length
 * bytes), hands it to the API, and closes the connection after the
 * response. The API runs requests that generate together on its own thread
 * and hands each connection its output to send. A client that stalls is cut
 * off after 30 seconds without a byte, and holds up no other; one that hangs
 * up mid-stream ends that generation without taking the server down. Each
 * request is logged to stderr as method, path, status and token counts;
 * prompts and completions are never logged.
 *
 * Once the GPU can run nothing more in the process (vitna_api_lost), the
 * server stops taking connections, lets the open ones finish, and returns.
 */

#include "server.h"
#include "api.h"
#include "compat.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(VITNA_OS_WINDOWS)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef SOCKET socket_t;
  #define IS_INVALID_SOCKET(s) ((s) == INVALID_SOCKET)
  #define CLOSE_SOCKET(s) closesocket(s)
  #define SHUTDOWN_SEND SD_SEND
#else
  #include <sys/select.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <signal.h>
  typedef int socket_t;
  #define IS_INVALID_SOCKET(s) ((s) < 0)
  #define CLOSE_SOCKET(s) close(s)
  #define SHUTDOWN_SEND SHUT_WR
#endif

#if defined(MSG_NOSIGNAL)
  #define SEND_FLAGS MSG_NOSIGNAL
#else
  #define SEND_FLAGS 0
#endif

#define MAX_HEADER_BYTES (64 * 1024)
#define MAX_BODY_BYTES (8 * 1024 * 1024)
#define IO_TIMEOUT_MS 30000

/* The most connections served at once; past it, the next waits to be
 * accepted until one closes. Each holds a thread, and one that is waiting
 * for its request to run holds only that. */
#define MAX_CONNECTIONS 64

/* How long the server waits for a connection before it looks again whether
 * the GPU has been lost, which no connection would tell it. */
#define ACCEPT_WAIT_MS 250

static bool sock_write(void* ctx, const void* data, size_t len) {
    socket_t s = *(socket_t*)ctx;
    const char* p = (const char*)data;
    while (len > 0) {
        int chunk = len > (size_t)INT_MAX ? INT_MAX : (int)len;
        int n = send(s, p, chunk, SEND_FLAGS);
        if (n <= 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static void send_simple(socket_t s, int status, const char* reason, const char* message) {
    char body[512], head[256];
    int bl = snprintf(body, sizeof(body), "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}", message);
    int hl = snprintf(head, sizeof(head), "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", status, reason, bl);
    sock_write(&s, head, (size_t)hl);
    sock_write(&s, body, (size_t)bl);
}

static void set_timeouts(socket_t s) {
#if defined(VITNA_OS_WINDOWS)
    DWORD ms = IO_TIMEOUT_MS;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
#else
    struct timeval tv;
    tv.tv_sec = IO_TIMEOUT_MS / 1000;
    tv.tv_usec = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  #if defined(SO_NOSIGPIPE)
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  #endif
#endif
}

typedef struct {
    char method[16];
    char path[1024];
    char* buf;
    size_t body_off;
    size_t body_len;
} request_t;

static bool header_is(const char* name, size_t n, const char* want) {
    size_t w = strlen(want);
    if (n != w) return false;
    for (size_t i = 0; i < n; i++) {
        char c = name[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != want[i]) return false;
    }
    return true;
}

/* Returns 0 with the request read, an HTTP status to answer with, or -1 when
 * the client went away before sending anything. */
static int read_request(socket_t s, request_t* req) {
    memset(req, 0, sizeof(*req));
    size_t cap = 8192, len = 0, head_end = 0, scan_from = 0;
    char* buf = (char*)malloc(cap + 1);
    if (!buf) return 500;
    req->buf = buf;
    for (;;) {
        /* The blank line that ends the headers can be anywhere in what has
         * arrived, with body bytes after it in the same read. */
        for (size_t i = scan_from; head_end == 0 && i + 4 <= len; i++) {
            if (memcmp(buf + i, "\r\n\r\n", 4) == 0) head_end = i + 4;
        }
        if (head_end) break;
        scan_from = len >= 3 ? len - 3 : 0;
        if (len >= MAX_HEADER_BYTES) return 431;
        if (len == cap) {
            char* grown = (char*)realloc(buf, cap * 2 + 1);
            if (!grown) return 500;
            req->buf = buf = grown;
            cap *= 2;
        }
        int n = recv(s, buf + len, (int)(cap - len), 0);
        if (n <= 0) return len == 0 ? -1 : 408;
        len += (size_t)n;
    }
    buf[len] = '\0';
    if (sscanf(buf, "%15s %1023s", req->method, req->path) != 2) return 400;

    bool expect_continue = false;
    long long content_length = 0;
    const char* line = strstr(buf, "\r\n") + 2;
    while (line < buf + head_end - 2) {
        const char* eol = strstr(line, "\r\n");
        if (!eol || eol > buf + head_end) return 400; /* a NUL inside the headers */
        const char* colon = memchr(line, ':', (size_t)(eol - line));
        if (colon) {
            const char* v = colon + 1;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            size_t name_len = (size_t)(colon - line);
            if (header_is(line, name_len, "content-length")) {
                char* end;
                content_length = strtoll(v, &end, 10);
                if (end == v || content_length < 0) return 400;
            } else if (header_is(line, name_len, "transfer-encoding")) {
                return 411; /* a chunked body; this server needs Content-Length */
            } else if (header_is(line, name_len, "expect")) {
                expect_continue = (size_t)(eol - v) >= 12 && (v[0] == '1');
            }
        }
        line = eol + 2;
    }
    if (content_length > MAX_BODY_BYTES) return 413;

    size_t need = head_end + (size_t)content_length;
    if (need > len && expect_continue) {
        const char* go = "HTTP/1.1 100 Continue\r\n\r\n";
        sock_write(&s, go, strlen(go));
    }
    if (need + 1 > cap) {
        char* grown = (char*)realloc(buf, need + 1);
        if (!grown) return 500;
        req->buf = buf = grown;
        cap = need;
    }
    while (len < need) {
        int n = recv(s, buf + len, (int)(need - len), 0);
        if (n <= 0) return 408;
        len += (size_t)n;
    }
    buf[need] = '\0';
    req->body_off = head_end;
    req->body_len = (size_t)content_length;
    return 0;
}

static const char* status_reason(int status) {
    switch (status) {
        case 400: return "Bad Request";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        default: return "Internal Server Error";
    }
}

/* A HEAD request is answered as GET would be, without the body: the bytes
 * of the response after its headers end are dropped, as HTTP requires. */
typedef struct {
    socket_t* s;
    int seen;   /* how much of the blank line that ends the headers has gone by */
    bool body;  /* the headers have ended */
} head_sink_t;

static bool head_write(void* ctx, const void* data, size_t len) {
    head_sink_t* h = (head_sink_t*)ctx;
    const char* p = (const char*)data;
    size_t i = 0;
    while (i < len && !h->body) {
        const char want = "\r\n\r\n"[h->seen];
        if (p[i] == want) h->body = ++h->seen == 4;
        else h->seen = p[i] == '\r' ? 1 : 0;
        i++;
    }
    return i == 0 || sock_write(h->s, data, i);
}

static void handle_client(socket_t client, vitna_api_t* api) {
    set_timeouts(client);
    request_t req;
    int rc = read_request(client, &req);
    if (rc > 0) {
        send_simple(client, rc, status_reason(rc),
                    rc == 411 ? "Send the body with a Content-Length header" : "The request could not be read");
        fprintf(stderr, "? ? %d\n", rc);
    } else if (rc == 0) {
        vitna_sink_t sink = { sock_write, &client, false };
        head_sink_t head = { &client, 0, false };
        const bool is_head = strcmp(req.method, "HEAD") == 0;
        if (is_head) {
            sink.write = head_write;
            sink.ctx = &head;
        }
        vitna_api_result_t r = vitna_api_handle(api, is_head ? "GET" : req.method, req.path, req.buf + req.body_off, req.body_len, &sink);
        char route[128];
        size_t n = strcspn(req.path, "?");
        snprintf(route, sizeof(route), "%.*s", (int)(n < 120 ? n : 120), req.path);
        if (r.prompt_tokens || r.completion_tokens) {
            fprintf(stderr, "%s %s %d prompt_tokens=%zu cached_tokens=%zu completion_tokens=%zu%s\n", req.method, route, r.status,
                    r.prompt_tokens, r.cached_tokens, r.completion_tokens, sink.failed ? " (the client went away)" : "");
        } else {
            fprintf(stderr, "%s %s %d\n", req.method, route, r.status);
        }
    }
    fflush(stderr);
    free(req.buf);
    shutdown(client, SHUTDOWN_SEND);
    CLOSE_SOCKET(client);
}

/* The connections being served, counted so there are never more than MAX_CONNECTIONS. */
typedef struct {
    vitna_mutex_t lock;
    vitna_cond_t freed;
    int active;
} conn_count_t;

typedef struct {
    socket_t client;
    vitna_api_t* api;
    conn_count_t* count;
} conn_t;

static void connection(void* arg) {
    conn_t* c = (conn_t*)arg;
    handle_client(c->client, c->api);
    vitna_mutex_lock(&c->count->lock);
    c->count->active--;
    vitna_cond_signal(&c->count->freed);
    vitna_mutex_unlock(&c->count->lock);
    free(c);
}

/* Whether a connection waits on the listening socket s to be accepted,
 * waiting up to ms for one: 1 if one does, 0 if none came, -1 on an error. */
static int connection_waiting(socket_t s, int ms) {
    fd_set ready;
    FD_ZERO(&ready);
    FD_SET(s, &ready);
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    /* Windows ignores the first argument. */
    const int n = select((int)(s + 1), &ready, NULL, NULL, &tv);
    return n < 0 ? -1 : n > 0;
}

int vitna_server_run(const vitna_server_config_t* config) {
#if defined(VITNA_OS_WINDOWS)
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return -1;
    }
#else
    /* A client that hangs up mid-stream must end one response, not the process. */
    signal(SIGPIPE, SIG_IGN);
#endif

    /* Port 0 asks the OS for any free port; main.c defaults to 8765. */
    uint16_t port = config ? config->port : 8765;
    const char* bind_ip = config && config->bind_addr ? config->bind_addr : "127.0.0.1";
    vitna_api_t* api = config ? (vitna_api_t*)config->engine_ctx : NULL;

    socket_t server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (IS_INVALID_SOCKET(server_sock)) {
        fprintf(stderr, "Failed to create socket\n");
#if defined(VITNA_OS_WINDOWS)
        WSACleanup();
#endif
        return -1;
    }

    int opt = 1;
#if defined(VITNA_OS_WINDOWS)
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(bind_ip);

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Failed to bind socket to %s:%u\n", bind_ip, port);
        CLOSE_SOCKET(server_sock);
#if defined(VITNA_OS_WINDOWS)
        WSACleanup();
#endif
        return -1;
    }

    if (listen(server_sock, 16) < 0) {
        fprintf(stderr, "Failed to listen on socket\n");
        CLOSE_SOCKET(server_sock);
#if defined(VITNA_OS_WINDOWS)
        WSACleanup();
#endif
        return -1;
    }

    /* Report the port actually bound, which differs from the one asked for
     * when that was 0 (any free port). */
    struct sockaddr_in bound;
    memset(&bound, 0, sizeof(bound));
#if defined(VITNA_OS_WINDOWS)
    int bound_len = sizeof(bound);
#else
    socklen_t bound_len = sizeof(bound);
#endif
    if (getsockname(server_sock, (struct sockaddr*)&bound, &bound_len) == 0) {
        port = ntohs(bound.sin_port);
    }

    printf("vitna-anchor listening on http://%s:%u\n", bind_ip, port);
    if (api && vitna_api_embeds(api)) {
        printf("Serving %s at /v1/embeddings, one request at a time.\n", vitna_api_model_id(api));
    } else if (api && vitna_api_parallel(api) > 1) {
        printf("Serving %s at /v1/chat/completions, /v1/completions, /v1/responses and /v1/messages, %zu requests at a time.\n", vitna_api_model_id(api),
               vitna_api_parallel(api));
    } else if (api) {
        printf("Serving %s at /v1/chat/completions, /v1/completions, /v1/responses and /v1/messages, one request at a time.\n", vitna_api_model_id(api));
    } else {
        printf("%s\n", vitna_api_no_model_message());
        printf("Generation endpoints answer 501.\n");
    }
    if (strcmp(bind_ip, "127.0.0.1") != 0) {
        printf("Warning: %s is not the loopback address. This server has no authentication.\n", bind_ip);
    }
    fflush(stdout);

    conn_count_t count;
    vitna_mutex_init(&count.lock);
    vitna_cond_init(&count.freed);
    count.active = 0;
    while (!(api && vitna_api_lost(api))) {
        vitna_mutex_lock(&count.lock);
        while (count.active >= MAX_CONNECTIONS) vitna_cond_wait(&count.freed, &count.lock);
        vitna_mutex_unlock(&count.lock);
        const int waiting = connection_waiting(server_sock, ACCEPT_WAIT_MS);
        if (waiting < 0) break;
        if (waiting == 0) continue;
        struct sockaddr_in client_addr;
#if defined(VITNA_OS_WINDOWS)
        int client_len = sizeof(client_addr);
        socket_t client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
#else
        socklen_t client_len = sizeof(client_addr);
        socket_t client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
#endif
        if (IS_INVALID_SOCKET(client_sock)) {
            break;
        }
        conn_t* c = (conn_t*)malloc(sizeof(conn_t));
        vitna_thread_t t;
        vitna_mutex_lock(&count.lock);
        count.active++;
        vitna_mutex_unlock(&count.lock);
        if (c) {
            c->client = client_sock;
            c->api = api;
            c->count = &count;
        }
        if (!c || !vitna_thread_start(&t, connection, c, true)) {
            /* No thread for it: serve it here, as the server once served every connection. */
            free(c);
            handle_client(client_sock, api);
            vitna_mutex_lock(&count.lock);
            count.active--;
            vitna_mutex_unlock(&count.lock);
        }
    }

    /* Closed first, so a client that comes now is refused at once rather
     * than left waiting. The connections still open then finish, each
     * sending its response: once the GPU is lost, the error each of its
     * requests ended with. */
    CLOSE_SOCKET(server_sock);
    vitna_mutex_lock(&count.lock);
    while (count.active > 0) vitna_cond_wait(&count.freed, &count.lock);
    vitna_mutex_unlock(&count.lock);
#if defined(VITNA_OS_WINDOWS)
    WSACleanup();
#endif
    return 0;
}
