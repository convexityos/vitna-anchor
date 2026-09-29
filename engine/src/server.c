/**
 * server.c - Local HTTP server for vitna-engine.
 *
 * No model runs yet, and this server says so. Generation endpoints answer
 * 501 with an OpenAI-style error, the model list is empty, and nothing is
 * reported that was not measured. Serving real completions is gate A3 in the
 * README, and it waits on a forward pass (gate A2).
 */

#include "server.h"
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(VITNA_OS_WINDOWS)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef SOCKET socket_t;
  #define IS_INVALID_SOCKET(s) ((s) == INVALID_SOCKET)
  #define CLOSE_SOCKET(s) closesocket(s)
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  typedef int socket_t;
  #define IS_INVALID_SOCKET(s) ((s) < 0)
  #define CLOSE_SOCKET(s) close(s)
#endif

#define VITNA_NO_MODEL_MESSAGE \
    "No model runs yet. This build of vitna-anchor has no forward pass, so it cannot generate text. " \
    "See the gate ladder in the README."

static void send_http_response(socket_t sock, int status_code, const char* status_text, const char* content_type, const char* body) {
    char header[512];
    size_t body_len = body ? strlen(body) : 0;
    snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text, content_type, body_len
    );
    send(sock, header, (int)strlen(header), 0);
    if (body_len > 0) {
        send(sock, body, (int)body_len, 0);
    }
}

static void handle_client(socket_t client_sock) {
    char buffer[4096];
    int bytes_received = recv(client_sock, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received <= 0) {
        CLOSE_SOCKET(client_sock);
        return;
    }
    buffer[bytes_received] = '\0';

    char method[16] = {0};
    char path[256] = {0};
    sscanf(buffer, "%15s %255s", method, path);

    if (strcmp(method, "GET") == 0 && (strcmp(path, "/health") == 0 || strcmp(path, "/v1/health") == 0)) {
        send_http_response(client_sock, 200, "OK", "application/json",
            "{\"ok\":true,\"engine\":\"vitna-anchor\",\"model\":null,\"generation\":false,"
            "\"message\":\"" VITNA_NO_MODEL_MESSAGE "\"}");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        send_http_response(client_sock, 200, "OK", "application/json", "{\"object\":\"list\",\"data\":[]}");
    } else if (strcmp(method, "POST") == 0 &&
               (strcmp(path, "/v1/chat/completions") == 0 ||
                strcmp(path, "/v1/completions") == 0 ||
                strcmp(path, "/v1/embeddings") == 0)) {
        send_http_response(client_sock, 501, "Not Implemented", "application/json",
            "{\"error\":{\"message\":\"" VITNA_NO_MODEL_MESSAGE "\",\"type\":\"not_implemented\",\"code\":\"no_model\"}}");
    } else {
        send_http_response(client_sock, 404, "Not Found", "application/json",
            "{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\",\"code\":\"not_found\"}}");
    }

    CLOSE_SOCKET(client_sock);
}

int vitna_server_run(const vitna_server_config_t* config) {
#if defined(VITNA_OS_WINDOWS)
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return -1;
    }
#endif

    /* Port 0 asks the OS for any free port; main.c defaults to 8765. */
    uint16_t port = config ? config->port : 8765;
    const char* bind_ip = config && config->bind_addr ? config->bind_addr : "127.0.0.1";

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
    printf("%s\n", VITNA_NO_MODEL_MESSAGE);
    printf("Generation endpoints answer 501.\n");
    fflush(stdout);

    while (1) {
        struct sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
#if defined(VITNA_OS_WINDOWS)
        socket_t client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
#else
        socket_t client_sock = accept(server_sock, (struct sockaddr*)&client_addr, (socklen_t*)&client_len);
#endif
        if (IS_INVALID_SOCKET(client_sock)) {
            break;
        }
        handle_client(client_sock);
    }

    CLOSE_SOCKET(server_sock);
#if defined(VITNA_OS_WINDOWS)
    WSACleanup();
#endif
    return 0;
}
