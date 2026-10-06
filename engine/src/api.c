/**
 * api.c - The OpenAI-compatible HTTP API over a loaded model.
 *
 * Requests run on one thread of their own, the scheduler, which owns the
 * model: each a sequence of its key-value cache, as many at once as it holds
 * (serve --parallel), with their next tokens run together in each round. A
 * request that finds no sequence free waits its turn. The connection that
 * asked for a request sends what the scheduler writes for it, so a slow
 * client holds up only itself.
 */

#include "api.h"
#include "chat.h"
#include "compat.h"
#include "convert.h"
#include "json.h"
#include "jsonpfx.h"
#include "sampler.h"
#include "strbuf.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Stop sequences a request may give: OpenAI's limit of 4, and more for
 * Anthropic's stop_sequences, which has none. */
#define OPENAI_MAX_STOPS 4
#define MAX_STOPS 16

/* JSON mode's mask is a function of the object so far, as the automaton's
 * state holds it, and of nothing else, and the same states come round again
 * and again: every token inside a string, for one. So the masks of the last
 * MASK_CACHE states are kept, a bit a token (mask_json). */
#define MASK_CACHE 32

/* The most threads a round's requests take their tokens on, the scheduler's
 * own among them (take_all). */
#define TAKE_THREADS 8

/* While other requests decode, a prompt runs a piece at a time between their
 * rounds, so none of them waits long for it: PROMPT_PIECE tokens where a
 * prompt's tokens run together (on a GPU), PROMPT_PIECE_STEPS where they run
 * a step at a time (on the CPU). With nothing decoding, a prompt runs whole.
 * Either way its keys, values and logits are the same, bit for bit
 * (vitna_llama_prompt_piece_min). */
#define PROMPT_PIECE 512
#define PROMPT_PIECE_STEPS 8

typedef struct {
    vitna_jsonpfx_t key; /* the state, with the stack's slots past its depth cleared */
    uint32_t* allowed;   /* bit (id & 31) of word id >> 5: whether token id can follow */
    size_t left;         /* how many can */
    uint64_t used;       /* when it last served, to replace the one unused longest */
} mask_entry_t;
#define NO_MODEL_MESSAGE \
    "This server was started without a model, so it serves none. Start it with: vitna-anchor serve --model <dir>"

/* A sequence of the model's cache, lent to one request at a time. */
typedef struct {
    vitna_token_list_t cached;  /* the tokens whose keys and values it holds, in order */
    uint64_t used;              /* when it was last lent, so the one unused longest goes first */
    bool busy;
} seq_t;

typedef struct job job_t;

struct vitna_api {
    vitna_llama_t* model;           /* a model that generates, or NULL when encoder is set */
    vitna_encoder_t* encoder;       /* an embedding model, served at /v1/embeddings in place of generation */
    vitna_mutex_t embed_lock;       /* one embedding request at a time: each runs on every thread the encoder has */
    const vitna_tokenizer_t* tok;
    char* model_id;
    long long created;
    bool prefix_cache;
    bool mask_cache;                /* whether JSON mode's masks are kept (vitna_api_set_mask_cache) */
    bool test_logits;               /* for tests: responses name their logits' hash (vitna_api_set_test_logits) */
    bool test_reply;                /* for tests: a request's vitna_test_reply is its reply (vitna_api_set_test_reply) */
    size_t speculate;               /* the most tokens drafted at once, 0 for none (vitna_api_set_speculate) */
    bool qwen3;                     /* conversations in Qwen3's chat template, else ChatML (vitna_api_set_qwen3_template) */
    int32_t tool_call_token;        /* <tool_call> as one token of the vocabulary, or -1 */

    /* The scheduler's alone, on its thread. */
    uint64_t counter;
    float* row;                     /* a prompt's last logits */
    seq_t* seqs;                    /* one per sequence of the model's cache */
    uint64_t seq_clock;
    size_t rows_cap;                /* the rows a round can hold (round_capacity) */
    vitna_llama_row_t* rows;        /* a round's rows, */
    float* logits;                  /* their logits, rows_cap x vocab, */
    bool* ran;                      /* and which ran */

    /* The threads a round's requests take their tokens on: the scheduler's,
     * sampler 0, and workers 1 to takers - 1, each with a sampler of its own,
     * whose state is each request's in turn (take). */
    size_t takers;
    vitna_sampler_t* samplers;
    vitna_thread_t* workers;
    size_t workers_started;
    vitna_mutex_t pool_lock;
    vitna_cond_t pool_work;         /* signalled when a round's requests are ready to take */
    vitna_cond_t pool_done;         /* signalled when the last of them is taken */
    job_t** pool_jobs;              /* the round's requests */
    size_t pool_n, pool_next, pool_pending;
    uint64_t pool_round;            /* counts rounds handed out, so a worker knows a new one */
    bool pool_stop;

    /* JSON mode's masks by state (mask_json), under mask_lock: requests
     * taking their tokens at once share them. */
    vitna_mutex_t mask_lock;
    mask_entry_t masks[MASK_CACHE];
    uint64_t mask_clock;

    /* Under lock. */
    vitna_mutex_t lock;
    vitna_cond_t work;              /* signalled when a request queues, and to stop */
    job_t* queue;                   /* requests waiting for a sequence, oldest first */
    job_t* queue_tail;
    bool stopping;
    bool started;                   /* the scheduler's thread is running */
    bool lost;                      /* the GPU can run nothing more, so requests are refused (lose) */
    vitna_thread_t thread;
};

/* The API a request came in on, which decides its parameters and the shape
 * of its response. */
typedef enum {
    API_COMPLETIONS,        /* OpenAI's /v1/completions */
    API_CHAT,               /* OpenAI's /v1/chat/completions */
    API_MESSAGES,           /* Anthropic's /v1/messages */
    API_RESPONSES,          /* OpenAI's /v1/responses */
} api_kind_t;

typedef struct {
    api_kind_t kind;
    bool chat;              /* a conversation: any kind but completions */
    size_t max_tokens;
    bool has_max_tokens;
    float temperature;
    float top_p;
    size_t top_k;
    uint64_t seed;
    bool has_seed;
    const char* stops[MAX_STOPS];
    size_t stop_lens[MAX_STOPS];
    size_t n_stops;
    bool stream;
    bool include_usage;
    bool has_stream_options;
    bool json;              /* response_format json_object: only a valid JSON object may be generated */

    /* A conversation in Qwen3's template, its reply read back (chat.h). */
    bool parse;             /* the reply is read into reasoning, text and tool calls */
    bool think;             /* it may open with <think> */
    bool tools;             /* its <tool_call> blocks are calls */
    int32_t ban;            /* a token never taken (<tool_call> under tool_choice none), or -1 */
    const char* prefill;    /* the reply's start, which the prompt already holds: a forced tool call's opening */
    size_t prefill_len;
    const int32_t* forced;  /* for tests: the reply's tokens, taken in place of the model's choices */
    size_t n_forced;
    bool has_test_prompt;   /* for tests: the prompt's hash goes in a header (vitna_api_set_test_reply) */
    uint64_t test_prompt;
    const vitna_json_value_t* root; /* the request, for the fields a Responses response repeats */
} params_t;

typedef struct {
    int status;
    const char* type;
    const char* code;
    const char* param;
    char message[512];
    char param_buf[64];     /* for a param made up on the spot, as "messages[2]" */
} api_error_t;

/* --- Output --- */

static void sink_out(vitna_sink_t* s, const void* data, size_t len) {
    if (!s->failed && len > 0 && !s->write(s->ctx, data, len)) s->failed = true;
}

static const char* reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

static void respond_as(vitna_sink_t* s, int status, const char* type, const char* headers, const char* body, size_t len) {
    vitna_strbuf_t h;
    vitna_sb_init(&h);
    vitna_sb_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n%s\r\n", status, reason_phrase(status), type,
                    len, headers ? headers : "");
    sink_out(s, h.data, h.len);
    sink_out(s, body, len);
    vitna_sb_free(&h);
}

static void respond(vitna_sink_t* s, int status, const char* headers, const char* body, size_t len) {
    respond_as(s, status, "application/json", headers, body, len);
}

/* The chat page (engine/web/chat.html, built in as chat_page.c). Its policy
 * lets it run its own script and styles and talk to this server alone: it
 * loads nothing from anywhere, and no other page may frame it. */
extern const char vitna_chat_page[];
extern const size_t vitna_chat_page_len;

#define CHAT_PAGE_HEADERS                                                                                                                   \
    "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' " \
    "data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n"                                                                 \
    "X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nCache-Control: no-cache\r\n"

static void set_error(api_error_t* e, int status, const char* type, const char* code, const char* param, const char* fmt, ...) {
    e->status = status;
    e->type = type;
    e->code = code;
    e->param = param;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->message, sizeof(e->message), fmt, ap);
    va_end(ap);
}

/* {"error":{...}}: the body of an error response, and of a stream's error event. */
static void error_json(vitna_strbuf_t* b, const api_error_t* e) {
    vitna_sb_puts(b, "{\"error\":{\"message\":");
    vitna_sb_json_string(b, (const unsigned char*)e->message, strlen(e->message));
    vitna_sb_puts(b, ",\"type\":");
    vitna_sb_json_string(b, (const unsigned char*)e->type, strlen(e->type));
    vitna_sb_puts(b, ",\"param\":");
    if (e->param) vitna_sb_json_string(b, (const unsigned char*)e->param, strlen(e->param));
    else vitna_sb_puts(b, "null");
    vitna_sb_puts(b, ",\"code\":");
    if (e->code) vitna_sb_json_string(b, (const unsigned char*)e->code, strlen(e->code));
    else vitna_sb_puts(b, "null");
    vitna_sb_puts(b, "}}");
}

/* Anthropic's error type for a status: its API's names for the same failures. */
static const char* anthropic_error_type(int status) {
    if (status == 404) return "not_found_error";
    if (status >= 500) return "api_error";
    return "invalid_request_error";
}

/* {"type":"error","error":{...}}: Anthropic's error body, and its stream's error event. */
static void anthropic_error_json(vitna_strbuf_t* b, const api_error_t* e) {
    vitna_sb_printf(b, "{\"type\":\"error\",\"error\":{\"type\":\"%s\",\"message\":", anthropic_error_type(e->status));
    vitna_sb_json_string(b, (const unsigned char*)e->message, strlen(e->message));
    vitna_sb_puts(b, "}}");
}

/* An error response in the shape of the API the request came in on. */
static int respond_error_as(vitna_sink_t* s, const api_error_t* e, api_kind_t kind) {
    vitna_strbuf_t b;
    vitna_sb_init(&b);
    if (kind == API_MESSAGES) anthropic_error_json(&b, e);
    else error_json(&b, e);
    respond(s, e->status, e->status == 405 ? "Allow: POST\r\n" : NULL, b.data, b.len);
    vitna_sb_free(&b);
    return e->status;
}

static int respond_error(vitna_sink_t* s, const api_error_t* e) {
    return respond_error_as(s, e, API_CHAT);
}

static void unsupported(api_error_t* e, const char* param, const char* why) {
    set_error(e, 400, "invalid_request_error", "unsupported_parameter", param, "`%s` is not supported by this server: %s", param, why);
}

/* --- Ids and seeds --- */

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t fresh_u64(vitna_api_t* api) {
    api->counter++;
    return mix64(vitna_time_nanos() ^ (api->counter * 0x9E3779B97F4A7C15ULL) ^ (uint64_t)(uintptr_t)api);
}

/* --- Parameters --- */

static bool integral(double d) {
    return d > -9.2e18 && d < 9.2e18 && d == (double)(long long)d;
}

static bool is_zero_or_null(const vitna_json_value_t* v) {
    double d;
    return vitna_json_is_null(v) || (vitna_json_as_number(v, &d) && d == 0.0);
}

/* The keys a request's converter reads (convert.h), or a completion's
 * prompt, which parse_params leaves to them. */
static bool conversation_key(api_kind_t kind, const char* k) {
    static const char* const completions[] = { "prompt", NULL };
    static const char* const chat[] = { "messages", "tools", "tool_choice", "functions", "function_call", "chat_template_kwargs", "reasoning_effort", NULL };
    static const char* const messages[] = { "messages", "system", "tools", "tool_choice", "thinking", NULL };
    static const char* const responses[] = { "input", "instructions", "tools", "tool_choice", "reasoning", NULL };
    const char* const* keys = kind == API_COMPLETIONS ? completions : kind == API_CHAT ? chat : kind == API_MESSAGES ? messages : responses;
    for (size_t i = 0; keys[i]; i++) {
        if (strcmp(k, keys[i]) == 0) return true;
    }
    return false;
}

/* Bookkeeping for a hosted API, which changes nothing here. */
static bool bookkeeping(api_kind_t kind, const char* k) {
    static const char* const keys[] = { "user", "metadata", "store", "service_tier", "parallel_tool_calls", "prompt_cache_key", "safety_identifier", NULL };
    for (size_t i = 0; keys[i]; i++) {
        if (strcmp(k, keys[i]) == 0) return true;
    }
    /* What a Responses client asks to have included: nothing kept server side here. */
    return kind == API_RESPONSES && strcmp(k, "include") == 0;
}

static bool parse_params(const vitna_api_t* api, const vitna_json_value_t* root, api_kind_t kind, params_t* p, api_error_t* e, vitna_strbuf_t* ignored) {
    memset(p, 0, sizeof(*p));
    p->kind = kind;
    p->chat = kind != API_COMPLETIONS;
    p->temperature = 1.0f;
    p->top_p = 1.0f;
    p->ban = -1;
    p->root = root;
    const bool openai = kind == API_COMPLETIONS || kind == API_CHAT; /* the two shapes with OpenAI's sampling parameters */
    for (size_t i = 0; i < root->u.object.count; i++) {
        const char* k = root->u.object.members[i].key;
        const vitna_json_value_t* v = root->u.object.members[i].value;
        double d;
        bool b;
        if (strcmp(k, "model") == 0 || conversation_key(kind, k) || bookkeeping(kind, k)) continue;
        if (api->test_reply && strcmp(k, "vitna_test_reply") == 0) continue;
        if (vitna_json_is_null(v)) continue;
        const bool max_key = kind == API_RESPONSES ? strcmp(k, "max_output_tokens") == 0
                                                   : strcmp(k, "max_tokens") == 0 || (kind == API_CHAT && strcmp(k, "max_completion_tokens") == 0);
        if (max_key) {
            if (!vitna_json_as_number(v, &d) || !integral(d) || d < 1) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`%s` must be a positive integer", k);
                return false;
            }
            p->max_tokens = (size_t)d;
            p->has_max_tokens = true;
        } else if (strcmp(k, "temperature") == 0) {
            /* Anthropic's scale ends at 1, OpenAI's at 2. */
            const double top = kind == API_MESSAGES ? 1 : 2;
            if (!vitna_json_as_number(v, &d) || d < 0 || d > top) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`temperature` must be a number from 0 to %d", (int)top);
                return false;
            }
            p->temperature = (float)d;
        } else if (strcmp(k, "top_p") == 0) {
            if (!vitna_json_as_number(v, &d) || !(d > 0) || d > 1) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`top_p` must be a number above 0 and at most 1");
                return false;
            }
            p->top_p = (float)d;
        } else if (strcmp(k, "top_k") == 0) {
            if (!vitna_json_as_number(v, &d) || !integral(d) || d < 0) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`top_k` must be a non-negative integer");
                return false;
            }
            p->top_k = (size_t)d;
        } else if (strcmp(k, "seed") == 0) {
            if (!vitna_json_as_number(v, &d) || !integral(d)) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`seed` must be an integer");
                return false;
            }
            p->seed = (uint64_t)(long long)d;
            p->has_seed = true;
        } else if (openai && strcmp(k, "stop") == 0) {
            if (v->type == VITNA_JSON_STRING) {
                if (v->u.string.len == 0) {
                    set_error(e, 400, "invalid_request_error", "invalid_value", k, "a stop sequence cannot be empty");
                    return false;
                }
                p->stops[0] = v->u.string.ptr;
                p->stop_lens[0] = v->u.string.len;
                p->n_stops = 1;
            } else if (v->type == VITNA_JSON_ARRAY && v->u.array.count <= OPENAI_MAX_STOPS) {
                for (size_t j = 0; j < v->u.array.count; j++) {
                    const vitna_json_value_t* s = v->u.array.items[j];
                    if (s->type != VITNA_JSON_STRING || s->u.string.len == 0) {
                        set_error(e, 400, "invalid_request_error", "invalid_value", k, "each stop sequence must be a non-empty string");
                        return false;
                    }
                    p->stops[p->n_stops] = s->u.string.ptr;
                    p->stop_lens[p->n_stops++] = s->u.string.len;
                }
            } else {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stop` must be a string or up to %d strings", OPENAI_MAX_STOPS);
                return false;
            }
        } else if (kind == API_MESSAGES && strcmp(k, "stop_sequences") == 0) {
            if (v->type != VITNA_JSON_ARRAY || v->u.array.count > MAX_STOPS) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stop_sequences` must be up to %d strings", MAX_STOPS);
                return false;
            }
            for (size_t j = 0; j < v->u.array.count; j++) {
                const vitna_json_value_t* s = v->u.array.items[j];
                if (s->type != VITNA_JSON_STRING || s->u.string.len == 0) {
                    set_error(e, 400, "invalid_request_error", "invalid_value", k, "each stop sequence must be a non-empty string");
                    return false;
                }
                p->stops[p->n_stops] = s->u.string.ptr;
                p->stop_lens[p->n_stops++] = s->u.string.len;
            }
        } else if (strcmp(k, "stream") == 0) {
            if (!vitna_json_as_bool(v, &p->stream)) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stream` must be true or false");
                return false;
            }
        } else if (kind != API_MESSAGES && strcmp(k, "stream_options") == 0) {
            if (v->type != VITNA_JSON_OBJECT) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stream_options` must be an object");
                return false;
            }
            /* Responses' only option hides the stream's lengths, and a local server has nothing to hide them from. */
            p->has_stream_options = openai;
            if (openai) vitna_json_as_bool(vitna_json_get(v, "include_usage"), &p->include_usage);
        } else if (openai && (strcmp(k, "n") == 0 || strcmp(k, "best_of") == 0)) {
            if (!vitna_json_as_number(v, &d) || d != 1) {
                unsupported(e, k, "it returns one choice per request");
                return false;
            }
        } else if (openai && strcmp(k, "logprobs") == 0) {
            if (!(vitna_json_as_bool(v, &b) && !b)) {
                unsupported(e, k, "it does not return log probabilities yet");
                return false;
            }
        } else if (kind != API_MESSAGES && strcmp(k, "top_logprobs") == 0) {
            if (!is_zero_or_null(v)) {
                unsupported(e, k, "it does not return log probabilities yet");
                return false;
            }
        } else if (openai && (strcmp(k, "presence_penalty") == 0 || strcmp(k, "frequency_penalty") == 0)) {
            if (!is_zero_or_null(v)) {
                unsupported(e, k, "it applies no penalties to logits");
                return false;
            }
        } else if (openai && strcmp(k, "logit_bias") == 0) {
            if (!(v->type == VITNA_JSON_OBJECT && v->u.object.count == 0)) {
                unsupported(e, k, "it applies no bias to logits");
                return false;
            }
        } else if ((openai && strcmp(k, "response_format") == 0) || (kind == API_RESPONSES && strcmp(k, "text") == 0)) {
            /* response_format, or Responses' text.format: the same types. */
            const vitna_json_value_t* f = kind == API_RESPONSES ? vitna_json_get(v, "format") : v;
            const char* t = vitna_json_as_string(vitna_json_get(f, "type"));
            if (kind == API_RESPONSES && (!f || vitna_json_is_null(f))) {
                /* text carries only its verbosity, which a model without the setting cannot follow */
            } else if (t && strcmp(t, "json_object") == 0) {
                p->json = true;
            } else if (t && strcmp(t, "json_schema") == 0) {
                unsupported(e, k, "it constrains output to a JSON object (type json_object) but not yet to a schema");
                return false;
            } else if (!t || strcmp(t, "text") != 0) {
                unsupported(e, k, "its types are text and json_object");
                return false;
            }
        } else if (kind == API_COMPLETIONS && (strcmp(k, "tools") == 0 || strcmp(k, "functions") == 0)) {
            if (!(v->type == VITNA_JSON_ARRAY && v->u.array.count == 0)) {
                unsupported(e, k, "a completion has no tool calling: send a conversation to /v1/chat/completions");
                return false;
            }
        } else if (kind == API_COMPLETIONS && (strcmp(k, "tool_choice") == 0 || strcmp(k, "function_call") == 0)) {
            const char* c = vitna_json_as_string(v);
            if (!c || (strcmp(c, "none") != 0 && strcmp(c, "auto") != 0)) {
                unsupported(e, k, "a completion has no tool calling: send a conversation to /v1/chat/completions");
                return false;
            }
        } else if (kind == API_COMPLETIONS && strcmp(k, "echo") == 0) {
            if (!(vitna_json_as_bool(v, &b) && !b)) {
                unsupported(e, k, "it returns only the completion");
                return false;
            }
        } else if (kind == API_COMPLETIONS && strcmp(k, "suffix") == 0) {
            unsupported(e, k, "it does not insert text");
            return false;
        } else if (kind == API_RESPONSES && (strcmp(k, "previous_response_id") == 0 || strcmp(k, "conversation") == 0)) {
            unsupported(e, k, "it keeps no responses or conversations, so send the whole conversation as `input`");
            return false;
        } else if (kind == API_RESPONSES && strcmp(k, "background") == 0) {
            if (!(vitna_json_as_bool(v, &b) && !b)) {
                unsupported(e, k, "it answers while the client waits");
                return false;
            }
        } else if (kind == API_RESPONSES && strcmp(k, "truncation") == 0) {
            const char* t = vitna_json_as_string(v);
            if (!t || strcmp(t, "disabled") != 0) {
                unsupported(e, k, "it never drops input to make it fit: a conversation too long for the context is refused");
                return false;
            }
        } else if (kind == API_MESSAGES && strcmp(k, "mcp_servers") == 0) {
            if (!(v->type == VITNA_JSON_ARRAY && v->u.array.count == 0)) {
                unsupported(e, k, "it connects to no MCP servers: the client runs its own tools");
                return false;
            }
        } else if (ignored->len < 400) {
            /* Named in a response header, so only header-safe characters:
             * a field name that is not [A-Za-z0-9_.-] is written with _ for
             * the rest, which keeps a client's CR or LF out of the headers. */
            if (ignored->len) vitna_sb_puts(ignored, ", ");
            for (size_t c = 0; k[c] && c < 64; c++) {
                char ch = k[c];
                bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-';
                vitna_sb_append(ignored, safe ? &ch : "_", 1);
            }
        }
    }
    if (p->has_stream_options && !p->stream) {
        set_error(e, 400, "invalid_request_error", "invalid_value", "stream_options", "`stream_options` is only allowed when `stream` is true");
        return false;
    }
    if (kind == API_MESSAGES && !p->has_max_tokens) {
        set_error(e, 400, "invalid_request_error", "missing_required_parameter", "max_tokens", "`max_tokens` is required");
        return false;
    }
    return true;
}

/* --- Prompts --- */

static void chatml_append(vitna_strbuf_t* sb, const char* role, const char* content, size_t len) {
    vitna_sb_puts(sb, "<|im_start|>");
    vitna_sb_puts(sb, role);
    vitna_sb_puts(sb, "\n");
    vitna_sb_append(sb, content, len);
    vitna_sb_puts(sb, "<|im_end|>\n");
}

bool vitna_chatml_format(const char* const* roles, const char* const* contents, size_t n, char** out, size_t* out_len) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    for (size_t i = 0; i < n; i++) chatml_append(&sb, roles[i], contents[i], strlen(contents[i]));
    vitna_sb_puts(&sb, "<|im_start|>assistant\n");
    if (!sb.ok) {
        vitna_sb_free(&sb);
        return false;
    }
    *out = sb.data;
    *out_len = sb.len;
    return true;
}

/* A converted conversation (convert.h) as ChatML: each
 * "<|im_start|>{role}\n{content}<|im_end|>\n", then "<|im_start|>assistant\n"
 * for the reply, or, to continue the last message, that message left open.
 * ChatML has no place for tools, tool calls or their results, so a
 * conversation with any is refused. An assistant's reasoning is left out:
 * a ChatML model writes none. */
static bool chatml_prompt(const vitna_chat_request_t* cr, api_kind_t kind, vitna_strbuf_t* out, api_error_t* e) {
    if (cr->tools.len) {
        unsupported(e, "tools", "it has no tool calling");
        return false;
    }
    char err[160];
    vitna_json_doc_t* doc = vitna_json_parse(cr->messages.data, cr->messages.len, err, sizeof(err));
    const vitna_json_value_t* messages = doc ? vitna_json_root(doc) : NULL;
    if (!messages) {
        set_error(e, 500, "server_error", NULL, NULL, "out of memory");
        return false;
    }
    const size_t n = messages->u.array.count;
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        const vitna_json_value_t* m = messages->u.array.items[i];
        const char* role = vitna_json_as_string(vitna_json_get(m, "role"));
        const vitna_json_value_t* content = vitna_json_get(m, "content");
        /* A chat request's messages convert one for one; the other shapes' do not. */
        if (kind == API_CHAT) snprintf(e->param_buf, sizeof(e->param_buf), "messages[%zu]", i);
        else snprintf(e->param_buf, sizeof(e->param_buf), "%s", kind == API_RESPONSES ? "input" : "messages");
        if (strcmp(role, "tool") == 0) {
            unsupported(e, e->param_buf, "it has no tool calling, so there are no tool results to read");
            ok = false;
        } else if (vitna_json_get(m, "tool_calls")) {
            unsupported(e, e->param_buf, "it has no tool calling");
            ok = false;
        } else {
            vitna_sb_puts(out, "<|im_start|>");
            vitna_sb_puts(out, role);
            vitna_sb_puts(out, "\n");
            vitna_sb_append(out, content->u.string.ptr, content->u.string.len);
            if (!(cr->continue_final && i == n - 1)) vitna_sb_puts(out, "<|im_end|>\n");
        }
    }
    if (ok && !cr->continue_final) vitna_sb_puts(out, "<|im_start|>assistant\n");
    vitna_json_free(doc);
    return ok;
}

/* The opening of a tool call the model is made to write: Qwen3's
 * <tool_call>, then the call's JSON up to the name the model chooses, or,
 * for a named tool, up to its arguments. */
static void forced_opening(vitna_strbuf_t* out, const vitna_chat_request_t* cr) {
    vitna_sb_puts(out, "<tool_call>\n{\"name\": ");
    if (cr->tool_choice == VITNA_TOOLS_NAMED) {
        vitna_json_value_t name;
        memset(&name, 0, sizeof(name));
        name.type = VITNA_JSON_STRING;
        name.u.string.ptr = cr->tool_name;
        name.u.string.len = strlen(cr->tool_name);
        vitna_json_write_py(out, &name);
        vitna_sb_puts(out, ", \"arguments\": ");
    } else {
        vitna_sb_puts(out, "\"");
    }
}

/* A conversation's prompt, and how its reply is read back, for a request of
 * any of the three shapes: converted (convert.h), then written in the
 * model's chat template. With Qwen3's, a tool_choice that forces a call
 * writes the call's opening into the prompt (prefill), and none bans the
 * <tool_call> token while the tools stay in the prompt, so its start is the
 * same whatever the choice and a cached prefix serves either. */
static bool chat_build(const vitna_api_t* api, const vitna_json_value_t* root, params_t* p, vitna_strbuf_t* prompt, vitna_strbuf_t* prefill,
                       vitna_strbuf_t* ignored, api_error_t* e) {
    vitna_chat_request_t cr;
    vitna_chat_request_init(&cr);
    vitna_json_doc_t* md = NULL;
    vitna_json_doc_t* td = NULL;
    char err[160];
    bool ok = p->kind == API_CHAT ? vitna_convert_chat(root, &cr) : p->kind == API_MESSAGES ? vitna_convert_messages(root, &cr) : vitna_convert_responses(root, &cr);
    if (!ok) {
        snprintf(e->param_buf, sizeof(e->param_buf), "%s", cr.param);
        if (!cr.param[0]) set_error(e, 500, "server_error", NULL, NULL, "%s", cr.message);
        else if (cr.unsupported) unsupported(e, e->param_buf, cr.message);
        else set_error(e, 400, "invalid_request_error", "invalid_value", e->param_buf, "%s", cr.message);
        goto done;
    }
    /* What the request asked for that the server left out, such as a tool
     * its API runs on its own side, is named as an ignored field is. */
    if (ignored && cr.ignored[0] && ignored->len < 400) {
        if (ignored->len) vitna_sb_puts(ignored, ", ");
        vitna_sb_puts(ignored, cr.ignored);
    }
    if (!api->qwen3) {
        ok = chatml_prompt(&cr, p->kind, prompt, e);
        goto done;
    }
    const bool has_tools = cr.tools.len > 0;
    const bool forced = cr.tool_choice == VITNA_TOOLS_REQUIRED || cr.tool_choice == VITNA_TOOLS_NAMED;
    const char* json_param = p->kind == API_RESPONSES ? "text" : "response_format";
    int thinking = cr.enable_thinking;
    ok = false;
    if (p->json && thinking == 1) {
        unsupported(e, json_param, "JSON mode holds the reply to a JSON object from its first token, so the model cannot think first");
        goto done;
    }
    if (p->json && forced) {
        unsupported(e, json_param, "JSON mode holds the reply to a JSON object, and a forced tool call is not one");
        goto done;
    }
    if (forced && thinking == 1) {
        set_error(e, 400, "invalid_request_error", "invalid_value", "tool_choice",
                  "a tool_choice that forces a tool call cannot be combined with thinking: the call is written from the reply's first token");
        goto done;
    }
    if (p->json || forced) thinking = 0;
    md = vitna_json_parse(cr.messages.data, cr.messages.len, err, sizeof(err));
    td = has_tools ? vitna_json_parse(cr.tools.data, cr.tools.len, err, sizeof(err)) : NULL;
    if (!md || (has_tools && !td)) {
        set_error(e, 500, "server_error", NULL, NULL, "out of memory");
        goto done;
    }
    if (!vitna_chat_qwen3(prompt, vitna_json_root(md), td ? vitna_json_root(td) : NULL, thinking, !cr.continue_final, err, sizeof(err))) {
        set_error(e, 400, "invalid_request_error", "invalid_value", p->kind == API_RESPONSES ? "input" : "messages", "%s", err);
        goto done;
    }
    if (cr.continue_final) {
        /* The last message, the assistant's, left open to be continued. */
        static const char end[] = "<|im_end|>\n";
        const size_t k = sizeof(end) - 1;
        if (prompt->len < k || memcmp(prompt->data + prompt->len - k, end, k) != 0) {
            set_error(e, 500, "server_error", NULL, NULL, "the chat template did not end the last message as it should");
            goto done;
        }
        prompt->len -= k;
        prompt->data[prompt->len] = '\0';
    }
    if (forced) {
        forced_opening(prefill, &cr);
        vitna_sb_append(prompt, prefill->data, prefill->len);
    }
    p->parse = true;
    p->think = thinking != 0 && !cr.continue_final;
    p->tools = has_tools && cr.tool_choice != VITNA_TOOLS_NONE && !p->json;
    p->ban = has_tools && cr.tool_choice == VITNA_TOOLS_NONE ? api->tool_call_token : -1;
    p->prefill = prefill->data;
    p->prefill_len = prefill->len;
    ok = prompt->ok && prefill->ok;
    if (!ok) set_error(e, 500, "server_error", NULL, NULL, "out of memory");
done:
    vitna_json_free(md);
    vitna_json_free(td);
    vitna_chat_request_free(&cr);
    return ok;
}

static bool completion_prompt(const vitna_api_t* api, const vitna_json_value_t* prompt, vitna_token_list_t* ids, api_error_t* e) {
    if (prompt && prompt->type == VITNA_JSON_ARRAY && prompt->u.array.count == 1 && prompt->u.array.items[0]->type == VITNA_JSON_STRING) {
        prompt = prompt->u.array.items[0];
    }
    if (prompt && prompt->type == VITNA_JSON_STRING) {
        if (!vitna_tokenizer_encode(api->tok, prompt->u.string.ptr, prompt->u.string.len, ids)) {
            set_error(e, 500, "server_error", NULL, NULL, "out of memory");
            return false;
        }
        return true;
    }
    if (prompt && prompt->type == VITNA_JSON_ARRAY && prompt->u.array.count > 0) {
        for (size_t i = 0; i < prompt->u.array.count; i++) {
            double d;
            if (!vitna_json_as_number(prompt->u.array.items[i], &d) || !integral(d) || d < 0 || d >= (double)api->model->cfg.vocab) {
                set_error(e, 400, "invalid_request_error", "invalid_value", "prompt",
                          "`prompt` must be a string, one string in an array, or an array of token ids below %zu", api->model->cfg.vocab);
                return false;
            }
            vitna_token_list_push(ids, (int32_t)d);
        }
        return true;
    }
    set_error(e, 400, "invalid_request_error", "invalid_value", "prompt",
              "`prompt` must be a string, one string in an array, or an array of token ids");
    return false;
}

/* --- Generation --- */

/* A part of a reply as it goes out: its reasoning or its text, gathered
 * while it lasts, or one tool call. A response sent whole is written from
 * the parts at the end; a stream sends each as it grows. */
typedef struct {
    vitna_reply_kind_t kind;
    vitna_strbuf_t data;    /* the reasoning or text, or the call's name */
    vitna_strbuf_t args;    /* a call's arguments, as JSON */
    bool open;              /* reasoning or text that may still grow */
    size_t call;            /* a call: which of the reply's calls it is */
} part_t;

typedef struct {
    vitna_api_t* api;
    vitna_sink_t* sink;
    const params_t* p;
    char id[48];
    uint64_t uid;           /* what the response's id, and its parts' and calls', are made from */
    long long created;
    vitna_strbuf_t text;    /* the reply: the prefill, then the generated bytes, cut at a stop sequence */
    size_t emitted;         /* bytes of text already streamed: a completion's, or a reply not read back */
    vitna_reply_parser_t rp;
    part_t* parts;
    size_t n_parts, cap_parts, n_calls;
    bool failed;            /* a part could not be kept: out of memory */
    size_t seq;             /* Responses: the events sent, so the next one's sequence_number */
    int stop;               /* the stop sequence that ended the reply, or -1 */
    size_t forced_at;       /* for tests: the forced tokens taken */
    bool window;            /* Anthropic's: the context's room, less than max_tokens, bounds the reply */
    bool think_closed;      /* </think> has been generated */
    size_t reasoning_tokens;/* the tokens up to and including the one that closed the reasoning */
} gen_t;

static void usage_json(vitna_strbuf_t* sb, const vitna_api_result_t* r) {
    vitna_sb_printf(sb, "{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,\"prompt_tokens_details\":{\"cached_tokens\":%zu}}",
                    r->prompt_tokens, r->completion_tokens, r->prompt_tokens + r->completion_tokens, r->cached_tokens);
}

static void put_string(vitna_strbuf_t* sb, const char* s, size_t n) {
    vitna_sb_json_string(sb, (const unsigned char*)(s ? s : ""), s ? n : 0);
}

/* One server-sent event. text is a delta, finish the finish reason or NULL,
 * usage NULL unless it goes on this chunk. role_only opens a chat stream. */
static void send_chunk(gen_t* g, const unsigned char* text, size_t n, const char* finish, const vitna_api_result_t* usage, bool role_only, bool usage_only) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_puts(&sb, "data: {\"id\":");
    vitna_sb_json_string(&sb, (const unsigned char*)g->id, strlen(g->id));
    vitna_sb_printf(&sb, ",\"object\":\"%s\",\"created\":%lld,\"model\":", g->p->chat ? "chat.completion.chunk" : "text_completion", g->created);
    vitna_sb_json_string(&sb, (const unsigned char*)g->api->model_id, strlen(g->api->model_id));
    if (usage_only) {
        vitna_sb_puts(&sb, ",\"choices\":[]");
    } else {
        vitna_sb_puts(&sb, ",\"choices\":[{\"index\":0,");
        if (g->p->chat) {
            if (role_only) vitna_sb_puts(&sb, "\"delta\":{\"role\":\"assistant\",\"content\":\"\"}");
            else if (finish) vitna_sb_puts(&sb, "\"delta\":{}");
            else {
                vitna_sb_puts(&sb, "\"delta\":{\"content\":");
                vitna_sb_json_string(&sb, text, n);
                vitna_sb_puts(&sb, "}");
            }
        } else {
            vitna_sb_puts(&sb, "\"text\":");
            vitna_sb_json_string(&sb, text ? text : (const unsigned char*)"", text ? n : 0);
        }
        vitna_sb_puts(&sb, ",\"logprobs\":null,\"finish_reason\":");
        if (finish) vitna_sb_printf(&sb, "\"%s\"", finish);
        else vitna_sb_puts(&sb, "null");
        vitna_sb_puts(&sb, "}]");
    }
    if (usage) {
        vitna_sb_puts(&sb, ",\"usage\":");
        usage_json(&sb, usage);
    }
    vitna_sb_puts(&sb, "}\n\n");
    sink_out(g->sink, sb.data, sb.len);
    vitna_sb_free(&sb);
}

/* A chat chunk whose delta holds the members given, as "content":"...". */
static void chat_chunk(gen_t* g, const vitna_strbuf_t* delta) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_puts(&sb, "data: {\"id\":");
    vitna_sb_json_string(&sb, (const unsigned char*)g->id, strlen(g->id));
    vitna_sb_printf(&sb, ",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":", g->created);
    vitna_sb_json_string(&sb, (const unsigned char*)g->api->model_id, strlen(g->api->model_id));
    vitna_sb_puts(&sb, ",\"choices\":[{\"index\":0,\"delta\":{");
    vitna_sb_append(&sb, delta->data, delta->len);
    vitna_sb_puts(&sb, "},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
    sink_out(g->sink, sb.data, sb.len);
    vitna_sb_free(&sb);
}

/* An event of Anthropic's stream: its type, then the members given. */
static void anthropic_event(gen_t* g, const char* type, const vitna_strbuf_t* members) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_printf(&sb, "event: %s\ndata: {\"type\":\"%s\"", type, type);
    if (members && members->len) {
        vitna_sb_puts(&sb, ",");
        vitna_sb_append(&sb, members->data, members->len);
    }
    vitna_sb_puts(&sb, "}\n\n");
    sink_out(g->sink, sb.data, sb.len);
    vitna_sb_free(&sb);
}

/* An event of a Responses stream: its type and sequence_number, then the members given. */
static void responses_event(gen_t* g, const char* type, const vitna_strbuf_t* members) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_printf(&sb, "event: %s\ndata: {\"type\":\"%s\",\"sequence_number\":%zu", type, type, g->seq++);
    if (members && members->len) {
        vitna_sb_puts(&sb, ",");
        vitna_sb_append(&sb, members->data, members->len);
    }
    vitna_sb_puts(&sb, "}\n\n");
    sink_out(g->sink, sb.data, sb.len);
    vitna_sb_free(&sb);
}

/* The id of a part of the reply (salt 1) or of a call (salt 2): a prefix
 * and 16 hex digits made from the response's own, so a stream's events and
 * the response it ends with name each the same. */
static void sub_id(char* buf, size_t len, const char* prefix, uint64_t uid, uint64_t salt, size_t i) {
    snprintf(buf, len, "%s%016llx", prefix, (unsigned long long)mix64(uid ^ (salt * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)(i + 1) << 20)));
}

static void call_id(const gen_t* g, const part_t* q, char* buf, size_t len) {
    sub_id(buf, len, g->p->kind == API_MESSAGES ? "toolu_" : "call_", g->uid, 2, q->call);
}

/* A Responses output item's id: rs_, msg_ or fc_ by its kind. */
static void item_id(const gen_t* g, size_t i, char* buf, size_t len) {
    const vitna_reply_kind_t kind = g->parts[i].kind;
    sub_id(buf, len, kind == VITNA_REPLY_REASONING ? "rs_" : kind == VITNA_REPLY_TEXT ? "msg_" : "fc_", g->uid, 1, i);
}

/* The text of a reply's parts of one kind, run together. */
static void parts_text(const gen_t* g, vitna_reply_kind_t kind, vitna_strbuf_t* out) {
    for (size_t i = 0; i < g->n_parts; i++) {
        if (g->parts[i].kind == kind) vitna_sb_append(out, g->parts[i].data.data, g->parts[i].data.len);
    }
}

static bool has_part(const gen_t* g, vitna_reply_kind_t kind) {
    for (size_t i = 0; i < g->n_parts; i++) {
        if (g->parts[i].kind == kind) return true;
    }
    return false;
}

/* --- Anthropic's Messages: a response's shape --- */

/* A content block: thinking, text or tool_use; at its start, empty. */
static void anthropic_block(const gen_t* g, size_t i, vitna_strbuf_t* b, bool start) {
    const part_t* q = &g->parts[i];
    if (q->kind == VITNA_REPLY_REASONING) {
        vitna_sb_puts(b, "{\"type\":\"thinking\",\"thinking\":");
        put_string(b, start ? "" : q->data.data, start ? 0 : q->data.len);
        vitna_sb_puts(b, ",\"signature\":\"\"}");
    } else if (q->kind == VITNA_REPLY_TEXT) {
        vitna_sb_puts(b, "{\"type\":\"text\",\"text\":");
        put_string(b, start ? "" : q->data.data, start ? 0 : q->data.len);
        vitna_sb_puts(b, "}");
    } else {
        char id[40];
        call_id(g, q, id, sizeof(id));
        vitna_sb_printf(b, "{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":", id);
        put_string(b, q->data.data, q->data.len);
        vitna_sb_puts(b, ",\"input\":");
        if (start) vitna_sb_puts(b, "{}");
        else vitna_sb_append(b, q->args.data, q->args.len);
        vitna_sb_puts(b, "}");
    }
}

/* Anthropic's usage: the prompt's tokens read from the cache counted apart
 * from the rest, as its API counts cache reads. */
static void anthropic_usage(vitna_strbuf_t* b, const vitna_api_result_t* r, bool start) {
    vitna_sb_printf(b, "{\"input_tokens\":%zu,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":%zu,\"output_tokens\":%zu}",
                    r->prompt_tokens - r->cached_tokens, r->cached_tokens, start ? (size_t)0 : r->completion_tokens);
}

static const char* anthropic_stop_reason(const gen_t* g, const char* finish) {
    if (strcmp(finish, "length") == 0) return g->window ? "model_context_window_exceeded" : "max_tokens";
    if (g->stop >= 0) return "stop_sequence";
    return g->n_calls ? "tool_use" : "end_turn";
}

static void anthropic_message(const gen_t* g, const vitna_api_result_t* r, const char* finish, vitna_strbuf_t* b) {
    vitna_sb_puts(b, "{\"id\":");
    put_string(b, g->id, strlen(g->id));
    vitna_sb_puts(b, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    put_string(b, g->api->model_id, strlen(g->api->model_id));
    vitna_sb_puts(b, ",\"content\":[");
    for (size_t i = 0; finish && i < g->n_parts; i++) {
        if (i) vitna_sb_puts(b, ",");
        anthropic_block(g, i, b, false);
    }
    vitna_sb_puts(b, "],\"stop_reason\":");
    if (finish) vitna_sb_printf(b, "\"%s\"", anthropic_stop_reason(g, finish));
    else vitna_sb_puts(b, "null");
    vitna_sb_puts(b, ",\"stop_sequence\":");
    if (finish && g->stop >= 0) put_string(b, g->p->stops[g->stop], g->p->stop_lens[g->stop]);
    else vitna_sb_puts(b, "null");
    vitna_sb_puts(b, ",\"usage\":");
    anthropic_usage(b, r, !finish);
    vitna_sb_puts(b, "}");
}

/* --- OpenAI's Responses: a response's shape --- */

static void output_text_part(vitna_strbuf_t* b, const char* text, size_t n) {
    vitna_sb_puts(b, "{\"type\":\"output_text\",\"text\":");
    put_string(b, text, n);
    vitna_sb_puts(b, ",\"annotations\":[],\"logprobs\":[]}");
}

/* An output item: reasoning, a message, or a function call; until done, without its content. */
static void responses_item(const gen_t* g, size_t i, vitna_strbuf_t* b, bool done) {
    const part_t* q = &g->parts[i];
    char id[40];
    item_id(g, i, id, sizeof(id));
    if (q->kind == VITNA_REPLY_REASONING) {
        vitna_sb_printf(b, "{\"type\":\"reasoning\",\"id\":\"%s\",\"summary\":[],\"content\":[", id);
        if (done) {
            vitna_sb_puts(b, "{\"type\":\"reasoning_text\",\"text\":");
            put_string(b, q->data.data, q->data.len);
            vitna_sb_puts(b, "}");
        }
        vitna_sb_puts(b, "]}");
    } else if (q->kind == VITNA_REPLY_TEXT) {
        vitna_sb_printf(b, "{\"type\":\"message\",\"id\":\"%s\",\"status\":\"%s\",\"role\":\"assistant\",\"content\":[", id, done ? "completed" : "in_progress");
        if (done) output_text_part(b, q->data.data, q->data.len);
        vitna_sb_puts(b, "]}");
    } else {
        char call[40];
        call_id(g, q, call, sizeof(call));
        vitna_sb_printf(b, "{\"type\":\"function_call\",\"id\":\"%s\",\"call_id\":\"%s\",\"name\":", id, call);
        put_string(b, q->data.data, q->data.len);
        /* A function of a namespace in the request is called by its own
         * name, with the namespace beside it, as OpenAI's API returns one. */
        const vitna_json_value_t* ns = vitna_responses_namespace(vitna_json_get(g->p->root, "tools"), q->data.data, q->data.len);
        if (ns) {
            vitna_sb_puts(b, ",\"namespace\":");
            put_string(b, ns->u.string.ptr, ns->u.string.len);
        }
        vitna_sb_puts(b, ",\"arguments\":");
        put_string(b, done ? q->args.data : "", done ? q->args.len : 0);
        vitna_sb_printf(b, ",\"status\":\"%s\"}", done ? "completed" : "in_progress");
    }
}

/* A member of the request repeated in the response, or fallback without one. */
static void echo(vitna_strbuf_t* b, const vitna_json_value_t* v, const char* fallback) {
    if (v && !vitna_json_is_null(v)) vitna_json_write_py(b, v);
    else vitna_sb_puts(b, fallback);
}

/* The response object: as created (in_progress, before any output), as it
 * ends (completed, or incomplete when it ran out of tokens), or failed. */
static void responses_object(const gen_t* g, const vitna_api_result_t* r, const char* status, const api_error_t* err, vitna_strbuf_t* b) {
    const params_t* p = g->p;
    const vitna_json_value_t* root = p->root;
    const vitna_json_value_t* reasoning = vitna_json_get(root, "reasoning");
    vitna_sb_puts(b, "{\"id\":");
    put_string(b, g->id, strlen(g->id));
    vitna_sb_printf(b, ",\"object\":\"response\",\"created_at\":%lld,\"status\":\"%s\",\"background\":false,\"error\":", g->created, status);
    if (err) {
        vitna_sb_puts(b, "{\"code\":\"server_error\",\"message\":");
        put_string(b, err->message, strlen(err->message));
        vitna_sb_puts(b, "}");
    } else {
        vitna_sb_puts(b, "null");
    }
    vitna_sb_puts(b, ",\"incomplete_details\":");
    vitna_sb_puts(b, strcmp(status, "incomplete") == 0 ? "{\"reason\":\"max_output_tokens\"}" : "null");
    vitna_sb_puts(b, ",\"instructions\":");
    echo(b, vitna_json_get(root, "instructions"), "null");
    vitna_sb_puts(b, ",\"max_output_tokens\":");
    if (p->has_max_tokens) vitna_sb_printf(b, "%zu", p->max_tokens);
    else vitna_sb_puts(b, "null");
    vitna_sb_puts(b, ",\"model\":");
    put_string(b, g->api->model_id, strlen(g->api->model_id));
    vitna_sb_puts(b, ",\"output\":[");
    for (size_t i = 0; r && i < g->n_parts; i++) {
        if (i) vitna_sb_puts(b, ",");
        responses_item(g, i, b, true);
    }
    vitna_sb_puts(b, "],\"parallel_tool_calls\":");
    echo(b, vitna_json_get(root, "parallel_tool_calls"), "true");
    vitna_sb_puts(b, ",\"previous_response_id\":null,\"reasoning\":{\"effort\":");
    echo(b, reasoning && reasoning->type == VITNA_JSON_OBJECT ? vitna_json_get(reasoning, "effort") : NULL, "null");
    vitna_sb_puts(b, ",\"summary\":null},\"store\":false,\"temperature\":");
    echo(b, vitna_json_get(root, "temperature"), "1.0");
    vitna_sb_printf(b, ",\"text\":{\"format\":{\"type\":\"%s\"}},\"tool_choice\":", p->json ? "json_object" : "text");
    echo(b, vitna_json_get(root, "tool_choice"), "\"auto\"");
    vitna_sb_puts(b, ",\"tools\":");
    echo(b, vitna_json_get(root, "tools"), "[]");
    vitna_sb_puts(b, ",\"top_p\":");
    echo(b, vitna_json_get(root, "top_p"), "1.0");
    vitna_sb_puts(b, ",\"truncation\":\"disabled\",\"usage\":");
    if (r) {
        vitna_sb_printf(b,
                        "{\"input_tokens\":%zu,\"input_tokens_details\":{\"cached_tokens\":%zu},\"output_tokens\":%zu,\"output_tokens_details\":"
                        "{\"reasoning_tokens\":%zu},\"total_tokens\":%zu}",
                        r->prompt_tokens, r->cached_tokens, r->completion_tokens, g->reasoning_tokens, r->prompt_tokens + r->completion_tokens);
    } else {
        vitna_sb_puts(b, "null");
    }
    vitna_sb_puts(b, ",\"user\":null,\"metadata\":{}}");
}

/* --- A reply's parts as a stream sends them --- */

/* A part opening: Anthropic's content block, or Responses' output item. */
static void stream_open(gen_t* g, size_t i) {
    vitna_strbuf_t m;
    vitna_sb_init(&m);
    if (g->p->kind == API_MESSAGES) {
        vitna_sb_printf(&m, "\"index\":%zu,\"content_block\":", i);
        anthropic_block(g, i, &m, true);
        anthropic_event(g, "content_block_start", &m);
    } else if (g->p->kind == API_RESPONSES) {
        vitna_sb_printf(&m, "\"output_index\":%zu,\"item\":", i);
        responses_item(g, i, &m, false);
        responses_event(g, "response.output_item.added", &m);
        if (g->parts[i].kind == VITNA_REPLY_TEXT) {
            char id[40];
            item_id(g, i, id, sizeof(id));
            vitna_sb_clear(&m);
            vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"content_index\":0,\"part\":", id, i);
            output_text_part(&m, "", 0);
            responses_event(g, "response.content_part.added", &m);
        }
    }
    vitna_sb_free(&m);
}

/* Bytes of a part: reasoning, text, or a call's arguments, whole. */
static void stream_delta(gen_t* g, size_t i, const char* s, size_t n) {
    const part_t* q = &g->parts[i];
    vitna_strbuf_t m;
    vitna_sb_init(&m);
    char id[40];
    if (g->p->kind == API_CHAT) {
        if (q->kind == VITNA_REPLY_TOOL_CALL) {
            call_id(g, q, id, sizeof(id));
            vitna_sb_printf(&m, "\"tool_calls\":[{\"index\":%zu,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":", q->call, id);
            put_string(&m, q->data.data, q->data.len);
            vitna_sb_puts(&m, ",\"arguments\":");
            put_string(&m, s, n);
            vitna_sb_puts(&m, "}}]");
        } else {
            vitna_sb_puts(&m, q->kind == VITNA_REPLY_REASONING ? "\"reasoning_content\":" : "\"content\":");
            put_string(&m, s, n);
        }
        chat_chunk(g, &m);
    } else if (g->p->kind == API_MESSAGES) {
        const char* type = q->kind == VITNA_REPLY_REASONING ? "thinking_delta" : q->kind == VITNA_REPLY_TEXT ? "text_delta" : "input_json_delta";
        const char* field = q->kind == VITNA_REPLY_REASONING ? "thinking" : q->kind == VITNA_REPLY_TEXT ? "text" : "partial_json";
        vitna_sb_printf(&m, "\"index\":%zu,\"delta\":{\"type\":\"%s\",\"%s\":", i, type, field);
        put_string(&m, s, n);
        vitna_sb_puts(&m, "}");
        anthropic_event(g, "content_block_delta", &m);
    } else if (g->p->kind == API_RESPONSES) {
        item_id(g, i, id, sizeof(id));
        if (q->kind == VITNA_REPLY_TOOL_CALL) {
            vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"delta\":", id, i);
            put_string(&m, s, n);
            responses_event(g, "response.function_call_arguments.delta", &m);
        } else {
            vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"content_index\":0,\"delta\":", id, i);
            put_string(&m, s, n);
            if (q->kind == VITNA_REPLY_TEXT) vitna_sb_puts(&m, ",\"logprobs\":[]");
            responses_event(g, q->kind == VITNA_REPLY_TEXT ? "response.output_text.delta" : "response.reasoning_text.delta", &m);
        }
    }
    vitna_sb_free(&m);
}

/* A part closing: Anthropic's block stops; Responses' item is done, with its text whole. */
static void stream_close(gen_t* g, size_t i) {
    const part_t* q = &g->parts[i];
    vitna_strbuf_t m;
    vitna_sb_init(&m);
    if (g->p->kind == API_MESSAGES) {
        vitna_sb_printf(&m, "\"index\":%zu", i);
        anthropic_event(g, "content_block_stop", &m);
    } else if (g->p->kind == API_RESPONSES) {
        char id[40];
        item_id(g, i, id, sizeof(id));
        if (q->kind == VITNA_REPLY_TOOL_CALL) {
            vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"name\":", id, i);
            put_string(&m, q->data.data, q->data.len);
            vitna_sb_puts(&m, ",\"arguments\":");
            put_string(&m, q->args.data, q->args.len);
            responses_event(g, "response.function_call_arguments.done", &m);
        } else {
            vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"content_index\":0,\"text\":", id, i);
            put_string(&m, q->data.data, q->data.len);
            if (q->kind == VITNA_REPLY_TEXT) vitna_sb_puts(&m, ",\"logprobs\":[]");
            responses_event(g, q->kind == VITNA_REPLY_TEXT ? "response.output_text.done" : "response.reasoning_text.done", &m);
            if (q->kind == VITNA_REPLY_TEXT) {
                vitna_sb_clear(&m);
                vitna_sb_printf(&m, "\"item_id\":\"%s\",\"output_index\":%zu,\"content_index\":0,\"part\":", id, i);
                output_text_part(&m, q->data.data, q->data.len);
                responses_event(g, "response.content_part.done", &m);
            }
        }
        vitna_sb_clear(&m);
        vitna_sb_printf(&m, "\"output_index\":%zu,\"item\":", i);
        responses_item(g, i, &m, true);
        responses_event(g, "response.output_item.done", &m);
    }
    vitna_sb_free(&m);
}

/* The last part, if it is reasoning or text still open, is done. */
static void close_part(gen_t* g) {
    if (!g->n_parts || !g->parts[g->n_parts - 1].open) return;
    g->parts[g->n_parts - 1].open = false;
    if (g->p->stream) stream_close(g, g->n_parts - 1);
}

static part_t* new_part(gen_t* g, vitna_reply_kind_t kind) {
    if (g->n_parts == g->cap_parts) {
        const size_t cap = g->cap_parts ? 2 * g->cap_parts : 4;
        part_t* parts = (part_t*)realloc(g->parts, cap * sizeof(part_t));
        if (!parts) {
            g->failed = true;
            return NULL;
        }
        g->parts = parts;
        g->cap_parts = cap;
    }
    part_t* q = &g->parts[g->n_parts++];
    memset(q, 0, sizeof(*q));
    q->kind = kind;
    vitna_sb_init(&q->data);
    vitna_sb_init(&q->args);
    return q;
}

/* A part of the reply as the parser reads it (vitna_reply_feed): kept, and
 * sent on in a stream. Reasoning or text joins the part before it while that
 * one is of its kind; a call is a part of its own, sent whole. */
static void on_part(void* ctx, const vitna_reply_part_t* rp) {
    gen_t* g = (gen_t*)ctx;
    const bool stream = g->p->stream;
    part_t* last = g->n_parts ? &g->parts[g->n_parts - 1] : NULL;
    if (rp->kind != VITNA_REPLY_TOOL_CALL && last && last->open && last->kind == rp->kind) {
        vitna_sb_append(&last->data, rp->data, rp->len);
        if (stream) stream_delta(g, g->n_parts - 1, rp->data, rp->len);
        return;
    }
    close_part(g);
    part_t* q = new_part(g, rp->kind);
    if (!q) return;
    const size_t i = g->n_parts - 1;
    vitna_sb_append(&q->data, rp->data, rp->len);
    if (rp->kind == VITNA_REPLY_TOOL_CALL) {
        vitna_sb_append(&q->args, rp->args, rp->args_len);
        q->call = g->n_calls++;
        if (stream) {
            stream_open(g, i);
            stream_delta(g, i, rp->args, rp->args_len);
            stream_close(g, i);
        }
        return;
    }
    q->open = true;
    if (stream) {
        stream_open(g, i);
        stream_delta(g, i, rp->data, rp->len);
    }
}

/* The last event of a stream that fails after its 200 has gone out, in the
 * shape of its API: an error event (OpenAI's chat and completions, and
 * Anthropic's), or Responses' response.failed. No final chunk, [DONE] or
 * message_stop follows it, so the reply so far cannot pass for a whole one. */
static void send_error_event(gen_t* g, const api_error_t* e) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    if (g->p->kind == API_RESPONSES) {
        vitna_sb_puts(&sb, "\"response\":");
        responses_object(g, NULL, "failed", e, &sb);
        responses_event(g, "response.failed", &sb);
    } else {
        vitna_sb_puts(&sb, g->p->kind == API_MESSAGES ? "event: error\ndata: " : "data: ");
        if (g->p->kind == API_MESSAGES) anthropic_error_json(&sb, e);
        else error_json(&sb, e);
        vitna_sb_puts(&sb, "\n\n");
        sink_out(g->sink, sb.data, sb.len);
    }
    vitna_sb_free(&sb);
}

/* Text that cannot yet be streamed: the longest end of it that begins a stop sequence. */
static size_t stop_holdback(const params_t* p, const char* text, size_t len) {
    size_t hold = 0;
    for (size_t i = 0; i < p->n_stops; i++) {
        size_t k = p->stop_lens[i] - 1 < len ? p->stop_lens[i] - 1 : len;
        for (; k > hold; k--) {
            if (memcmp(text + len - k, p->stops[i], k) == 0) {
                hold = k;
                break;
            }
        }
    }
    return hold;
}

/* Where the earliest stop sequence begins, looking only where bytes added
 * since from could complete one and never before floor (the prefill, which
 * the model did not write); -1 for none, else with *which the sequence. */
static long long find_stop(const params_t* p, const char* text, size_t len, size_t from, size_t floor, int* which) {
    long long best = -1;
    for (size_t i = 0; i < p->n_stops; i++) {
        size_t sl = p->stop_lens[i];
        size_t start = from >= sl ? from - sl + 1 : 0;
        if (start < floor) start = floor;
        for (size_t q = start; q + sl <= len; q++) {
            if (memcmp(text + q, p->stops[i], sl) == 0) {
                if (best < 0 || (long long)q < best) {
                    best = (long long)q;
                    *which = (int)i;
                }
                break;
            }
        }
    }
    return best;
}

/* A completion's text as far as it can be streamed. */
static void stream_ready(gen_t* g, bool final) {
    size_t safe = g->text.len;
    if (!final) {
        safe -= stop_holdback(g->p, g->text.data, g->text.len);
        safe = vitna_utf8_complete_prefix((const unsigned char*)g->text.data, safe);
    }
    if (safe > g->emitted) {
        send_chunk(g, (const unsigned char*)g->text.data + g->emitted, safe - g->emitted, NULL, NULL, false, false);
        g->emitted = safe;
    }
}

/* A conversation's reply as far as it can be read: through the parser into
 * reasoning, text and calls, or, for a template whose replies are not read
 * back, as text. Without final, what could still begin a stop sequence, or
 * end inside a character, waits. */
static void reply_ready(gen_t* g, bool final) {
    size_t safe = g->text.len;
    if (!final) {
        safe -= stop_holdback(g->p, g->text.data, g->text.len);
        safe = vitna_utf8_complete_prefix((const unsigned char*)g->text.data, safe);
    }
    if (g->p->parse) {
        vitna_reply_feed(&g->rp, g->text.data ? g->text.data : "", safe, final, on_part, g);
    } else if (safe > g->emitted) {
        const vitna_reply_part_t part = { VITNA_REPLY_TEXT, g->text.data + g->emitted, safe - g->emitted, NULL, 0 };
        on_part(g, &part);
        g->emitted = safe;
    }
}

/* JSON mode: set the logit of every token that could not continue a JSON
 * object to minus infinity, so neither greedy decoding nor sampling can
 * choose it. Special tokens are masked too; the object's closing brace ends
 * the reply instead. Returns false when no token is left.
 *
 * Which tokens can follow depends only on the automaton's state, so a
 * state's answer is kept (MASK_CACHE) and serves again when the state comes
 * round again: the same tokens masked, found once. */
static bool mask_json_locked(vitna_api_t* api, float* row, const vitna_jsonpfx_t* state) {
    const size_t V = api->model->cfg.vocab;
    const size_t words = (V + 31) / 32;
    vitna_jsonpfx_t key = *state;
    memset(key.stack + key.depth, 0, sizeof(key.stack) - key.depth); /* what a pop leaves behind means nothing */
    mask_entry_t* e = NULL;
    mask_entry_t* spare = &api->masks[0];
    for (size_t i = 0; i < MASK_CACHE; i++) {
        mask_entry_t* c = &api->masks[i];
        if (c->allowed && memcmp(&c->key, &key, sizeof(key)) == 0) {
            e = c;
            break;
        }
        /* The slot to fill if the state is new: an empty one, else the one unused longest. */
        if (spare->allowed && (!c->allowed || c->used < spare->used)) spare = c;
    }
    if (!e) {
        if (api->mask_cache && !spare->allowed) spare->allowed = (uint32_t*)malloc(words * sizeof(uint32_t));
        if (!api->mask_cache || !spare->allowed) {
            /* Not kept, for a test, or nowhere to keep it: mask directly. */
            uint8_t allowed[32];
            vitna_jsonpfx_next_bytes(state, allowed);
            size_t left = 0;
            for (size_t id = 0; id < V; id++) {
                size_t n = 0;
                const unsigned char* b = vitna_tokenizer_token_bytes(api->tok, (int32_t)id, &n);
                bool ok = b && n > 0 && !vitna_tokenizer_is_special(api->tok, (int32_t)id) &&
                          (allowed[b[0] >> 3] & (1u << (b[0] & 7))) && vitna_jsonpfx_accepts(state, b, n);
                if (ok) left++;
                else row[id] = -INFINITY;
            }
            return left > 0;
        }
        e = spare;
        memset(e->allowed, 0, words * sizeof(uint32_t));
        e->left = 0;
        uint8_t allowed[32];
        vitna_jsonpfx_next_bytes(state, allowed);
        for (size_t id = 0; id < V; id++) {
            size_t n = 0;
            const unsigned char* b = vitna_tokenizer_token_bytes(api->tok, (int32_t)id, &n);
            bool ok = b && n > 0 && !vitna_tokenizer_is_special(api->tok, (int32_t)id) &&
                      (allowed[b[0] >> 3] & (1u << (b[0] & 7))) && vitna_jsonpfx_accepts(state, b, n);
            if (ok) {
                e->allowed[id >> 5] |= 1u << (id & 31);
                e->left++;
            }
        }
        e->key = key;
    }
    e->used = ++api->mask_clock;
    for (size_t id = 0; id < V; id++) {
        if (!((e->allowed[id >> 5] >> (id & 31)) & 1u)) row[id] = -INFINITY;
    }
    return e->left > 0;
}

/* mask_json_locked, holding the lock the masks are kept under. A kept mask
 * is a function of its state alone, so which thread found it, and which
 * masks are kept, never change what is masked. */
static bool mask_json(vitna_api_t* api, float* row, const vitna_jsonpfx_t* state) {
    vitna_mutex_lock(&api->mask_lock);
    const bool left = mask_json_locked(api, row, state);
    vitna_mutex_unlock(&api->mask_lock);
    return left;
}

/* What taking a token needs, for take_token. */
typedef struct {
    vitna_api_t* api;
    gen_t* g;
    const params_t* p;
    const vitna_sampling_t* cfg;
    vitna_api_result_t* r;
    const char** finish;
    vitna_jsonpfx_t json;   /* JSON mode: the object so far */
} take_t;

enum { TAKE_GO, TAKE_END, TAKE_END_NONE };

/* One token from a row of logits, as a reply takes every one: the JSON mask,
 * the choice, the text, the stop checks and the stream. TAKE_GO to go on;
 * TAKE_END when the reply ends with this token (a special token, a stop
 * sequence or a closed JSON object), with *finish set; TAKE_END_NONE when the
 * JSON mask left no token to take. */
static int take_token(take_t* tk, vitna_sampler_t* sampler, float* row, int32_t* chosen) {
    const params_t* p = tk->p;
    gen_t* g = tk->g;
    if (p->json && !mask_json(tk->api, row, &tk->json)) {
        /* No token can continue the object. The pinned model's vocabulary
         * never gets here: the 21 bytes it lacks are control characters,
         * which JSON escapes, and UTF-8 lead bytes a string can do without. */
        *tk->finish = "stop";
        return TAKE_END_NONE;
    }
    if (p->ban >= 0) row[p->ban] = -INFINITY;
    int32_t next;
    if (p->forced) {
        /* For tests: the reply given, a token at a time, then its end, as
         * the model's own end-of-text token would end it. */
        if (g->forced_at == p->n_forced) {
            *chosen = 0;
            tk->r->completion_tokens++;
            *tk->finish = "stop";
            return TAKE_END;
        }
        next = p->forced[g->forced_at++];
    } else {
        next = vitna_sample(sampler, row, tk->cfg);
    }
    *chosen = next;
    tk->r->completion_tokens++;
    if (vitna_tokenizer_is_special(tk->api->tok, next)) {
        *tk->finish = "stop";
        return TAKE_END;
    }
    size_t n = 0;
    const unsigned char* bytes = vitna_tokenizer_token_bytes(tk->api->tok, next, &n);
    size_t before = g->text.len;
    if (bytes) {
        vitna_sb_append(&g->text, bytes, n);
        if (p->json) vitna_jsonpfx_feed(&tk->json, bytes, n);
    }
    long long cut = find_stop(p, g->text.data ? g->text.data : "", g->text.len, before, p->prefill_len, &g->stop);
    if (cut >= 0) {
        g->text.len = (size_t)cut;
        *tk->finish = "stop";
        return TAKE_END;
    }
    if (p->json && vitna_jsonpfx_complete(&tk->json)) {
        *tk->finish = "stop"; /* the object has closed, and nothing may follow it */
        return TAKE_END;
    }
    if (p->think && !g->think_closed) {
        /* The reasoning's tokens: those up to the one that closes it. */
        static const char close[] = "</think>";
        const size_t k = sizeof(close) - 1;
        for (size_t i = before >= k - 1 ? before - (k - 1) : 0; i + k <= g->text.len && !g->think_closed; i++) {
            if (memcmp(g->text.data + i, close, k) == 0) {
                g->think_closed = true;
                g->reasoning_tokens = tk->r->completion_tokens;
            }
        }
    }
    if (p->stream) {
        if (p->chat) reply_ready(g, false);
        else stream_ready(g, false);
    }
    return TAKE_GO;
}

/* --- Drafting (vitna_drafter_t) --- */

void vitna_drafter_init(vitna_drafter_t* d, size_t k) {
    d->k = k;
    d->skip = 0;
    d->backoff = 1;
}

size_t vitna_drafter_draft(vitna_drafter_t* d, const int32_t* text, size_t len, int32_t* out) {
    if (!d->k) return 0;
    if (d->skip) {
        d->skip--;
        return 0;
    }
    for (size_t n = 3; n >= 2; n--) {
        if (len <= n) continue;
        const int32_t* key = text + len - n;
        for (size_t i = len - n; i-- > 0;) {
            if (memcmp(text + i, key, n * sizeof(int32_t)) != 0) continue;
            size_t got = 0;
            while (got < d->k && i + n + got < len) {
                out[got] = text[i + n + got];
                got++;
            }
            if (got > 0) return got;
        }
    }
    return 0;
}

void vitna_drafter_taken(vitna_drafter_t* d, size_t taken) {
    if (taken == 0) {
        d->skip = d->backoff;
        d->backoff = d->backoff < 16 ? 2 * d->backoff : 16;
    } else {
        d->backoff = 1;
    }
}

/* --- Requests at once (the scheduler) --- */

/* A request that generates, from the connection that asked for it to the
 * scheduler that runs it. The connection fills the first part and queues
 * the job; the scheduler writes the response into out as a sink would, and
 * the connection sends it on (run_job). */
struct job {
    vitna_api_t* api;
    const params_t* p;
    const vitna_token_list_t* ids;
    const char* ignored;
    size_t max_new;

    /* Under api->lock. */
    vitna_strbuf_t out;     /* written by the scheduler, not yet sent */
    bool done;              /* the scheduler has finished with the job */
    bool gone;              /* the client went away, so writing more is pointless */
    vitna_cond_t cv;        /* signalled when out grows or done is set */
    job_t* next;            /* in the queue, then in the scheduler's list */

    /* The scheduler's. */
    vitna_sink_t sink;      /* writes to out (job_write) */
    size_t seq;             /* the sequence of the model's cache lent to it */
    bool begun, ended;
    bool prompting;         /* its prompt has more to run (job_prompt) */
    size_t prompt_at;       /* the prompt tokens its sequence holds */
    gen_t g;
    take_t tk;
    vitna_sampling_t cfg;
    uint64_t rng;           /* the sampler's state while this request's tokens are taken */
    vitna_strbuf_t headers;
    vitna_api_result_t r;
    const char* finish;
    bool ok;                /* every step so far ran */
    int st;                 /* the last take_token's answer */
    int32_t t;              /* the token taken last, which runs next */
    int32_t* text;          /* with speculation: the prompt and the tokens taken, for drafting */
    size_t text_len;
    vitna_drafter_t drafter;
    int32_t* pass;          /* with speculation: t and the tokens drafted after it */
    size_t d, row0, start;  /* this round: drafts, the job's first row, its sequence's position before */
    size_t passes, drafted, taken;
    uint64_t logits_hash;   /* for tests: FNV-1a over every row a token was taken from */
};

/* The job's sink: out, for the connection to send. Once the client has gone
 * a write fails, so the generation stops as it did when the scheduler wrote
 * to the socket itself. */
static bool job_write(void* ctx, const void* data, size_t len) {
    job_t* j = (job_t*)ctx;
    vitna_mutex_lock(&j->api->lock);
    bool ok = !j->gone;
    if (ok) {
        vitna_sb_append(&j->out, data, len);
        ok = j->out.ok;
        vitna_cond_signal(&j->cv);
    }
    vitna_mutex_unlock(&j->api->lock);
    return ok;
}

/* take_token with the sampler of the thread taking it, holding this
 * request's state: each request draws from its own stream, as it would
 * alone, whichever thread takes its tokens. */
static int take(job_t* j, vitna_sampler_t* sampler, float* row, int32_t* chosen) {
    if (j->api->test_logits) {
        /* The row as the model gave it, before JSON mode masks it. */
        const unsigned char* b = (const unsigned char*)row;
        const size_t n = j->api->model->cfg.vocab * sizeof(float);
        uint64_t h = j->logits_hash;
        for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ULL;
        j->logits_hash = h;
    }
    sampler->state = j->rng;
    const int st = take_token(&j->tk, sampler, row, chosen);
    j->rng = sampler->state;
    return st;
}

/* Room for a round's rows: one for each request, as many as the passes they
 * fill hold, or one request and k drafts. */
static bool round_capacity(vitna_api_t* api, size_t k) {
    const size_t cap = vitna_llama_exact_max(api->model);
    const size_t seqs = api->model->seqs;
    size_t n = (seqs + cap - 1) / cap * cap;
    if (n < k + 1) n = k + 1;
    if (n == api->rows_cap && api->rows) return true;
    vitna_llama_row_t* rows = (vitna_llama_row_t*)malloc(n * sizeof(vitna_llama_row_t));
    float* logits = (float*)malloc(n * api->model->cfg.vocab * sizeof(float));
    bool* ran = (bool*)malloc(n * sizeof(bool));
    if (!rows || !logits || !ran) {
        free(rows);
        free(logits);
        free(ran);
        return false;
    }
    free(api->rows);
    free(api->logits);
    free(api->ran);
    api->rows = rows;
    api->logits = logits;
    api->ran = ran;
    api->rows_cap = n;
    return true;
}

/* The sequence to lend a request, or SIZE_MAX with none free: of the free
 * ones, the one whose cache holds the longest prefix of its prompt, and of
 * those the one unused longest. Called under api->lock. */
static size_t pick_seq(vitna_api_t* api, const vitna_token_list_t* ids) {
    size_t best = SIZE_MAX, best_len = 0;
    for (size_t s = 0; s < api->model->seqs; s++) {
        const seq_t* q = &api->seqs[s];
        if (q->busy) continue;
        size_t n = 0;
        if (api->prefix_cache) {
            const size_t limit = ids->count - 1 < q->cached.count ? ids->count - 1 : q->cached.count;
            while (n < limit && q->cached.ids[n] == ids->ids[n]) n++;
        }
        if (best == SIZE_MAX || n > best_len || (n == best_len && q->used < api->seqs[best].used)) {
            best = s;
            best_len = n;
        }
    }
    return best;
}

/* A request's start, once it has a sequence: its headers, the prompt (what
 * the sequence already holds of it reused), and its first token. */
static void job_begin(vitna_api_t* api, job_t* j) {
    vitna_llama_t* m = api->model;
    const params_t* p = j->p;
    const vitna_token_list_t* ids = j->ids;
    seq_t* q = &api->seqs[j->seq];
    j->begun = true;
    j->r.status = 200;
    j->r.prompt_tokens = ids->count;

    gen_t* g = &j->g;
    g->api = api;
    g->sink = &j->sink;
    g->p = p;
    g->created = (long long)time(NULL);
    g->uid = fresh_u64(api);
    static const char* const id_prefix[] = { "cmpl-", "chatcmpl-", "msg_", "resp_" };
    snprintf(g->id, sizeof(g->id), "%s%016llx", id_prefix[p->kind], (unsigned long long)g->uid);
    vitna_sb_init(&g->text);
    if (p->prefill_len) vitna_sb_append(&g->text, p->prefill, p->prefill_len);
    g->stop = -1;
    g->window = p->kind == API_MESSAGES && p->has_max_tokens && j->max_new < p->max_tokens;
    vitna_reply_init(&g->rp, p->think, p->tools);

    const uint64_t seed = p->has_seed ? p->seed : fresh_u64(api);
    j->rng = seed;
    j->cfg.temperature = p->temperature;
    j->cfg.top_k = p->top_k;
    j->cfg.top_p = p->top_p;
    j->cfg.seed = seed;

    vitna_sb_init(&j->headers);
    vitna_sb_printf(&j->headers, "x-vitna-seed: %llu\r\n", (unsigned long long)seed);
    if (j->ignored && *j->ignored) vitna_sb_printf(&j->headers, "x-vitna-ignored: %s\r\n", j->ignored);
    if (p->has_test_prompt) vitna_sb_printf(&j->headers, "x-vitna-test-prompt: %016llx\r\n", (unsigned long long)p->test_prompt);

    if (p->stream) {
        vitna_strbuf_t h;
        vitna_sb_init(&h);
        vitna_sb_printf(&h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n%s\r\n", j->headers.data);
        sink_out(&j->sink, h.data, h.len);
        vitna_sb_free(&h);
    }

    /* Reuse the longest prefix the sequence's cache already holds from the
     * request it served before. The last prompt token is always fed again,
     * for the logits that choose the first new token. */
    size_t reuse = 0;
    if (api->prefix_cache) {
        const size_t limit = ids->count - 1 < q->cached.count ? ids->count - 1 : q->cached.count;
        while (reuse < limit && q->cached.ids[reuse] == ids->ids[reuse]) reuse++;
    }
    vitna_llama_truncate(m, j->seq, reuse);
    q->cached.count = reuse;
    j->r.cached_tokens = reuse;

    /* A stream opens: chat's first chunk names the role; Anthropic's
     * message_start carries the usage of the prompt, cache reads apart; a
     * Responses stream says the response is created and in progress. */
    if (p->stream) {
        vitna_strbuf_t ev;
        vitna_sb_init(&ev);
        if (p->kind == API_CHAT) {
            send_chunk(g, NULL, 0, NULL, NULL, true, false);
        } else if (p->kind == API_MESSAGES) {
            vitna_sb_puts(&ev, "\"message\":");
            anthropic_message(g, &j->r, NULL, &ev);
            anthropic_event(g, "message_start", &ev);
        } else if (p->kind == API_RESPONSES) {
            vitna_sb_puts(&ev, "\"response\":");
            responses_object(g, NULL, "in_progress", NULL, &ev);
            responses_event(g, "response.created", &ev);
            responses_event(g, "response.in_progress", &ev);
        }
        vitna_sb_free(&ev);
    }

    /* The rest of the prompt runs next (job_prompt), then a token at a time.
     * ok stays true while every step runs. */
    j->finish = "length";
    j->prompt_at = reuse;
    j->prompting = true;
    j->logits_hash = 0xcbf29ce484222325ULL;
    j->ok = true;
    j->tk.api = api;
    j->tk.g = g;
    j->tk.p = p;
    j->tk.cfg = &j->cfg;
    j->tk.r = &j->r;
    j->tk.finish = &j->finish;
    vitna_jsonpfx_init(&j->tk.json);

    /* With speculation, the text so far, for drafting: the prompt, then each
     * token taken. */
    const size_t spec = api->speculate;
    if (spec) {
        j->text = (int32_t*)malloc((ids->count + j->max_new) * sizeof(int32_t));
        j->pass = (int32_t*)malloc((spec + 1) * sizeof(int32_t));
        if (!j->text || !j->pass) {
            free(j->text);
            free(j->pass);
            j->text = NULL;
            j->pass = NULL;
        } else {
            memcpy(j->text, ids->ids, ids->count * sizeof(int32_t));
            j->text_len = ids->count;
        }
    }
    vitna_drafter_init(&j->drafter, j->text ? spec : 0);
}

/* The next piece of a request's prompt: the rest of it when whole, or when
 * too little is left to split; otherwise PROMPT_PIECE tokens, leaving at
 * least half that and never fewer than vitna_llama_prompt_piece_min, so the
 * pieces run exactly as the prompt would in one call. After the last, the
 * request takes its first token.
 *
 * A step that fails, as one on a GPU does when the device reports an error,
 * ends the request with an error rather than with the reply so far, and its
 * token never joins the sequence's cached list; the prompt tokens that ran
 * before it do. */
static void job_prompt(vitna_api_t* api, job_t* j, bool whole) {
    vitna_llama_t* m = api->model;
    const vitna_token_list_t* ids = j->ids;
    seq_t* q = &api->seqs[j->seq];
    const size_t left = ids->count - j->prompt_at;
    size_t n = left;
    if (!whole) {
        const size_t min = vitna_llama_prompt_piece_min(m);
        const size_t piece = min > 1 ? PROMPT_PIECE : PROMPT_PIECE_STEPS;
        /* A piece only when what is left after it is at least half one, so
         * a prompt a little longer than a piece is not split for nothing. */
        const size_t after = piece / 2 > min ? piece / 2 : min;
        if (left >= piece + after) n = piece;
    }
    const bool last = n == left;
    const size_t ran = vitna_llama_steps(m, j->seq, ids->ids + j->prompt_at, n, last ? api->row : NULL, last ? 1 : 0);
    for (size_t t = 0; t < ran; t++) vitna_token_list_push(&q->cached, ids->ids[j->prompt_at + t]);
    j->prompt_at += ran;
    if (ran < n) {
        j->ok = false;
        j->prompting = false;
        j->ended = true;
        return;
    }
    if (!last) return;
    j->prompting = false;

    /* Every token is taken as one step at a time takes it, and a token taken
     * then needs the logits after it: from a step, or from a pass over it and
     * the tokens drafted after it, whose rows serve while the tokens taken
     * agree with the drafts. So a reply is the one without drafts, and the
     * cache ends holding what it would hold: every token fed that one step at
     * a time would have fed, and no other. */
    j->st = j->max_new > 0 && !j->sink.failed ? take(j, &api->samplers[0], api->row, &j->t) : TAKE_END;
    if (j->text && j->st != TAKE_END_NONE) j->text[j->text_len++] = j->t;
    j->ended = !(j->st == TAKE_GO && j->r.completion_tokens < j->max_new);
}

/* After a round's rows ran: the request takes the tokens its rows give, as a
 * token at a time would. It touches only its own request, its own sequence
 * and the masks under their lock, so a round's requests can take their
 * tokens at once, each with a sampler of its own. */
static void job_take(vitna_api_t* api, job_t* j, vitna_sampler_t* sampler) {
    vitna_llama_t* m = api->model;
    seq_t* q = &api->seqs[j->seq];
    const size_t V = m->cfg.vocab;
    float* rows = api->logits + j->row0 * V;
    const bool* ran = api->ran + j->row0;
    if (j->d == 0) {
        j->ok = ran[0];
        if (!j->ok) {
            j->ended = true;
            return;
        }
        vitna_token_list_push(&q->cached, j->t);
        if (j->sink.failed) {
            j->ended = true;
            return;
        }
        j->st = take(j, sampler, rows, &j->t);
        if (j->text && j->st != TAKE_END_NONE) j->text[j->text_len++] = j->t;
    } else {
        j->passes++;
        j->drafted += j->d;
        size_t fed = 0;
        while (fed <= j->d && ran[fed]) fed++;
        if (fed < j->d + 1) {
            /* As a step that fails: the tokens before it stay, it does not. */
            for (size_t i = 0; i < fed; i++) vitna_token_list_push(&q->cached, j->pass[i]);
            j->ok = false;
            j->ended = true;
            return;
        }
        /* Row i holds the logits after j->pass[i], with j->pass[0..i] fed.
         * keep says how many of the pass's tokens one step at a time would
         * have fed by the time it stops using the pass. */
        size_t keep = 0;
        for (size_t i = 0;; i++) {
            if (j->sink.failed) {
                keep = i + 1;
                break;
            }
            int32_t u = 0;
            j->st = take(j, sampler, rows + i * V, &u);
            if (j->st != TAKE_END_NONE && j->text) j->text[j->text_len++] = u;
            if (j->st == TAKE_GO && j->r.completion_tokens < j->max_new && i < j->d && u == j->pass[i + 1]) continue;
            keep = i + 1;
            j->t = u;
            break;
        }
        j->taken += keep - 1;
        vitna_drafter_taken(&j->drafter, keep - 1);
        vitna_llama_truncate(m, j->seq, j->start + keep);
        for (size_t i = 0; i < keep; i++) vitna_token_list_push(&q->cached, j->pass[i]);
        if (j->sink.failed) {
            j->ended = true;
            return;
        }
    }
    j->ended = !(j->ok && j->st == TAKE_GO && j->r.completion_tokens < j->max_new);
}

/* A worker thread: it takes the requests of each round handed out, one at a
 * time, until none is left. */
typedef struct {
    vitna_api_t* api;
    size_t index; /* its sampler */
} worker_t;

static void worker(void* arg) {
    worker_t w = *(worker_t*)arg;
    free(arg);
    vitna_api_t* api = w.api;
    uint64_t seen = 0;
    vitna_mutex_lock(&api->pool_lock);
    for (;;) {
        while (!api->pool_stop && api->pool_round == seen) vitna_cond_wait(&api->pool_work, &api->pool_lock);
        if (api->pool_stop) break;
        seen = api->pool_round;
        while (api->pool_next < api->pool_n) {
            job_t* j = api->pool_jobs[api->pool_next++];
            vitna_mutex_unlock(&api->pool_lock);
            job_take(api, j, &api->samplers[w.index]);
            vitna_mutex_lock(&api->pool_lock);
            if (--api->pool_pending == 0) vitna_cond_signal(&api->pool_done);
        }
    }
    vitna_mutex_unlock(&api->pool_lock);
}

/* Start the workers, once; fewer than asked if threads run short, and none
 * is needed: the scheduler takes whatever they do not. */
static void start_workers(vitna_api_t* api) {
    while (api->workers_started + 1 < api->takers) {
        worker_t* w = (worker_t*)malloc(sizeof(worker_t));
        if (!w) break;
        w->api = api;
        w->index = api->workers_started + 1;
        if (!vitna_thread_start(&api->workers[api->workers_started], worker, w, false)) {
            free(w);
            break;
        }
        api->workers_started++;
    }
}

/* The round's requests take the tokens its rows gave them. A token taken by
 * sampling costs about as much as a step of the model (sorting the
 * vocabulary by probability), so when more than one request samples, or
 * masks for JSON mode, they take theirs on several threads at once; greedy
 * tokens cost little, and go in turn on the scheduler's. Which thread takes
 * a request's tokens changes nothing: its state is its own (take). */
static void take_all(vitna_api_t* api, job_t* running, size_t live) {
    size_t costly = 0, n = 0;
    for (job_t* j = running; j; j = j->next) {
        if (!j->ended && !j->prompting) costly += j->p->temperature > 0.0f || j->p->json;
    }
    if (live < 2 || costly < 2 || api->takers < 2) {
        for (job_t* j = running; j; j = j->next) {
            if (!j->ended && !j->prompting) job_take(api, j, &api->samplers[0]);
        }
        return;
    }
    if (api->workers_started + 1 < api->takers) start_workers(api);
    for (job_t* j = running; j; j = j->next) {
        if (!j->ended && !j->prompting) api->pool_jobs[n++] = j;
    }
    vitna_mutex_lock(&api->pool_lock);
    api->pool_n = n;
    api->pool_next = 0;
    api->pool_pending = n;
    api->pool_round++;
    vitna_cond_broadcast(&api->pool_work);
    while (api->pool_next < api->pool_n) {
        job_t* j = api->pool_jobs[api->pool_next++];
        vitna_mutex_unlock(&api->pool_lock);
        job_take(api, j, &api->samplers[0]);
        vitna_mutex_lock(&api->pool_lock);
        api->pool_pending--;
    }
    while (api->pool_pending > 0) vitna_cond_wait(&api->pool_done, &api->pool_lock);
    vitna_mutex_unlock(&api->pool_lock);
}

/* One round: every running request's next token, and any drafts after it,
 * as rows of one vitna_llama_step_rows, which runs them in passes and gives
 * each row the logits a step would. Drafts take only the rows the passes
 * have room for beyond one a request, so they add no pass; a request running
 * alone drafts as many as --speculate allows. */
static void run_round(vitna_api_t* api, job_t* running) {
    vitna_llama_t* m = api->model;
    size_t live = 0;
    for (job_t* j = running; j; j = j->next) live += !j->ended && !j->prompting;
    if (live == 0) return;
    const size_t cap = vitna_llama_exact_max(m);
    size_t spare = live == 1 ? api->speculate : (live + cap - 1) / cap * cap - live;
    size_t n = 0;
    for (job_t* j = running; j; j = j->next) {
        if (j->ended || j->prompting) continue;
        const size_t past = m->past[j->seq];
        size_t d = 0;
        if (j->text && spare > 0) {
            d = vitna_drafter_draft(&j->drafter, j->text, j->text_len, j->pass + 1);
            if (d > spare) d = spare;
            if (d) {
                const size_t left = j->max_new - j->r.completion_tokens; /* tokens still to take, 1 or more */
                if (d > left - 1) d = left - 1;
                if (past + d + 1 > m->ctx) d = past + 1 < m->ctx ? m->ctx - past - 1 : 0;
                /* A step a test makes fail (vitna_llama_fail_step_once) fails
                 * only where one step at a time would have run it: a pass stops
                 * short of it, and the step itself runs alone. */
                if (m->fail_armed && m->fail_at >= past && m->fail_at <= past + d) d = m->fail_at > past ? m->fail_at - past - 1 : 0;
            }
            spare -= d;
        }
        j->d = d;
        j->row0 = n;
        j->start = past;
        if (j->pass) j->pass[0] = j->t;
        api->rows[n].seq = j->seq;
        api->rows[n++].token = j->t;
        for (size_t i = 1; i <= d; i++) {
            api->rows[n].seq = j->seq;
            api->rows[n++].token = j->pass[i];
        }
    }
    vitna_llama_step_rows(m, api->rows, n, api->logits, api->ran);
    take_all(api, running, live);
}

/* Chat's finish reason: tool_calls when the reply ended on its own after calling one. */
static const char* chat_finish(const gen_t* g, const char* finish) {
    return g->n_calls && strcmp(finish, "stop") == 0 ? "tool_calls" : finish;
}

/* A Responses response's status at its end. */
static const char* responses_status(const char* finish) {
    return strcmp(finish, "length") == 0 ? "incomplete" : "completed";
}

/* The end of a stream that ran to its end. */
static void stream_end(gen_t* g, job_t* j) {
    const params_t* p = j->p;
    vitna_strbuf_t ev;
    vitna_sb_init(&ev);
    if (p->kind == API_MESSAGES) {
        vitna_sb_printf(&ev, "\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":", anthropic_stop_reason(g, j->finish));
        if (g->stop >= 0) put_string(&ev, p->stops[g->stop], p->stop_lens[g->stop]);
        else vitna_sb_puts(&ev, "null");
        vitna_sb_printf(&ev, "},\"usage\":{\"output_tokens\":%zu}", j->r.completion_tokens);
        anthropic_event(g, "message_delta", &ev);
        anthropic_event(g, "message_stop", NULL);
    } else if (p->kind == API_RESPONSES) {
        const char* status = responses_status(j->finish);
        vitna_sb_puts(&ev, "\"response\":");
        responses_object(g, &j->r, status, NULL, &ev);
        responses_event(g, strcmp(status, "completed") == 0 ? "response.completed" : "response.incomplete", &ev);
    } else {
        if (!p->chat) stream_ready(g, true);
        send_chunk(g, NULL, 0, p->chat ? chat_finish(g, j->finish) : j->finish, &j->r, false, false);
        if (p->include_usage) send_chunk(g, NULL, 0, NULL, &j->r, false, true);
        sink_out(&j->sink, "data: [DONE]\n\n", 14);
    }
    vitna_sb_free(&ev);
}

/* The body of a response sent whole. */
static void response_body(const gen_t* g, const job_t* j, vitna_strbuf_t* b) {
    const params_t* p = j->p;
    if (p->kind == API_MESSAGES) {
        anthropic_message(g, &j->r, j->finish, b);
        return;
    }
    if (p->kind == API_RESPONSES) {
        responses_object(g, &j->r, responses_status(j->finish), NULL, b);
        return;
    }
    vitna_sb_puts(b, "{\"id\":");
    vitna_sb_json_string(b, (const unsigned char*)g->id, strlen(g->id));
    vitna_sb_printf(b, ",\"object\":\"%s\",\"created\":%lld,\"model\":", p->chat ? "chat.completion" : "text_completion", g->created);
    vitna_sb_json_string(b, (const unsigned char*)g->api->model_id, strlen(g->api->model_id));
    vitna_sb_puts(b, ",\"choices\":[{\"index\":0,");
    if (p->chat && p->parse) {
        /* Read back: the text, null when the reply is only calls; the
         * reasoning, if any, beside it; then the calls. */
        vitna_strbuf_t t;
        vitna_sb_init(&t);
        vitna_sb_puts(b, "\"message\":{\"role\":\"assistant\",\"content\":");
        parts_text(g, VITNA_REPLY_TEXT, &t);
        if (!has_part(g, VITNA_REPLY_TEXT) && g->n_calls) vitna_sb_puts(b, "null");
        else put_string(b, t.data, t.len);
        if (has_part(g, VITNA_REPLY_REASONING)) {
            vitna_sb_clear(&t);
            parts_text(g, VITNA_REPLY_REASONING, &t);
            vitna_sb_puts(b, ",\"reasoning_content\":");
            put_string(b, t.data, t.len);
        }
        vitna_sb_free(&t);
        if (g->n_calls) {
            vitna_sb_puts(b, ",\"tool_calls\":[");
            for (size_t i = 0, k = 0; i < g->n_parts; i++) {
                const part_t* q = &g->parts[i];
                if (q->kind != VITNA_REPLY_TOOL_CALL) continue;
                char id[40];
                call_id(g, q, id, sizeof(id));
                vitna_sb_printf(b, "%s{\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":", k++ ? "," : "", id);
                put_string(b, q->data.data, q->data.len);
                vitna_sb_puts(b, ",\"arguments\":");
                put_string(b, q->args.data, q->args.len);
                vitna_sb_puts(b, "}}");
            }
            vitna_sb_puts(b, "]");
        }
        vitna_sb_puts(b, "}");
    } else if (p->chat) {
        vitna_sb_puts(b, "\"message\":{\"role\":\"assistant\",\"content\":");
        vitna_sb_json_string(b, (const unsigned char*)(g->text.data ? g->text.data : ""), g->text.len);
        vitna_sb_puts(b, "}");
    } else {
        vitna_sb_puts(b, "\"text\":");
        vitna_sb_json_string(b, (const unsigned char*)(g->text.data ? g->text.data : ""), g->text.len);
    }
    vitna_sb_printf(b, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":", p->chat ? chat_finish(g, j->finish) : j->finish);
    usage_json(b, &j->r);
    vitna_sb_puts(b, "}");
}

/* A request's end: the rest of its response. */
static void job_end(vitna_api_t* api, job_t* j) {
    const params_t* p = j->p;
    gen_t* g = &j->g;
    api_error_t e;
    /* One line a request on stderr while drafting is on, as generate --timing
     * prints it: what drafting did, which never changes the response. */
    if (j->text) fprintf(stderr, "speculation: %zu passes, %zu tokens drafted, %zu of them taken\n", j->passes, j->drafted, j->taken);
    free(j->text);
    free(j->pass);
    j->text = NULL;
    j->pass = NULL;

    if (j->ok && g->text.ok && p->chat) {
        /* The rest of the reply, read to its end. */
        reply_ready(g, true);
        close_part(g);
        /* Reasoning tokens are counted only for a reply that opened with
         * <think>: to the token that closed it, or every one if none did. */
        if (!(p->think && g->text.len >= 7 && memcmp(g->text.data, "<think>", 7) == 0)) g->reasoning_tokens = 0;
        else if (!g->think_closed) g->reasoning_tokens = j->r.completion_tokens;
        for (size_t i = 0; i < g->n_parts; i++) g->failed = g->failed || !g->parts[i].data.ok || !g->parts[i].args.ok;
    }
    if (!j->ok) {
        set_error(&e, 500, "server_error", NULL, NULL, "The model failed to run a step, so the reply could not be completed.");
        if (p->stream) send_error_event(g, &e);
        else j->r.status = respond_error_as(&j->sink, &e, p->kind);
    } else if (!g->text.ok || g->failed) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        if (p->stream) send_error_event(g, &e);
        else j->r.status = respond_error_as(&j->sink, &e, p->kind);
    } else if (p->stream) {
        stream_end(g, j);
    } else {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        response_body(g, j, &b);
        if (api->test_logits) vitna_sb_printf(&j->headers, "x-vitna-test-logits: %016llx\r\n", (unsigned long long)j->logits_hash);
        respond(&j->sink, 200, j->headers.data, b.data, b.len);
        vitna_sb_free(&b);
    }
    for (size_t i = 0; i < g->n_parts; i++) {
        vitna_sb_free(&g->parts[i].data);
        vitna_sb_free(&g->parts[i].args);
    }
    free(g->parts);
    g->parts = NULL;
    vitna_sb_free(&j->headers);
    vitna_sb_free(&g->text);
}

/* What a request is refused with once the GPU can run nothing more (lose).
 * The model's device_error was written before lost was set, and is not
 * written again. */
static void lost_error(const vitna_api_t* api, api_error_t* e) {
    set_error(e, 503, "server_error", "device_lost", NULL,
              "The GPU can run nothing more in this process after this error: %s. The server is stopping; start it again to use the GPU.",
              api->model->device_error);
}

/* After a step that lost the GPU (vitna_llama_t.device_lost), which only a
 * new process can use again: every request still running ends with the
 * error a failed step gives, every one waiting for a sequence is refused, as
 * is every one that comes later (run_job), and the server stops taking
 * connections (vitna_api_lost). */
static void lose(vitna_api_t* api, job_t* running) {
    fprintf(stderr,
            "The GPU can run nothing more in this process after this error: %s. Each request still open is answered with an error, "
            "and the server stops: start it again to use the GPU.\n",
            api->model->device_error);
    vitna_mutex_lock(&api->lock);
    api->lost = true;
    job_t* waiting = api->queue;
    api->queue = NULL;
    api->queue_tail = NULL;
    vitna_mutex_unlock(&api->lock);
    for (job_t* j = running; j; j = j->next) {
        if (!j->ended) {
            j->ok = false;
            j->ended = true;
        }
    }
    while (waiting) {
        job_t* j = waiting;
        waiting = j->next; /* first: once done is set, its connection frees j */
        api_error_t e;
        lost_error(api, &e);
        j->r.prompt_tokens = j->ids->count; /* as run_job reports a request it refuses */
        j->r.status = respond_error_as(&j->sink, &e, j->p->kind);
        vitna_mutex_lock(&api->lock);
        j->done = true;
        vitna_cond_signal(&j->cv);
        vitna_mutex_unlock(&api->lock);
    }
}

/* The scheduler's thread: it lends free sequences to waiting requests, oldest
 * first, begins each (its prompt), runs rounds over every request that has
 * one, and hands each back to its connection when it ends. */
static void scheduler(void* arg) {
    vitna_api_t* api = (vitna_api_t*)arg;
    const vitna_llama_t* m = api->model;
    job_t* running = NULL; /* the requests with a sequence, in the order they got one */
    for (;;) {
        vitna_mutex_lock(&api->lock);
        while (!api->stopping && !running && !api->queue) vitna_cond_wait(&api->work, &api->lock);
        if (api->stopping) {
            vitna_mutex_unlock(&api->lock);
            return;
        }
        job_t** tail = &running;
        while (*tail) tail = &(*tail)->next;
        while (api->queue) {
            const size_t s = pick_seq(api, api->queue->ids);
            if (s == SIZE_MAX) break;
            job_t* j = api->queue;
            api->queue = j->next;
            if (!api->queue) api->queue_tail = NULL;
            j->next = NULL;
            j->seq = s;
            api->seqs[s].busy = true;
            api->seqs[s].used = ++api->seq_clock;
            *tail = j;
            tail = &j->next;
        }
        vitna_mutex_unlock(&api->lock);

        for (job_t* j = running; j; j = j->next) {
            if (!j->begun) job_begin(api, j);
        }
        /* Prompts run whole while no request is decoding, since then no one
         * waits for them; otherwise a piece of the oldest between rounds. */
        bool decoding = false;
        for (job_t* j = running; j; j = j->next) decoding = decoding || (!j->ended && !j->prompting);
        for (job_t* j = running; j && !m->device_lost; j = j->next) {
            if (j->ended || !j->prompting) continue;
            job_prompt(api, j, !decoding);
            if (decoding) break;
        }
        if (!m->device_lost) run_round(api, running);
        if (m->device_lost) lose(api, running);

        /* Hand back the requests that ended; a connection frees its job once
         * done is set, so each leaves the list first. */
        job_t** at = &running;
        while (*at) {
            job_t* j = *at;
            if (!j->ended) {
                at = &j->next;
                continue;
            }
            job_end(api, j);
            *at = j->next;
            vitna_mutex_lock(&api->lock);
            api->seqs[j->seq].busy = false;
            j->done = true;
            vitna_cond_signal(&j->cv);
            vitna_mutex_unlock(&api->lock);
        }
    }
}

/* A request that generates: checked here, on the connection's thread, then
 * queued for the scheduler, whose output this thread sends on as it comes. */
static vitna_api_result_t run_job(vitna_api_t* api, vitna_sink_t* sink, const params_t* p, const vitna_token_list_t* ids, const char* ignored) {
    vitna_api_result_t r = { 200, ids->count, 0, 0 };
    const vitna_llama_t* m = api->model;
    api_error_t e;
    /* The parameters a refusal names, in the request's own shape. */
    const char* conversation = p->kind == API_COMPLETIONS ? "prompt" : p->kind == API_RESPONSES ? "input" : "messages";
    const char* max_param = p->kind == API_RESPONSES ? "max_output_tokens" : "max_tokens";
    if (ids->count == 0) {
        set_error(&e, 400, "invalid_request_error", "invalid_value", conversation, "the prompt has no tokens");
        r.status = respond_error_as(sink, &e, p->kind);
        return r;
    }
    size_t room = ids->count < m->ctx ? m->ctx - ids->count : 0;
    size_t max_new = p->has_max_tokens ? p->max_tokens : (p->chat ? room : 16);
    /* Anthropic's API, from Claude Sonnet 4.5 on, takes a max_tokens larger
     * than the context has room for and stops when the context is full, with
     * stop_reason model_context_window_exceeded; its clients send the most
     * output they will take, whatever the prompt. OpenAI's refuses. */
    if (p->kind == API_MESSAGES && room > 0 && max_new > room) max_new = room;
    if (room == 0 && p->kind == API_MESSAGES) {
        /* Anthropic's own words, which its clients look for. */
        set_error(&e, 400, "invalid_request_error", "context_length_exceeded", conversation, "prompt is too long: %zu tokens > %zu maximum", ids->count,
                  m->ctx - 1);
        r.status = respond_error_as(sink, &e, p->kind);
        return r;
    }
    if (room == 0 || max_new > room) {
        set_error(&e, 400, "invalid_request_error", "context_length_exceeded", p->has_max_tokens ? max_param : conversation,
                  "This model's context holds %zu tokens. The prompt is %zu tokens, which leaves room for %zu, and %zu were asked for.",
                  m->ctx, ids->count, room, max_new);
        r.status = respond_error_as(sink, &e, p->kind);
        return r;
    }

    job_t* j = (job_t*)calloc(1, sizeof(job_t));
    if (!j) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        r.status = respond_error_as(sink, &e, p->kind);
        return r;
    }
    j->api = api;
    j->p = p;
    j->ids = ids;
    j->ignored = ignored;
    j->max_new = max_new;
    vitna_sb_init(&j->out);
    vitna_cond_init(&j->cv);
    j->sink.write = job_write;
    j->sink.ctx = j;

    vitna_mutex_lock(&api->lock);
    if (!api->started) api->started = vitna_thread_start(&api->thread, scheduler, api, false);
    if (!api->started || api->lost) {
        /* Refused under the lock that queues, so no request joins the queue
         * after lose has emptied it. */
        const bool lost = api->lost;
        vitna_mutex_unlock(&api->lock);
        vitna_cond_destroy(&j->cv);
        free(j);
        if (lost) lost_error(api, &e);
        else set_error(&e, 500, "server_error", NULL, NULL, "the server could not start the thread that runs requests");
        r.status = respond_error_as(sink, &e, p->kind);
        return r;
    }
    if (api->queue_tail) api->queue_tail->next = j;
    else api->queue = j;
    api->queue_tail = j;
    vitna_cond_signal(&api->work);
    /* Send what the scheduler writes until it is done. Bytes are taken out
     * of the job under the lock and sent without it, so the scheduler never
     * waits on this client. */
    for (;;) {
        while (j->out.len == 0 && j->out.ok && !j->done) vitna_cond_wait(&j->cv, &api->lock);
        if (j->out.len == 0 && j->out.ok) break; /* done, and everything sent */
        vitna_strbuf_t chunk = j->out;
        vitna_sb_init(&j->out);
        vitna_mutex_unlock(&api->lock);
        if (chunk.ok) sink_out(sink, chunk.data, chunk.len);
        else sink->failed = true;
        vitna_sb_free(&chunk);
        vitna_mutex_lock(&api->lock);
        if (sink->failed) j->gone = true;
    }
    vitna_mutex_unlock(&api->lock);
    r = j->r;
    vitna_sb_free(&j->out);
    vitna_cond_destroy(&j->cv);
    free(j);
    return r;
}

/* --- Routing --- */

const char* vitna_api_no_model_message(void) {
    return NO_MODEL_MESSAGE;
}

const char* vitna_api_model_id(const vitna_api_t* api) {
    return api ? api->model_id : "";
}

vitna_api_t* vitna_api_create(vitna_llama_t* model, const vitna_tokenizer_t* tok, const char* model_id) {
    vitna_api_t* api = (vitna_api_t*)calloc(1, sizeof(vitna_api_t));
    if (!api) return NULL;
    api->model = model;
    api->tok = tok;
    api->created = (long long)time(NULL);
    vitna_mutex_init(&api->lock);
    vitna_cond_init(&api->work);
    vitna_mutex_init(&api->pool_lock);
    vitna_cond_init(&api->pool_work);
    vitna_cond_init(&api->pool_done);
    vitna_mutex_init(&api->mask_lock);
    api->takers = model->seqs < TAKE_THREADS ? model->seqs : TAKE_THREADS;
    api->samplers = (vitna_sampler_t*)calloc(api->takers, sizeof(vitna_sampler_t));
    api->workers = (vitna_thread_t*)calloc(api->takers, sizeof(vitna_thread_t));
    api->pool_jobs = (job_t**)calloc(model->seqs, sizeof(job_t*));
    bool samplers = api->samplers && api->workers && api->pool_jobs;
    for (size_t i = 0; samplers && i < api->takers; i++) samplers = vitna_sampler_init(&api->samplers[i], model->cfg.vocab, 0);
    size_t n = strlen(model_id) + 1;
    api->model_id = (char*)malloc(n);
    api->row = (float*)malloc(model->cfg.vocab * sizeof(float));
    api->seqs = (seq_t*)calloc(model->seqs, sizeof(seq_t));
    if (!api->model_id || !api->row || !api->seqs || !samplers || !round_capacity(api, 0)) {
        vitna_api_free(api);
        return NULL;
    }
    memcpy(api->model_id, model_id, n);
    api->prefix_cache = true;
    api->mask_cache = true;
    vitna_api_set_qwen3_template(api, model->cfg.qwen3);
    /* The cache starts empty, and each sequence's cached list says so. */
    vitna_llama_reset(model);
    return api;
}

vitna_api_t* vitna_api_create_encoder(vitna_encoder_t* encoder, const vitna_tokenizer_t* tok, const char* model_id) {
    vitna_api_t* api = (vitna_api_t*)calloc(1, sizeof(vitna_api_t));
    if (!api) return NULL;
    api->encoder = encoder;
    api->tok = tok;
    api->created = (long long)time(NULL);
    vitna_mutex_init(&api->lock);
    vitna_cond_init(&api->work);
    vitna_mutex_init(&api->pool_lock);
    vitna_cond_init(&api->pool_work);
    vitna_cond_init(&api->pool_done);
    vitna_mutex_init(&api->mask_lock);
    vitna_mutex_init(&api->embed_lock);
    size_t n = strlen(model_id) + 1;
    api->model_id = (char*)malloc(n);
    if (!api->model_id) {
        vitna_api_free(api);
        return NULL;
    }
    memcpy(api->model_id, model_id, n);
    return api;
}

size_t vitna_api_parallel(const vitna_api_t* api) {
    return api ? (api->model ? api->model->seqs : 1) : 0;
}

bool vitna_api_embeds(const vitna_api_t* api) {
    return api && api->encoder;
}

bool vitna_api_lost(vitna_api_t* api) {
    vitna_mutex_lock(&api->lock);
    const bool lost = api->lost;
    vitna_mutex_unlock(&api->lock);
    return lost;
}

void vitna_api_set_test_logits(vitna_api_t* api, bool on) {
    api->test_logits = on;
}

void vitna_api_set_test_reply(vitna_api_t* api, bool on) {
    api->test_reply = on;
}

void vitna_api_set_qwen3_template(vitna_api_t* api, bool on) {
    api->qwen3 = on;
    /* <tool_call> as the one token Qwen3's vocabulary has for it, which
     * tool_choice none bans; a vocabulary that spells it in pieces has none. */
    api->tool_call_token = -1;
    vitna_token_list_t ids = {0};
    if (on && vitna_tokenizer_encode(api->tok, "<tool_call>", 11, &ids) && ids.count == 1) api->tool_call_token = ids.ids[0];
    vitna_token_list_free(&ids);
}

void vitna_api_set_mask_cache(vitna_api_t* api, bool on) {
    api->mask_cache = on;
}

void vitna_api_set_prefix_cache(vitna_api_t* api, bool on) {
    api->prefix_cache = on;
}

bool vitna_api_set_speculate(vitna_api_t* api, size_t k) {
    if (!api->model) return false;
    const size_t pass_max = vitna_llama_exact_max(api->model);
    if (k && pass_max > 1 && k > pass_max - 1) k = pass_max - 1;
    vitna_mutex_lock(&api->lock);
    const bool ok = round_capacity(api, k);
    api->speculate = ok ? k : 0;
    if (!ok) round_capacity(api, 0);
    vitna_mutex_unlock(&api->lock);
    return ok;
}

void vitna_api_free(vitna_api_t* api) {
    if (!api) return;
    vitna_mutex_lock(&api->lock);
    api->stopping = true;
    vitna_cond_broadcast(&api->work);
    const bool started = api->started;
    vitna_mutex_unlock(&api->lock);
    if (started) vitna_thread_join(api->thread);
    vitna_mutex_lock(&api->pool_lock);
    api->pool_stop = true;
    vitna_cond_broadcast(&api->pool_work);
    vitna_mutex_unlock(&api->pool_lock);
    for (size_t i = 0; i < api->workers_started; i++) vitna_thread_join(api->workers[i]);
    if (api->samplers) {
        for (size_t i = 0; i < api->takers; i++) vitna_sampler_free(&api->samplers[i]);
    }
    free(api->samplers);
    free(api->workers);
    free(api->pool_jobs);
    if (api->seqs) {
        for (size_t s = 0; s < api->model->seqs; s++) vitna_token_list_free(&api->seqs[s].cached);
    }
    for (size_t i = 0; i < MASK_CACHE; i++) free(api->masks[i].allowed);
    free(api->seqs);
    free(api->rows);
    free(api->logits);
    free(api->ran);
    free(api->row);
    free(api->model_id);
    vitna_cond_destroy(&api->work);
    vitna_mutex_destroy(&api->lock);
    vitna_cond_destroy(&api->pool_work);
    vitna_cond_destroy(&api->pool_done);
    vitna_mutex_destroy(&api->pool_lock);
    vitna_mutex_destroy(&api->mask_lock);
    if (api->encoder) vitna_mutex_destroy(&api->embed_lock);
    free(api);
}

/* --- Embeddings --- */

/* The most texts, and tokens across them, one request may ask for: OpenAI's
 * 2,048 inputs, and enough tokens for 256 texts of 512, which a CPU embeds in
 * seconds rather than minutes. A larger request is refused, not cut short. */
#define EMBED_MAX_INPUTS 2048
#define EMBED_MAX_TOKENS 131072

static void base64_f32(vitna_strbuf_t* b, const float* v, size_t n) {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsigned char bytes[3];
    size_t have = 0;
    char quad[4];
    for (size_t i = 0; i < n * 4; i++) {
        /* float32 little-endian, as OpenAI's base64 encoding_format sends it */
        uint32_t bits;
        memcpy(&bits, &v[i / 4], 4);
        bytes[have++] = (unsigned char)(bits >> (8 * (i % 4)));
        if (have == 3) {
            quad[0] = alphabet[bytes[0] >> 2];
            quad[1] = alphabet[((bytes[0] & 3) << 4) | (bytes[1] >> 4)];
            quad[2] = alphabet[((bytes[1] & 15) << 2) | (bytes[2] >> 6)];
            quad[3] = alphabet[bytes[2] & 63];
            vitna_sb_append(b, quad, 4);
            have = 0;
        }
    }
    if (have) {
        if (have == 1) bytes[1] = 0;
        quad[0] = alphabet[bytes[0] >> 2];
        quad[1] = alphabet[((bytes[0] & 3) << 4) | (bytes[1] >> 4)];
        quad[2] = have == 2 ? alphabet[(bytes[1] & 15) << 2] : '=';
        quad[3] = '=';
        vitna_sb_append(b, quad, 4);
    }
}

/* Name a field the embeddings route ignores, header-safe, as parse_params does. */
static void note_ignored(vitna_strbuf_t* ignored, const char* k) {
    if (ignored->len >= 400) return;
    if (ignored->len) vitna_sb_puts(ignored, ", ");
    for (size_t c = 0; k[c] && c < 64; c++) {
        char ch = k[c];
        bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-';
        vitna_sb_append(ignored, safe ? &ch : "_", 1);
    }
}

/* POST /v1/embeddings: OpenAI's request and response. input is a string or
 * a list of them; each is tokenized with the model's tokenizer, [CLS] and
 * [SEP] included, embedded, pooled and normalized as the model's
 * sentence-transformers files say, and returned in order. usage counts the
 * tokens the model read. A text longer than the model reads is refused, not
 * cut short: truncation would embed a text other than the one sent. Token-id
 * input is refused too, since a list of ids would have to carry the model's
 * own [CLS] and [SEP], which a client sending OpenAI's ids would not know. */
static vitna_api_result_t embeddings_route(vitna_api_t* api, const char* body, size_t body_len, vitna_sink_t* sink) {
    vitna_api_result_t r = { 200, 0, 0, 0 };
    api_error_t e;
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(body ? body : "", body_len, jerr, sizeof(jerr));
    const vitna_json_value_t* root = doc ? vitna_json_root(doc) : NULL;
    vitna_strbuf_t ignored, out, headers;
    vitna_sb_init(&ignored);
    vitna_sb_init(&out);
    vitna_sb_init(&headers);
    vitna_encoder_t* enc = api->encoder;
    const size_t H = enc->cfg.hidden;
    size_t n = 0, tokens = 0;
    vitna_token_list_t* ids = NULL;
    const int32_t** rows = NULL;
    size_t* lens = NULL;
    float* vecs = NULL;
    bool base64 = false;

    if (!root || root->type != VITNA_JSON_OBJECT) {
        set_error(&e, 400, "invalid_request_error", "invalid_json", NULL, "The body must be a JSON object%s%s", doc ? "" : ": ", doc ? "" : jerr);
        goto fail;
    }
    const char* model = vitna_json_as_string(vitna_json_get(root, "model"));
    if (model && strcmp(model, api->model_id) != 0) {
        set_error(&e, 404, "invalid_request_error", "model_not_found", "model", "The model `%s` does not exist. This server serves `%s`.", model,
                  api->model_id);
        goto fail;
    }
    const vitna_json_value_t* input = NULL;
    for (size_t i = 0; i < root->u.object.count; i++) {
        const char* k = root->u.object.members[i].key;
        const vitna_json_value_t* v = root->u.object.members[i].value;
        double d;
        if (strcmp(k, "model") == 0 || strcmp(k, "user") == 0) continue;
        if (strcmp(k, "input") == 0) {
            input = v;
        } else if (strcmp(k, "encoding_format") == 0) {
            const char* f = vitna_json_as_string(v);
            if (vitna_json_is_null(v) || (f && strcmp(f, "float") == 0)) {
                base64 = false;
            } else if (f && strcmp(f, "base64") == 0) {
                base64 = true;
            } else {
                set_error(&e, 400, "invalid_request_error", "invalid_value", k, "`encoding_format` must be float or base64");
                goto fail;
            }
        } else if (strcmp(k, "dimensions") == 0) {
            if (vitna_json_is_null(v)) continue;
            if (!vitna_json_as_number(v, &d) || !integral(d) || d < 1) {
                set_error(&e, 400, "invalid_request_error", "invalid_value", k, "`dimensions` must be a positive integer");
                goto fail;
            }
            if ((size_t)d != H) {
                set_error(&e, 400, "invalid_request_error", "unsupported_parameter", k,
                          "`dimensions` is not supported by this model: its embeddings have %zu dimensions, and it was not trained to be cut shorter", H);
                goto fail;
            }
        } else {
            note_ignored(&ignored, k);
        }
    }
    if (!input || vitna_json_is_null(input)) {
        set_error(&e, 400, "invalid_request_error", "missing_required_parameter", "input", "`input` is required");
        goto fail;
    }
    const bool one = input->type == VITNA_JSON_STRING;
    if (!one && input->type != VITNA_JSON_ARRAY) {
        set_error(&e, 400, "invalid_request_error", "invalid_value", "input", "`input` must be a string or an array of strings");
        goto fail;
    }
    n = one ? 1 : input->u.array.count;
    if (n == 0 || n > EMBED_MAX_INPUTS) {
        set_error(&e, 400, "invalid_request_error", "invalid_value", "input", "`input` must hold from 1 to %d texts", EMBED_MAX_INPUTS);
        goto fail;
    }
    for (size_t i = 0; !one && i < n; i++) {
        const vitna_json_value_t* item = input->u.array.items[i];
        if (item->type == VITNA_JSON_STRING) continue;
        if (item->type == VITNA_JSON_NUMBER || item->type == VITNA_JSON_ARRAY) {
            unsupported(&e, "input", "token ids are not accepted; send text, which the server tokenizes with the model's own tokenizer");
        } else {
            set_error(&e, 400, "invalid_request_error", "invalid_value", "input", "`input` must be a string or an array of strings");
        }
        goto fail;
    }
    ids = (vitna_token_list_t*)calloc(n, sizeof(vitna_token_list_t));
    rows = (const int32_t**)malloc(n * sizeof(int32_t*));
    lens = (size_t*)malloc(n * sizeof(size_t));
    if (!ids || !rows || !lens) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        goto fail;
    }
    for (size_t i = 0; i < n; i++) {
        const vitna_json_value_t* item = one ? input : input->u.array.items[i];
        if (!vitna_tokenizer_encode(api->tok, item->u.string.ptr, item->u.string.len, &ids[i])) {
            set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
            goto fail;
        }
        if (ids[i].count > enc->cfg.max_tokens) {
            char which[32] = "input";
            if (!one) snprintf(which, sizeof(which), "input[%zu]", i);
            set_error(&e, 400, "invalid_request_error", "context_length_exceeded", "input",
                      "This model reads at most %zu tokens a text, [CLS] and [SEP] included, and %s is %zu tokens. Send shorter texts; nothing is cut short.",
                      enc->cfg.max_tokens, which, ids[i].count);
            goto fail;
        }
        rows[i] = ids[i].ids;
        lens[i] = ids[i].count;
        tokens += ids[i].count;
    }
    if (tokens > EMBED_MAX_TOKENS) {
        set_error(&e, 400, "invalid_request_error", "invalid_value", "input",
                  "These texts are %zu tokens together, and one request may hold %d. Send them in smaller requests.", tokens, EMBED_MAX_TOKENS);
        goto fail;
    }
    vecs = (float*)malloc(n * H * sizeof(float));
    if (!vecs) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        goto fail;
    }
    vitna_mutex_lock(&api->embed_lock);
    const bool ran = vitna_encoder_embed(enc, rows, lens, n, enc->cfg.pooling, enc->cfg.normalize, vecs, NULL);
    vitna_mutex_unlock(&api->embed_lock);
    if (!ran) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        goto fail;
    }
    for (size_t i = 0; i < n * H; i++) {
        if (!isfinite(vecs[i])) {
            set_error(&e, 500, "server_error", NULL, NULL, "the model produced a value that is not a finite number");
            goto fail;
        }
    }

    vitna_sb_puts(&out, "{\"object\":\"list\",\"data\":[");
    for (size_t i = 0; i < n; i++) {
        vitna_sb_printf(&out, "%s{\"object\":\"embedding\",\"index\":%zu,\"embedding\":", i ? "," : "", i);
        if (base64) {
            vitna_sb_puts(&out, "\"");
            base64_f32(&out, vecs + i * H, H);
            vitna_sb_puts(&out, "\"");
        } else {
            vitna_sb_puts(&out, "[");
            for (size_t k = 0; k < H; k++) vitna_sb_printf(&out, k ? ",%.9g" : "%.9g", (double)vecs[i * H + k]);
            vitna_sb_puts(&out, "]");
        }
        vitna_sb_puts(&out, "}");
    }
    vitna_sb_puts(&out, "],\"model\":");
    vitna_sb_json_string(&out, (const unsigned char*)api->model_id, strlen(api->model_id));
    vitna_sb_printf(&out, ",\"usage\":{\"prompt_tokens\":%zu,\"total_tokens\":%zu}}", tokens, tokens);
    if (!out.ok) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        goto fail;
    }
    if (ignored.len) vitna_sb_printf(&headers, "x-vitna-ignored: %s\r\n", ignored.data);
    respond(sink, 200, headers.len ? headers.data : NULL, out.data, out.len);
    r.prompt_tokens = tokens;
    goto done;

fail:
    r.status = respond_error(sink, &e);
done:
    for (size_t i = 0; ids && i < n; i++) vitna_token_list_free(&ids[i]);
    free(ids);
    free(rows);
    free(lens);
    free(vecs);
    vitna_sb_free(&ignored);
    vitna_sb_free(&out);
    vitna_sb_free(&headers);
    vitna_json_free(doc);
    return r;
}

/* The body as a JSON object naming this server's model, if it names one;
 * NULL, with the error set, otherwise. */
static const vitna_json_value_t* request_root(const vitna_api_t* api, vitna_json_doc_t* doc, const char* jerr, api_error_t* e) {
    const vitna_json_value_t* root = doc ? vitna_json_root(doc) : NULL;
    if (!root || root->type != VITNA_JSON_OBJECT) {
        set_error(e, 400, "invalid_request_error", "invalid_json", NULL, "The body must be a JSON object%s%s", doc ? "" : ": ", doc ? "" : jerr);
        return NULL;
    }
    const char* model = vitna_json_as_string(vitna_json_get(root, "model"));
    if (model && strcmp(model, api->model_id) != 0) {
        set_error(e, 404, "invalid_request_error", "model_not_found", "model", "The model `%s` does not exist. This server serves `%s`.", model,
                  api->model_id);
        return NULL;
    }
    return root;
}

/* A request that generates, of any of the four kinds: its parameters, its
 * prompt (a completion's own, or a conversation in the model's template),
 * then its job. */
static vitna_api_result_t generate_route(vitna_api_t* api, api_kind_t kind, const char* body, size_t body_len, vitna_sink_t* sink) {
    vitna_api_result_t r = { 200, 0, 0, 0 };
    api_error_t e;
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(body ? body : "", body_len, jerr, sizeof(jerr));
    params_t p;
    vitna_strbuf_t ignored, prompt, prefill;
    vitna_token_list_t ids = {0};
    vitna_token_list_t forced = {0};
    vitna_sb_init(&ignored);
    vitna_sb_init(&prompt);
    vitna_sb_init(&prefill);

    const vitna_json_value_t* root = request_root(api, doc, jerr, &e);
    if (!root || !parse_params(api, root, kind, &p, &e, &ignored)) goto fail;
    if (kind == API_COMPLETIONS) {
        if (!completion_prompt(api, vitna_json_get(root, "prompt"), &ids, &e)) goto fail;
    } else {
        if (!chat_build(api, root, &p, &prompt, &prefill, &ignored, &e)) goto fail;
        if (!vitna_tokenizer_encode(api->tok, prompt.data, prompt.len, &ids)) {
            set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
            goto fail;
        }
    }
    /* For tests: an FNV-1a hash of the prompt a conversation was written
     * as, and the reply a request gives, taken in place of the model's
     * (vitna_api_set_test_reply). */
    if (api->test_reply && kind != API_COMPLETIONS) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < prompt.len; i++) h = (h ^ (unsigned char)prompt.data[i]) * 0x100000001b3ULL;
        p.test_prompt = h;
        p.has_test_prompt = true;
    }
    const vitna_json_value_t* reply = api->test_reply ? vitna_json_get(root, "vitna_test_reply") : NULL;
    if (reply && reply->type == VITNA_JSON_STRING) {
        static const int32_t none = 0;
        if (!vitna_tokenizer_encode(api->tok, reply->u.string.ptr, reply->u.string.len, &forced)) {
            set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
            goto fail;
        }
        p.forced = forced.count ? forced.ids : &none;
        p.n_forced = forced.count;
    }
    r = run_job(api, sink, &p, &ids, ignored.data);
    goto done;

fail:
    r.status = respond_error_as(sink, &e, kind);
done:
    vitna_token_list_free(&ids);
    vitna_token_list_free(&forced);
    vitna_sb_free(&ignored);
    vitna_sb_free(&prompt);
    vitna_sb_free(&prefill);
    vitna_json_free(doc);
    return r;
}

/* POST /v1/messages/count_tokens, Anthropic's, and POST
 * /v1/responses/input_tokens, OpenAI's: the tokens a request's prompt holds
 * as the model reads it, its template and any prefill included, in that
 * API's response. Nothing runs. A Messages count takes no max_tokens, so its
 * parameters are not read; a Responses count reads them, since JSON mode
 * changes the prompt. */
static vitna_api_result_t count_tokens_route(vitna_api_t* api, api_kind_t kind, const char* body, size_t body_len, vitna_sink_t* sink) {
    vitna_api_result_t r = { 200, 0, 0, 0 };
    api_error_t e;
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(body ? body : "", body_len, jerr, sizeof(jerr));
    params_t p;
    memset(&p, 0, sizeof(p));
    p.kind = kind;
    p.chat = true;
    p.ban = -1;
    vitna_strbuf_t prompt, prefill, out, ignored;
    vitna_token_list_t ids = {0};
    vitna_sb_init(&prompt);
    vitna_sb_init(&prefill);
    vitna_sb_init(&out);
    vitna_sb_init(&ignored);

    const vitna_json_value_t* root = request_root(api, doc, jerr, &e);
    if (!root || (kind == API_RESPONSES && !parse_params(api, root, kind, &p, &e, &ignored))) goto fail;
    p.root = root;
    if (!chat_build(api, root, &p, &prompt, &prefill, NULL, &e)) goto fail;
    if (!vitna_tokenizer_encode(api->tok, prompt.data, prompt.len, &ids)) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        goto fail;
    }
    if (kind == API_RESPONSES) vitna_sb_printf(&out, "{\"object\":\"response.input_tokens\",\"input_tokens\":%zu}", ids.count);
    else vitna_sb_printf(&out, "{\"input_tokens\":%zu}", ids.count);
    respond(sink, 200, NULL, out.data, out.len);
    r.prompt_tokens = ids.count;
    goto done;

fail:
    r.status = respond_error_as(sink, &e, kind);
done:
    vitna_token_list_free(&ids);
    vitna_sb_free(&prompt);
    vitna_sb_free(&prefill);
    vitna_sb_free(&out);
    vitna_sb_free(&ignored);
    vitna_json_free(doc);
    return r;
}

vitna_api_result_t vitna_api_handle(vitna_api_t* api, const char* method, const char* path,
                                    const char* body, size_t body_len, vitna_sink_t* sink) {
    vitna_api_result_t r = { 200, 0, 0, 0 };
    api_error_t e;
    char route[256];
    size_t plen = strcspn(path, "?");
    if (plen >= sizeof(route)) plen = sizeof(route) - 1;
    memcpy(route, path, plen);
    route[plen] = '\0';
    bool get = strcmp(method, "GET") == 0;
    bool post = strcmp(method, "POST") == 0;
    bool chat = strcmp(route, "/v1/chat/completions") == 0;
    bool completions = strcmp(route, "/v1/completions") == 0;
    bool embeddings = strcmp(route, "/v1/embeddings") == 0;
    bool messages = strcmp(route, "/v1/messages") == 0;
    bool count = strcmp(route, "/v1/messages/count_tokens") == 0;
    bool responses = strcmp(route, "/v1/responses") == 0;
    bool input_tokens = strcmp(route, "/v1/responses/input_tokens") == 0;
    const bool generates = chat || completions || messages || count || responses || input_tokens;
    /* Errors answer in the shape of the API asked: Anthropic's for its routes, OpenAI's for the rest. */
    const api_kind_t kind = messages || count ? API_MESSAGES : responses || input_tokens ? API_RESPONSES : completions ? API_COMPLETIONS : API_CHAT;

    if (get && (strcmp(route, "/") == 0 || strcmp(route, "/chat") == 0)) {
        respond_as(sink, 200, "text/html; charset=utf-8", CHAT_PAGE_HEADERS, vitna_chat_page, vitna_chat_page_len);
        return r;
    }
    if (get && (strcmp(route, "/health") == 0 || strcmp(route, "/v1/health") == 0)) {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        if (api && api->encoder) {
            const vitna_encoder_config_t* c = &api->encoder->cfg;
            vitna_sb_puts(&b, "{\"ok\":true,\"engine\":\"vitna-anchor\",\"model\":");
            vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
            vitna_sb_printf(&b, ",\"generation\":false,\"embeddings\":true,\"dimensions\":%zu,\"max_tokens\":%zu,\"pooling\":\"%s\",\"normalized\":%s,\"threads\":%zu}",
                            c->hidden, c->max_tokens, c->pooling == VITNA_POOL_CLS ? "cls" : "mean", c->normalize ? "true" : "false", api->encoder->threads);
        } else if (api) {
            vitna_sb_puts(&b, "{\"ok\":true,\"engine\":\"vitna-anchor\",\"model\":");
            vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
            vitna_sb_printf(&b, ",\"generation\":true,\"context\":%zu,\"parallel\":%zu,\"chat_template\":\"%s\"}", api->model->ctx, api->model->seqs,
                            api->qwen3 ? "qwen3" : "chatml");
        } else {
            vitna_sb_puts(&b, "{\"ok\":true,\"engine\":\"vitna-anchor\",\"model\":null,\"generation\":false,\"message\":\"" NO_MODEL_MESSAGE "\"}");
        }
        respond(sink, 200, NULL, b.data, b.len);
        vitna_sb_free(&b);
        return r;
    }
    if (get && strcmp(route, "/v1/models") == 0) {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        vitna_sb_puts(&b, "{\"object\":\"list\",\"data\":[");
        if (api) {
            vitna_sb_puts(&b, "{\"id\":");
            vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
            vitna_sb_printf(&b, ",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"vitna-anchor\"}", api->created);
        }
        vitna_sb_puts(&b, "]}");
        respond(sink, 200, NULL, b.data, b.len);
        vitna_sb_free(&b);
        return r;
    }
    if ((generates || embeddings) && !post) {
        set_error(&e, 405, "invalid_request_error", "method_not_allowed", NULL, "%s takes POST", route);
        r.status = respond_error_as(sink, &e, kind);
        return r;
    }
    if (generates || embeddings) {
        if (!api) {
            set_error(&e, 501, "not_implemented", "no_model", NULL, NO_MODEL_MESSAGE);
            r.status = respond_error_as(sink, &e, kind);
            return r;
        }
        /* A model does one or the other: an embedding model generates no
         * text, and a model that generates is not served as an embedding
         * model, whose pooling and training it does not have. */
        if (embeddings && !api->encoder) {
            set_error(&e, 404, "invalid_request_error", "model_not_supported", "model",
                      "The model `%s` generates text and produces no embeddings. Start the server with an embedding model to serve /v1/embeddings.",
                      api->model_id);
            r.status = respond_error(sink, &e);
            return r;
        }
        if (!embeddings && api->encoder) {
            set_error(&e, 404, "invalid_request_error", "model_not_supported", "model",
                      "The model `%s` produces embeddings and generates no text. Send it to /v1/embeddings.", api->model_id);
            r.status = respond_error_as(sink, &e, kind);
            return r;
        }
        if (embeddings) return embeddings_route(api, body, body_len, sink);
        if (count || input_tokens) return count_tokens_route(api, kind, body, body_len, sink);
        return generate_route(api, kind, body, body_len, sink);
    }
    set_error(&e, 404, "invalid_request_error", "not_found", NULL, "Not found: %s %s", method, route);
    r.status = respond_error(sink, &e);
    return r;
}
