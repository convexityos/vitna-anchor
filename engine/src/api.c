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
#include "compat.h"
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

#define MAX_STOPS 4

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
    size_t speculate;               /* the most tokens drafted at once, 0 for none (vitna_api_set_speculate) */

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

typedef struct {
    bool chat;
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

static void respond(vitna_sink_t* s, int status, const char* headers, const char* body, size_t len) {
    vitna_strbuf_t h;
    vitna_sb_init(&h);
    vitna_sb_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n%s\r\n",
                    status, reason_phrase(status), len, headers ? headers : "");
    sink_out(s, h.data, h.len);
    sink_out(s, body, len);
    vitna_sb_free(&h);
}

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

static int respond_error(vitna_sink_t* s, const api_error_t* e) {
    vitna_strbuf_t b;
    vitna_sb_init(&b);
    error_json(&b, e);
    respond(s, e->status, e->status == 405 ? "Allow: POST\r\n" : NULL, b.data, b.len);
    vitna_sb_free(&b);
    return e->status;
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

static bool parse_params(const vitna_json_value_t* root, bool chat, params_t* p, api_error_t* e, vitna_strbuf_t* ignored) {
    memset(p, 0, sizeof(*p));
    p->chat = chat;
    p->temperature = 1.0f;
    p->top_p = 1.0f;
    for (size_t i = 0; i < root->u.object.count; i++) {
        const char* k = root->u.object.members[i].key;
        const vitna_json_value_t* v = root->u.object.members[i].value;
        double d;
        bool b;
        if (strcmp(k, "model") == 0 || strcmp(k, (chat ? "messages" : "prompt")) == 0) continue;
        if (strcmp(k, "user") == 0 || strcmp(k, "metadata") == 0 || strcmp(k, "store") == 0 || strcmp(k, "service_tier") == 0 ||
            strcmp(k, "parallel_tool_calls") == 0 || strcmp(k, "prompt_cache_key") == 0 || strcmp(k, "safety_identifier") == 0) {
            continue; /* bookkeeping for a hosted API; changes nothing here */
        }
        if (vitna_json_is_null(v)) continue;
        if (strcmp(k, "max_tokens") == 0 || (chat && strcmp(k, "max_completion_tokens") == 0)) {
            if (!vitna_json_as_number(v, &d) || !integral(d) || d < 1) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`%s` must be a positive integer", k);
                return false;
            }
            p->max_tokens = (size_t)d;
            p->has_max_tokens = true;
        } else if (strcmp(k, "temperature") == 0) {
            if (!vitna_json_as_number(v, &d) || d < 0 || d > 2) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`temperature` must be a number from 0 to 2");
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
        } else if (strcmp(k, "stop") == 0) {
            if (v->type == VITNA_JSON_STRING) {
                if (v->u.string.len == 0) {
                    set_error(e, 400, "invalid_request_error", "invalid_value", k, "a stop sequence cannot be empty");
                    return false;
                }
                p->stops[0] = v->u.string.ptr;
                p->stop_lens[0] = v->u.string.len;
                p->n_stops = 1;
            } else if (v->type == VITNA_JSON_ARRAY && v->u.array.count <= MAX_STOPS) {
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
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stop` must be a string or up to %d strings", MAX_STOPS);
                return false;
            }
        } else if (strcmp(k, "stream") == 0) {
            if (!vitna_json_as_bool(v, &p->stream)) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stream` must be true or false");
                return false;
            }
        } else if (strcmp(k, "stream_options") == 0) {
            if (v->type != VITNA_JSON_OBJECT) {
                set_error(e, 400, "invalid_request_error", "invalid_value", k, "`stream_options` must be an object");
                return false;
            }
            p->has_stream_options = true;
            vitna_json_as_bool(vitna_json_get(v, "include_usage"), &p->include_usage);
        } else if (strcmp(k, "n") == 0 || strcmp(k, "best_of") == 0) {
            if (!vitna_json_as_number(v, &d) || d != 1) {
                unsupported(e, k, "it returns one choice per request");
                return false;
            }
        } else if (strcmp(k, "logprobs") == 0) {
            if (!(vitna_json_as_bool(v, &b) && !b)) {
                unsupported(e, k, "it does not return log probabilities yet");
                return false;
            }
        } else if (strcmp(k, "top_logprobs") == 0) {
            unsupported(e, k, "it does not return log probabilities yet");
            return false;
        } else if (strcmp(k, "presence_penalty") == 0 || strcmp(k, "frequency_penalty") == 0) {
            if (!is_zero_or_null(v)) {
                unsupported(e, k, "it applies no penalties to logits");
                return false;
            }
        } else if (strcmp(k, "logit_bias") == 0) {
            if (!(v->type == VITNA_JSON_OBJECT && v->u.object.count == 0)) {
                unsupported(e, k, "it applies no bias to logits");
                return false;
            }
        } else if (strcmp(k, "response_format") == 0) {
            const char* t = vitna_json_as_string(vitna_json_get(v, "type"));
            if (t && strcmp(t, "json_object") == 0) {
                p->json = true;
            } else if (t && strcmp(t, "json_schema") == 0) {
                unsupported(e, k, "it constrains output to a JSON object (type json_object) but not yet to a schema");
                return false;
            } else if (!t || strcmp(t, "text") != 0) {
                unsupported(e, k, "its types are text and json_object");
                return false;
            }
        } else if (strcmp(k, "tools") == 0 || strcmp(k, "functions") == 0) {
            if (!(v->type == VITNA_JSON_ARRAY && v->u.array.count == 0)) {
                unsupported(e, k, "it has no tool calling");
                return false;
            }
        } else if (strcmp(k, "tool_choice") == 0 || strcmp(k, "function_call") == 0) {
            const char* c = vitna_json_as_string(v);
            if (!c || (strcmp(c, "none") != 0 && strcmp(c, "auto") != 0)) {
                unsupported(e, k, "it has no tool calling");
                return false;
            }
        } else if (!chat && strcmp(k, "echo") == 0) {
            if (!(vitna_json_as_bool(v, &b) && !b)) {
                unsupported(e, k, "it returns only the completion");
                return false;
            }
        } else if (!chat && strcmp(k, "suffix") == 0) {
            unsupported(e, k, "it does not insert text");
            return false;
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

/* Messages as ChatML: each "<|im_start|>{role}\n{content}<|im_end|>\n", then
 * "<|im_start|>assistant\n" for the reply. A developer message is a system one. */
static bool chat_prompt(const vitna_json_value_t* messages, vitna_strbuf_t* out, api_error_t* e) {
    char* param = e->param_buf;
    if (!messages || messages->type != VITNA_JSON_ARRAY || messages->u.array.count == 0) {
        set_error(e, 400, "invalid_request_error", "invalid_value", "messages", "`messages` must be a non-empty array");
        return false;
    }
    for (size_t i = 0; i < messages->u.array.count; i++) {
        const vitna_json_value_t* m = messages->u.array.items[i];
        const char* role = vitna_json_as_string(vitna_json_get(m, "role"));
        snprintf(param, sizeof(e->param_buf), "messages[%zu]", i);
        if (m->type != VITNA_JSON_OBJECT || !role) {
            set_error(e, 400, "invalid_request_error", "invalid_value", param, "each message needs a role");
            return false;
        }
        if (strcmp(role, "developer") == 0) role = "system";
        if (strcmp(role, "tool") == 0 || strcmp(role, "function") == 0) {
            unsupported(e, param, "it has no tool calling, so there are no tool results to read");
            return false;
        }
        if (strcmp(role, "system") != 0 && strcmp(role, "user") != 0 && strcmp(role, "assistant") != 0) {
            set_error(e, 400, "invalid_request_error", "invalid_value", param, "unknown role \"%s\"", role);
            return false;
        }
        const vitna_json_value_t* name = vitna_json_get(m, "name");
        const vitna_json_value_t* calls = vitna_json_get(m, "tool_calls");
        if (name && !vitna_json_is_null(name)) {
            unsupported(e, param, "a message's `name` has no place in this model's chat format");
            return false;
        }
        if (calls && !vitna_json_is_null(calls) && !(calls->type == VITNA_JSON_ARRAY && calls->u.array.count == 0)) {
            unsupported(e, param, "it has no tool calling");
            return false;
        }
        const vitna_json_value_t* content = vitna_json_get(m, "content");
        if (content && content->type == VITNA_JSON_STRING) {
            chatml_append(out, role, content->u.string.ptr, content->u.string.len);
        } else if (content && content->type == VITNA_JSON_ARRAY) {
            vitna_strbuf_t text;
            vitna_sb_init(&text);
            for (size_t j = 0; j < content->u.array.count; j++) {
                const vitna_json_value_t* part = content->u.array.items[j];
                const char* type = vitna_json_as_string(vitna_json_get(part, "type"));
                const vitna_json_value_t* t = vitna_json_get(part, "text");
                if (!type || strcmp(type, "text") != 0 || !t || t->type != VITNA_JSON_STRING) {
                    vitna_sb_free(&text);
                    unsupported(e, param, "the model reads text only");
                    return false;
                }
                vitna_sb_append(&text, t->u.string.ptr, t->u.string.len);
            }
            chatml_append(out, role, text.data ? text.data : "", text.len);
            vitna_sb_free(&text);
        } else {
            set_error(e, 400, "invalid_request_error", "invalid_value", param, "each message needs text content");
            return false;
        }
    }
    vitna_sb_puts(out, "<|im_start|>assistant\n");
    return true;
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

typedef struct {
    vitna_api_t* api;
    vitna_sink_t* sink;
    const params_t* p;
    char id[48];
    long long created;
    vitna_strbuf_t text;   /* generated bytes, cut at a stop sequence */
    size_t emitted;        /* bytes of text already streamed */
} gen_t;

static void usage_json(vitna_strbuf_t* sb, const vitna_api_result_t* r) {
    vitna_sb_printf(sb, "{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,\"prompt_tokens_details\":{\"cached_tokens\":%zu}}",
                    r->prompt_tokens, r->completion_tokens, r->prompt_tokens + r->completion_tokens, r->cached_tokens);
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

/* The last event of a stream that fails after its 200 has gone out. Its data
 * is the body an error response would have had, and no final chunk or [DONE]
 * follows it, so the reply so far cannot pass for a whole one. */
static void send_error_event(gen_t* g, const api_error_t* e) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_puts(&sb, "data: ");
    error_json(&sb, e);
    vitna_sb_puts(&sb, "\n\n");
    sink_out(g->sink, sb.data, sb.len);
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
 * since from could complete one; -1 for none. */
static long long find_stop(const params_t* p, const char* text, size_t len, size_t from) {
    long long best = -1;
    for (size_t i = 0; i < p->n_stops; i++) {
        size_t sl = p->stop_lens[i];
        size_t start = from >= sl ? from - sl + 1 : 0;
        for (size_t q = start; q + sl <= len; q++) {
            if (memcmp(text + q, p->stops[i], sl) == 0) {
                if (best < 0 || (long long)q < best) best = (long long)q;
                break;
            }
        }
    }
    return best;
}

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
    const int32_t next = vitna_sample(sampler, row, tk->cfg);
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
    long long cut = find_stop(p, g->text.data ? g->text.data : "", g->text.len, before);
    if (cut >= 0) {
        g->text.len = (size_t)cut;
        *tk->finish = "stop";
        return TAKE_END;
    }
    if (p->json && vitna_jsonpfx_complete(&tk->json)) {
        *tk->finish = "stop"; /* the object has closed, and nothing may follow it */
        return TAKE_END;
    }
    if (p->stream) stream_ready(g, false);
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
    snprintf(g->id, sizeof(g->id), "%s%016llx", p->chat ? "chatcmpl-" : "cmpl-", (unsigned long long)fresh_u64(api));
    vitna_sb_init(&g->text);

    const uint64_t seed = p->has_seed ? p->seed : fresh_u64(api);
    j->rng = seed;
    j->cfg.temperature = p->temperature;
    j->cfg.top_k = p->top_k;
    j->cfg.top_p = p->top_p;
    j->cfg.seed = seed;

    vitna_sb_init(&j->headers);
    vitna_sb_printf(&j->headers, "x-vitna-seed: %llu\r\n", (unsigned long long)seed);
    if (j->ignored && *j->ignored) vitna_sb_printf(&j->headers, "x-vitna-ignored: %s\r\n", j->ignored);

    if (p->stream) {
        vitna_strbuf_t h;
        vitna_sb_init(&h);
        vitna_sb_printf(&h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n%s\r\n", j->headers.data);
        sink_out(&j->sink, h.data, h.len);
        vitna_sb_free(&h);
        if (p->chat) send_chunk(g, NULL, 0, NULL, NULL, true, false);
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

    if (!j->ok) {
        set_error(&e, 500, "server_error", NULL, NULL, "The model failed to run a step, so the reply could not be completed.");
        if (p->stream) send_error_event(g, &e);
        else j->r.status = respond_error(&j->sink, &e);
    } else if (!g->text.ok) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        if (!p->stream) j->r.status = respond_error(&j->sink, &e);
    } else if (p->stream) {
        stream_ready(g, true);
        send_chunk(g, NULL, 0, j->finish, &j->r, false, false);
        if (p->include_usage) send_chunk(g, NULL, 0, NULL, &j->r, false, true);
        sink_out(&j->sink, "data: [DONE]\n\n", 14);
    } else {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        vitna_sb_puts(&b, "{\"id\":");
        vitna_sb_json_string(&b, (const unsigned char*)g->id, strlen(g->id));
        vitna_sb_printf(&b, ",\"object\":\"%s\",\"created\":%lld,\"model\":", p->chat ? "chat.completion" : "text_completion", g->created);
        vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
        vitna_sb_puts(&b, ",\"choices\":[{\"index\":0,");
        if (p->chat) {
            vitna_sb_puts(&b, "\"message\":{\"role\":\"assistant\",\"content\":");
            vitna_sb_json_string(&b, (const unsigned char*)(g->text.data ? g->text.data : ""), g->text.len);
            vitna_sb_puts(&b, "}");
        } else {
            vitna_sb_puts(&b, "\"text\":");
            vitna_sb_json_string(&b, (const unsigned char*)(g->text.data ? g->text.data : ""), g->text.len);
        }
        vitna_sb_printf(&b, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":", j->finish);
        usage_json(&b, &j->r);
        vitna_sb_puts(&b, "}");
        if (api->test_logits) vitna_sb_printf(&j->headers, "x-vitna-test-logits: %016llx\r\n", (unsigned long long)j->logits_hash);
        respond(&j->sink, 200, j->headers.data, b.data, b.len);
        vitna_sb_free(&b);
    }
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
        j->r.status = respond_error(&j->sink, &e);
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
    if (ids->count == 0) {
        set_error(&e, 400, "invalid_request_error", "invalid_value", p->chat ? "messages" : "prompt", "the prompt has no tokens");
        r.status = respond_error(sink, &e);
        return r;
    }
    size_t room = ids->count < m->ctx ? m->ctx - ids->count : 0;
    size_t max_new = p->has_max_tokens ? p->max_tokens : (p->chat ? room : 16);
    if (room == 0 || max_new > room) {
        set_error(&e, 400, "invalid_request_error", "context_length_exceeded", p->has_max_tokens ? "max_tokens" : (p->chat ? "messages" : "prompt"),
                  "This model's context holds %zu tokens. The prompt is %zu tokens, which leaves room for %zu, and %zu were asked for.",
                  m->ctx, ids->count, room, max_new);
        r.status = respond_error(sink, &e);
        return r;
    }

    job_t* j = (job_t*)calloc(1, sizeof(job_t));
    if (!j) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        r.status = respond_error(sink, &e);
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
        r.status = respond_error(sink, &e);
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

static vitna_api_result_t completion_route(vitna_api_t* api, bool chat, const char* body, size_t body_len, vitna_sink_t* sink) {
    vitna_api_result_t r = { 200, 0, 0, 0 };
    api_error_t e;
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(body ? body : "", body_len, jerr, sizeof(jerr));
    const vitna_json_value_t* root = doc ? vitna_json_root(doc) : NULL;
    params_t p;
    vitna_strbuf_t ignored, prompt;
    vitna_token_list_t ids = {0};
    vitna_sb_init(&ignored);
    vitna_sb_init(&prompt);

    if (!root || root->type != VITNA_JSON_OBJECT) {
        set_error(&e, 400, "invalid_request_error", "invalid_json", NULL, "The body must be a JSON object%s%s", doc ? "" : ": ", doc ? "" : jerr);
        r.status = respond_error(sink, &e);
        goto done;
    }
    const char* model = vitna_json_as_string(vitna_json_get(root, "model"));
    if (model && strcmp(model, api->model_id) != 0) {
        set_error(&e, 404, "invalid_request_error", "model_not_found", "model",
                  "The model `%s` does not exist. This server serves `%s`.", model, api->model_id);
        r.status = respond_error(sink, &e);
        goto done;
    }
    if (!parse_params(root, chat, &p, &e, &ignored)) {
        r.status = respond_error(sink, &e);
        goto done;
    }
    if (chat) {
        if (!chat_prompt(vitna_json_get(root, "messages"), &prompt, &e)) {
            r.status = respond_error(sink, &e);
            goto done;
        }
        if (!vitna_tokenizer_encode(api->tok, prompt.data, prompt.len, &ids)) {
            set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
            r.status = respond_error(sink, &e);
            goto done;
        }
    } else if (!completion_prompt(api, vitna_json_get(root, "prompt"), &ids, &e)) {
        r.status = respond_error(sink, &e);
        goto done;
    }
    r = run_job(api, sink, &p, &ids, ignored.data);

done:
    vitna_token_list_free(&ids);
    vitna_sb_free(&ignored);
    vitna_sb_free(&prompt);
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
            vitna_sb_printf(&b, ",\"generation\":true,\"context\":%zu,\"parallel\":%zu}", api->model->ctx, api->model->seqs);
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
    if ((chat || completions || embeddings) && !post) {
        set_error(&e, 405, "invalid_request_error", "method_not_allowed", NULL, "%s takes POST", route);
        r.status = respond_error(sink, &e);
        return r;
    }
    if (chat || completions || embeddings) {
        if (!api) {
            set_error(&e, 501, "not_implemented", "no_model", NULL, NO_MODEL_MESSAGE);
            r.status = respond_error(sink, &e);
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
            r.status = respond_error(sink, &e);
            return r;
        }
        return embeddings ? embeddings_route(api, body, body_len, sink) : completion_route(api, chat, body, body_len, sink);
    }
    set_error(&e, 404, "invalid_request_error", "not_found", NULL, "Not found: %s %s", method, route);
    r.status = respond_error(sink, &e);
    return r;
}
