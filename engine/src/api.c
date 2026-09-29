/**
 * api.c - The OpenAI-compatible HTTP API over a loaded model.
 *
 * One request at a time: the model has one key-value cache, which each
 * request starts afresh (reusing a prefix across requests is gate A6).
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
#define NO_MODEL_MESSAGE \
    "This server was started without a model, so it serves none. Start it with: vitna-anchor serve --model <dir>"

struct vitna_api {
    vitna_llama_t* model;
    const vitna_tokenizer_t* tok;
    char* model_id;
    long long created;
    uint64_t counter;
    vitna_sampler_t sampler;
    float* row;
    bool prefix_cache;
    vitna_token_list_t cached;  /* the tokens whose keys and values are in the model's cache, in order */
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

static int respond_error(vitna_sink_t* s, const api_error_t* e) {
    vitna_strbuf_t b;
    vitna_sb_init(&b);
    vitna_sb_puts(&b, "{\"error\":{\"message\":");
    vitna_sb_json_string(&b, (const unsigned char*)e->message, strlen(e->message));
    vitna_sb_puts(&b, ",\"type\":");
    vitna_sb_json_string(&b, (const unsigned char*)e->type, strlen(e->type));
    vitna_sb_puts(&b, ",\"param\":");
    if (e->param) vitna_sb_json_string(&b, (const unsigned char*)e->param, strlen(e->param));
    else vitna_sb_puts(&b, "null");
    vitna_sb_puts(&b, ",\"code\":");
    if (e->code) vitna_sb_json_string(&b, (const unsigned char*)e->code, strlen(e->code));
    else vitna_sb_puts(&b, "null");
    vitna_sb_puts(&b, "}}");
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
    static char param[64];
    if (!messages || messages->type != VITNA_JSON_ARRAY || messages->u.array.count == 0) {
        set_error(e, 400, "invalid_request_error", "invalid_value", "messages", "`messages` must be a non-empty array");
        return false;
    }
    for (size_t i = 0; i < messages->u.array.count; i++) {
        const vitna_json_value_t* m = messages->u.array.items[i];
        const char* role = vitna_json_as_string(vitna_json_get(m, "role"));
        snprintf(param, sizeof(param), "messages[%zu]", i);
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
 * the reply instead. Returns false when no token is left. */
static bool mask_json(vitna_api_t* api, const vitna_jsonpfx_t* state) {
    uint8_t allowed[32];
    vitna_jsonpfx_next_bytes(state, allowed);
    size_t left = 0;
    for (size_t id = 0; id < api->model->cfg.vocab; id++) {
        size_t n = 0;
        const unsigned char* b = vitna_tokenizer_token_bytes(api->tok, (int32_t)id, &n);
        bool ok = b && n > 0 && !vitna_tokenizer_is_special(api->tok, (int32_t)id) &&
                  (allowed[b[0] >> 3] & (1u << (b[0] & 7))) && vitna_jsonpfx_accepts(state, b, n);
        if (ok) left++;
        else api->row[id] = -INFINITY;
    }
    return left > 0;
}

static vitna_api_result_t generate(vitna_api_t* api, vitna_sink_t* sink, const params_t* p, const vitna_token_list_t* ids, const char* ignored) {
    vitna_api_result_t r = { 200, ids->count, 0, 0 };
    vitna_llama_t* m = api->model;
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

    gen_t g;
    memset(&g, 0, sizeof(g));
    g.api = api;
    g.sink = sink;
    g.p = p;
    g.created = (long long)time(NULL);
    snprintf(g.id, sizeof(g.id), "%s%016llx", p->chat ? "chatcmpl-" : "cmpl-", (unsigned long long)fresh_u64(api));
    vitna_sb_init(&g.text);

    uint64_t seed = p->has_seed ? p->seed : fresh_u64(api);
    api->sampler.state = seed;
    vitna_sampling_t cfg = { p->temperature, p->top_k, p->top_p, seed };

    vitna_strbuf_t headers;
    vitna_sb_init(&headers);
    vitna_sb_printf(&headers, "x-vitna-seed: %llu\r\n", (unsigned long long)seed);
    if (ignored && *ignored) vitna_sb_printf(&headers, "x-vitna-ignored: %s\r\n", ignored);

    if (p->stream) {
        vitna_strbuf_t h;
        vitna_sb_init(&h);
        vitna_sb_printf(&h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n%s\r\n", headers.data);
        sink_out(sink, h.data, h.len);
        vitna_sb_free(&h);
        if (p->chat) send_chunk(&g, NULL, 0, NULL, NULL, true, false);
    }

    /* Reuse the longest prefix the model's cache already holds from the
     * previous request. The last prompt token is always fed again, for the
     * logits that choose the first new token. */
    size_t reuse = 0;
    if (api->prefix_cache) {
        size_t limit = ids->count - 1 < api->cached.count ? ids->count - 1 : api->cached.count;
        while (reuse < limit && api->cached.ids[reuse] == ids->ids[reuse]) reuse++;
    }
    vitna_llama_truncate(m, reuse);
    api->cached.count = reuse;
    r.cached_tokens = reuse;

    /* Prefill the rest, then one token at a time. */
    const char* finish = "length";
    bool ok = true;
    for (size_t t = reuse; t < ids->count && ok; t++) {
        ok = vitna_llama_step(m, ids->ids[t], t + 1 == ids->count ? api->row : NULL);
        if (ok) vitna_token_list_push(&api->cached, ids->ids[t]);
    }
    vitna_jsonpfx_t json_state;
    vitna_jsonpfx_init(&json_state);
    for (size_t s = 0; ok && s < max_new && !sink->failed; s++) {
        if (p->json && !mask_json(api, &json_state)) {
            /* No token can continue the object. The pinned model's vocabulary
             * never gets here: the 21 bytes it lacks are control characters,
             * which JSON escapes, and UTF-8 lead bytes a string can do without. */
            finish = "stop";
            break;
        }
        int32_t next = vitna_sample(&api->sampler, api->row, &cfg);
        r.completion_tokens++;
        if (vitna_tokenizer_is_special(api->tok, next)) {
            finish = "stop";
            break;
        }
        size_t n = 0;
        const unsigned char* bytes = vitna_tokenizer_token_bytes(api->tok, next, &n);
        size_t before = g.text.len;
        if (bytes) {
            vitna_sb_append(&g.text, bytes, n);
            if (p->json) vitna_jsonpfx_feed(&json_state, bytes, n);
        }
        long long cut = find_stop(p, g.text.data ? g.text.data : "", g.text.len, before);
        if (cut >= 0) {
            g.text.len = (size_t)cut;
            finish = "stop";
            break;
        }
        if (p->json && vitna_jsonpfx_complete(&json_state)) {
            finish = "stop"; /* the object has closed, and nothing may follow it */
            break;
        }
        if (p->stream) stream_ready(&g, false);
        if (s + 1 < max_new) {
            if (!vitna_llama_step(m, next, api->row)) break;
            vitna_token_list_push(&api->cached, next);
        }
    }

    if (!g.text.ok) {
        set_error(&e, 500, "server_error", NULL, NULL, "out of memory");
        if (!p->stream) r.status = respond_error(sink, &e);
    } else if (p->stream) {
        stream_ready(&g, true);
        send_chunk(&g, NULL, 0, finish, &r, false, false);
        if (p->include_usage) send_chunk(&g, NULL, 0, NULL, &r, false, true);
        sink_out(sink, "data: [DONE]\n\n", 14);
    } else {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        vitna_sb_puts(&b, "{\"id\":");
        vitna_sb_json_string(&b, (const unsigned char*)g.id, strlen(g.id));
        vitna_sb_printf(&b, ",\"object\":\"%s\",\"created\":%lld,\"model\":", p->chat ? "chat.completion" : "text_completion", g.created);
        vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
        vitna_sb_puts(&b, ",\"choices\":[{\"index\":0,");
        if (p->chat) {
            vitna_sb_puts(&b, "\"message\":{\"role\":\"assistant\",\"content\":");
            vitna_sb_json_string(&b, (const unsigned char*)(g.text.data ? g.text.data : ""), g.text.len);
            vitna_sb_puts(&b, "}");
        } else {
            vitna_sb_puts(&b, "\"text\":");
            vitna_sb_json_string(&b, (const unsigned char*)(g.text.data ? g.text.data : ""), g.text.len);
        }
        vitna_sb_printf(&b, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":", finish);
        usage_json(&b, &r);
        vitna_sb_puts(&b, "}");
        respond(sink, 200, headers.data, b.data, b.len);
        vitna_sb_free(&b);
    }
    vitna_sb_free(&headers);
    vitna_sb_free(&g.text);
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
    size_t n = strlen(model_id) + 1;
    api->model_id = (char*)malloc(n);
    api->row = (float*)malloc(model->cfg.vocab * sizeof(float));
    if (!api->model_id || !api->row || !vitna_sampler_init(&api->sampler, model->cfg.vocab, 0)) {
        vitna_api_free(api);
        return NULL;
    }
    memcpy(api->model_id, model_id, n);
    api->prefix_cache = true;
    /* The cache starts empty, and api->cached says so. */
    vitna_llama_reset(model);
    return api;
}

void vitna_api_set_prefix_cache(vitna_api_t* api, bool on) {
    api->prefix_cache = on;
}

void vitna_api_free(vitna_api_t* api) {
    if (!api) return;
    vitna_sampler_free(&api->sampler);
    vitna_token_list_free(&api->cached);
    free(api->row);
    free(api->model_id);
    free(api);
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
    r = generate(api, sink, &p, &ids, ignored.data);

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

    if (get && (strcmp(route, "/health") == 0 || strcmp(route, "/v1/health") == 0)) {
        vitna_strbuf_t b;
        vitna_sb_init(&b);
        if (api) {
            vitna_sb_puts(&b, "{\"ok\":true,\"engine\":\"vitna-anchor\",\"model\":");
            vitna_sb_json_string(&b, (const unsigned char*)api->model_id, strlen(api->model_id));
            vitna_sb_printf(&b, ",\"generation\":true,\"context\":%zu}", api->model->ctx);
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
    if ((chat || completions) && !post) {
        set_error(&e, 405, "invalid_request_error", "method_not_allowed", NULL, "%s takes POST", route);
        r.status = respond_error(sink, &e);
        return r;
    }
    if (post && strcmp(route, "/v1/embeddings") == 0) {
        set_error(&e, 501, "not_implemented", "not_implemented", NULL, "Embeddings are not implemented.");
        r.status = respond_error(sink, &e);
        return r;
    }
    if (chat || completions) {
        if (!api) {
            set_error(&e, 501, "not_implemented", "no_model", NULL, NO_MODEL_MESSAGE);
            r.status = respond_error(sink, &e);
            return r;
        }
        return completion_route(api, chat, body, body_len, sink);
    }
    set_error(&e, 404, "invalid_request_error", "not_found", NULL, "Not found: %s %s", method, route);
    r.status = respond_error(sink, &e);
    return r;
}
