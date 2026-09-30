/**
 * main.c - CLI entry point for vitna-engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compat.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "ops.h"
#include "api.h"
#include "safetensors.h"
#include "sampler.h"
#include "server.h"
#include "strbuf.h"
#include "tokenizer.h"

#if defined(VITNA_OS_WINDOWS)
  #include <fcntl.h>
#endif

static void print_usage(const char* prog) {
    if (vitna_llama_cuda_built()) {
        printf("vitna-anchor engine: a dense Llama-architecture model in float32, on the CPU or, with --device cuda, on an NVIDIA GPU.\n\n");
    } else {
        printf("vitna-anchor engine: a dense Llama-architecture model on the CPU, in float32. This build has no CUDA path.\n\n");
    }
    printf("Usage:\n");
    printf("  %s run      --model <dir> --prompt <text> [--max-new <n>] [sampling]\n", prog);
    printf("  %s generate --model <dir> (--prompt <text> | --ids <a,b,...>) [--max-new <n>] [sampling]\n", prog);
    printf("               [--logits-out <file>] [--stop-at-special] [--timing] [--speculate <k>]\n");
    printf("  %s logits   --model <dir> (--prompt <text> | --ids <a,b,...>) --out <file>\n", prog);
    printf("  %s tokenize --model <dir> [--text <text>]\n", prog);
    printf("  %s serve    [--model <dir>] [--model-id <id>] [--host <ip>] [--port <port>] [--no-prefix-cache]\n", prog);
    printf("  %s info     --model <file.safetensors>\n", prog);
    printf("  %s bench    [--iterations <n>]\n\n", prog);
    printf("Sampling: --greedy (the default), or --temperature <t> [--top-k <k>] [--top-p <p>] [--seed <s>].\n");
    printf("--ctx <n> sets how many positions the key-value cache holds (default: the model's maximum, at most 4096).\n");
    printf("--device cpu|cuda runs the model on the CPU (the default) or on the first CUDA device, which needs an\n");
    printf("engine built with the CUDA path. Asked for a device it cannot use, the engine says why and stops.\n\n");
    printf("run       prints the prompt's continuation as it is generated\n");
    printf("generate  prints JSON: the prompt's ids, the new ids and their text. --logits-out writes each\n");
    printf("          step's logits as float32, little-endian, steps x vocab. --timing prints to stderr how\n");
    printf("          long the prompt and the new tokens took, measured on this machine. --speculate k drafts up\n");
    printf("          to k tokens where the text's last two or three tokens occur earlier in it, and checks them\n");
    printf("          together: the same tokens and logits as without it. On a GPU they run in one pass (k at\n");
    printf("          most 7); on the CPU a step at a time, which is no faster\n");
    printf("logits    writes the logits at every position of the prompt, positions x vocab, float32 LE\n");
    printf("tokenize  prints the ids of --text as a JSON array; without --text it reads one JSON string\n");
    printf("          per line from stdin and prints one array per line\n");
    printf("serve     serves the model over an OpenAI-compatible HTTP API at /v1, on 127.0.0.1:8765 unless told\n");
    printf("          otherwise, under the model directory's name unless --model-id says another. Without\n");
    printf("          --model its generation endpoints answer 501\n");
    printf("info      lists the tensors in a SafeTensors file\n");
    printf("bench     times the int4 matrix-vector kernel on synthetic data\n");
}

typedef struct {
    const char* model;
    const char* prompt;
    const char* ids;
    const char* out;
    const char* logits_out;
    const char* text;
    const char* model_id;
    const char* host;
    const char* device;
    size_t max_new;
    size_t ctx;
    int iterations;
    uint16_t port;
    bool stop_at_special;
    bool no_prefix_cache;
    bool timing;
    size_t speculate;
    vitna_sampling_t sampling;
} args_t;

static bool parse_args(int argc, char** argv, args_t* a) {
    memset(a, 0, sizeof(*a));
    a->max_new = 64;
    a->iterations = 100;
    a->port = 8765;
    a->sampling.top_p = 1.0f;
    for (int i = 2; i < argc; i++) {
        const char* k = argv[i];
        const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define TAKE(name) (strcmp(k, name) == 0 && v && (++i, true))
        if (TAKE("--model")) a->model = v;
        else if (TAKE("--prompt")) a->prompt = v;
        else if (TAKE("--ids")) a->ids = v;
        else if (TAKE("--out")) a->out = v;
        else if (TAKE("--logits-out")) a->logits_out = v;
        else if (TAKE("--text")) a->text = v;
        else if (TAKE("--model-id")) a->model_id = v;
        else if (TAKE("--host")) a->host = v;
        else if (TAKE("--device")) a->device = v;
        else if (TAKE("--max-new")) a->max_new = (size_t)strtoull(v, NULL, 10);
        else if (TAKE("--speculate")) a->speculate = (size_t)strtoull(v, NULL, 10);
        else if (TAKE("--ctx")) a->ctx = (size_t)strtoull(v, NULL, 10);
        else if (TAKE("--iterations")) a->iterations = atoi(v);
        else if (TAKE("--port")) a->port = (uint16_t)atoi(v);
        else if (TAKE("--temperature")) a->sampling.temperature = (float)atof(v);
        else if (TAKE("--top-k")) a->sampling.top_k = (size_t)strtoull(v, NULL, 10);
        else if (TAKE("--top-p")) a->sampling.top_p = (float)atof(v);
        else if (TAKE("--seed")) a->sampling.seed = strtoull(v, NULL, 10);
        else if (strcmp(k, "--greedy") == 0) a->sampling.temperature = 0.0f;
        else if (strcmp(k, "--stop-at-special") == 0) a->stop_at_special = true;
        else if (strcmp(k, "--no-prefix-cache") == 0) a->no_prefix_cache = true;
        else if (strcmp(k, "--timing") == 0) a->timing = true;
        else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", k);
            return false;
        }
#undef TAKE
    }
    return true;
}

static void binary_stdio(void) {
#if defined(VITNA_OS_WINDOWS)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

/* JSON string output. Invalid UTF-8 becomes U+FFFD, as a lossy decoder would. */
static void print_json_string(FILE* f, const unsigned char* s, size_t n) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    vitna_sb_json_string(&sb, s, n);
    if (sb.ok) fwrite(sb.data, 1, sb.len, f);
    vitna_sb_free(&sb);
}

static void print_ids(FILE* f, const int32_t* ids, size_t n) {
    fputc('[', f);
    for (size_t i = 0; i < n; i++) fprintf(f, i ? ",%d" : "%d", ids[i]);
    fputc(']', f);
}

static char* join_path(const char* dir, const char* name) {
    size_t n = strlen(dir) + strlen(name) + 2;
    char* p = (char*)malloc(n);
    if (p) snprintf(p, n, "%s/%s", dir, name);
    return p;
}

static vitna_tokenizer_t* load_tokenizer(const char* dir) {
    char* path = join_path(dir, "tokenizer.json");
    char err[512];
    vitna_tokenizer_t* tok = path ? vitna_tokenizer_load(path, err, sizeof(err)) : NULL;
    if (!tok) fprintf(stderr, "%s\n", path ? err : "out of memory");
    free(path);
    return tok;
}

static bool parse_id_list(const char* s, vitna_token_list_t* out) {
    const char* p = s;
    while (*p) {
        char* end;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > 0x7FFFFFFF) return false;
        if (!vitna_token_list_push(out, (int32_t)v)) return false;
        p = end;
        if (*p == ',') p++;
        else if (*p) return false;
    }
    return true;
}

/* The prompt's ids, from --ids or by tokenizing --prompt. */
static bool prompt_ids(const args_t* a, const vitna_tokenizer_t* tok, vitna_token_list_t* out) {
    if (a->ids) {
        if (!parse_id_list(a->ids, out)) {
            fprintf(stderr, "--ids must be a comma-separated list of token ids\n");
            return false;
        }
    } else if (a->prompt) {
        if (!vitna_tokenizer_encode(tok, a->prompt, strlen(a->prompt), out)) {
            fprintf(stderr, "out of memory\n");
            return false;
        }
    } else {
        fprintf(stderr, "--prompt or --ids is required\n");
        return false;
    }
    if (out->count == 0) {
        fprintf(stderr, "the prompt has no tokens\n");
        return false;
    }
    return true;
}

static int cmd_tokenize(const args_t* a) {
    if (!a->model) { fprintf(stderr, "--model <dir> is required\n"); return 1; }
    vitna_tokenizer_t* tok = load_tokenizer(a->model);
    if (!tok) return 1;
    binary_stdio();
    int rc = 0;
    vitna_token_list_t ids = {0};
    if (a->text) {
        if (!vitna_tokenizer_encode(tok, a->text, strlen(a->text), &ids)) rc = 1;
        print_ids(stdout, ids.ids, ids.count);
        fputc('\n', stdout);
    } else {
        /* One JSON string per line in, one array per line out. */
        size_t cap = 1 << 16, len = 0;
        char* buf = (char*)malloc(cap);
        for (size_t got; buf && (got = fread(buf + len, 1, cap - len, stdin)) > 0;) {
            len += got;
            if (len == cap) {
                char* grown = (char*)realloc(buf, cap *= 2);
                if (!grown) { free(buf); buf = NULL; }
                buf = grown;
            }
        }
        if (!buf) { fprintf(stderr, "out of memory\n"); vitna_tokenizer_free(tok); return 1; }
        size_t start = 0;
        for (size_t i = 0; i <= len && rc == 0; i++) {
            if (i < len && buf[i] != '\n') continue;
            size_t end = i;
            if (end > start && buf[end - 1] == '\r') end--;
            if (end > start) {
                char err[160];
                vitna_json_doc_t* doc = vitna_json_parse(buf + start, end - start, err, sizeof(err));
                const vitna_json_value_t* v = doc ? vitna_json_root(doc) : NULL;
                if (!v || v->type != VITNA_JSON_STRING) {
                    fprintf(stderr, "line is not a JSON string: %s\n", doc ? "wrong type" : err);
                    rc = 1;
                } else {
                    ids.count = 0;
                    if (!vitna_tokenizer_encode(tok, v->u.string.ptr, v->u.string.len, &ids)) rc = 1;
                    print_ids(stdout, ids.ids, ids.count);
                    fputc('\n', stdout);
                }
                vitna_json_free(doc);
            }
            start = i + 1;
        }
        free(buf);
    }
    vitna_token_list_free(&ids);
    vitna_tokenizer_free(tok);
    fflush(stdout);
    return rc;
}

static bool wants_cuda(const args_t* a) {
    return a->device && strcmp(a->device, "cuda") == 0;
}

static bool load_model(const args_t* a, vitna_llama_t* m) {
    char err[512];
    if (!vitna_llama_load(m, a->model, a->ctx, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return false;
    }
    /* Asked for the GPU, the model runs there or not at all. */
    if (wants_cuda(a) && !vitna_llama_use_cuda(m, err, sizeof(err))) {
        fprintf(stderr, "--device cuda: %s\n", err);
        vitna_llama_free(m);
        return false;
    }
    return true;
}

static int cmd_logits(const args_t* a) {
    if (!a->model || !a->out) { fprintf(stderr, "--model <dir> and --out <file> are required\n"); return 1; }
    vitna_tokenizer_t* tok = a->ids ? NULL : load_tokenizer(a->model);
    if (!a->ids && !tok) return 1;
    vitna_token_list_t ids = {0};
    int rc = 1;
    vitna_llama_t m;
    if (prompt_ids(a, tok, &ids) && load_model(a, &m)) {
        /* Every position's logits, up to 256 positions at a time. */
        const size_t chunk = 256, V = m.cfg.vocab;
        FILE* f = fopen(a->out, "wb");
        float* rows = (float*)malloc(chunk * V * sizeof(float));
        if (!f || !rows) {
            fprintf(stderr, "cannot write %s\n", a->out);
        } else {
            rc = 0;
            for (size_t t = 0; t < ids.count && rc == 0; t += chunk) {
                const size_t n = ids.count - t < chunk ? ids.count - t : chunk;
                const size_t ran = vitna_llama_steps(&m, ids.ids + t, n, rows, n);
                if (ran < n) {
                    fprintf(stderr, "token %d at position %zu: out of range or past --ctx\n", ids.ids[t + ran], t + ran);
                    rc = 1;
                } else if (fwrite(rows, sizeof(float), n * V, f) != n * V) {
                    fprintf(stderr, "cannot write %s\n", a->out);
                    rc = 1;
                }
            }
        }
        if (f) fclose(f);
        free(rows);
        vitna_llama_free(&m);
    }
    vitna_token_list_free(&ids);
    vitna_tokenizer_free(tok);
    return rc;
}

/* Prefill the prompt, then choose max_new tokens one at a time. Calls
 * on_token after each, and writes each step's logits to logits_out. */
/* Drafts for speculative decoding, by prompt lookup: the latest earlier place
 * in ctx where its last n tokens occur (n = 3, then 2), and up to k of the
 * tokens that followed them there, into out. Returns how many. A match of one
 * token drafts nothing: measured, its drafts were taken too seldom to pay for
 * the pass that checks them. */
static size_t draft_by_lookup(const int32_t* ctx, size_t len, size_t k, int32_t* out) {
    for (size_t n = 3; n >= 2; n--) {
        if (len <= n) continue;
        const int32_t* key = ctx + len - n;
        for (size_t i = len - n; i-- > 0;) {
            if (memcmp(ctx + i, key, n * sizeof(int32_t)) != 0) continue;
            size_t d = 0;
            while (d < k && i + n + d < len) {
                out[d] = ctx[i + n + d];
                d++;
            }
            if (d > 0) return d;
        }
    }
    return 0;
}

/* One new token from a row of logits, as generate takes every one: the row
 * to --logits-out, the choice, the list, the callback. Returns 1 to go on, 0
 * to stop (a special token under --stop-at-special), -1 on an error. */
static int take_token(const args_t* a, const vitna_tokenizer_t* tok, vitna_sampler_t* sampler, const float* row, size_t V, FILE* logits_out,
                      vitna_token_list_t* out, void (*on_token)(const vitna_tokenizer_t*, int32_t), int32_t* chosen) {
    if (logits_out && fwrite(row, sizeof(float), V, logits_out) != V) {
        fprintf(stderr, "cannot write the logits\n");
        return -1;
    }
    const int32_t next = vitna_sample(sampler, row, &a->sampling);
    if (!vitna_token_list_push(out, next)) return -1;
    if (on_token) on_token(tok, next);
    *chosen = next;
    return a->stop_at_special && tok && vitna_tokenizer_is_special(tok, next) ? 0 : 1;
}

static int generate(const args_t* a, vitna_llama_t* m, const vitna_tokenizer_t* tok, const vitna_token_list_t* prompt,
                    vitna_token_list_t* out, FILE* logits_out, void (*on_token)(const vitna_tokenizer_t*, int32_t)) {
    const size_t V = m->cfg.vocab;
    /* With --speculate k, each token taken is followed by up to k drafted
     * ones, checked in one call of vitna_llama_steps_exact: as many as a pass
     * takes on a GPU; on the CPU as asked, a step at a time. */
    size_t k = a->speculate;
    const size_t pass_max = vitna_llama_exact_max(m);
    if (k && pass_max > 1 && k > pass_max - 1) k = pass_max - 1;
    float* row = (float*)malloc(V * sizeof(float));
    float* rows = k ? (float*)malloc((k + 1) * V * sizeof(float)) : NULL;
    int32_t* pass = k ? (int32_t*)malloc((k + 1) * sizeof(int32_t)) : NULL;
    int32_t* text = k ? (int32_t*)malloc((prompt->count + a->max_new + 1) * sizeof(int32_t)) : NULL;
    vitna_sampler_t sampler;
    if (!row || (k && (!rows || !pass || !text)) || !vitna_sampler_init(&sampler, V, a->sampling.seed)) {
        free(row);
        free(rows);
        free(pass);
        free(text);
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    size_t text_len = 0;
    if (k) {
        memcpy(text, prompt->ids, prompt->count * sizeof(int32_t));
        text_len = prompt->count;
    }
    size_t passes = 0, drafted = 0, accepted = 0; /* for --timing */
    /* After a pass that took none of its drafts, the next skip tokens are
     * not drafted for, the pause doubling each time that happens again, up to
     * 16, and starting over from 1 once a pass takes one. */
    size_t skip = 0, backoff = 1;
    int rc = 0;
    const double t_prompt = vitna_time_ms();
    const size_t ran = vitna_llama_steps(m, prompt->ids, prompt->count, row, 1);
    if (ran < prompt->count) {
        fprintf(stderr, "token %d at position %zu: out of range or past --ctx\n", prompt->ids[ran], ran);
        rc = 1;
    }
    const double t_first = vitna_time_ms();
    double t_after_first = t_first;

    /* The first new token, from the prompt's logits. */
    int32_t t = 0;
    int go = rc == 0 && a->max_new > 0 ? take_token(a, tok, &sampler, row, V, logits_out, out, on_token, &t) : 0;
    t_after_first = vitna_time_ms();
    if (go < 0) rc = 1;
    if (k && go >= 0) text[text_len++] = t;

    /* Then each token t taken needs the logits after it: from a step, or from
     * a pass over t and the tokens drafted after it, whose rows serve while
     * the choices agree with the drafts. Every row is the one a step would
     * give, and every choice is taken from it in turn, so the reply is the
     * same with or without drafts. */
    while (go > 0 && out->count < a->max_new) {
        size_t d = 0;
        if (k && skip) {
            skip--;
        } else if (k) {
            d = draft_by_lookup(text, text_len, k, pass + 1);
            const size_t left = a->max_new - out->count; /* choices still to take, 1 or more */
            if (d > left - 1) d = left - 1;
            if (m->n_past + d + 1 > m->ctx) d = m->n_past + 1 < m->ctx ? m->ctx - m->n_past - 1 : 0;
        }
        if (d == 0) {
            if (!vitna_llama_step(m, t, row)) {
                fprintf(stderr, "the key-value cache is full at %zu positions; raise --ctx\n", m->ctx);
                rc = 1;
                break;
            }
            go = take_token(a, tok, &sampler, row, V, logits_out, out, on_token, &t);
            if (go < 0) rc = 1;
            if (k && go >= 0) text[text_len++] = t;
            continue;
        }
        pass[0] = t;
        passes++;
        drafted += d;
        const size_t start = m->n_past;
        if (vitna_llama_steps_exact(m, pass, d + 1, rows) < d + 1) {
            fprintf(stderr, "the forward pass failed at position %zu\n", m->n_past);
            rc = 1;
            break;
        }
        for (size_t i = 0; i <= d; i++) {
            int32_t u = 0;
            go = take_token(a, tok, &sampler, rows + i * V, V, logits_out, out, on_token, &u);
            if (go < 0) {
                rc = 1;
                break;
            }
            text[text_len++] = u;
            /* The next row is ready while u is the next draft. Otherwise the
             * pass's positions after pass[i] held drafts not taken: forget them. */
            if (go > 0 && out->count < a->max_new && i < d && u == pass[i + 1]) {
                accepted++;
                continue;
            }
            if (i == 0) {
                skip = backoff;
                backoff = backoff < 16 ? 2 * backoff : 16;
            } else {
                backoff = 1;
            }
            vitna_llama_truncate(m, start + i + 1);
            t = u;
            break;
        }
    }
    /* Each token after the first new one cost one forward step and one choice. */
    if (a->timing && rc == 0 && out->count > 1) {
        const double ms = vitna_time_ms() - t_after_first;
        fprintf(stderr, "timing: %zu prompt tokens in %.3f ms; %zu tokens after the first new one in %.3f ms, %.4f ms each\n",
                prompt->count, t_first - t_prompt, out->count - 1, ms, ms / (double)(out->count - 1));
        if (k) fprintf(stderr, "speculation: %zu passes, %zu tokens drafted, %zu of them taken\n", passes, drafted, accepted);
    }
    vitna_sampler_free(&sampler);
    free(row);
    free(rows);
    free(pass);
    free(text);
    return rc;
}

static void append_bytes(unsigned char** buf, size_t* len, size_t* cap, const unsigned char* p, size_t n) {
    if (*len + n > *cap) {
        size_t c = *cap ? *cap : 256;
        while (c < *len + n) c *= 2;
        unsigned char* g = (unsigned char*)realloc(*buf, c);
        if (!g) return;
        *buf = g;
        *cap = c;
    }
    memcpy(*buf + *len, p, n);
    *len += n;
}

static int cmd_generate(const args_t* a) {
    if (!a->model) { fprintf(stderr, "--model <dir> is required\n"); return 1; }
    vitna_tokenizer_t* tok = load_tokenizer(a->model);
    if (!tok) return 1;
    binary_stdio();
    vitna_token_list_t prompt = {0}, out = {0};
    int rc = 1;
    vitna_llama_t m;
    if (prompt_ids(a, tok, &prompt) && load_model(a, &m)) {
        FILE* lf = a->logits_out ? fopen(a->logits_out, "wb") : NULL;
        if (a->logits_out && !lf) {
            fprintf(stderr, "cannot write %s\n", a->logits_out);
        } else {
            rc = generate(a, &m, tok, &prompt, &out, lf, NULL);
        }
        if (lf) fclose(lf);
        vitna_llama_free(&m);
    }
    if (rc == 0) {
        unsigned char* text = NULL;
        size_t len = 0, cap = 0;
        for (size_t i = 0; i < out.count; i++) {
            size_t n;
            const unsigned char* b = vitna_tokenizer_token_bytes(tok, out.ids[i], &n);
            if (b) append_bytes(&text, &len, &cap, b, n);
        }
        printf("{\"prompt_ids\":");
        print_ids(stdout, prompt.ids, prompt.count);
        printf(",\"ids\":");
        print_ids(stdout, out.ids, out.count);
        printf(",\"text\":");
        print_json_string(stdout, text ? text : (const unsigned char*)"", len);
        printf("}\n");
        free(text);
    }
    vitna_token_list_free(&prompt);
    vitna_token_list_free(&out);
    vitna_tokenizer_free(tok);
    fflush(stdout);
    return rc;
}

static void print_token(const vitna_tokenizer_t* tok, int32_t id) {
    size_t n;
    const unsigned char* b = vitna_tokenizer_token_bytes(tok, id, &n);
    if (b && !vitna_tokenizer_is_special(tok, id)) {
        fwrite(b, 1, n, stdout);
        fflush(stdout);
    }
}

static int cmd_run(args_t* a) {
    if (!a->model || !a->prompt) { fprintf(stderr, "--model <dir> and --prompt <text> are required\n"); return 1; }
    vitna_tokenizer_t* tok = load_tokenizer(a->model);
    if (!tok) return 1;
#if defined(VITNA_OS_WINDOWS)
    SetConsoleOutputCP(65001); /* UTF-8 */
#endif
    a->stop_at_special = true;
    vitna_token_list_t prompt = {0}, out = {0};
    int rc = 1;
    vitna_llama_t m;
    if (prompt_ids(a, tok, &prompt) && load_model(a, &m)) {
        fputs(a->prompt, stdout);
        rc = generate(a, &m, tok, &prompt, &out, NULL, print_token);
        fputc('\n', stdout);
        vitna_llama_free(&m);
    }
    vitna_token_list_free(&prompt);
    vitna_token_list_free(&out);
    vitna_tokenizer_free(tok);
    return rc;
}

static int cmd_info(const char* model_path) {
    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required\n");
        return 1;
    }

    vitna_safetensors_t st;
    char err[256];
    if (!vitna_safetensors_open_ex(model_path, &st, err, sizeof(err))) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", err);
        return 1;
    }

    printf("Safetensors model: %s\n", model_path);
    printf("Header size: %llu bytes\n", (unsigned long long)st.header_len);
    printf("Tensor count: %zu\n\n", st.tensor_count);

    for (size_t i = 0; i < st.tensor_count && i < 25; i++) {
        const vitna_tensor_desc_t* t = &st.tensors[i];
        printf("  [%3zu] %-50s %-4s shape: [", i, t->name, vitna_dtype_name(t->dtype));
        for (size_t d = 0; d < t->ndim; d++) {
            printf("%zu%s", t->shape[d], (d + 1 < t->ndim) ? ", " : "");
        }
        printf("] offsets: [%llu..%llu]\n",
            (unsigned long long)t->offset_begin,
            (unsigned long long)t->offset_end
        );
    }
    if (st.tensor_count > 25) {
        printf("  ... and %zu more tensors\n", st.tensor_count - 25);
    }

    vitna_safetensors_close(&st);
    return 0;
}

static int cmd_bench(int iterations) {
    vitna_simd_capabilities_t caps = vitna_detect_simd_capabilities();
    printf("int4 matrix-vector kernel, 2048 x 2048, synthetic weights, %d iterations\n", iterations);
    printf("CPU features detected: AVX2 %s, AVX-512F %s, NEON %s, FMA %s\n",
        caps.has_avx2 ? "yes" : "no",
        caps.has_avx512 ? "yes" : "no",
        caps.has_neon ? "yes" : "no",
        caps.has_fma ? "yes" : "no"
    );
    double t0 = vitna_time_ms();

    /* Time the int4 GEMV kernel on constant synthetic weights. This measures
     * one kernel on this machine; it says nothing about running a model. */
    const size_t rows = 2048;
    const size_t cols = 2048;
    size_t packed_bytes = (rows * cols) / 2;
    uint8_t* packed_w = (uint8_t*)vitna_aligned_alloc(64, packed_bytes);
    float* scales = (float*)vitna_aligned_alloc(64, (rows * cols / 128) * sizeof(float));
    float* x = (float*)vitna_aligned_alloc(64, cols * sizeof(float));
    float* y = (float*)vitna_aligned_alloc(64, rows * sizeof(float));

    if (packed_w && scales && x && y) {
        memset(packed_w, 0x88, packed_bytes);
        for (size_t i = 0; i < cols; i++) x[i] = 1.0f;
        for (size_t i = 0; i < (rows * cols / 128); i++) scales[i] = 0.05f;

        for (int it = 0; it < iterations; it++) {
            vitna_gemv_int4(packed_w, scales, 128, x, y, rows, cols);
        }
    }

    vitna_aligned_free(packed_w);
    vitna_aligned_free(scales);
    vitna_aligned_free(x);
    vitna_aligned_free(y);

    double t1 = vitna_time_ms();
    printf("%d int4 matrix-vector products in %.2f ms (%.2f per second)\n",
        iterations, t1 - t0, (iterations / ((t1 - t0) / 1000.0))
    );
    return 0;
}

/* The last path component of dir, without trailing slashes: the default model id. */
static void dir_basename(const char* dir, char* out, size_t n) {
    size_t end = strlen(dir);
    while (end > 0 && (dir[end - 1] == '/' || dir[end - 1] == '\\')) end--;
    size_t start = end;
    while (start > 0 && dir[start - 1] != '/' && dir[start - 1] != '\\') start--;
    size_t len = end - start < n - 1 ? end - start : n - 1;
    memcpy(out, dir + start, len);
    out[len] = '\0';
}

static int cmd_serve(const args_t* a) {
    vitna_server_config_t cfg;
    cfg.port = a->port;
    cfg.bind_addr = a->host ? a->host : "127.0.0.1";
    cfg.engine_ctx = NULL;
    if (!a->model) return vitna_server_run(&cfg);

    vitna_tokenizer_t* tok = load_tokenizer(a->model);
    if (!tok) return 1;
    vitna_llama_t m;
    if (!load_model(a, &m)) {
        vitna_tokenizer_free(tok);
        return 1;
    }
    /* For tests only: VITNA_TEST_FAIL_STEP=<position> makes the step at that
     * position fail once, so tests/step-failure.test.mjs can check what a
     * client sees when a step fails, with no GPU error to cause one. */
    const char* fail = getenv("VITNA_TEST_FAIL_STEP");
    if (fail && *fail) {
        char* end;
        unsigned long long pos = strtoull(fail, &end, 10);
        if (fail[0] < '0' || fail[0] > '9' || *end) {
            fprintf(stderr, "VITNA_TEST_FAIL_STEP must be a position, a whole number, not %s\n", fail);
            vitna_llama_free(&m);
            vitna_tokenizer_free(tok);
            return 1;
        }
        vitna_llama_fail_step_once(&m, (size_t)pos);
        fprintf(stderr, "VITNA_TEST_FAIL_STEP is set, for a test: the step at position %llu will fail once.\n", pos);
    }
    char id[128];
    if (a->model_id) snprintf(id, sizeof(id), "%s", a->model_id);
    else dir_basename(a->model, id, sizeof(id));
    vitna_api_t* api = vitna_api_create(&m, tok, id[0] ? id : "model");
    int rc = 1;
    if (!api) {
        fprintf(stderr, "out of memory\n");
    } else {
        vitna_api_set_prefix_cache(api, !a->no_prefix_cache);
        char device[400];
        printf("Loaded %s: %zu layers, %zu-token context, %s.\n", vitna_api_model_id(api), m.cfg.n_layers, m.ctx, vitna_llama_device(&m, device, sizeof(device)));
        cfg.engine_ctx = api;
        rc = vitna_server_run(&cfg);
        vitna_api_free(api);
    }
    vitna_llama_free(&m);
    vitna_tokenizer_free(tok);
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* cmd = argv[1];
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0 || strcmp(cmd, "help") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    args_t a;
    if (!parse_args(argc, argv, &a)) return 1;
    /* A device the engine cannot use is refused before anything is loaded,
     * and nothing falls back to the CPU in its place. */
    if (a.device && strcmp(a.device, "cpu") != 0) {
        char err[512];
        if (!wants_cuda(&a)) {
            fprintf(stderr, "--device must be cpu or cuda, not %s\n", a.device);
            return 1;
        }
        if (!vitna_llama_cuda_probe(err, sizeof(err))) {
            fprintf(stderr, "--device cuda: %s\n", err);
            return 1;
        }
    }

    if (strcmp(cmd, "run") == 0) return cmd_run(&a);
    if (strcmp(cmd, "generate") == 0) return cmd_generate(&a);
    if (strcmp(cmd, "logits") == 0) return cmd_logits(&a);
    if (strcmp(cmd, "tokenize") == 0) return cmd_tokenize(&a);
    if (strcmp(cmd, "info") == 0) return cmd_info(a.model);
    if (strcmp(cmd, "bench") == 0) return cmd_bench(a.iterations);
    if (strcmp(cmd, "serve") == 0) return cmd_serve(&a);
    print_usage(argv[0]);
    return 1;
}
