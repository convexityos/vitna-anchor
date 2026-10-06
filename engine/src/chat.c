/**
 * chat.c - A conversation in a model's own chat template (chat.h), and JSON
 * written as Python writes it.
 */

#include "chat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- JSON as Python's json.dumps(v, ensure_ascii=False) writes it --- */

/* A string: the quote, the backslash and the control characters escaped,
 * \n \r \t \b \f by name and the rest as \u00xx, lowercase; everything else,
 * DEL and every byte of UTF-8 included, as it is. */
static void write_py_string(vitna_strbuf_t* out, const char* s, size_t n) {
    vitna_sb_append(out, "\"", 1);
    size_t from = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        const char* esc = NULL;
        char u[8];
        switch (c) {
            case '"': esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\n': esc = "\\n"; break;
            case '\r': esc = "\\r"; break;
            case '\t': esc = "\\t"; break;
            case '\b': esc = "\\b"; break;
            case '\f': esc = "\\f"; break;
            default:
                if (c < 0x20) {
                    snprintf(u, sizeof(u), "\\u%04x", c);
                    esc = u;
                }
        }
        if (esc) {
            vitna_sb_append(out, s + from, i - from);
            vitna_sb_puts(out, esc);
            from = i + 1;
        }
    }
    vitna_sb_append(out, s + from, n - from);
    vitna_sb_append(out, "\"", 1);
}

void vitna_py_float_repr(double x, char* buf, size_t len) {
    if (isnan(x)) {
        snprintf(buf, len, "NaN");
        return;
    }
    if (isinf(x)) {
        snprintf(buf, len, x < 0 ? "-Infinity" : "Infinity");
        return;
    }
    if (x == 0.0) {
        snprintf(buf, len, signbit(x) ? "-0.0" : "0.0");
        return;
    }
    /* The fewest significant digits that read back as x, each correctly
     * rounded: what repr(float) prints. */
    char e[48];
    for (int p = 1; p <= 17; p++) {
        snprintf(e, sizeof(e), "%.*e", p - 1, x);
        if (strtod(e, NULL) == x) break;
    }
    const char* q = e;
    const bool neg = *q == '-';
    if (neg) q++;
    char digits[24];
    int nd = 0;
    for (; *q && *q != 'e'; q++) {
        if (*q >= '0' && *q <= '9' && nd < (int)sizeof(digits) - 1) digits[nd++] = *q;
    }
    const int exp10 = *q == 'e' ? atoi(q + 1) : 0;
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    digits[nd] = '\0';
    /* Python writes x in exponent notation when its decimal point would sit
     * more than 4 places left of the first digit or more than 16 right of it. */
    const int decpt = exp10 + 1;
    char body[64];
    if (decpt <= -4 || decpt > 16) {
        snprintf(body, sizeof(body), "%c%s%se%c%02d", digits[0], nd > 1 ? "." : "", digits + 1, exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    } else if (decpt <= 0) {
        snprintf(body, sizeof(body), "0.%.*s%s", -decpt, "0000", digits);
    } else if (decpt >= nd) {
        snprintf(body, sizeof(body), "%s%.*s.0", digits, decpt - nd, "0000000000000000");
    } else {
        snprintf(body, sizeof(body), "%.*s.%s", decpt, digits, digits + decpt);
    }
    snprintf(buf, len, "%s%s", neg ? "-" : "", body);
}

/* Whether two keys are the same. */
static bool same_key(const vitna_json_member_t* a, const vitna_json_member_t* b) {
    return a->key_len == b->key_len && memcmp(a->key, b->key, a->key_len) == 0;
}

void vitna_json_write_py(vitna_strbuf_t* out, const vitna_json_value_t* v) {
    char num[48];
    switch (v->type) {
        case VITNA_JSON_NULL: vitna_sb_puts(out, "null"); return;
        case VITNA_JSON_FALSE: vitna_sb_puts(out, "false"); return;
        case VITNA_JSON_TRUE: vitna_sb_puts(out, "true"); return;
        case VITNA_JSON_NUMBER: {
            /* json.loads makes an int of a number with no fraction and no
             * exponent, written as it was but for -0, and a float of any
             * other, written as its repr. */
            const bool is_int = v->lit && !memchr(v->lit, '.', v->lit_len) && !memchr(v->lit, 'e', v->lit_len) && !memchr(v->lit, 'E', v->lit_len);
            if (is_int) {
                if (v->lit_len == 2 && v->lit[0] == '-' && v->lit[1] == '0') vitna_sb_puts(out, "0");
                else vitna_sb_append(out, v->lit, v->lit_len);
            } else {
                vitna_py_float_repr(v->u.number, num, sizeof(num));
                vitna_sb_puts(out, num);
            }
            return;
        }
        case VITNA_JSON_STRING: write_py_string(out, v->u.string.ptr, v->u.string.len); return;
        case VITNA_JSON_ARRAY:
            vitna_sb_puts(out, "[");
            for (size_t i = 0; i < v->u.array.count; i++) {
                if (i) vitna_sb_puts(out, ", ");
                vitna_json_write_py(out, v->u.array.items[i]);
            }
            vitna_sb_puts(out, "]");
            return;
        case VITNA_JSON_OBJECT: {
            /* A dict keeps a repeated key where it first came, with its last value. */
            const vitna_json_member_t* m = v->u.object.members;
            const size_t n = v->u.object.count;
            bool first = true;
            vitna_sb_puts(out, "{");
            for (size_t i = 0; i < n; i++) {
                bool seen = false;
                for (size_t j = 0; j < i && !seen; j++) seen = same_key(&m[j], &m[i]);
                if (seen) continue;
                size_t last = i;
                for (size_t j = i + 1; j < n; j++) {
                    if (same_key(&m[j], &m[i])) last = j;
                }
                if (!first) vitna_sb_puts(out, ", ");
                first = false;
                write_py_string(out, m[i].key, m[i].key_len);
                vitna_sb_puts(out, ": ");
                vitna_json_write_py(out, m[last].value);
            }
            vitna_sb_puts(out, "}");
            return;
        }
    }
}

/* --- Qwen3's chat template --- */

typedef struct {
    const char* p;
    size_t n;
} slice_t;

static bool fail(char* err, size_t err_len, const char* msg) {
    if (err && err_len) snprintf(err, err_len, "%s", msg);
    return false;
}

/* A member that is a string, or none. */
static bool str_member(const vitna_json_value_t* obj, const char* key, slice_t* s) {
    const vitna_json_value_t* v = vitna_json_get(obj, key);
    if (!v || v->type != VITNA_JSON_STRING) return false;
    s->p = v->u.string.ptr;
    s->n = v->u.string.len;
    return true;
}

static bool role_is(const vitna_json_value_t* m, const char* role) {
    slice_t r;
    return str_member(m, "role", &r) && r.n == strlen(role) && memcmp(r.p, role, r.n) == 0;
}

static void put(vitna_strbuf_t* out, slice_t s) { vitna_sb_append(out, s.p, s.n); }

/* Python's str.find: the first place sub begins in s, or -1. */
static long find_first(slice_t s, const char* sub) {
    const size_t k = strlen(sub);
    for (size_t i = 0; k <= s.n && i + k <= s.n; i++) {
        if (memcmp(s.p + i, sub, k) == 0) return (long)i;
    }
    return -1;
}

static long find_last(slice_t s, const char* sub) {
    const size_t k = strlen(sub);
    long at = -1;
    for (size_t i = 0; k <= s.n && i + k <= s.n; i++) {
        if (memcmp(s.p + i, sub, k) == 0) at = (long)i;
    }
    return at;
}

static slice_t lstrip_nl(slice_t s) {
    while (s.n && s.p[0] == '\n') {
        s.p++;
        s.n--;
    }
    return s;
}

static slice_t rstrip_nl(slice_t s) {
    while (s.n && s.p[s.n - 1] == '\n') s.n--;
    return s;
}

static bool starts_with(slice_t s, const char* pre) {
    const size_t k = strlen(pre);
    return s.n >= k && memcmp(s.p, pre, k) == 0;
}

static bool ends_with(slice_t s, const char* suf) {
    const size_t k = strlen(suf);
    return s.n >= k && memcmp(s.p + s.n - k, suf, k) == 0;
}

/* Jinja's truthiness of a list: present and not empty. */
static bool nonempty_array(const vitna_json_value_t* v) {
    return v && v->type == VITNA_JSON_ARRAY && v->u.array.count > 0;
}

bool vitna_chat_qwen3(vitna_strbuf_t* out, const vitna_json_value_t* messages, const vitna_json_value_t* tools, int enable_thinking,
                      bool add_generation_prompt, char* err, size_t err_len) {
    if (!nonempty_array(messages)) return fail(err, err_len, "there are no messages");
    const size_t n = messages->u.array.count;
    vitna_json_value_t* const* msg = messages->u.array.items;
    for (size_t i = 0; i < n; i++) {
        if (msg[i]->type != VITNA_JSON_OBJECT) return fail(err, err_len, "a message is not an object");
    }
    const slice_t none = { "", 0 };
    slice_t sys0 = none;
    const bool first_is_system = role_is(msg[0], "system");
    if (first_is_system && !str_member(msg[0], "content", &sys0)) sys0 = none;

    if (nonempty_array(tools)) {
        vitna_sb_puts(out, "<|im_start|>system\n");
        if (first_is_system) {
            put(out, sys0);
            vitna_sb_puts(out, "\n\n");
        }
        vitna_sb_puts(out, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                           "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for (size_t t = 0; t < tools->u.array.count; t++) {
            vitna_sb_puts(out, "\n");
            vitna_json_write_py(out, tools->u.array.items[t]);
        }
        vitna_sb_puts(out, "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                           "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                           "<args-json-object>}\n</tool_call><|im_end|>\n");
    } else if (first_is_system) {
        vitna_sb_puts(out, "<|im_start|>system\n");
        put(out, sys0);
        vitna_sb_puts(out, "<|im_end|>\n");
    }

    /* The last message that is the user's own query, not tool responses
     * handed back in a user turn: an assistant turn after it keeps its
     * reasoning. With none, the last message. */
    size_t last_query = n - 1;
    for (size_t k = n; k-- > 0;) {
        slice_t c;
        if (role_is(msg[k], "user") && str_member(msg[k], "content", &c) && !(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
            last_query = k;
            break;
        }
    }

    for (size_t i = 0; i < n; i++) {
        const vitna_json_value_t* m = msg[i];
        slice_t content;
        if (!str_member(m, "content", &content)) content = none;
        if (role_is(m, "user") || (role_is(m, "system") && i > 0)) {
            slice_t role;
            str_member(m, "role", &role);
            vitna_sb_puts(out, "<|im_start|>");
            put(out, role);
            vitna_sb_puts(out, "\n");
            put(out, content);
            vitna_sb_puts(out, "<|im_end|>\n");
        } else if (role_is(m, "assistant")) {
            slice_t reasoning = none;
            if (!str_member(m, "reasoning_content", &reasoning)) {
                reasoning = none;
                const long close = find_first(content, "</think>");
                if (close >= 0) {
                    /* content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') */
                    slice_t before = { content.p, (size_t)close };
                    before = rstrip_nl(before);
                    const long open = find_last(before, "<think>");
                    if (open >= 0) {
                        before.p += open + 7;
                        before.n -= (size_t)open + 7;
                    }
                    reasoning = lstrip_nl(before);
                    /* content.split('</think>')[-1].lstrip('\n') */
                    const long last_close = find_last(content, "</think>");
                    slice_t after = { content.p + last_close + 8, content.n - (size_t)last_close - 8 };
                    content = lstrip_nl(after);
                }
            }
            vitna_sb_puts(out, "<|im_start|>assistant\n");
            if (i > last_query && (i == n - 1 || reasoning.n > 0)) {
                vitna_sb_puts(out, "<think>\n");
                put(out, rstrip_nl(lstrip_nl(reasoning)));
                vitna_sb_puts(out, "\n</think>\n\n");
                put(out, lstrip_nl(content));
            } else {
                put(out, content);
            }
            const vitna_json_value_t* calls = vitna_json_get(m, "tool_calls");
            if (nonempty_array(calls)) {
                for (size_t j = 0; j < calls->u.array.count; j++) {
                    if ((j == 0 && content.n > 0) || j > 0) vitna_sb_puts(out, "\n");
                    const vitna_json_value_t* call = calls->u.array.items[j];
                    const vitna_json_value_t* fn = call->type == VITNA_JSON_OBJECT ? vitna_json_get(call, "function") : NULL;
                    /* {% if tool_call.function %}: a function object that is not empty */
                    if (fn && fn->type == VITNA_JSON_OBJECT && fn->u.object.count > 0) call = fn;
                    vitna_sb_puts(out, "<tool_call>\n{\"name\": \"");
                    const vitna_json_value_t* name = call->type == VITNA_JSON_OBJECT ? vitna_json_get(call, "name") : NULL;
                    if (name && name->type == VITNA_JSON_STRING) vitna_sb_append(out, name->u.string.ptr, name->u.string.len);
                    else if (name) vitna_json_write_py(out, name);
                    vitna_sb_puts(out, "\", \"arguments\": ");
                    const vitna_json_value_t* args = call->type == VITNA_JSON_OBJECT ? vitna_json_get(call, "arguments") : NULL;
                    if (args && args->type == VITNA_JSON_STRING) vitna_sb_append(out, args->u.string.ptr, args->u.string.len);
                    else if (args) vitna_json_write_py(out, args);
                    else vitna_sb_puts(out, "{}");
                    vitna_sb_puts(out, "}\n</tool_call>");
                }
            }
            vitna_sb_puts(out, "<|im_end|>\n");
        } else if (role_is(m, "tool")) {
            if (i == 0 || !role_is(msg[i - 1], "tool")) vitna_sb_puts(out, "<|im_start|>user");
            vitna_sb_puts(out, "\n<tool_response>\n");
            put(out, content);
            vitna_sb_puts(out, "\n</tool_response>");
            if (i == n - 1 || !role_is(msg[i + 1], "tool")) vitna_sb_puts(out, "<|im_end|>\n");
        }
    }
    if (add_generation_prompt) {
        vitna_sb_puts(out, "<|im_start|>assistant\n");
        if (enable_thinking == 0) vitna_sb_puts(out, "<think>\n\n</think>\n\n");
    }
    if (!out->ok) return fail(err, err_len, "out of memory");
    return true;
}
