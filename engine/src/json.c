/**
 * json.c - A small JSON parser (RFC 8259) into an arena-held tree.
 */

#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Arena --- */

typedef struct arena_block {
    struct arena_block* next;
    size_t used;
    size_t cap;
    /* data follows */
} arena_block_t;

struct vitna_json_doc {
    arena_block_t* blocks;
    vitna_json_value_t* root;
};

static void* arena_alloc(vitna_json_doc_t* doc, size_t size) {
    size = (size + 15) & ~(size_t)15;
    arena_block_t* b = doc->blocks;
    if (!b || b->used + size > b->cap) {
        size_t cap = size > (1u << 20) ? size : (1u << 20);
        arena_block_t* nb = (arena_block_t*)malloc(sizeof(arena_block_t) + cap);
        if (!nb) return NULL;
        nb->next = doc->blocks;
        nb->used = 0;
        nb->cap = cap;
        doc->blocks = nb;
        b = nb;
    }
    void* p = (char*)(b + 1) + b->used;
    b->used += size;
    return p;
}

/* --- Parser --- */

typedef struct {
    const char* p;
    const char* end;
    vitna_json_doc_t* doc;
    char* err;
    size_t err_len;
    const char* start;
    int depth;
} parser_t;

static bool fail(parser_t* ps, const char* msg) {
    if (ps->err && ps->err_len > 0) {
        snprintf(ps->err, ps->err_len, "JSON: %s at byte %zu", msg, (size_t)(ps->p - ps->start));
    }
    return false;
}

static void skip_ws(parser_t* ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool read_hex4(parser_t* ps, uint32_t* out) {
    if (ps->end - ps->p < 4) return fail(ps, "truncated \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hexval(ps->p[i]);
        if (h < 0) return fail(ps, "bad \\u escape");
        v = (v << 4) | (uint32_t)h;
    }
    ps->p += 4;
    *out = v;
    return true;
}

static size_t put_utf8(char* out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Parse a string starting at the opening quote. The decoded form is never
 * longer than the encoded one, so the output is sized by the input. */
static bool parse_string(parser_t* ps, const char** out, size_t* out_len) {
    ps->p++; /* opening quote */
    const char* s = ps->p;
    /* Find the closing quote to size the buffer. */
    const char* q = s;
    while (q < ps->end && *q != '"') {
        if (*q == '\\') {
            if (ps->end - q < 2) break;
            q++;
        }
        q++;
    }
    if (q >= ps->end) return fail(ps, "unterminated string");
    char* buf = (char*)arena_alloc(ps->doc, (size_t)(q - s) + 1);
    if (!buf) return fail(ps, "out of memory");
    size_t n = 0;
    while (ps->p < ps->end && *ps->p != '"') {
        unsigned char c = (unsigned char)*ps->p;
        if (c < 0x20) return fail(ps, "control character in string");
        if (c != '\\') {
            buf[n++] = (char)c;
            ps->p++;
            continue;
        }
        ps->p++;
        if (ps->p >= ps->end) return fail(ps, "truncated escape");
        char e = *ps->p++;
        switch (e) {
            case '"': buf[n++] = '"'; break;
            case '\\': buf[n++] = '\\'; break;
            case '/': buf[n++] = '/'; break;
            case 'b': buf[n++] = '\b'; break;
            case 'f': buf[n++] = '\f'; break;
            case 'n': buf[n++] = '\n'; break;
            case 'r': buf[n++] = '\r'; break;
            case 't': buf[n++] = '\t'; break;
            case 'u': {
                uint32_t cp;
                if (!read_hex4(ps, &cp)) return false;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    /* A high surrogate needs a low one after it. */
                    uint32_t lo = 0;
                    if (ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
                        const char* save = ps->p;
                        ps->p += 2;
                        if (!read_hex4(ps, &lo)) return false;
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            ps->p = save;
                            cp = 0xFFFD;
                        }
                    } else {
                        cp = 0xFFFD;
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                n += put_utf8(buf + n, cp);
                break;
            }
            default:
                return fail(ps, "bad escape");
        }
    }
    if (ps->p >= ps->end) return fail(ps, "unterminated string");
    ps->p++; /* closing quote */
    buf[n] = '\0';
    *out = buf;
    *out_len = n;
    return true;
}

static bool parse_number(parser_t* ps, double* out) {
    const char* s = ps->p;
    const char* p = s;
    if (p < ps->end && *p == '-') p++;
    if (p >= ps->end) return fail(ps, "bad number");
    if (*p == '0') {
        p++;
    } else if (*p >= '1' && *p <= '9') {
        while (p < ps->end && *p >= '0' && *p <= '9') p++;
    } else {
        return fail(ps, "bad number");
    }
    if (p < ps->end && *p == '.') {
        p++;
        if (p >= ps->end || *p < '0' || *p > '9') return fail(ps, "bad fraction");
        while (p < ps->end && *p >= '0' && *p <= '9') p++;
    }
    if (p < ps->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < ps->end && (*p == '+' || *p == '-')) p++;
        if (p >= ps->end || *p < '0' || *p > '9') return fail(ps, "bad exponent");
        while (p < ps->end && *p >= '0' && *p <= '9') p++;
    }
    char tmp[64];
    size_t n = (size_t)(p - s);
    if (n >= sizeof(tmp)) {
        /* A number this long is legal; strtod reads it from a heap copy. */
        char* big = (char*)malloc(n + 1);
        if (!big) return fail(ps, "out of memory");
        memcpy(big, s, n);
        big[n] = '\0';
        *out = strtod(big, NULL);
        free(big);
    } else {
        memcpy(tmp, s, n);
        tmp[n] = '\0';
        *out = strtod(tmp, NULL);
    }
    ps->p = p;
    return true;
}

static vitna_json_value_t* parse_value(parser_t* ps);

/* A growable list of pointers, on the heap while parsing, copied into the
 * arena once its length is known. */
typedef struct {
    void** items;
    size_t count;
    size_t cap;
} ptr_list_t;

static bool ptr_list_push(ptr_list_t* l, void* item) {
    if (l->count == l->cap) {
        size_t cap = l->cap ? l->cap * 2 : 8;
        void** grown = (void**)realloc(l->items, cap * sizeof(void*));
        if (!grown) return false;
        l->items = grown;
        l->cap = cap;
    }
    l->items[l->count++] = item;
    return true;
}

static vitna_json_value_t* parse_array(parser_t* ps, vitna_json_value_t* v) {
    ps->p++; /* [ */
    ptr_list_t list = {0};
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') {
        ps->p++;
    } else {
        for (;;) {
            vitna_json_value_t* item = parse_value(ps);
            if (!item || !ptr_list_push(&list, item)) { free(list.items); if (item) fail(ps, "out of memory"); return NULL; }
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == ']') { ps->p++; break; }
            free(list.items);
            fail(ps, "expected , or ] in array");
            return NULL;
        }
    }
    v->type = VITNA_JSON_ARRAY;
    v->u.array.count = list.count;
    v->u.array.items = NULL;
    if (list.count > 0) {
        v->u.array.items = (vitna_json_value_t**)arena_alloc(ps->doc, list.count * sizeof(vitna_json_value_t*));
        if (!v->u.array.items) { free(list.items); fail(ps, "out of memory"); return NULL; }
        memcpy(v->u.array.items, list.items, list.count * sizeof(vitna_json_value_t*));
    }
    free(list.items);
    return v;
}

static vitna_json_value_t* parse_object(parser_t* ps, vitna_json_value_t* v) {
    ps->p++; /* { */
    ptr_list_t keys = {0}, lens = {0}, values = {0};
    skip_ws(ps);
    bool ok = true;
    if (ps->p < ps->end && *ps->p == '}') {
        ps->p++;
    } else {
        for (;;) {
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != '"') { ok = fail(ps, "expected a string key"); break; }
            const char* key; size_t key_len;
            if (!parse_string(ps, &key, &key_len)) { ok = false; break; }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { ok = fail(ps, "expected : after key"); break; }
            ps->p++;
            vitna_json_value_t* val = parse_value(ps);
            if (!val) { ok = false; break; }
            if (!ptr_list_push(&keys, (void*)key) || !ptr_list_push(&lens, (void*)(uintptr_t)key_len) || !ptr_list_push(&values, val)) {
                ok = fail(ps, "out of memory");
                break;
            }
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == '}') { ps->p++; break; }
            ok = fail(ps, "expected , or } in object");
            break;
        }
    }
    if (ok) {
        v->type = VITNA_JSON_OBJECT;
        v->u.object.count = keys.count;
        v->u.object.members = NULL;
        if (keys.count > 0) {
            v->u.object.members = (vitna_json_member_t*)arena_alloc(ps->doc, keys.count * sizeof(vitna_json_member_t));
            if (!v->u.object.members) {
                ok = fail(ps, "out of memory");
            } else {
                for (size_t i = 0; i < keys.count; i++) {
                    v->u.object.members[i].key = (const char*)keys.items[i];
                    v->u.object.members[i].key_len = (size_t)(uintptr_t)lens.items[i];
                    v->u.object.members[i].value = (vitna_json_value_t*)values.items[i];
                }
            }
        }
    }
    free(keys.items);
    free(lens.items);
    free(values.items);
    return ok ? v : NULL;
}

static bool match_literal(parser_t* ps, const char* lit) {
    size_t n = strlen(lit);
    if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, lit, n) != 0) return false;
    ps->p += n;
    return true;
}

static vitna_json_value_t* parse_value(parser_t* ps) {
    if (++ps->depth > VITNA_JSON_MAX_DEPTH) { fail(ps, "nested too deeply"); return NULL; }
    skip_ws(ps);
    vitna_json_value_t* v = (vitna_json_value_t*)arena_alloc(ps->doc, sizeof(vitna_json_value_t));
    if (!v) { fail(ps, "out of memory"); return NULL; }
    memset(v, 0, sizeof(*v));
    vitna_json_value_t* result = NULL;
    if (ps->p >= ps->end) {
        fail(ps, "unexpected end of input");
    } else if (*ps->p == '{') {
        result = parse_object(ps, v);
    } else if (*ps->p == '[') {
        result = parse_array(ps, v);
    } else if (*ps->p == '"') {
        const char* s; size_t n;
        if (parse_string(ps, &s, &n)) {
            v->type = VITNA_JSON_STRING;
            v->u.string.ptr = s;
            v->u.string.len = n;
            result = v;
        }
    } else if (*ps->p == '-' || (*ps->p >= '0' && *ps->p <= '9')) {
        if (parse_number(ps, &v->u.number)) {
            v->type = VITNA_JSON_NUMBER;
            result = v;
        }
    } else if (match_literal(ps, "true")) {
        v->type = VITNA_JSON_TRUE;
        result = v;
    } else if (match_literal(ps, "false")) {
        v->type = VITNA_JSON_FALSE;
        result = v;
    } else if (match_literal(ps, "null")) {
        v->type = VITNA_JSON_NULL;
        result = v;
    } else {
        fail(ps, "unexpected character");
    }
    ps->depth--;
    return result;
}

vitna_json_doc_t* vitna_json_parse(const char* text, size_t len, char* err, size_t err_len) {
    if (err && err_len > 0) err[0] = '\0';
    vitna_json_doc_t* doc = (vitna_json_doc_t*)calloc(1, sizeof(vitna_json_doc_t));
    if (!doc) return NULL;
    parser_t ps = { text, text + len, doc, err, err_len, text, 0 };
    /* A UTF-8 byte order mark is allowed before the value. */
    if (len >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) {
        ps.p += 3;
    }
    doc->root = parse_value(&ps);
    if (doc->root) {
        skip_ws(&ps);
        if (ps.p != ps.end) {
            fail(&ps, "trailing characters after the value");
            doc->root = NULL;
        }
    }
    if (!doc->root) {
        vitna_json_free(doc);
        return NULL;
    }
    return doc;
}

void vitna_json_free(vitna_json_doc_t* doc) {
    if (!doc) return;
    arena_block_t* b = doc->blocks;
    while (b) {
        arena_block_t* next = b->next;
        free(b);
        b = next;
    }
    free(doc);
}

const vitna_json_value_t* vitna_json_root(const vitna_json_doc_t* doc) {
    return doc ? doc->root : NULL;
}

const vitna_json_value_t* vitna_json_get(const vitna_json_value_t* object, const char* key) {
    if (!object || object->type != VITNA_JSON_OBJECT || !key) return NULL;
    size_t n = strlen(key);
    for (size_t i = 0; i < object->u.object.count; i++) {
        const vitna_json_member_t* m = &object->u.object.members[i];
        if (m->key_len == n && memcmp(m->key, key, n) == 0) return m->value;
    }
    return NULL;
}

bool vitna_json_as_number(const vitna_json_value_t* v, double* out) {
    if (!v || v->type != VITNA_JSON_NUMBER) return false;
    *out = v->u.number;
    return true;
}

bool vitna_json_as_bool(const vitna_json_value_t* v, bool* out) {
    if (!v || (v->type != VITNA_JSON_TRUE && v->type != VITNA_JSON_FALSE)) return false;
    *out = v->type == VITNA_JSON_TRUE;
    return true;
}

const char* vitna_json_as_string(const vitna_json_value_t* v) {
    return (v && v->type == VITNA_JSON_STRING) ? v->u.string.ptr : NULL;
}

bool vitna_json_is_null(const vitna_json_value_t* v) {
    return v && v->type == VITNA_JSON_NULL;
}

char* vitna_read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    char* buf = NULL;
    size_t len = 0, cap = 0;
    for (;;) {
        if (cap - len < 65536) {
            size_t ncap = cap ? cap * 2 : 1 << 20;
            char* grown = (char*)realloc(buf, ncap + 1);
            if (!grown) { free(buf); fclose(f); return NULL; }
            buf = grown;
            cap = ncap;
        }
        size_t got = fread(buf + len, 1, cap - len, f);
        len += got;
        if (got == 0) break;
    }
    int bad = ferror(f);
    fclose(f);
    if (bad) { free(buf); return NULL; }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}
