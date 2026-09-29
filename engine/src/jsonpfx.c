/**
 * jsonpfx.c - Is this text the start of a JSON object? See jsonpfx.h.
 *
 * A pushdown automaton over bytes, following RFC 8259's grammar: the stack
 * holds the open objects and arrays, and the state says what may come next.
 * Numbers, literals, escapes and multi-byte UTF-8 characters are followed a
 * byte at a time, so a candidate token that would end half-way through one
 * is judged on whether it can still be finished.
 */

#include "jsonpfx.h"
#include <string.h>

enum {
    S_START,     /* before the top-level object: whitespace or '{' */
    S_OBJ_OPEN,  /* after '{': a key or '}' */
    S_OBJ_KEY,   /* after ',' in an object: a key */
    S_COLON,     /* after a key: ':' */
    S_VALUE,     /* a value */
    S_ARR_OPEN,  /* after '[': a value or ']' */
    S_AFTER,     /* after a value in a container: ',' or its closer */
    S_STRING,
    S_NUMBER,
    S_LITERAL,
    S_DONE       /* the top-level object has closed: whitespace only */
};

enum { N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_EXP, N_EXP_SIGN, N_EXP_DIGITS };
enum { STR_NORMAL, STR_ESCAPE, STR_U1, STR_U2, STR_U3, STR_U4 };

static bool fail(vitna_jsonpfx_t* p) {
    p->failed = true;
    return false;
}

static bool is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static bool is_hex(unsigned char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static void value_done(vitna_jsonpfx_t* p) {
    if (p->depth == 0) {
        p->done = true;
        p->state = S_DONE;
    } else {
        p->state = S_AFTER;
    }
}

static bool push(vitna_jsonpfx_t* p, uint8_t kind, uint8_t next) {
    if (p->depth >= VITNA_JSONPFX_MAX_DEPTH) return fail(p);
    p->stack[p->depth++] = kind;
    p->state = next;
    return true;
}

static bool close_container(vitna_jsonpfx_t* p) {
    p->depth--;
    value_done(p);
    return true;
}

static bool start_string(vitna_jsonpfx_t* p, bool key) {
    p->in_key = key;
    p->state = S_STRING;
    p->sub = STR_NORMAL;
    return true;
}

static bool start_value(vitna_jsonpfx_t* p, unsigned char c) {
    switch (c) {
        case '{': return push(p, 'O', S_OBJ_OPEN);
        case '[': return push(p, 'A', S_ARR_OPEN);
        case '"': return start_string(p, false);
        case '-': p->state = S_NUMBER; p->sub = N_MINUS; return true;
        case '0': p->state = S_NUMBER; p->sub = N_ZERO; return true;
        case 't': case 'f': case 'n': p->state = S_LITERAL; p->lit = c; p->sub = 1; return true;
        default:
            if (c >= '1' && c <= '9') {
                p->state = S_NUMBER;
                p->sub = N_INT;
                return true;
            }
            return fail(p);
    }
}

/* Whether c continues the number, updating the sub-state if so. */
static bool number_continue(vitna_jsonpfx_t* p, unsigned char c) {
    switch (p->sub) {
        case N_MINUS:
            if (c == '0') { p->sub = N_ZERO; return true; }
            if (c >= '1' && c <= '9') { p->sub = N_INT; return true; }
            return false;
        case N_ZERO:
            if (c == '.') { p->sub = N_DOT; return true; }
            if (c == 'e' || c == 'E') { p->sub = N_EXP; return true; }
            return false;
        case N_INT:
            if (is_digit(c)) return true;
            if (c == '.') { p->sub = N_DOT; return true; }
            if (c == 'e' || c == 'E') { p->sub = N_EXP; return true; }
            return false;
        case N_DOT:
            if (is_digit(c)) { p->sub = N_FRAC; return true; }
            return false;
        case N_FRAC:
            if (is_digit(c)) return true;
            if (c == 'e' || c == 'E') { p->sub = N_EXP; return true; }
            return false;
        case N_EXP:
            if (c == '+' || c == '-') { p->sub = N_EXP_SIGN; return true; }
            if (is_digit(c)) { p->sub = N_EXP_DIGITS; return true; }
            return false;
        case N_EXP_SIGN:
            if (is_digit(c)) { p->sub = N_EXP_DIGITS; return true; }
            return false;
        default: /* N_EXP_DIGITS */
            return is_digit(c);
    }
}

static bool number_complete(uint8_t sub) {
    return sub == N_ZERO || sub == N_INT || sub == N_FRAC || sub == N_EXP_DIGITS;
}

static bool string_byte(vitna_jsonpfx_t* p, unsigned char c) {
    if (p->utf8_left) {
        if (c < p->utf8_lo || c > p->utf8_hi) return fail(p);
        p->utf8_left--;
        p->utf8_lo = 0x80;
        p->utf8_hi = 0xBF;
        return true;
    }
    switch (p->sub) {
        case STR_ESCAPE:
            if (c == 'u') { p->sub = STR_U1; return true; }
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
                p->sub = STR_NORMAL;
                return true;
            }
            return fail(p);
        case STR_U1: case STR_U2: case STR_U3: case STR_U4:
            if (!is_hex(c)) return fail(p);
            p->sub = (p->sub == STR_U4) ? STR_NORMAL : (uint8_t)(p->sub + 1);
            return true;
        default:
            break;
    }
    if (c == '"') {
        if (p->in_key) p->state = S_COLON;
        else value_done(p);
        return true;
    }
    if (c == '\\') { p->sub = STR_ESCAPE; return true; }
    if (c < 0x20) return fail(p); /* control characters must be escaped */
    if (c < 0x80) return true;
    /* The lead byte of a UTF-8 sequence, and the range its first
     * continuation byte must fall in (no overlongs, no surrogates, nothing
     * past U+10FFFF). */
    p->utf8_lo = 0x80;
    p->utf8_hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) p->utf8_left = 1;
    else if (c == 0xE0) { p->utf8_left = 2; p->utf8_lo = 0xA0; }
    else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) p->utf8_left = 2;
    else if (c == 0xED) { p->utf8_left = 2; p->utf8_hi = 0x9F; }
    else if (c == 0xF0) { p->utf8_left = 3; p->utf8_lo = 0x90; }
    else if (c >= 0xF1 && c <= 0xF3) p->utf8_left = 3;
    else if (c == 0xF4) { p->utf8_left = 3; p->utf8_hi = 0x8F; }
    else return fail(p);
    return true;
}

static bool feed_byte(vitna_jsonpfx_t* p, unsigned char c) {
    if (p->failed) return false;
    switch (p->state) {
        case S_STRING:
            return string_byte(p, c);
        case S_NUMBER:
            if (number_continue(p, c)) return true;
            if (!number_complete(p->sub)) return fail(p);
            value_done(p);
            return feed_byte(p, c); /* c ends the number, and is read in the state after it */
        case S_LITERAL: {
            const char* word = p->lit == 't' ? "true" : p->lit == 'f' ? "false" : "null";
            if (c != (unsigned char)word[p->sub]) return fail(p);
            p->sub++;
            if (word[p->sub] == '\0') value_done(p);
            return true;
        }
        default:
            break;
    }
    if (is_ws(c)) {
        if (++p->ws > VITNA_JSONPFX_MAX_WS) return fail(p);
        return true;
    }
    p->ws = 0;
    switch (p->state) {
        case S_START:
            if (c == '{') return push(p, 'O', S_OBJ_OPEN);
            return fail(p);
        case S_OBJ_OPEN:
            if (c == '}') return close_container(p);
            if (c == '"') return start_string(p, true);
            return fail(p);
        case S_OBJ_KEY:
            if (c == '"') return start_string(p, true);
            return fail(p);
        case S_COLON:
            if (c == ':') { p->state = S_VALUE; return true; }
            return fail(p);
        case S_VALUE:
            return start_value(p, c);
        case S_ARR_OPEN:
            if (c == ']') return close_container(p);
            return start_value(p, c);
        case S_AFTER: {
            uint8_t top = p->stack[p->depth - 1];
            if (c == ',') {
                p->state = (top == 'O') ? S_OBJ_KEY : S_VALUE;
                return true;
            }
            if ((c == '}' && top == 'O') || (c == ']' && top == 'A')) return close_container(p);
            return fail(p);
        }
        default: /* S_DONE */
            return fail(p);
    }
}

void vitna_jsonpfx_init(vitna_jsonpfx_t* p) {
    memset(p, 0, sizeof(*p));
    p->state = S_START;
    p->utf8_lo = 0x80;
    p->utf8_hi = 0xBF;
}

bool vitna_jsonpfx_feed(vitna_jsonpfx_t* p, const unsigned char* bytes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!feed_byte(p, bytes[i])) return false;
    }
    return !p->failed;
}

bool vitna_jsonpfx_accepts(const vitna_jsonpfx_t* p, const unsigned char* bytes, size_t n) {
    vitna_jsonpfx_t copy = *p;
    return vitna_jsonpfx_feed(&copy, bytes, n);
}

void vitna_jsonpfx_next_bytes(const vitna_jsonpfx_t* p, uint8_t allowed[32]) {
    memset(allowed, 0, 32);
    for (int b = 0; b < 256; b++) {
        vitna_jsonpfx_t copy = *p;
        if (feed_byte(&copy, (unsigned char)b)) allowed[b >> 3] |= (uint8_t)(1u << (b & 7));
    }
}

bool vitna_jsonpfx_complete(const vitna_jsonpfx_t* p) {
    return p->done && !p->failed;
}
