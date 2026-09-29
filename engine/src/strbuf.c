/**
 * strbuf.c - A growable byte buffer, and JSON output into it.
 */

#include "strbuf.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vitna_sb_init(vitna_strbuf_t* sb) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->ok = true;
}

void vitna_sb_free(vitna_strbuf_t* sb) {
    free(sb->data);
    vitna_sb_init(sb);
}

void vitna_sb_clear(vitna_strbuf_t* sb) {
    sb->len = 0;
    if (sb->data) sb->data[0] = '\0';
}

static bool reserve(vitna_strbuf_t* sb, size_t extra) {
    if (!sb->ok) return false;
    if (sb->len + extra + 1 <= sb->cap) return true;
    size_t cap = sb->cap ? sb->cap : 256;
    while (cap < sb->len + extra + 1) cap *= 2;
    char* grown = (char*)realloc(sb->data, cap);
    if (!grown) {
        sb->ok = false;
        return false;
    }
    sb->data = grown;
    sb->cap = cap;
    return true;
}

void vitna_sb_append(vitna_strbuf_t* sb, const void* data, size_t len) {
    if (!reserve(sb, len)) return;
    memcpy(sb->data + sb->len, data, len);
    sb->len += len;
    sb->data[sb->len] = '\0';
}

void vitna_sb_puts(vitna_strbuf_t* sb, const char* s) {
    vitna_sb_append(sb, s, strlen(s));
}

void vitna_sb_printf(vitna_strbuf_t* sb, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char small[256];
    int n = vsnprintf(small, sizeof(small), fmt, ap);
    va_end(ap);
    if (n < 0) {
        sb->ok = false;
        return;
    }
    if ((size_t)n < sizeof(small)) {
        vitna_sb_append(sb, small, (size_t)n);
        return;
    }
    if (!reserve(sb, (size_t)n)) return;
    va_start(ap, fmt);
    vsnprintf(sb->data + sb->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb->len += (size_t)n;
}

/* The length of a valid UTF-8 sequence starting at s, or 0 if there is none. */
static size_t utf8_valid_len(const unsigned char* s, size_t avail) {
    unsigned char c = s[0];
    size_t n;
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c >= 0xE0 && c <= 0xEF) n = 3;
    else if (c >= 0xF0 && c <= 0xF4) n = 4;
    else return 0;
    if (avail < n) return 0;
    for (size_t k = 1; k < n; k++) if ((s[k] & 0xC0) != 0x80) return 0;
    if (n == 3) {
        unsigned cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
    } else if (n == 4) {
        unsigned cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return 0;
    }
    return n;
}

void vitna_sb_json_string(vitna_strbuf_t* sb, const unsigned char* s, size_t n) {
    vitna_sb_append(sb, "\"", 1);
    size_t i = 0;
    while (i < n) {
        unsigned char c = s[i];
        if (c == '"') { vitna_sb_append(sb, "\\\"", 2); i++; continue; }
        if (c == '\\') { vitna_sb_append(sb, "\\\\", 2); i++; continue; }
        if (c == '\n') { vitna_sb_append(sb, "\\n", 2); i++; continue; }
        if (c == '\r') { vitna_sb_append(sb, "\\r", 2); i++; continue; }
        if (c == '\t') { vitna_sb_append(sb, "\\t", 2); i++; continue; }
        if (c < 0x20) { vitna_sb_printf(sb, "\\u%04x", c); i++; continue; }
        size_t l = utf8_valid_len(s + i, n - i);
        if (l == 0) {
            vitna_sb_append(sb, "\\ufffd", 6);
            i++;
        } else {
            vitna_sb_append(sb, s + i, l);
            i += l;
        }
    }
    vitna_sb_append(sb, "\"", 1);
}

size_t vitna_utf8_complete_prefix(const unsigned char* s, size_t len) {
    /* Look back at most three bytes for the start of an unfinished sequence. */
    size_t start = len;
    for (size_t back = 1; back <= 3 && back <= len; back++) {
        unsigned char c = s[len - back];
        if ((c & 0xC0) == 0x80) continue; /* a continuation byte */
        size_t need = 0;
        if (c >= 0xC2 && c <= 0xDF) need = 2;
        else if (c >= 0xE0 && c <= 0xEF) need = 3;
        else if (c >= 0xF0 && c <= 0xF4) need = 4;
        if (need > back) start = len - back; /* started, not finished */
        break;
    }
    return start;
}
