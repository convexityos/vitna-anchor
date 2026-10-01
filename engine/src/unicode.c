/**
 * unicode.c - Character classes for the tokenizer, a UTF-8 decoder, and
 * Normalization Form C.
 */

#include "unicode.h"
#include "unicode_data.h"
#include <stdlib.h>
#include <string.h>

static bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool vitna_uni_is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return in_ranges(VITNA_UNI_LETTER, VITNA_UNI_LETTER_COUNT, cp);
}

bool vitna_uni_is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(VITNA_UNI_NUMBER, VITNA_UNI_NUMBER_COUNT, cp);
}

/* The White_Space property (Unicode PropList.txt), which has not changed
 * since Unicode 6.0: tab to carriage return, space, next line, no-break
 * space, ogham space mark, the spaces from en quad to hair space, line and
 * paragraph separators, narrow no-break space, medium mathematical space
 * and ideographic space. */
bool vitna_uni_is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

const char* vitna_uni_version(void) {
    return VITNA_UNICODE_VERSION;
}

size_t vitna_utf8_decode(const unsigned char* s, size_t len, uint32_t* cp) {
    unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    size_t n;
    uint32_t v, min;
    if (c >= 0xC2 && c <= 0xDF) { n = 2; v = c & 0x1F; min = 0x80; }
    else if (c >= 0xE0 && c <= 0xEF) { n = 3; v = c & 0x0F; min = 0x800; }
    else if (c >= 0xF0 && c <= 0xF4) { n = 4; v = c & 0x07; min = 0x10000; }
    else { *cp = 0x110000u + c; return 1; }
    if (len < n) { *cp = 0x110000u + c; return 1; }
    for (size_t i = 1; i < n; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = 0x110000u + c; return 1; }
        v = (v << 6) | (s[i] & 0x3F);
    }
    if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) { *cp = 0x110000u + c; return 1; }
    *cp = v;
    return n;
}

/* --- Normalization Form C --- */

uint8_t vitna_uni_combining_class(uint32_t cp) {
    size_t lo = 0, hi = VITNA_UNI_CCC_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < VITNA_UNI_CCC[mid][0]) hi = mid;
        else if (cp > VITNA_UNI_CCC[mid][1]) lo = mid + 1;
        else return (uint8_t)VITNA_UNI_CCC[mid][2];
    }
    return 0;
}

/* Hangul syllables decompose and compose by formula (Unicode, section 3.12). */
#define S_BASE 0xAC00u
#define L_BASE 0x1100u
#define V_BASE 0x1161u
#define T_BASE 0x11A7u
#define L_COUNT 19u
#define V_COUNT 21u
#define T_COUNT 28u
#define N_COUNT (V_COUNT * T_COUNT)
#define S_COUNT (L_COUNT * N_COUNT)

/* Code points with their combining classes. A byte that is not valid UTF-8
 * is held as 0x110000 + the byte, as vitna_utf8_decode gives it, class 0. */
typedef struct {
    uint32_t* cp;
    uint8_t* cc;
    size_t n, cap;
} cp_buf_t;

static bool cp_push(cp_buf_t* b, uint32_t cp) {
    if (b->n == b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 64;
        uint32_t* c = (uint32_t*)realloc(b->cp, cap * sizeof(uint32_t));
        if (!c) return false;
        b->cp = c;
        uint8_t* k = (uint8_t*)realloc(b->cc, cap);
        if (!k) return false;
        b->cc = k;
        b->cap = cap;
    }
    b->cp[b->n] = cp;
    b->cc[b->n] = cp < 0x110000u ? vitna_uni_combining_class(cp) : 0;
    b->n++;
    return true;
}

static const uint32_t* find_decomposition(uint32_t cp) {
    size_t lo = 0, hi = VITNA_UNI_DECOMP_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < VITNA_UNI_DECOMP[mid][0]) hi = mid;
        else if (cp > VITNA_UNI_DECOMP[mid][0]) lo = mid + 1;
        else return VITNA_UNI_DECOMP[mid];
    }
    return NULL;
}

/* Append cp's full canonical decomposition: its one-level decomposition,
 * each part decomposed in turn. */
static bool decompose(cp_buf_t* b, uint32_t cp) {
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        const uint32_t s = cp - S_BASE;
        if (!cp_push(b, L_BASE + s / N_COUNT) || !cp_push(b, V_BASE + (s % N_COUNT) / T_COUNT)) return false;
        return s % T_COUNT == 0 || cp_push(b, T_BASE + s % T_COUNT);
    }
    const uint32_t* d = cp < 0x110000u ? find_decomposition(cp) : NULL;
    if (!d) return cp_push(b, cp);
    return decompose(b, d[1]) && (d[2] == 0 || decompose(b, d[2]));
}

/* The primary composite of the pair a, b, if there is one. */
static bool compose_pair(uint32_t a, uint32_t b, uint32_t* out) {
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT) {
        *out = S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
        return true;
    }
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE && b < T_BASE + T_COUNT) {
        *out = a + (b - T_BASE);
        return true;
    }
    size_t lo = 0, hi = VITNA_UNI_COMPOSE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const uint32_t* r = VITNA_UNI_COMPOSE[mid];
        if (a < r[0] || (a == r[0] && b < r[1])) hi = mid;
        else if (a > r[0] || b > r[1]) lo = mid + 1;
        else {
            *out = r[2];
            return true;
        }
    }
    return false;
}

static size_t utf8_encode(uint32_t cp, unsigned char* out) {
    if (cp >= 0x110000u) { out[0] = (unsigned char)(cp - 0x110000u); return 1; }
    if (cp < 0x80) { out[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (unsigned char)(0xC0 | (cp >> 6));
        out[1] = (unsigned char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (unsigned char)(0xE0 | (cp >> 12));
        out[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (unsigned char)(0xF0 | (cp >> 18));
    out[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

unsigned char* vitna_uni_nfc(const unsigned char* s, size_t len, size_t* out_len) {
    /* Below U+0300 nothing decomposes that NFC would not compose again, and
     * nothing combines with what precedes it, so such text is its own NFC. */
    bool simple = true;
    for (size_t i = 0; i < len && simple;) {
        uint32_t cp;
        i += vitna_utf8_decode(s + i, len - i, &cp);
        simple = cp < 0x300 || cp >= 0x110000u;
    }
    if (simple) {
        unsigned char* copy = (unsigned char*)malloc(len ? len : 1);
        if (copy) memcpy(copy, s, len);
        *out_len = len;
        return copy;
    }

    cp_buf_t b = {0};
    bool ok = true;
    for (size_t i = 0; i < len && ok;) {
        uint32_t cp;
        i += vitna_utf8_decode(s + i, len - i, &cp);
        ok = decompose(&b, cp);
    }
    unsigned char* out = NULL;
    if (ok) {
        /* Canonical ordering: each mark moves before the marks of a higher
         * class ahead of it, and never past a starter. */
        for (size_t i = 1; i < b.n; i++) {
            const uint8_t k = b.cc[i];
            if (k == 0) continue;
            for (size_t j = i; j > 0 && b.cc[j - 1] > k; j--) {
                const uint32_t c = b.cp[j - 1];
                const uint8_t kc = b.cc[j - 1];
                b.cp[j - 1] = b.cp[j];
                b.cc[j - 1] = b.cc[j];
                b.cp[j] = c;
                b.cc[j] = kc;
            }
        }
        /* Canonical composition: each character joins the last starter when
         * nothing between them blocks it, which a starter between does, or a
         * mark of the same or a higher class. */
        size_t n = 0;
        size_t starter = 0;
        bool have_starter = false;
        int last = -1; /* the class of the last character kept after the starter; -1 if none */
        for (size_t i = 0; i < b.n; i++) {
            const uint32_t c = b.cp[i];
            const uint8_t k = b.cc[i];
            uint32_t composite;
            if (have_starter && (last < 0 || (last > 0 && last < k)) && compose_pair(b.cp[starter], c, &composite)) {
                b.cp[starter] = composite;
                b.cc[starter] = vitna_uni_combining_class(composite);
                continue;
            }
            if (k == 0) {
                starter = n;
                have_starter = true;
                last = -1;
            } else {
                last = k;
            }
            b.cp[n] = c;
            b.cc[n] = k;
            n++;
        }
        out = (unsigned char*)malloc(n * 4 + 1);
        if (out) {
            size_t o = 0;
            for (size_t i = 0; i < n; i++) o += utf8_encode(b.cp[i], out + o);
            *out_len = o;
        }
    }
    free(b.cp);
    free(b.cc);
    return out;
}
