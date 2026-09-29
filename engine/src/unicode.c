/**
 * unicode.c - Character classes for the tokenizer, and a UTF-8 decoder.
 */

#include "unicode.h"
#include "unicode_data.h"

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
