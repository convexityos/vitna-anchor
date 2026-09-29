/**
 * strbuf.h - A growable byte buffer, and JSON output into it.
 */

#ifndef VITNA_STRBUF_H
#define VITNA_STRBUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char* data;   /* NUL-terminated while ok */
    size_t len;
    size_t cap;
    bool ok;      /* false once an allocation has failed */
} vitna_strbuf_t;

void vitna_sb_init(vitna_strbuf_t* sb);
void vitna_sb_free(vitna_strbuf_t* sb);
void vitna_sb_clear(vitna_strbuf_t* sb);
void vitna_sb_append(vitna_strbuf_t* sb, const void* data, size_t len);
void vitna_sb_puts(vitna_strbuf_t* sb, const char* s);
void vitna_sb_printf(vitna_strbuf_t* sb, const char* fmt, ...);

/**
 * Append s[0..len) as a JSON string, quotes included. Invalid UTF-8 becomes
 * U+FFFD, as a lossy decoder would; control characters are escaped.
 */
void vitna_sb_json_string(vitna_strbuf_t* sb, const unsigned char* s, size_t len);

/** The length of the longest prefix of s[0..len) that ends on a UTF-8 character boundary. */
size_t vitna_utf8_complete_prefix(const unsigned char* s, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_STRBUF_H */
