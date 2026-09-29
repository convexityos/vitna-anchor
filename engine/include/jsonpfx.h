/**
 * jsonpfx.h - Is this text the start of a JSON object? An incremental check
 * for constrained decoding.
 *
 * Fed bytes one at a time, it answers whether everything so far can still be
 * extended into a valid JSON object (RFC 8259, with an object at the top
 * level), and whether that object is complete. Strings must be valid UTF-8
 * with control characters escaped. Outside strings, a run of whitespace is
 * limited to VITNA_JSONPFX_MAX_WS bytes, so a model cannot pad forever.
 *
 * The state is a small plain struct, so a caller tests a candidate by
 * copying it and feeding the copy.
 */

#ifndef VITNA_JSONPFX_H
#define VITNA_JSONPFX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_JSONPFX_MAX_DEPTH 64
#define VITNA_JSONPFX_MAX_WS 16

typedef struct {
    uint8_t stack[VITNA_JSONPFX_MAX_DEPTH]; /* 'O' for an object, 'A' for an array */
    uint8_t depth;
    uint8_t state;
    uint8_t sub;        /* within a number, literal, escape or UTF-8 sequence */
    uint8_t lit;        /* the literal being read: 't', 'f' or 'n' */
    uint8_t utf8_left;  /* continuation bytes still owed in a string */
    uint8_t utf8_lo;    /* the range the next continuation byte must fall in */
    uint8_t utf8_hi;
    uint8_t ws;         /* whitespace bytes in the current run */
    bool in_key;        /* the string being read is an object key */
    bool done;          /* the top-level object is complete */
    bool failed;
} vitna_jsonpfx_t;

void vitna_jsonpfx_init(vitna_jsonpfx_t* p);

/** Feed bytes. Returns false, and marks the state failed, at the first byte no valid object could continue with. */
bool vitna_jsonpfx_feed(vitna_jsonpfx_t* p, const unsigned char* bytes, size_t n);

/** Whether feeding these bytes would keep it valid. p is not changed. */
bool vitna_jsonpfx_accepts(const vitna_jsonpfx_t* p, const unsigned char* bytes, size_t n);

/** The bytes that could come next, as a 256-bit set: bit b of allowed[b >> 3] is byte b. */
void vitna_jsonpfx_next_bytes(const vitna_jsonpfx_t* p, uint8_t allowed[32]);

/** Whether the top-level object has closed. */
bool vitna_jsonpfx_complete(const vitna_jsonpfx_t* p);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_JSONPFX_H */
