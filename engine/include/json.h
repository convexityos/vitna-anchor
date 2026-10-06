/**
 * json.h - A small JSON parser for the engine's own inputs: a model's
 * config.json, its tokenizer.json and a SafeTensors header.
 *
 * It parses a whole document into a tree held in one arena, which
 * vitna_json_free releases at once. Strings are decoded to UTF-8 and
 * NUL-terminated; a lone surrogate in a \u escape becomes U+FFFD. Nesting
 * deeper than VITNA_JSON_MAX_DEPTH is an error.
 */

#ifndef VITNA_JSON_H
#define VITNA_JSON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_JSON_MAX_DEPTH 256

typedef enum {
    VITNA_JSON_NULL = 0,
    VITNA_JSON_FALSE,
    VITNA_JSON_TRUE,
    VITNA_JSON_NUMBER,
    VITNA_JSON_STRING,
    VITNA_JSON_ARRAY,
    VITNA_JSON_OBJECT
} vitna_json_type_t;

typedef struct vitna_json_value vitna_json_value_t;

typedef struct {
    const char* key;      /* decoded UTF-8, NUL-terminated */
    size_t key_len;
    vitna_json_value_t* value;
} vitna_json_member_t;

struct vitna_json_value {
    vitna_json_type_t type;
    union {
        double number;
        struct { const char* ptr; size_t len; } string;
        struct { vitna_json_value_t** items; size_t count; } array;
        struct { vitna_json_member_t* members; size_t count; } object;
    } u;
    /* A number's text as the document wrote it, NUL-terminated: what tells 1
     * from 1.0, which a double cannot (chat.c writes JSON as Python does). */
    const char* lit;
    size_t lit_len;
};

typedef struct vitna_json_doc vitna_json_doc_t;

/**
 * Parse len bytes of JSON. Returns NULL on error, with a message in err.
 */
vitna_json_doc_t* vitna_json_parse(const char* text, size_t len, char* err, size_t err_len);

void vitna_json_free(vitna_json_doc_t* doc);

const vitna_json_value_t* vitna_json_root(const vitna_json_doc_t* doc);

/** The member of an object with this key, or NULL. The first one wins. */
const vitna_json_value_t* vitna_json_get(const vitna_json_value_t* object, const char* key);

/** Typed readers: return false if v is missing or of another type. */
bool vitna_json_as_number(const vitna_json_value_t* v, double* out);
bool vitna_json_as_bool(const vitna_json_value_t* v, bool* out);
const char* vitna_json_as_string(const vitna_json_value_t* v); /* NULL unless a string */
bool vitna_json_is_null(const vitna_json_value_t* v);

/** Read a whole file into a NUL-terminated heap buffer. Returns NULL on failure. */
char* vitna_read_file(const char* path, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_JSON_H */
