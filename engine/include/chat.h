/**
 * chat.h - A conversation written into a model's prompt in its own chat
 * template's words (gate A11).
 *
 * A model is trained on its chat template's rendering of a conversation, so
 * the engine writes the same text the template would. Qwen3's template is
 * written here in C, line for line, and held to transformers' rendering of
 * it (reference/qwen3-30b-a3b/chat-template.json). Its tools and tool calls
 * are JSON as transformers' tojson writes it, which is Python's json.dumps:
 * vitna_json_write_py.
 */

#ifndef VITNA_CHAT_H
#define VITNA_CHAT_H

#include <stdbool.h>
#include <stddef.h>

#include "json.h"
#include "strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append v as Python's json.dumps(v, ensure_ascii=False) writes it, with
 * its default separators: ", " between items and ": " after a key, keys in
 * the order the document gave them, a string's quote, backslash and control
 * characters escaped and the rest as UTF-8, an integer as written and any
 * other number as Python's float repr (1.0, 1e-05, 1e+20).
 */
void vitna_json_write_py(vitna_strbuf_t* out, const vitna_json_value_t* v);

/** A double as Python's repr(float) writes it, into buf (32 bytes is enough). */
void vitna_py_float_repr(double x, char* buf, size_t len);

/**
 * Append the prompt Qwen3's chat template renders for messages (an array of
 * objects in OpenAI's shape: role, content, and for an assistant
 * reasoning_content and tool_calls, each a function's name and arguments,
 * the arguments a JSON string or an object) and tools (an array, or NULL
 * for none, each written as given). enable_thinking is 0 for the template's
 * enable_thinking=False, 1 for True and -1 for leaving it undefined; with
 * add_generation_prompt the prompt ends ready for the assistant's reply.
 * A message whose content is not a string has empty content, as in the
 * template. Returns false, with the reason in err, for a conversation the
 * template cannot render: no messages, or a message that is not an object.
 */
bool vitna_chat_qwen3(vitna_strbuf_t* out, const vitna_json_value_t* messages, const vitna_json_value_t* tools, int enable_thinking,
                      bool add_generation_prompt, char* err, size_t err_len);

/*
 * A reply in Qwen3's own format, read back as it is written: its reasoning
 * between <think> and </think> at the start, then its text, then tool calls,
 * each <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>. The
 * parser is fed the whole reply so far each time it grows, and reports each
 * part once it is sure of it: it holds back only the end that could still
 * begin a tag, and the newlines the template strips around the parts (the
 * reasoning's at both ends, the text's before a tool call and at the end).
 */

typedef enum {
    VITNA_REPLY_REASONING,  /* bytes of reasoning */
    VITNA_REPLY_TEXT,       /* bytes of the reply's text */
    VITNA_REPLY_TOOL_CALL,  /* a whole tool call: name, and arguments as JSON text (Python's writing of them) */
} vitna_reply_kind_t;

typedef struct {
    vitna_reply_kind_t kind;
    const char* data;       /* REASONING and TEXT: the bytes; TOOL_CALL: the name */
    size_t len;
    const char* args;       /* TOOL_CALL: the arguments as JSON text */
    size_t args_len;
} vitna_reply_part_t;

typedef void (*vitna_reply_fn)(void* ctx, const vitna_reply_part_t* part);

typedef struct {
    bool think;             /* the reply may open with <think> */
    bool tools;             /* <tool_call> blocks are tool calls, not text */
    int phase;              /* where the parser is (chat.c) */
    size_t at;              /* bytes of the reply read */
    size_t nl;              /* newlines read and held back, not yet known to be the part's */
    bool any_text;          /* text has been reported, so later newlines are not leading ones */
    size_t calls;           /* tool calls reported */
} vitna_reply_parser_t;

void vitna_reply_init(vitna_reply_parser_t* rp, bool think, bool tools);

/**
 * Read the reply so far, text[0..len), from where the last call left off,
 * calling fn for each part it is now sure of. With final, the reply has
 * ended: everything held back is reported as what it is, a tool call left
 * open or one whose JSON is not a call is reported as text, and the
 * newlines at the ends of the parts are dropped.
 */
void vitna_reply_feed(vitna_reply_parser_t* rp, const char* text, size_t len, bool final, vitna_reply_fn fn, void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_CHAT_H */
