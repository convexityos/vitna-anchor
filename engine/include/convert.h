/**
 * convert.h - Requests in the three shapes the server takes, rewritten as
 * one: the conversation and tools of OpenAI's chat shape, which the engine's
 * chat template reads (chat.h). OpenAI's chat completions, Anthropic's
 * Messages and OpenAI's Responses each carry the same things, a system
 * prompt, turns, tool calls and their results, and tools, in a shape of
 * their own (gate A11).
 */

#ifndef VITNA_CONVERT_H
#define VITNA_CONVERT_H

#include <stdbool.h>
#include <stddef.h>

#include "json.h"
#include "strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VITNA_TOOLS_AUTO = 0,   /* the model calls a tool or not, as it chooses */
    VITNA_TOOLS_NONE,       /* it is not offered the tools */
    VITNA_TOOLS_REQUIRED,   /* it must call one */
    VITNA_TOOLS_NAMED,      /* it must call the one named */
} vitna_tool_choice_t;

typedef struct {
    vitna_strbuf_t messages;  /* a JSON array of messages in OpenAI's chat shape, each content a string */
    vitna_strbuf_t tools;     /* a JSON array of tools in OpenAI's chat shape, empty for none */
    int enable_thinking;      /* the chat template's switch: -1 as the model chooses, 0 off, 1 on */
    vitna_tool_choice_t tool_choice;
    char tool_name[128];      /* VITNA_TOOLS_NAMED: the tool */
    bool continue_final;      /* the conversation ends with the assistant's reply, to be continued (Anthropic's prefill) */
    char ignored[256];        /* what was left out, named for x-vitna-ignored: a tool its API runs on its own side */
    /* Why a request was refused: the parameter, and what is wrong with it. */
    char param[64];
    char message[384];
    bool unsupported;         /* refused for asking what the server does not do, rather than for being malformed */
} vitna_chat_request_t;

void vitna_chat_request_init(vitna_chat_request_t* cr);
void vitna_chat_request_free(vitna_chat_request_t* cr);

/** OpenAI's chat completions: messages, tools, tool_choice and chat_template_kwargs. */
bool vitna_convert_chat(const vitna_json_value_t* req, vitna_chat_request_t* cr);

/** Anthropic's Messages: system, messages of content blocks, tools, tool_choice and thinking. */
bool vitna_convert_messages(const vitna_json_value_t* req, vitna_chat_request_t* cr);

/**
 * OpenAI's Responses: instructions, input items, tools, tool_choice and
 * reasoning. A namespace's functions are offered as functions; a tool OpenAI
 * runs on its own side (web search, file search and the like) is left out
 * and named in cr->ignored.
 */
bool vitna_convert_responses(const vitna_json_value_t* req, vitna_chat_request_t* cr);

/** The name of the namespace in a Responses request's tools that holds the function called name, or NULL. */
const vitna_json_value_t* vitna_responses_namespace(const vitna_json_value_t* tools, const char* name, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_CONVERT_H */
