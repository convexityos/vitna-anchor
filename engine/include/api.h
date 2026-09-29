/**
 * api.h - The OpenAI-compatible HTTP API over a loaded model.
 *
 * Independent of sockets: a request comes in as method, path and body, and a
 * complete HTTP response goes out through a sink, so the whole API can be
 * driven from a test. server.c does the network half.
 *
 * Served: GET /v1/models, GET /v1/health (and /health), POST
 * /v1/chat/completions and POST /v1/completions, each streamed as
 * server-sent events when asked. Usage is counted from tokens: prompt_tokens
 * is the number of tokens the model read, after the chat format is applied,
 * and completion_tokens the number it generated, counting the end-of-text or
 * other special token it stopped on. Parameters this API does not implement
 * are refused with a 400 that names them, never ignored in silence; fields it
 * does not know are ignored and named in an x-vitna-ignored header.
 */

#ifndef VITNA_API_H
#define VITNA_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "model.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vitna_api vitna_api_t;

/** Where a response goes. write returns false once the client is gone. */
typedef struct {
    bool (*write)(void* ctx, const void* data, size_t len);
    void* ctx;
    bool failed;
} vitna_sink_t;

typedef struct {
    int status;
    size_t prompt_tokens;
    size_t completion_tokens;
} vitna_api_result_t;

/** Serve model under model_id. The API keeps the pointers; it frees neither the model nor the tokenizer. */
vitna_api_t* vitna_api_create(vitna_llama_t* model, const vitna_tokenizer_t* tok, const char* model_id);

void vitna_api_free(vitna_api_t* api);

/**
 * Answer one request with a complete HTTP response through sink. api may be
 * NULL, for a server started without a model: health and the model list say
 * so, and generation answers 501.
 */
vitna_api_result_t vitna_api_handle(vitna_api_t* api, const char* method, const char* path,
                                    const char* body, size_t body_len, vitna_sink_t* sink);

/** The id the model is served under. */
const char* vitna_api_model_id(const vitna_api_t* api);

/** What a server started without a model says about it. */
const char* vitna_api_no_model_message(void);

/** Format chat messages as ChatML, the format of the SmolLM2 family's instruct models. Exposed for tests. */
bool vitna_chatml_format(const char* const* roles, const char* const* contents, size_t n, char** out, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_API_H */
