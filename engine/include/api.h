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
 * does not know are ignored and named in an x-vitna-ignored header. A step the
 * model fails to run ends the request with an error, never with the reply so
 * far: a 500 of type server_error or, once a stream's 200 has gone out, an
 * event carrying that error, with no final chunk or [DONE] after it. A step
 * that loses the GPU (vitna_api_lost) ends every request running that way,
 * and every other is refused with a 503.
 *
 * Requests that generate run together, as many at once as the model's cache
 * has sequences (vitna_llama_load's seqs), each in a sequence of its own; the
 * rest wait their turn. Every one is answered exactly as it would be alone,
 * a token at a time: its tokens' logits are those steps', bit for bit.
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
    size_t cached_tokens;   /* prompt tokens whose keys and values were reused, not recomputed */
} vitna_api_result_t;

/** Serve model under model_id. The API keeps the pointers; it frees neither the model nor the tokenizer. */
vitna_api_t* vitna_api_create(vitna_llama_t* model, const vitna_tokenizer_t* tok, const char* model_id);

void vitna_api_free(vitna_api_t* api);

/** The requests it runs at once: the model's sequences. */
size_t vitna_api_parallel(const vitna_api_t* api);

/**
 * Whether the GPU can run nothing more in this process: a step failed with an
 * error CUDA calls sticky (vitna_llama_t.device_lost), after which only a new
 * process can use the device. From then on every request that was running
 * ends with the error a failed step gives, and every request waiting for a
 * sequence, or sent later, is refused with a 503 that says so. The server
 * stops taking connections when this turns true, so that something can start
 * a new one.
 */
bool vitna_api_lost(vitna_api_t* api);

/**
 * Whether a request may reuse the key-value cache an earlier one left in
 * the sequence it is given (on by default): of the sequences free, the one
 * holding the longest prefix of its prompt. That prefix is kept rather than
 * recomputed, and reported as usage.prompt_tokens_details.cached_tokens.
 * Reuse cannot change a response: the keys and values it keeps are the ones
 * the same tokens would produce again.
 */
void vitna_api_set_prefix_cache(vitna_api_t* api, bool on);

/**
 * Whether a request drafts tokens and checks them several at a time (off by
 * default): up to k drafted after each token taken, by vitna_drafter_t, and
 * checked with vitna_llama_step_rows, which gives every position the logits
 * a step there would. Each token is still chosen from those logits in turn,
 * with the JSON mask, the stop sequences and the sampler exactly as without
 * it, and the cache left as it would be, so no response changes, usage
 * included. On a GPU k is at most vitna_llama_exact_max - 1. With several
 * requests running, drafts take only the rows a round's passes have spare.
 * Call it before serving. Returns false if it could not allocate for k, and
 * leaves it off.
 */
bool vitna_api_set_speculate(vitna_api_t* api, size_t k);

/**
 * For tests: whether JSON mode keeps each state's mask for when the state
 * comes round again (on by default). Off, every mask is found anew; the
 * responses must be the same either way.
 */
void vitna_api_set_mask_cache(vitna_api_t* api, bool on);

/**
 * For tests: whether a response that is not streamed names, in an
 * x-vitna-test-logits header, an FNV-1a hash of the bytes of every row of
 * logits its tokens were taken from, in order, as the model gave them (off
 * by default). Two servers' hashes for a request agree only if their logits
 * did, bit for bit, where the replies alone would agree through any change
 * too small to move a token.
 */
void vitna_api_set_test_logits(vitna_api_t* api, bool on);

/**
 * Drafts for speculative decoding, by prompt lookup: the tokens that followed
 * the latest earlier place in the text where its last three, or two, tokens
 * occur. A match of one token drafts nothing, and after a pass takes none of
 * its drafts none are made for 1 token, then 2, doubling to 16, and back to 1
 * once a pass takes one: both chosen by measurement, as drafts refused cost a
 * pass that gains nothing. generate --speculate and the API draft the same way.
 */
typedef struct {
    size_t k;       /* the most drafts at once */
    size_t skip;    /* tokens still to go without drafting */
    size_t backoff; /* the next pause */
} vitna_drafter_t;

void vitna_drafter_init(vitna_drafter_t* d, size_t k);

/** Up to d->k drafts to follow text[len - 1], into out; 0 while pausing or with nothing found. */
size_t vitna_drafter_draft(vitna_drafter_t* d, const int32_t* text, size_t len, int32_t* out);

/** After a pass: how many of its drafts were taken. */
void vitna_drafter_taken(vitna_drafter_t* d, size_t taken);

/**
 * Answer one request with a complete HTTP response through sink, returning
 * once it has all gone. Several threads may call it at once, one a
 * connection: a request that generates waits for a sequence, and its output
 * goes out through its own caller's sink. api may be NULL, for a server
 * started without a model: health and the model list say so, and generation
 * answers 501.
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
