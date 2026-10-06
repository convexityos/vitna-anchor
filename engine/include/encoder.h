/**
 * encoder.h - A BERT-architecture text encoder in float32 on the CPU, and the
 * embedding sentence-transformers makes of its last hidden state.
 *
 * Reads config.json (model_type "bert"), model.safetensors, and the
 * sentence-transformers files that say how a text becomes one vector:
 * modules.json, the pooling module's config.json (the [CLS] position, or the
 * mean over every position) and whether a Normalize module divides the vector
 * by its L2 norm; sentence_bert_config.json's max_seq_length, when smaller
 * than the model's positions, is the most tokens a text may have.
 *
 * Computes Hugging Face's BertModel with eager attention: the word, position
 * and token-type (always 0) embeddings added and put through LayerNorm, then
 * per layer self-attention in which every position attends to every position
 * of its own text, a residual add and LayerNorm, an MLP with GELU in its erf
 * form, and a residual add and LayerNorm.
 *
 * Several texts run together: their positions are the rows of the same
 * matrix products, and each attends only to its own, so none is padded and
 * none changes another's result. Each output of a product is one dot product
 * done by one thread, in a fixed order, so the result is the same, bit for
 * bit, however many threads run and however the texts are grouped.
 *
 * What the config asks for and this does not do (another activation, relative
 * position embeddings, a dense layer after pooling, pooling other than [CLS]
 * or mean) is refused on load.
 */

#ifndef VITNA_ENCODER_H
#define VITNA_ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VITNA_POOL_CLS = 0,     /* the last hidden state at the first position, [CLS] */
    VITNA_POOL_MEAN = 1     /* the mean of the last hidden state over every position */
} vitna_pooling_t;

typedef struct {
    size_t n_layers;
    size_t hidden;
    size_t n_heads;
    size_t head_dim;
    size_t intermediate;
    size_t vocab;
    size_t max_positions;
    size_t type_vocab;
    size_t max_tokens;          /* the most tokens a text may have, [CLS] and [SEP] included */
    float ln_eps;
    vitna_pooling_t pooling;    /* as the model's pooling module says */
    bool normalize;             /* whether a Normalize module follows the pooling */
} vitna_encoder_config_t;

/** y = x W^T + b, W out x in, stored as float32. */
typedef struct {
    const float* w;
    const float* b;
    size_t out, in;
} vitna_dense_t;

typedef struct {
    vitna_dense_t q, k, v, o, up, down;
    const float *attn_ln_w, *attn_ln_b;   /* after attention */
    const float *out_ln_w, *out_ln_b;     /* after the MLP */
} vitna_encoder_layer_t;

typedef struct vitna_encoder_pool vitna_encoder_pool_t;

typedef struct {
    vitna_encoder_config_t cfg;
    vitna_safetensors_t st;
    bool st_open;
    float** owned;              /* weights widened to float32, or copied to be aligned */
    size_t n_owned;
    const float *word, *position, *token_type;
    const float *emb_ln_w, *emb_ln_b;
    vitna_encoder_layer_t* layers;
    size_t threads;
    vitna_encoder_pool_t* pool;
} vitna_encoder_t;

/** Whether dir holds a model this loads: config.json with model_type "bert". */
bool vitna_encoder_wanted(const char* dir);

/**
 * Load the model in dir, to run on threads threads (0 for every processor
 * this machine has). Returns false, with a reason in err, if the model cannot
 * be read or asks for something unsupported.
 */
bool vitna_encoder_load(vitna_encoder_t* e, const char* dir, size_t threads, char* err, size_t err_len);

void vitna_encoder_free(vitna_encoder_t* e);

/**
 * Embed n texts, each given as its token ids, the tokenizer's [CLS] and
 * [SEP] included: ids[i] has lens[i] ids, from 1 to cfg.max_tokens, each
 * below cfg.vocab. Writes n x hidden floats to out, pooled as pooling says
 * and divided by their L2 norm when normalize is true; norms, unless NULL,
 * gets each pooled vector's L2 norm before that division. Returns false,
 * with out unwritten, if memory runs out or an input is out of range.
 */
bool vitna_encoder_embed(vitna_encoder_t* e, const int32_t* const* ids, const size_t* lens, size_t n, vitna_pooling_t pooling,
                         bool normalize, float* out, float* norms);

/** The path the matrix products take on this CPU: "avx2+fma" or "scalar". */
const char* vitna_encoder_path(void);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_ENCODER_H */
