/**
 * sampler.h - Choosing the next token from a row of logits.
 *
 * Greedy decoding takes the largest logit, and the lowest token id among
 * equal ones. Sampling applies temperature, then top-k, then top-p (the
 * order transformers applies its logits warpers in), and draws from what is
 * left with a seeded generator, so a seed reproduces a run on any machine.
 */

#ifndef VITNA_SAMPLER_H
#define VITNA_SAMPLER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature;   /* 0 or less: greedy */
    size_t top_k;        /* 0: no limit */
    float top_p;         /* 1 or more: no limit */
    uint64_t seed;
} vitna_sampling_t;

typedef struct {
    uint64_t state;
    size_t n;
    float* probs;        /* n */
    int32_t* order;      /* n */
} vitna_sampler_t;

bool vitna_sampler_init(vitna_sampler_t* s, size_t vocab, uint64_t seed);
void vitna_sampler_free(vitna_sampler_t* s);

/** The largest logit's id; the lowest id among ties. */
int32_t vitna_argmax(const float* logits, size_t n);

/** The next token under the settings: greedy when temperature <= 0. */
int32_t vitna_sample(vitna_sampler_t* s, const float* logits, const vitna_sampling_t* cfg);

/** The generator's next value in [0, 1). */
double vitna_sampler_uniform(vitna_sampler_t* s);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SAMPLER_H */
