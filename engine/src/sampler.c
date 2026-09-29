/**
 * sampler.c - Choosing the next token from a row of logits.
 */

#include "sampler.h"
#include <math.h>
#include <stdlib.h>

bool vitna_sampler_init(vitna_sampler_t* s, size_t vocab, uint64_t seed) {
    s->state = seed;
    s->n = vocab;
    s->probs = (float*)malloc(vocab * sizeof(float));
    s->order = (int32_t*)malloc(vocab * sizeof(int32_t));
    return s->probs && s->order;
}

void vitna_sampler_free(vitna_sampler_t* s) {
    free(s->probs);
    free(s->order);
    s->probs = NULL;
    s->order = NULL;
}

int32_t vitna_argmax(const float* logits, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return (int32_t)best;
}

/* splitmix64: small, fast, and the same sequence on every platform. */
static uint64_t next_u64(vitna_sampler_t* s) {
    uint64_t z = (s->state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double vitna_sampler_uniform(vitna_sampler_t* s) {
    return (double)(next_u64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Sort order[0..n) by probability, largest first, lower id first on ties. */
static const float* g_probs;
static int by_prob_desc(const void* a, const void* b) {
    int32_t ia = *(const int32_t*)a, ib = *(const int32_t*)b;
    if (g_probs[ia] > g_probs[ib]) return -1;
    if (g_probs[ia] < g_probs[ib]) return 1;
    return (ia > ib) - (ia < ib);
}

int32_t vitna_sample(vitna_sampler_t* s, const float* logits, const vitna_sampling_t* cfg) {
    const size_t n = s->n;
    if (!cfg || cfg->temperature <= 0.0f) return vitna_argmax(logits, n);

    /* Temperature, then softmax over the whole vocabulary. */
    float max = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        s->probs[i] = expf((logits[i] - max) / cfg->temperature);
        sum += s->probs[i];
        s->order[i] = (int32_t)i;
    }
    for (size_t i = 0; i < n; i++) s->probs[i] = (float)(s->probs[i] / sum);

    g_probs = s->probs;
    qsort(s->order, n, sizeof(int32_t), by_prob_desc);

    /* Top-k, then top-p on what remains: keep the smallest prefix whose
     * mass reaches top_p, always at least one token. */
    size_t keep = (cfg->top_k > 0 && cfg->top_k < n) ? cfg->top_k : n;
    double kept_mass = 0.0;
    for (size_t i = 0; i < keep; i++) kept_mass += s->probs[s->order[i]];
    if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = keep;
        for (size_t i = 0; i < keep; i++) {
            cum += s->probs[s->order[i]] / kept_mass;
            if (cum >= cfg->top_p) {
                cut = i + 1;
                break;
            }
        }
        keep = cut;
        kept_mass = 0.0;
        for (size_t i = 0; i < keep; i++) kept_mass += s->probs[s->order[i]];
    }

    double u = vitna_sampler_uniform(s) * kept_mass;
    double cum = 0.0;
    for (size_t i = 0; i < keep; i++) {
        cum += s->probs[s->order[i]];
        if (u < cum) return s->order[i];
    }
    return s->order[keep - 1];
}
