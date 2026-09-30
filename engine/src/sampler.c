/**
 * sampler.c - Choosing the next token from a row of logits.
 */

#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

bool vitna_sampler_init(vitna_sampler_t* s, size_t vocab, uint64_t seed) {
    s->state = seed;
    s->n = vocab;
    s->probs = (float*)malloc(vocab * sizeof(float));
    s->order = (int32_t*)malloc(vocab * sizeof(int32_t));
    s->spare = (int32_t*)malloc(vocab * sizeof(int32_t));
    s->keys = (uint32_t*)malloc(2 * vocab * sizeof(uint32_t));
    return s->probs && s->order && s->spare && s->keys;
}

void vitna_sampler_free(vitna_sampler_t* s) {
    free(s->probs);
    free(s->order);
    free(s->spare);
    free(s->keys);
    s->probs = NULL;
    s->order = NULL;
    s->spare = NULL;
    s->keys = NULL;
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

/* order[0..n) set to the ids by probability, largest first, lower id first on
 * ties: one total order, so any correct sort gives the same one. This is a
 * least-significant-digit radix sort, three passes of 11 bits, in place of a
 * comparison sort's n log n comparisons, which cost several milliseconds a
 * token over a vocabulary of 49,152. The probabilities are never negative,
 * and floats that are not negative order as their bit patterns do, so the
 * patterns' complements sorted in increasing order put the largest first.
 * Every pass is stable and the ids start in increasing order, so equal
 * probabilities keep the lower id first. A NaN, which the probabilities never
 * are, would sort before every number. */
#define RADIX_BITS 11
#define RADIX_SIZE (1u << RADIX_BITS)

static void sort_by_prob(vitna_sampler_t* s) {
    const size_t n = s->n;
    uint32_t* key = s->keys;
    uint32_t* key_next = s->keys + n;
    int32_t* id = s->order;
    int32_t* id_next = s->spare;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &s->probs[i], sizeof(bits));
        key[i] = ~bits;
        id[i] = (int32_t)i;
    }
    for (unsigned shift = 0; shift < 32; shift += RADIX_BITS) {
        size_t start[RADIX_SIZE + 1];
        memset(start, 0, sizeof(start));
        for (size_t i = 0; i < n; i++) start[((key[i] >> shift) & (RADIX_SIZE - 1)) + 1]++;
        for (unsigned d = 0; d < RADIX_SIZE; d++) start[d + 1] += start[d];
        for (size_t i = 0; i < n; i++) {
            const size_t at = start[(key[i] >> shift) & (RADIX_SIZE - 1)]++;
            key_next[at] = key[i];
            id_next[at] = id[i];
        }
        uint32_t* kt = key;
        key = key_next;
        key_next = kt;
        int32_t* it = id;
        id = id_next;
        id_next = it;
    }
    if (id != s->order) memcpy(s->order, id, n * sizeof(int32_t));
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
    }
    for (size_t i = 0; i < n; i++) s->probs[i] = (float)(s->probs[i] / sum);

    sort_by_prob(s);

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
