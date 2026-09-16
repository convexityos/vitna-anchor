#ifndef VITNA_CRYPTO_H
#define VITNA_CRYPTO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SHA-256 state context for rolling token trajectory hashing.
 * Pure C11 with zero external runtime dependencies.
 */
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} vitna_sha256_ctx_t;

/**
 * Sovereign air-gap execution proof.
 * Binds token generation trajectory to a cryptographic hash proving local air-gapped compute.
 */
typedef struct {
    char trajectory_hash[65];     /* 64-char hex string + null terminator */
    uint32_t token_count;         /* Number of tokens hashed in trajectory */
    bool airgap_verified;         /* Verified 0 external socket bytes */
    uint64_t duration_us;         /* Microsecond execution duration */
} vitna_sovereign_proof_t;

/**
 * Initialize a SHA-256 context.
 */
void vitna_sha256_init(vitna_sha256_ctx_t *ctx);

/**
 * Feed bytes into the SHA-256 context.
 */
void vitna_sha256_update(vitna_sha256_ctx_t *ctx, const void *data, size_t len);

/**
 * Finalize SHA-256 and produce 32-byte binary digest.
 */
void vitna_sha256_final(vitna_sha256_ctx_t *ctx, uint8_t digest[32]);

/**
 * Finalize SHA-256 and produce 64-character null-terminated hex string.
 */
void vitna_sha256_final_hex(vitna_sha256_ctx_t *ctx, char hex_output[65]);

/**
 * Accumulate a generated token ID and top logit into the rolling trajectory hash.
 */
void vitna_trajectory_feed(vitna_sha256_ctx_t *ctx, int32_t token_id, float top_logit);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_CRYPTO_H */
