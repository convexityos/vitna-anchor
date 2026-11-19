#include "crypto.h"
#include <string.h>
#include <stdio.h>

/* SHA-256 standard constants */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (i = 16; i < 64; i++) {
        w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (i = 0; i < 64; i++) {
        uint32_t t1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + w[i];
        uint32_t t2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void vitna_sha256_init(vitna_sha256_ctx_t *ctx) {
    if (!ctx) return;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

void vitna_sha256_update(vitna_sha256_ctx_t *ctx, const void *data, size_t len) {
    if (!ctx || !data || len == 0) return;
    const uint8_t *p = (const uint8_t *)data;
    size_t buffer_index = (size_t)(ctx->count & 0x3f);
    ctx->count += len;

    if (buffer_index > 0) {
        size_t left = 64 - buffer_index;
        if (len < left) {
            memcpy(&ctx->buffer[buffer_index], p, len);
            return;
        }
        memcpy(&ctx->buffer[buffer_index], p, left);
        transform(ctx->state, ctx->buffer);
        p += left;
        len -= left;
    }

    while (len >= 64) {
        transform(ctx->state, p);
        p += 64;
        len -= 64;
    }

    if (len > 0) {
        memcpy(ctx->buffer, p, len);
    }
}

void vitna_sha256_final(vitna_sha256_ctx_t *ctx, uint8_t digest[32]) {
    if (!ctx || !digest) return;
    uint8_t pad = 0x80;
    uint64_t total_bits = ctx->count * 8;
    int i;

    vitna_sha256_update(ctx, &pad, 1);
    while ((ctx->count & 0x3f) != 56) {
        uint8_t zero = 0;
        vitna_sha256_update(ctx, &zero, 1);
    }

    uint8_t len_bytes[8];
    for (i = 0; i < 8; i++) {
        len_bytes[i] = (uint8_t)((total_bits >> ((7 - i) * 8)) & 0xff);
    }
    vitna_sha256_update(ctx, len_bytes, 8);

    for (i = 0; i < 8; i++) {
        digest[i * 4] = (uint8_t)((ctx->state[i] >> 24) & 0xff);
        digest[i * 4 + 1] = (uint8_t)((ctx->state[i] >> 16) & 0xff);
        digest[i * 4 + 2] = (uint8_t)((ctx->state[i] >> 8) & 0xff);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i] & 0xff);
    }
}

void vitna_sha256_final_hex(vitna_sha256_ctx_t *ctx, char hex_output[65]) {
    if (!ctx || !hex_output) return;
    uint8_t digest[32];
    vitna_sha256_final(ctx, digest);
    for (int i = 0; i < 32; i++) {
        sprintf(&hex_output[i * 2], "%02x", digest[i]);
    }
    hex_output[64] = '\0';
}

void vitna_trajectory_feed(vitna_sha256_ctx_t *ctx, int32_t token_id, float top_logit) {
    if (!ctx) return;
    struct {
        int32_t id;
        float logit;
    } entry;
    entry.id = token_id;
    entry.logit = top_logit;
    vitna_sha256_update(ctx, &entry, sizeof(entry));
}

#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#endif

static void set_ckpt_err(char *err, size_t err_cap, const char *msg) {
    if (err && err_cap > 0) snprintf(err, err_cap, "%s", msg ? msg : "checkpoint hash error");
}

static bool has_suffix(const char *s, const char *suffix) {
    size_t sl = strlen(s), xl = strlen(suffix);
    return sl >= xl && strcmp(s + (sl - xl), suffix) == 0;
}

static bool is_checkpoint_artifact(const char *name) {
    return has_suffix(name, ".gguf") ||
           has_suffix(name, ".safetensors") ||
           strcmp(name, "config.json") == 0 ||
           strcmp(name, "tokenizer.json") == 0 ||
           strcmp(name, "tokenizer_config.json") == 0;
}

static bool hash_single_file(const char *path, char hex_output[65], char *err, size_t err_cap) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err && err_cap > 0) snprintf(err, err_cap, "cannot open checkpoint file %.400s", path);
        return false;
    }
    vitna_sha256_ctx_t ctx;
    vitna_sha256_init(&ctx);
    uint8_t *buf = (uint8_t *)malloc(65536);
    if (!buf) {
        fclose(f);
        set_ckpt_err(err, err_cap, "out of memory while hashing checkpoint");
        return false;
    }
    size_t got;
    while ((got = fread(buf, 1, 65536, f)) > 0) {
        vitna_sha256_update(&ctx, buf, got);
    }
    const bool io_err = ferror(f) != 0;
    free(buf);
    fclose(f);
    if (io_err) {
        if (err && err_cap > 0) snprintf(err, err_cap, "I/O error reading checkpoint file %.400s", path);
        return false;
    }
    vitna_sha256_final_hex(&ctx, hex_output);
    return true;
}

static int cmp_cstr_ptr(const void *a, const void *b) {
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    return strcmp(sa, sb);
}

bool vitna_checkpoint_sha256(const char *path, char hex_output[65], char *err, size_t err_cap) {
    if (!path || !*path || !hex_output) {
        set_ckpt_err(err, err_cap, "checkpoint path is empty");
        return false;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        if (err && err_cap > 0) snprintf(err, err_cap, "cannot stat checkpoint %.400s", path);
        return false;
    }
    if (S_ISREG(st.st_mode)) {
        return hash_single_file(path, hex_output, err, err_cap);
    }
    if (!S_ISDIR(st.st_mode)) {
        if (err && err_cap > 0) snprintf(err, err_cap, "checkpoint %.400s is neither a regular file nor a directory", path);
        return false;
    }

    char **names = NULL;
    size_t count = 0, cap = 0;

#ifdef _WIN32
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
        if (err && err_cap > 0) snprintf(err, err_cap, "cannot read checkpoint directory %.400s", path);
        return false;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) continue;
        if (!is_checkpoint_artifact(fd.cFileName)) continue;
        if (count == cap) {
            size_t next_cap = cap ? cap * 2 : 16;
            char **grown = (char **)realloc(names, next_cap * sizeof(char *));
            if (!grown) { FindClose(h); goto oom; }
            names = grown;
            cap = next_cap;
        }
        size_t nlen = strlen(fd.cFileName) + 1;
        names[count] = (char *)malloc(nlen);
        if (!names[count]) { FindClose(h); goto oom; }
        memcpy(names[count], fd.cFileName, nlen);
        count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *dir = opendir(path);
    if (!dir) {
        if (err && err_cap > 0) snprintf(err, err_cap, "cannot read checkpoint directory %.400s", path);
        return false;
    }
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!is_checkpoint_artifact(ent->d_name)) continue;
        char full[1536];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        struct stat est;
        if (stat(full, &est) != 0 || !S_ISREG(est.st_mode)) continue;
        if (count == cap) {
            size_t next_cap = cap ? cap * 2 : 16;
            char **grown = (char **)realloc(names, next_cap * sizeof(char *));
            if (!grown) { closedir(dir); goto oom; }
            names = grown;
            cap = next_cap;
        }
        size_t nlen = strlen(ent->d_name) + 1;
        names[count] = (char *)malloc(nlen);
        if (!names[count]) { closedir(dir); goto oom; }
        memcpy(names[count], ent->d_name, nlen);
        count++;
    }
    closedir(dir);
#endif

    if (count == 0) {
        free(names);
        if (err && err_cap > 0) {
            snprintf(err, err_cap, "checkpoint directory %.400s contains no recognized model artifacts", path);
        }
        return false;
    }

    qsort(names, count, sizeof(char *), cmp_cstr_ptr);

    vitna_sha256_ctx_t manifest;
    vitna_sha256_init(&manifest);
    bool ok = true;
    for (size_t i = 0; i < count; i++) {
        char full[1536];
        snprintf(full, sizeof(full), "%s/%s", path, names[i]);
        char file_hex[65];
        if (!hash_single_file(full, file_hex, err, err_cap)) {
            ok = false;
            break;
        }
        vitna_sha256_update(&manifest, names[i], strlen(names[i]));
        vitna_sha256_update(&manifest, ":", 1);
        vitna_sha256_update(&manifest, file_hex, 64);
        vitna_sha256_update(&manifest, "\n", 1);
    }
    for (size_t i = 0; i < count; i++) free(names[i]);
    free(names);
    if (!ok) return false;

    vitna_sha256_final_hex(&manifest, hex_output);
    return true;

oom:
    for (size_t i = 0; i < count; i++) free(names[i]);
    free(names);
    set_ckpt_err(err, err_cap, "out of memory listing checkpoint directory");
    return false;
}
