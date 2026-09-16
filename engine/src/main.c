/**
 * main.c - CLI entry point for vitna-engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compat.h"
#include "safetensors.h"
#include "expert_store.h"
#include "router.h"
#include "kernels.h"
#include "server.h"

static void print_usage(const char* prog) {
    printf("vitna-anchor - Sovereign MoE inference daemon\n\n");
    printf("Usage:\n");
    printf("  %s serve [--port <port>] [--model <path>] [--ram-gb <gb>]\n", prog);
    printf("  %s info  --model <path>\n", prog);
    printf("  %s bench [--model <path>] [--iterations <n>]\n", prog);
}

static int cmd_info(const char* model_path) {
    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required\n");
        return 1;
    }

    vitna_safetensors_t st;
    if (!vitna_safetensors_open(model_path, &st)) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", model_path);
        return 1;
    }

    printf("Safetensors model: %s\n", model_path);
    printf("Header size: %llu bytes\n", (unsigned long long)st.header_len);
    printf("Tensor count: %zu\n\n", st.tensor_count);

    for (size_t i = 0; i < st.tensor_count && i < 25; i++) {
        const vitna_tensor_desc_t* t = &st.tensors[i];
        printf("  [%3zu] %-50s shape: [", i, t->name);
        for (size_t d = 0; d < t->ndim; d++) {
            printf("%zu%s", t->shape[d], (d + 1 < t->ndim) ? ", " : "");
        }
        printf("] offsets: [%llu..%llu]\n",
            (unsigned long long)t->offset_begin,
            (unsigned long long)t->offset_end
        );
    }
    if (st.tensor_count > 25) {
        printf("  ... and %zu more tensors\n", st.tensor_count - 25);
    }

    vitna_safetensors_close(&st);
    return 0;
}

static int cmd_bench(const char* model_path, int iterations) {
    (void)model_path;
    vitna_simd_capabilities_t caps = vitna_detect_simd_capabilities();
    printf("Running vitna-anchor streaming benchmark (%d iterations)...\n", iterations);
    printf("Hardware acceleration: [AVX2: %s] [AVX512: %s] [ARM NEON: %s] [FMA: %s]\n",
        caps.has_avx2 ? "active" : "no",
        caps.has_avx512 ? "active" : "no",
        caps.has_neon ? "active" : "no",
        caps.has_fma ? "active" : "no"
    );
    double t0 = vitna_time_ms();

    /* Benchmark synthetic routing & GEMV int4 operations */
    const size_t rows = 2048;
    const size_t cols = 2048;
    size_t packed_bytes = (rows * cols) / 2;
    uint8_t* packed_w = (uint8_t*)vitna_aligned_alloc(64, packed_bytes);
    float* scales = (float*)vitna_aligned_alloc(64, (rows * cols / 128) * sizeof(float));
    float* x = (float*)vitna_aligned_alloc(64, cols * sizeof(float));
    float* y = (float*)vitna_aligned_alloc(64, rows * sizeof(float));

    if (packed_w && scales && x && y) {
        memset(packed_w, 0x88, packed_bytes);
        for (size_t i = 0; i < cols; i++) x[i] = 1.0f;
        for (size_t i = 0; i < (rows * cols / 128); i++) scales[i] = 0.05f;

        for (int it = 0; it < iterations; it++) {
            vitna_gemv_int4(packed_w, scales, 128, x, y, rows, cols);
        }
    }

    vitna_aligned_free(packed_w);
    vitna_aligned_free(scales);
    vitna_aligned_free(x);
    vitna_aligned_free(y);

    double t1 = vitna_time_ms();
    printf("Completed %d int4 matrix-vector ops in %.2f ms (%.2f ops/sec)\n",
        iterations, t1 - t0, (iterations / ((t1 - t0) / 1000.0))
    );
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* cmd = argv[1];
    uint16_t port = 8765;
    const char* model_path = NULL;
    int iterations = 100;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        }
    }

    if (strcmp(cmd, "serve") == 0) {
        vitna_server_config_t cfg;
        cfg.port = port;
        cfg.bind_addr = "127.0.0.1";
        cfg.engine_ctx = NULL;
        return vitna_server_run(&cfg);
    } else if (strcmp(cmd, "info") == 0) {
        return cmd_info(model_path);
    } else if (strcmp(cmd, "bench") == 0) {
        return cmd_bench(model_path, iterations);
    } else {
        print_usage(argv[0]);
        return 1;
    }
}
