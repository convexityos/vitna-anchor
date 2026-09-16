/**
 * server.h - Embedded OpenAI-compatible HTTP server for vitna-engine.
 */

#ifndef VITNA_SERVER_H
#define VITNA_SERVER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t port;
    const char* bind_addr;
    void* engine_ctx;
} vitna_server_config_t;

/**
 * Start the HTTP server. Blocks until stopped.
 */
int vitna_server_run(const vitna_server_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SERVER_H */
