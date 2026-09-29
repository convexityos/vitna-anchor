/**
 * server.h - The engine's HTTP server: the network half of api.h.
 */

#ifndef VITNA_SERVER_H
#define VITNA_SERVER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t port;          /* 0 asks the OS for any free port */
    const char* bind_addr;
    void* engine_ctx;       /* the vitna_api_t to serve, or NULL for a server with no model */
} vitna_server_config_t;

/**
 * Start the HTTP server. Answers one request at a time, and blocks until the
 * listening socket fails.
 */
int vitna_server_run(const vitna_server_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SERVER_H */
