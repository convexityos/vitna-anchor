/**
 * install.h - What the installer asks of the engine (gate A12): the machine
 * it is on, which of the catalogue's models fits it, and whether the files
 * fetched for that model are the pinned files.
 *
 * The installer scripts (install/install.sh and install/install.ps1) fetch
 * the engine, then ask it for a plan: the model chosen, the files to fetch,
 * each with its URL, size and SHA-256, and the arguments to serve it with.
 * They fetch the files, resuming any part already there, have the engine
 * check them, and start the server. The catalogue (install/catalog.json,
 * built in as catalog.c) lists the models in order of preference, each with
 * what it needs of the machine.
 */

#ifndef VITNA_INSTALL_H
#define VITNA_INSTALL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "json.h"
#include "strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_MAX_GPUS 8

typedef struct {
    char name[128];
    uint64_t memory_bytes;  /* the device's whole memory */
    uint64_t free_bytes;    /* free when asked */
    int major, minor;       /* compute capability */
} vitna_gpu_t;

typedef struct {
    char os[16];            /* windows, linux or macos */
    char arch[16];          /* x86_64 or arm64 */
    size_t cpu_threads;
    char matvec[16];        /* the CPU's matrix-vector path: avx2+fma, neon or scalar */
    uint64_t memory_bytes;  /* the machine's physical memory */
    uint64_t disk_free_bytes; /* free where the models go */
    bool cuda_built;        /* this engine was built with the CUDA path */
    size_t n_gpus;
    vitna_gpu_t gpus[VITNA_MAX_GPUS];
} vitna_hardware_t;

/** The machine this runs on, with the free space of the drive dir is on (or its nearest existing parent). */
bool vitna_hardware_detect(const char* dir, vitna_hardware_t* hw, char* err, size_t err_len);

/** hw as JSON, as `vitna-anchor hardware` prints it. */
void vitna_hardware_json(const vitna_hardware_t* hw, vitna_strbuf_t* out);

/** hw from that JSON, for tests that plan for a machine other than this one. */
bool vitna_hardware_read(const vitna_json_value_t* v, vitna_hardware_t* hw, char* err, size_t err_len);

/** The catalogue built into the engine (catalog.c), as JSON text. */
const char* vitna_catalog_text(void);

/**
 * The plan for a machine: the first model of the catalogue whose needs hw
 * meets, or want's if given, and every file of it to fetch into dir, as
 * lines of tab-separated fields:
 *
 *   model  <id>
 *   title  <title>
 *   note   <note>
 *   why    <why this model, in a sentence>
 *   file   <url>  <path under dir>  <size>  <sha256>
 *   fetch  <bytes still to fetch, beyond what dir already holds>
 *   serve  <argument>  <argument>  ...   (what to serve it with, but the port)
 *
 * Returns false, with the reason in err, when no model fits, or want is not
 * in the catalogue or does not fit.
 */
bool vitna_plan(const vitna_json_value_t* catalog, const vitna_hardware_t* hw, const char* dir, const char* want, vitna_strbuf_t* out, char* err,
                size_t err_len);

/**
 * Check model id's files under dir against the catalogue: a line for each,
 * "ok <path>" or "bad <path> <why>" (missing, short, long, or a SHA-256 other
 * than the pinned one), tab-separated. Returns true when every one is ok;
 * false, with err set and no lines, when id is not in the catalogue.
 */
bool vitna_verify(const vitna_json_value_t* catalog, const char* dir, const char* id, vitna_strbuf_t* out, bool* all_ok, char* err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_INSTALL_H */
