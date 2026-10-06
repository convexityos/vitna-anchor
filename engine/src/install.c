/**
 * install.c - The machine, the plan for it, and the files' check (install.h).
 */

#include "install.h"
#include "compat.h"
#include "crypto.h"
#include "ops.h"
#if defined(VITNA_CUDA)
#include "model_cuda.h"
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(VITNA_OS_WINDOWS)
#include <windows.h>
#else
#include <sys/statvfs.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

#define GIB 1073741824.0

/* What the drive must keep free beyond the files fetched. */
#define DISK_MARGIN ((uint64_t)1 << 30)

extern const char vitna_catalog_json[];

const char* vitna_catalog_text(void) {
    return vitna_catalog_json;
}

static bool fail(char* err, size_t err_len, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
    return false;
}

/* A file's size, or false when there is no file there. */
static bool file_size(const char* path, uint64_t* size) {
#if defined(VITNA_OS_WINDOWS)
    struct _stat64 st;
    if (_stat64(path, &st) != 0 || (st.st_mode & _S_IFREG) == 0) return false;
#else
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return false;
#endif
    *size = (uint64_t)st.st_size;
    return true;
}

static bool path_exists(const char* path) {
#if defined(VITNA_OS_WINDOWS)
    struct _stat64 st;
    return _stat64(path, &st) == 0;
#else
    struct stat st;
    return stat(path, &st) == 0;
#endif
}

/* Free bytes on the drive path is on, or on its nearest parent that exists. */
static uint64_t disk_free(const char* dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s", dir && *dir ? dir : ".");
    while (!path_exists(path)) {
        char* cut = NULL;
        for (char* c = path; *c; c++) {
            if (*c == '/' || *c == '\\') cut = c;
        }
        if (!cut) {
            snprintf(path, sizeof(path), ".");
            break;
        }
        if (cut == path) {
            cut[1] = '\0'; /* the root */
            break;
        }
        *cut = '\0';
        if (cut > path && cut[-1] == ':') {
            cut[0] = '\\'; /* C: is the drive's current directory; C:\ its root */
            cut[1] = '\0';
        }
    }
#if defined(VITNA_OS_WINDOWS)
    ULARGE_INTEGER avail;
    if (!GetDiskFreeSpaceExA(path, &avail, NULL, NULL)) return 0;
    return (uint64_t)avail.QuadPart;
#else
    struct statvfs s;
    if (statvfs(path, &s) != 0) return 0;
    return (uint64_t)s.f_bavail * (uint64_t)s.f_frsize;
#endif
}

bool vitna_hardware_detect(const char* dir, vitna_hardware_t* hw, char* err, size_t err_len) {
    (void)err;
    (void)err_len;
    memset(hw, 0, sizeof(*hw));
#if defined(VITNA_OS_WINDOWS)
    snprintf(hw->os, sizeof(hw->os), "windows");
#elif defined(__APPLE__)
    snprintf(hw->os, sizeof(hw->os), "macos");
#else
    snprintf(hw->os, sizeof(hw->os), "linux");
#endif
#if defined(__x86_64__) || defined(_M_X64)
    snprintf(hw->arch, sizeof(hw->arch), "x86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
    snprintf(hw->arch, sizeof(hw->arch), "arm64");
#else
    snprintf(hw->arch, sizeof(hw->arch), "other");
#endif
    snprintf(hw->matvec, sizeof(hw->matvec), "%s", vitna_matvec_path());
#if defined(VITNA_OS_WINDOWS)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    hw->cpu_threads = si.dwNumberOfProcessors;
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) hw->memory_bytes = ms.ullTotalPhys;
#else
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    hw->cpu_threads = cpus > 0 ? (size_t)cpus : 1;
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t mem_len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &mem_len, NULL, 0) == 0) hw->memory_bytes = mem;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) hw->memory_bytes = (uint64_t)pages * (uint64_t)page;
#endif
#endif
    hw->disk_free_bytes = disk_free(dir);
#if defined(VITNA_CUDA)
    hw->cuda_built = true;
    vitna_cuda_device_t devs[VITNA_MAX_GPUS];
    const int n = vitna_cuda_devices(devs, VITNA_MAX_GPUS);
    for (int i = 0; i < n; i++) {
        vitna_gpu_t* g = &hw->gpus[hw->n_gpus++];
        snprintf(g->name, sizeof(g->name), "%s", devs[i].name);
        g->memory_bytes = devs[i].total_bytes;
        g->free_bytes = devs[i].free_bytes;
        g->major = devs[i].major;
        g->minor = devs[i].minor;
    }
#endif
    return true;
}

void vitna_hardware_json(const vitna_hardware_t* hw, vitna_strbuf_t* out) {
    vitna_sb_printf(out, "{\"os\":\"%s\",\"arch\":\"%s\",\"cpu_threads\":%zu,\"matvec\":\"%s\",\"memory_bytes\":%llu,\"disk_free_bytes\":%llu,", hw->os,
                    hw->arch, hw->cpu_threads, hw->matvec, (unsigned long long)hw->memory_bytes, (unsigned long long)hw->disk_free_bytes);
    vitna_sb_printf(out, "\"cuda_built\":%s,\"gpus\":[", hw->cuda_built ? "true" : "false");
    for (size_t i = 0; i < hw->n_gpus; i++) {
        const vitna_gpu_t* g = &hw->gpus[i];
        vitna_sb_puts(out, i ? ",{\"name\":" : "{\"name\":");
        vitna_sb_json_string(out, (const unsigned char*)g->name, strlen(g->name));
        vitna_sb_printf(out, ",\"memory_bytes\":%llu,\"free_bytes\":%llu,\"compute\":\"%d.%d\"}", (unsigned long long)g->memory_bytes,
                        (unsigned long long)g->free_bytes, g->major, g->minor);
    }
    vitna_sb_puts(out, "]}");
}

static uint64_t number_u64(const vitna_json_value_t* v) {
    double d = 0;
    return vitna_json_as_number(v, &d) && d > 0 ? (uint64_t)d : 0;
}

static void copy_string(char* dst, size_t len, const vitna_json_value_t* v, const char* fallback) {
    const char* s = vitna_json_as_string(v);
    snprintf(dst, len, "%s", s ? s : fallback);
}

bool vitna_hardware_read(const vitna_json_value_t* v, vitna_hardware_t* hw, char* err, size_t err_len) {
    memset(hw, 0, sizeof(*hw));
    if (!v || v->type != VITNA_JSON_OBJECT) return fail(err, err_len, "a machine is described by a JSON object, as `hardware` prints one");
    copy_string(hw->os, sizeof(hw->os), vitna_json_get(v, "os"), "linux");
    copy_string(hw->arch, sizeof(hw->arch), vitna_json_get(v, "arch"), "x86_64");
    copy_string(hw->matvec, sizeof(hw->matvec), vitna_json_get(v, "matvec"), "scalar");
    hw->cpu_threads = (size_t)number_u64(vitna_json_get(v, "cpu_threads"));
    hw->memory_bytes = number_u64(vitna_json_get(v, "memory_bytes"));
    hw->disk_free_bytes = number_u64(vitna_json_get(v, "disk_free_bytes"));
    vitna_json_as_bool(vitna_json_get(v, "cuda_built"), &hw->cuda_built);
    const vitna_json_value_t* gpus = vitna_json_get(v, "gpus");
    for (size_t i = 0; gpus && gpus->type == VITNA_JSON_ARRAY && i < gpus->u.array.count && hw->n_gpus < VITNA_MAX_GPUS; i++) {
        const vitna_json_value_t* g = gpus->u.array.items[i];
        vitna_gpu_t* d = &hw->gpus[hw->n_gpus++];
        copy_string(d->name, sizeof(d->name), vitna_json_get(g, "name"), "GPU");
        d->memory_bytes = number_u64(vitna_json_get(g, "memory_bytes"));
        d->free_bytes = number_u64(vitna_json_get(g, "free_bytes"));
        const char* compute = vitna_json_as_string(vitna_json_get(g, "compute"));
        if (!compute || sscanf(compute, "%d.%d", &d->major, &d->minor) != 2) d->major = d->minor = 0;
    }
    return true;
}

/* --- The plan --- */

static const vitna_json_value_t* models_of(const vitna_json_value_t* catalog) {
    const vitna_json_value_t* m = catalog ? vitna_json_get(catalog, "models") : NULL;
    return m && m->type == VITNA_JSON_ARRAY ? m : NULL;
}

static const char* str_of(const vitna_json_value_t* obj, const char* key) {
    const char* s = vitna_json_as_string(vitna_json_get(obj, key));
    return s ? s : "";
}

static double num_of(const vitna_json_value_t* obj, const char* key) {
    double d = 0;
    vitna_json_as_number(vitna_json_get(obj, key), &d);
    return d;
}

static void join_path(char* out, size_t len, const char* dir, const char* rel) {
    const size_t n = strlen(dir);
    const bool sep = n > 0 && (dir[n - 1] == '/' || dir[n - 1] == '\\');
    snprintf(out, len, "%s%s%s", dir, sep ? "" : "/", rel);
}

/* The bytes of a file still to fetch: what dir lacks of it, or the whole of
 * it when what is there is longer than it should be. */
static uint64_t still_to_fetch(const char* dir, const char* rel, uint64_t size) {
    char path[4096];
    join_path(path, sizeof(path), dir, rel);
    uint64_t have = 0;
    if (!file_size(path, &have) || have > size) return size;
    return size - have;
}

static const vitna_gpu_t* largest_gpu(const vitna_hardware_t* hw) {
    const vitna_gpu_t* best = NULL;
    for (size_t i = 0; i < hw->n_gpus; i++) {
        if (!best || hw->gpus[i].memory_bytes > best->memory_bytes) best = &hw->gpus[i];
    }
    return best;
}

/* Whether hw meets model's needs, with what is still to fetch into dir in
 * *fetch, or, in why, what it lacks: "it has 16 GiB of memory". */
static bool meets(const vitna_json_value_t* model, const vitna_hardware_t* hw, const char* dir, uint64_t* fetch, char* why, size_t why_len) {
    const vitna_json_value_t* needs = vitna_json_get(model, "needs");
    bool cuda = false;
    vitna_json_as_bool(vitna_json_get(needs, "cuda"), &cuda);
    const double gpu_gib = num_of(needs, "gpu_gib"), memory_gib = num_of(needs, "memory_gib");
    *fetch = 0;
    const vitna_json_value_t* files = vitna_json_get(model, "files");
    for (size_t i = 0; files && files->type == VITNA_JSON_ARRAY && i < files->u.array.count; i++) {
        const vitna_json_value_t* f = files->u.array.items[i];
        *fetch += still_to_fetch(dir, str_of(f, "path"), (uint64_t)num_of(f, "size"));
    }
    if (cuda) {
        const vitna_gpu_t* g = largest_gpu(hw);
        if (!hw->cuda_built) return fail(why, why_len, "this engine was built without the CUDA path");
        if (!g) return fail(why, why_len, "no NVIDIA GPU was found");
        if ((double)g->memory_bytes < gpu_gib * GIB) return fail(why, why_len, "its largest NVIDIA GPU, %s, has %.1f GiB", g->name, g->memory_bytes / GIB);
    }
    if ((double)hw->memory_bytes < memory_gib * GIB) return fail(why, why_len, "it has %.0f GiB of memory", hw->memory_bytes / GIB);
    if (hw->disk_free_bytes < *fetch + DISK_MARGIN) {
        return fail(why, why_len, "it has %.1f GiB free where the models go, and the files need %.1f GiB more besides 1 GiB to spare", hw->disk_free_bytes / GIB,
                    *fetch / GIB);
    }
    return true;
}

/* What a model needs, as words: "an NVIDIA GPU with 7.5 GiB and 30 GiB of memory". */
static void needs_text(const vitna_json_value_t* model, char* out, size_t len) {
    const vitna_json_value_t* needs = vitna_json_get(model, "needs");
    bool cuda = false;
    vitna_json_as_bool(vitna_json_get(needs, "cuda"), &cuda);
    const double gpu_gib = num_of(needs, "gpu_gib"), memory_gib = num_of(needs, "memory_gib");
    if (cuda) snprintf(out, len, "an NVIDIA GPU with %.1f GiB and %.0f GiB of memory", gpu_gib, memory_gib);
    else snprintf(out, len, "%.0f GiB of memory", memory_gib);
}

/* A field of a plan's line: the catalogue's own words, which hold no tab or line break. */
static bool plain(const char* s) {
    return !strpbrk(s, "\t\r\n");
}

bool vitna_plan(const vitna_json_value_t* catalog, const vitna_hardware_t* hw, const char* dir, const char* want, vitna_strbuf_t* out, char* err,
                size_t err_len) {
    const vitna_json_value_t* models = models_of(catalog);
    if (!models || models->u.array.count == 0) return fail(err, err_len, "the catalogue lists no models");
    if (!plain(dir)) return fail(err, err_len, "the directory's name holds a tab or a line break");
    vitna_strbuf_t skipped;
    vitna_sb_init(&skipped);
    const vitna_json_value_t* chosen = NULL;
    uint64_t fetch = 0;
    bool found_want = false;
    for (size_t i = 0; i < models->u.array.count && !chosen; i++) {
        const vitna_json_value_t* m = models->u.array.items[i];
        const char* id = str_of(m, "id");
        if (want && strcmp(id, want) != 0) continue;
        found_want = true;
        char why[256], needs[128];
        if (meets(m, hw, dir, &fetch, why, sizeof(why))) {
            chosen = m;
            break;
        }
        needs_text(m, needs, sizeof(needs));
        if (want) {
            vitna_sb_free(&skipped);
            return fail(err, err_len, "%s needs %s, and %s", str_of(m, "title"), needs, why);
        }
        vitna_sb_printf(&skipped, "%s needs %s, and %s. ", str_of(m, "title"), needs, why);
    }
    if (want && !found_want) {
        vitna_sb_free(&skipped);
        return fail(err, err_len, "the catalogue has no model %s", want);
    }
    if (!chosen) {
        fail(err, err_len, "no model in the catalogue fits this machine: %s", skipped.data ? skipped.data : "");
        vitna_sb_free(&skipped);
        return false;
    }
    const char* id = str_of(chosen, "id");
    const char* device = str_of(chosen, "device");
    const char* weights = str_of(chosen, "weights");
    if (!plain(id) || !plain(str_of(chosen, "title")) || !plain(str_of(chosen, "note")) || !plain(weights)) {
        vitna_sb_free(&skipped);
        return fail(err, err_len, "the catalogue's entry for %s holds a tab or a line break", id);
    }

    /* Why: what was passed over, then what this machine has. */
    vitna_strbuf_t why;
    vitna_sb_init(&why);
    if (skipped.len) vitna_sb_append(&why, skipped.data, skipped.len);
    const vitna_gpu_t* g = largest_gpu(hw);
    if (strcmp(device, "cuda") == 0 && g) {
        vitna_sb_printf(&why, "This machine has an NVIDIA %s with %.1f GiB, %.0f GiB of memory and %.0f GiB free where the models go.",
                        strncmp(g->name, "NVIDIA ", 7) == 0 ? g->name + 7 : g->name, g->memory_bytes / GIB, hw->memory_bytes / GIB,
                        hw->disk_free_bytes / GIB);
    } else {
        vitna_sb_printf(&why, "This machine has %.0f GiB of memory and %.0f GiB free where the models go.", hw->memory_bytes / GIB, hw->disk_free_bytes / GIB);
    }

    /* The largest context its memory affords. */
    size_t ctx = 4096;
    const vitna_json_value_t* ctxs = vitna_json_get(chosen, "ctx");
    for (size_t i = 0; ctxs && ctxs->type == VITNA_JSON_ARRAY && i < ctxs->u.array.count; i++) {
        const vitna_json_value_t* c = ctxs->u.array.items[i];
        if ((double)hw->memory_bytes >= num_of(c, "memory_gib") * GIB) {
            ctx = (size_t)num_of(c, "ctx");
            break;
        }
    }

    vitna_sb_printf(out, "model\t%s\ntitle\t%s\nnote\t%s\nwhy\t%s\n", id, str_of(chosen, "title"), str_of(chosen, "note"), why.data);
    const vitna_json_value_t* files = vitna_json_get(chosen, "files");
    for (size_t i = 0; files && files->type == VITNA_JSON_ARRAY && i < files->u.array.count; i++) {
        const vitna_json_value_t* f = files->u.array.items[i];
        vitna_sb_printf(out, "file\t%s\t%s\t%llu\t%s\n", str_of(f, "url"), str_of(f, "path"), (unsigned long long)num_of(f, "size"), str_of(f, "sha256"));
    }
    char model_dir[4096];
    join_path(model_dir, sizeof(model_dir), dir, id);
    vitna_sb_printf(out, "fetch\t%llu\nserve\t--model\t%s\t--model-id\t%s\t--ctx\t%zu", (unsigned long long)fetch, model_dir, id, ctx);
    if (*weights) {
        char path[4096];
        join_path(path, sizeof(path), model_dir, weights);
        vitna_sb_printf(out, "\t--weights\t%s", path);
    }
    if (*device) vitna_sb_printf(out, "\t--device\t%s", device);
    vitna_sb_puts(out, "\n");
    vitna_sb_free(&why);
    vitna_sb_free(&skipped);
    if (!out->ok) return fail(err, err_len, "out of memory");
    return true;
}

/* --- The files' check --- */

static bool sha256_file(const char* path, char hex[65]) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    const size_t chunk = (size_t)4 << 20;
    unsigned char* buf = (unsigned char*)malloc(chunk);
    if (!buf) {
        fclose(f);
        return false;
    }
    vitna_sha256_ctx_t ctx;
    vitna_sha256_init(&ctx);
    size_t got;
    while ((got = fread(buf, 1, chunk, f)) > 0) vitna_sha256_update(&ctx, buf, got);
    const bool ok = !ferror(f);
    fclose(f);
    free(buf);
    if (ok) vitna_sha256_final_hex(&ctx, hex);
    return ok;
}

bool vitna_verify(const vitna_json_value_t* catalog, const char* dir, const char* id, vitna_strbuf_t* out, bool* all_ok, char* err, size_t err_len) {
    const vitna_json_value_t* models = models_of(catalog);
    const vitna_json_value_t* model = NULL;
    for (size_t i = 0; models && i < models->u.array.count && !model; i++) {
        if (strcmp(str_of(models->u.array.items[i], "id"), id) == 0) model = models->u.array.items[i];
    }
    if (!model) return fail(err, err_len, "the catalogue has no model %s", id);
    *all_ok = true;
    const vitna_json_value_t* files = vitna_json_get(model, "files");
    for (size_t i = 0; files && files->type == VITNA_JSON_ARRAY && i < files->u.array.count; i++) {
        const vitna_json_value_t* f = files->u.array.items[i];
        const char* rel = str_of(f, "path");
        const uint64_t size = (uint64_t)num_of(f, "size");
        char path[4096], hex[65];
        join_path(path, sizeof(path), dir, rel);
        uint64_t have = 0;
        if (!file_size(path, &have)) {
            vitna_sb_printf(out, "bad\t%s\tmissing\n", rel);
        } else if (have < size) {
            vitna_sb_printf(out, "bad\t%s\tshort: %llu of %llu bytes\n", rel, (unsigned long long)have, (unsigned long long)size);
        } else if (have > size) {
            vitna_sb_printf(out, "bad\t%s\tlong: %llu bytes, pinned %llu\n", rel, (unsigned long long)have, (unsigned long long)size);
        } else if (!sha256_file(path, hex)) {
            vitna_sb_printf(out, "bad\t%s\tunreadable\n", rel);
        } else if (strcmp(hex, str_of(f, "sha256")) != 0) {
            vitna_sb_printf(out, "bad\t%s\tsha256 %s, pinned %s\n", rel, hex, str_of(f, "sha256"));
        } else {
            vitna_sb_printf(out, "ok\t%s\n", rel);
            continue;
        }
        *all_ok = false;
    }
    if (!out->ok) return fail(err, err_len, "out of memory");
    return true;
}
