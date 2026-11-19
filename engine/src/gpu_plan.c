/**
 * gpu_plan.c - Multi-GPU device selection and pipeline-parallel layer splitter.
 */

#include "gpu_plan.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool set_err(char* err, size_t err_len, const char* msg) {
    if (err && err_len > 0) snprintf(err, err_len, "%s", msg);
    return false;
}

bool vitna_gpu_parse_device_list(const char* spec, int* out_ids, size_t max_ids, size_t* out_count, char* err,
                                 size_t err_len) {
    if (out_count) *out_count = 0;
    if (err && err_len) err[0] = '\0';
    if (!spec || !*spec || !out_ids || !out_count || max_ids == 0) {
        return set_err(err, err_len, "device list must not be empty");
    }

    /* Strip optional "cuda:" or "hip:" prefix. */
    const char* p = spec;
    if (strncmp(p, "cuda:", 5) == 0) {
        p += 5;
    } else if (strncmp(p, "hip:", 4) == 0) {
        p += 4;
    }
    if (!*p) return set_err(err, err_len, "missing GPU device index after prefix");

    size_t count = 0;
    while (*p) {
        if (!isdigit((unsigned char)*p)) {
            return set_err(err, err_len, "GPU device id must be a non-negative integer");
        }
        unsigned long val = 0;
        while (isdigit((unsigned char)*p)) {
            val = val * 10ul + (unsigned long)(*p - '0');
            if (val >= VITNA_MAX_GPUS) {
                return set_err(err, err_len, "GPU device id exceeds VITNA_MAX_GPUS");
            }
            p++;
        }
        if (*p != '\0' && *p != ',') {
            return set_err(err, err_len, "expected comma between GPU device ids");
        }
        if (*p == ',') {
            p++;
            if (*p == '\0') return set_err(err, err_len, "trailing comma in GPU device list");
        }
        const int id = (int)val;
        for (size_t i = 0; i < count; i++) {
            if (out_ids[i] == id) return set_err(err, err_len, "duplicate GPU device id");
        }
        if (count >= max_ids || count >= VITNA_MAX_GPUS) {
            return set_err(err, err_len, "too many GPU devices listed");
        }
        out_ids[count++] = id;
    }

    *out_count = count;
    return count > 0;
}

bool vitna_gpu_parse_ratios(const char* spec, size_t expected_count, double* out_ratios, char* err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!spec || !*spec || !out_ratios || expected_count == 0 || expected_count > VITNA_MAX_GPUS) {
        return set_err(err, err_len, "invalid GPU split ratio specification");
    }

    double raw[VITNA_MAX_GPUS];
    size_t count = 0;
    double sum = 0.0;
    const char* p = spec;

    while (*p) {
        if (count >= expected_count) {
            return set_err(err, err_len, "more GPU split ratios given than devices");
        }
        if (isspace((unsigned char)*p)) {
            return set_err(err, err_len, "unexpected whitespace in GPU split ratios");
        }
        char* end = NULL;
        const double v = strtod(p, &end);
        if (end == p || !isfinite(v) || v <= 0.0) {
            return set_err(err, err_len, "each GPU split ratio must be a positive finite number");
        }
        raw[count++] = v;
        sum += v;
        p = end;
        if (*p == ',') {
            p++;
            if (*p == '\0') return set_err(err, err_len, "trailing comma in GPU split ratios");
        } else if (*p != '\0') {
            return set_err(err, err_len, "unexpected character in GPU split ratios");
        }
    }

    if (count != expected_count) {
        return set_err(err, err_len, "number of GPU split ratios must match number of devices");
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        return set_err(err, err_len, "GPU split ratios sum is invalid");
    }
    for (size_t i = 0; i < count; i++) out_ratios[i] = raw[i] / sum;
    return true;
}

bool vitna_gpu_plan_split(const vitna_gpu_device_spec_t* devs, size_t n_devs, const vitna_gpu_model_budget_t* model,
                          const double* explicit_ratios, vitna_gpu_split_plan_t* plan) {
    memset(plan, 0, sizeof(*plan));
    if (!devs || !model || n_devs == 0 || n_devs > VITNA_MAX_GPUS) {
        snprintf(plan->reason, sizeof(plan->reason), "need between 1 and %d GPU devices", VITNA_MAX_GPUS);
        return false;
    }
    if (model->n_layers == 0) {
        snprintf(plan->reason, sizeof(plan->reason), "model must have at least 1 layer");
        return false;
    }

    const size_t reserve = model->reserve_bytes ? model->reserve_bytes : VITNA_GPU_DEFAULT_RESERVE;
    const size_t per_layer = model->layer_weights_bytes + model->layer_kv_bytes;
    const size_t used_devs = n_devs < model->n_layers ? n_devs : model->n_layers;

    /* Compute usable VRAM on each active GPU after reserve and fixed embed/head weights. */
    size_t usable_total[VITNA_MAX_GPUS];
    size_t usable_for_layers[VITNA_MAX_GPUS];
    double weights[VITNA_MAX_GPUS];
    double weight_sum = 0.0;

    for (size_t i = 0; i < used_devs; i++) {
        const size_t fixed = (i == 0 ? model->embed_bytes : 0) + (i + 1 == used_devs ? model->head_bytes : 0);
        if (devs[i].free_bytes <= reserve) {
            snprintf(plan->reason, sizeof(plan->reason),
                     "GPU %d has only %.0f MiB free, below the %.0f MiB reserve", devs[i].index,
                     (double)devs[i].free_bytes / 1048576.0, (double)reserve / 1048576.0);
            return false;
        }
        usable_total[i] = devs[i].free_bytes - reserve;
        if (usable_total[i] < fixed) {
            snprintf(plan->reason, sizeof(plan->reason),
                     "GPU %d lacks VRAM for fixed %s weights (%.0f MiB needed, %.0f MiB usable)", devs[i].index,
                     i == 0 ? "embedding" : "output head", (double)fixed / 1048576.0,
                     (double)usable_total[i] / 1048576.0);
            return false;
        }
        usable_for_layers[i] = usable_total[i] - fixed;

        if (explicit_ratios) {
            if (!isfinite(explicit_ratios[i]) || explicit_ratios[i] <= 0.0) {
                snprintf(plan->reason, sizeof(plan->reason), "explicit ratio for GPU %d must be positive", devs[i].index);
                return false;
            }
            weights[i] = explicit_ratios[i];
        } else {
            weights[i] = (double)usable_for_layers[i];
        }
        weight_sum += weights[i];
    }

    if (!(weight_sum > 0.0)) {
        snprintf(plan->reason, sizeof(plan->reason), "no GPU has usable VRAM left for model layers");
        return false;
    }

    /* Largest-remainder (Hamilton) apportionment of n_layers across used_devs.
     * Guarantees every layer is assigned to exactly one GPU with >= 1 layer per active GPU. */
    size_t counts[VITNA_MAX_GPUS];
    double remainders[VITNA_MAX_GPUS];
    size_t assigned = 0;

    for (size_t i = 0; i < used_devs; i++) {
        const double exact = (double)model->n_layers * (weights[i] / weight_sum);
        size_t base = (size_t)floor(exact);
        if (base == 0) base = 1;
        counts[i] = base;
        remainders[i] = exact - (double)base;
        assigned += base;
    }

    /* Adjust if clamping min=1 overshot n_layers. */
    while (assigned > model->n_layers) {
        size_t best = 0;
        bool found = false;
        for (size_t i = 0; i < used_devs; i++) {
            if (counts[i] > 1 && (!found || remainders[i] < remainders[best])) {
                best = i;
                found = true;
            }
        }
        if (!found) break;
        counts[best]--;
        remainders[best] += 1.0;
        assigned--;
    }

    /* Distribute leftover layers to highest fractional remainders. */
    while (assigned < model->n_layers) {
        size_t best = 0;
        for (size_t i = 1; i < used_devs; i++) {
            if (remainders[i] > remainders[best]) best = i;
        }
        counts[best]++;
        remainders[best] = -1e9;
        assigned++;
    }

    /* Materialize shards and verify resident memory fits on every GPU. */
    size_t cursor = 0;
    size_t total_resident = 0;
    size_t total_ecache = 0;
    size_t max_shortfall = 0;
    int worst_gpu = -1;

    for (size_t i = 0; i < used_devs; i++) {
        vitna_gpu_shard_t* s = &plan->shards[i];
        s->device_index = devs[i].index;
        s->layer_begin = cursor;
        s->layer_end = cursor + counts[i];
        cursor = s->layer_end;
        s->holds_embed = (i == 0);
        s->holds_head = (i + 1 == used_devs);

        const size_t fixed = (s->holds_embed ? model->embed_bytes : 0) + (s->holds_head ? model->head_bytes : 0);
        const size_t layer_mem = counts[i] * per_layer;
        s->resident_bytes = fixed + layer_mem;
        total_resident += s->resident_bytes;

        if (s->resident_bytes > usable_total[i]) {
            const size_t shortfall = s->resident_bytes - usable_total[i];
            if (shortfall > max_shortfall) {
                max_shortfall = shortfall;
                worst_gpu = devs[i].index;
            }
            s->expert_cache_bytes = 0;
        } else if (model->expert_bytes_per_layer > 0) {
            const size_t avail = usable_total[i] - s->resident_bytes;
            const size_t needed = counts[i] * model->expert_bytes_per_layer;
            s->expert_cache_bytes = avail < needed ? avail : needed;
            total_ecache += s->expert_cache_bytes;
        } else {
            s->expert_cache_bytes = 0;
        }
    }

    plan->n_shards = used_devs;
    plan->total_resident_bytes = total_resident;
    plan->total_expert_cache_bytes = total_ecache;

    if (max_shortfall > 0) {
        plan->fits = false;
        snprintf(plan->reason, sizeof(plan->reason),
                 "GPU %d short by %.0f MiB for its assigned layers and KV cache", worst_gpu,
                 (double)max_shortfall / 1048576.0);
        return false;
    }

    plan->fits = true;
    plan->reason[0] = '\0';
    return true;
}

int vitna_gpu_layer_device(const vitna_gpu_split_plan_t* plan, size_t layer) {
    if (!plan || !plan->fits) return -1;
    for (size_t i = 0; i < plan->n_shards; i++) {
        if (layer >= plan->shards[i].layer_begin && layer < plan->shards[i].layer_end) {
            return plan->shards[i].device_index;
        }
    }
    return -1;
}

const char* vitna_gpu_plan_format(const vitna_gpu_split_plan_t* plan, char* buf, size_t len) {
    if (!buf || len == 0) return buf;
    buf[0] = '\0';
    if (!plan || plan->n_shards == 0) return buf;

    size_t pos = 0;
    for (size_t i = 0; i < plan->n_shards && pos < len; i++) {
        const vitna_gpu_shard_t* s = &plan->shards[i];
        const char* role = (s->holds_embed && s->holds_head) ? "embed+head, "
                         : s->holds_embed ? "embed, "
                         : s->holds_head ? "head, " : "";
        int n;
        if (s->expert_cache_bytes > 0) {
            n = snprintf(buf + pos, len - pos, "%sgpu %d: layers %zu..%zu (%s%.0f MiB resident, %.0f MiB expert cache)",
                         i > 0 ? "; " : "", s->device_index, s->layer_begin, s->layer_end - 1, role,
                         (double)s->resident_bytes / 1048576.0, (double)s->expert_cache_bytes / 1048576.0);
        } else {
            n = snprintf(buf + pos, len - pos, "%sgpu %d: layers %zu..%zu (%s%.0f MiB resident)",
                         i > 0 ? "; " : "", s->device_index, s->layer_begin, s->layer_end - 1, role,
                         (double)s->resident_bytes / 1048576.0);
        }
        if (n < 0) break;
        pos += (size_t)n < len - pos ? (size_t)n : len - pos;
    }
    return buf;
}
