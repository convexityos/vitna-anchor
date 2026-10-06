/**
 * tokenizer.c - Byte-level BPE, read from a Hugging Face tokenizer.json.
 *
 * See tokenizer.h for the steps. Each follows the tokenizers library
 * (added_vocabulary.rs, pre_tokenizers/digits.rs and byte_level.rs,
 * models/bpe/word.rs), and tests/reference.test.mjs compares the result with
 * that library's ids for the reference corpus.
 */

#include "tokenizer.h"
#include "json.h"
#include "unicode.h"
#include "wordpiece.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char* bytes;
    uint32_t len;
} tok_entry_t;

typedef struct {
    unsigned char* bytes;        /* what is matched: the content, normalized when the token is */
    size_t len;
    int32_t id;
    bool special;
    bool normalized;             /* matched in normalized text, not as written */
} added_token_t;

struct vitna_tokenizer {
    size_t vocab_size;
    tok_entry_t* tokens;         /* by id; bytes NULL where no token has this id */
    unsigned char* is_special;   /* by id */

    int32_t* map_ids;            /* open addressing: raw bytes -> id, -1 for an empty slot */
    size_t map_cap;              /* a power of two */

    uint64_t* merge_keys;        /* open addressing: (left << 32 | right) -> merge */
    int32_t* merge_ids;
    uint32_t* merge_ranks;
    unsigned char* merge_used;
    size_t merge_cap;

    int32_t byte_token[256];     /* the single-byte token of each byte, or -1 */

    added_token_t* added;
    size_t n_added;

    int digits;                  /* 0: no Digits step; 1: each digit alone; 2: runs of digits */
    bool add_prefix_space;
    bool ignore_merges;
    bool nfc;                    /* the normalizer is NFC; without it there is none */

    /* A WordPiece tokenizer.json (a BERT model's), which wordpiece.c runs in
     * place of everything above. */
    vitna_wordpiece_t* wordpiece;
};

/* --- Token lists --- */

bool vitna_token_list_push(vitna_token_list_t* list, int32_t id) {
    if (list->count == list->cap) {
        size_t cap = list->cap ? list->cap * 2 : 64;
        int32_t* grown = (int32_t*)realloc(list->ids, cap * sizeof(int32_t));
        if (!grown) return false;
        list->ids = grown;
        list->cap = cap;
    }
    list->ids[list->count++] = id;
    return true;
}

void vitna_token_list_free(vitna_token_list_t* list) {
    free(list->ids);
    list->ids = NULL;
    list->count = list->cap = 0;
}

/* --- The byte-level alphabet (GPT-2's bytes_to_unicode) --- */

/* Printable bytes stand for themselves; the other 68 are shifted to
 * U+0100..U+0143 in byte order. */
static void byte_level_maps(uint32_t byte_to_cp[256], int cp_to_byte[324]) {
    for (int i = 0; i < 324; i++) cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = direct ? (uint32_t)b : (uint32_t)(256 + n++);
        byte_to_cp[b] = cp;
        cp_to_byte[cp] = b;
    }
}

/* The raw bytes a byte-level string stands for. Returns false if a character
 * is outside the alphabet. out must hold at least len bytes. */
static bool byte_level_decode(const char* s, size_t len, const int cp_to_byte[324], unsigned char* out, size_t* out_len) {
    size_t i = 0, n = 0;
    while (i < len) {
        uint32_t cp;
        i += vitna_utf8_decode((const unsigned char*)s + i, len - i, &cp);
        if (cp >= 324 || cp_to_byte[cp] < 0) return false;
        out[n++] = (unsigned char)cp_to_byte[cp];
    }
    *out_len = n;
    return true;
}

/* --- Hash maps --- */

static uint64_t fnv1a(const unsigned char* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t pow2_at_least(size_t n) {
    size_t c = 16;
    while (c < n) c <<= 1;
    return c;
}

static int32_t map_find(const vitna_tokenizer_t* t, const unsigned char* p, size_t n) {
    size_t mask = t->map_cap - 1;
    for (size_t slot = (size_t)fnv1a(p, n) & mask;; slot = (slot + 1) & mask) {
        int32_t id = t->map_ids[slot];
        if (id < 0) return -1;
        const tok_entry_t* e = &t->tokens[id];
        if (e->len == n && memcmp(e->bytes, p, n) == 0) return id;
    }
}

/* Insert id under its bytes. A second id with the same bytes keeps the first. */
static void map_insert(vitna_tokenizer_t* t, int32_t id) {
    const tok_entry_t* e = &t->tokens[id];
    size_t mask = t->map_cap - 1;
    for (size_t slot = (size_t)fnv1a(e->bytes, e->len) & mask;; slot = (slot + 1) & mask) {
        int32_t other = t->map_ids[slot];
        if (other < 0) {
            t->map_ids[slot] = id;
            return;
        }
        const tok_entry_t* o = &t->tokens[other];
        if (o->len == e->len && memcmp(o->bytes, e->bytes, e->len) == 0) return;
    }
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static bool merge_find(const vitna_tokenizer_t* t, int32_t left, int32_t right, int32_t* merged, uint32_t* rank) {
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    size_t mask = t->merge_cap - 1;
    for (size_t slot = (size_t)mix64(key) & mask;; slot = (slot + 1) & mask) {
        if (!t->merge_used[slot]) return false;
        if (t->merge_keys[slot] == key) {
            *merged = t->merge_ids[slot];
            *rank = t->merge_ranks[slot];
            return true;
        }
    }
}

/* Insert a merge. A pair listed twice keeps its first (lower) rank. */
static void merge_insert(vitna_tokenizer_t* t, int32_t left, int32_t right, int32_t merged, uint32_t rank) {
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    size_t mask = t->merge_cap - 1;
    for (size_t slot = (size_t)mix64(key) & mask;; slot = (slot + 1) & mask) {
        if (!t->merge_used[slot]) {
            t->merge_used[slot] = 1;
            t->merge_keys[slot] = key;
            t->merge_ids[slot] = merged;
            t->merge_ranks[slot] = rank;
            return;
        }
        if (t->merge_keys[slot] == key) return;
    }
}

/* --- Loading --- */

static bool errf(char* err, size_t err_len, const char* msg, const char* detail) {
    if (err && err_len > 0) snprintf(err, err_len, "tokenizer.json: %s%s%s", msg, detail ? ": " : "", detail ? detail : "");
    return false;
}

static const char* type_of(const vitna_json_value_t* v) {
    return vitna_json_as_string(vitna_json_get(v, "type"));
}

static bool is_false_or_missing(const vitna_json_value_t* v) {
    bool b;
    return !v || vitna_json_is_null(v) || (vitna_json_as_bool(v, &b) && !b);
}

static bool configure_pretokenizer(vitna_tokenizer_t* t, const vitna_json_value_t* pre, char* err, size_t err_len) {
    const vitna_json_value_t* byte_level = NULL;
    const char* type = type_of(pre);
    if (!type) return errf(err, err_len, "pre_tokenizer has no type", NULL);
    if (strcmp(type, "ByteLevel") == 0) {
        byte_level = pre;
    } else if (strcmp(type, "Sequence") == 0) {
        const vitna_json_value_t* list = vitna_json_get(pre, "pretokenizers");
        if (!list || list->type != VITNA_JSON_ARRAY) return errf(err, err_len, "Sequence pre_tokenizer without a list", NULL);
        for (size_t i = 0; i < list->u.array.count; i++) {
            const vitna_json_value_t* step = list->u.array.items[i];
            const char* st = type_of(step);
            if (st && strcmp(st, "Digits") == 0 && i + 1 < list->u.array.count && !t->digits) {
                bool individual = false;
                vitna_json_as_bool(vitna_json_get(step, "individual_digits"), &individual);
                t->digits = individual ? 1 : 2;
            } else if (st && strcmp(st, "ByteLevel") == 0 && i + 1 == list->u.array.count) {
                byte_level = step;
            } else {
                return errf(err, err_len, "unsupported pre_tokenizer step", st ? st : "(no type)");
            }
        }
    } else {
        return errf(err, err_len, "unsupported pre_tokenizer", type);
    }
    if (!byte_level) return errf(err, err_len, "the pre_tokenizer has no ByteLevel step", NULL);
    bool use_regex = true;
    vitna_json_as_bool(vitna_json_get(byte_level, "use_regex"), &use_regex);
    if (!use_regex) return errf(err, err_len, "ByteLevel without its regex is not supported", NULL);
    vitna_json_as_bool(vitna_json_get(byte_level, "add_prefix_space"), &t->add_prefix_space);
    return true;
}

/* A post-processor that changes nothing in the encoding of one text: none,
 * ByteLevel (whose offset trimming touches no id), or a TemplateProcessing
 * whose single-sequence template is the sequence alone, with no special
 * tokens to add. */
static bool post_adds_nothing(const vitna_json_value_t* post) {
    if (!post || vitna_json_is_null(post)) return true;
    const char* type = type_of(post);
    if (type && strcmp(type, "ByteLevel") == 0) return true;
    if (!type || strcmp(type, "TemplateProcessing") != 0) return false;
    const vitna_json_value_t* single = vitna_json_get(post, "single");
    if (!single || single->type != VITNA_JSON_ARRAY || single->u.array.count != 1) return false;
    const vitna_json_value_t* seq = vitna_json_get(single->u.array.items[0], "Sequence");
    const char* id = vitna_json_as_string(vitna_json_get(seq, "id"));
    if (!id || strcmp(id, "A") != 0) return false;
    const vitna_json_value_t* specials = vitna_json_get(post, "special_tokens");
    return !specials || vitna_json_is_null(specials) || (specials->type == VITNA_JSON_OBJECT && specials->u.object.count == 0);
}

vitna_tokenizer_t* vitna_tokenizer_load(const char* path, char* err, size_t err_len) {
    size_t text_len = 0;
    char* text = vitna_read_file(path, &text_len);
    if (!text) {
        errf(err, err_len, "cannot read", path);
        return NULL;
    }
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(text, text_len, jerr, sizeof(jerr));
    free(text);
    if (!doc) {
        errf(err, err_len, "not valid JSON", jerr);
        return NULL;
    }

    vitna_tokenizer_t* t = (vitna_tokenizer_t*)calloc(1, sizeof(vitna_tokenizer_t));
    unsigned char* scratch = NULL;
    bool ok = false;
    if (!t) goto done;

    const vitna_json_value_t* root = vitna_json_root(doc);
    if (vitna_wordpiece_wanted(root)) {
        t->wordpiece = vitna_wordpiece_from_json(root, err, err_len);
        ok = t->wordpiece != NULL;
        goto done;
    }
    const vitna_json_value_t* norm = vitna_json_get(root, "normalizer");
    if (norm && !vitna_json_is_null(norm)) {
        if (!type_of(norm) || strcmp(type_of(norm), "NFC") != 0) { errf(err, err_len, "normalizers other than NFC are not supported", type_of(norm)); goto done; }
        t->nfc = true;
    }
    const vitna_json_value_t* post = vitna_json_get(root, "post_processor");
    if (!post_adds_nothing(post)) {
        errf(err, err_len, "a post_processor that adds tokens is not supported", type_of(post));
        goto done;
    }
    const vitna_json_value_t* dec = vitna_json_get(root, "decoder");
    if (dec && !vitna_json_is_null(dec) && !(type_of(dec) && strcmp(type_of(dec), "ByteLevel") == 0)) {
        errf(err, err_len, "decoders other than ByteLevel are not supported", type_of(dec));
        goto done;
    }
    const vitna_json_value_t* pre = vitna_json_get(root, "pre_tokenizer");
    if (!pre || vitna_json_is_null(pre)) { errf(err, err_len, "no pre_tokenizer", NULL); goto done; }
    if (!configure_pretokenizer(t, pre, err, err_len)) goto done;

    const vitna_json_value_t* model = vitna_json_get(root, "model");
    const char* mtype = type_of(model);
    if (!mtype || strcmp(mtype, "BPE") != 0) { errf(err, err_len, "only BPE models are supported", mtype); goto done; }
    if (!(vitna_json_get(model, "dropout") == NULL || vitna_json_is_null(vitna_json_get(model, "dropout")))) {
        errf(err, err_len, "BPE dropout is not supported", NULL);
        goto done;
    }
    if (!is_false_or_missing(vitna_json_get(model, "byte_fallback")) || !is_false_or_missing(vitna_json_get(model, "fuse_unk"))) {
        errf(err, err_len, "byte_fallback and fuse_unk are not supported", NULL);
        goto done;
    }
    const vitna_json_value_t* unk = vitna_json_get(model, "unk_token");
    const vitna_json_value_t* cont = vitna_json_get(model, "continuing_subword_prefix");
    const vitna_json_value_t* eow = vitna_json_get(model, "end_of_word_suffix");
    if ((unk && !vitna_json_is_null(unk)) || (cont && !vitna_json_is_null(cont)) || (eow && !vitna_json_is_null(eow))) {
        errf(err, err_len, "unk_token, continuing_subword_prefix and end_of_word_suffix are not supported", NULL);
        goto done;
    }
    vitna_json_as_bool(vitna_json_get(model, "ignore_merges"), &t->ignore_merges);

    const vitna_json_value_t* vocab = vitna_json_get(model, "vocab");
    const vitna_json_value_t* merges = vitna_json_get(model, "merges");
    const vitna_json_value_t* added = vitna_json_get(root, "added_tokens");
    if (!vocab || vocab->type != VITNA_JSON_OBJECT || !merges || merges->type != VITNA_JSON_ARRAY) {
        errf(err, err_len, "model.vocab or model.merges is missing", NULL);
        goto done;
    }

    /* The vocabulary size is one more than the largest id anywhere. */
    double d;
    size_t max_id = 0;
    for (size_t i = 0; i < vocab->u.object.count; i++) {
        if (!vitna_json_as_number(vocab->u.object.members[i].value, &d) || d < 0 || d > 16777216.0 || d != (double)(int32_t)d) {
            errf(err, err_len, "a vocab id is not a small non-negative integer", vocab->u.object.members[i].key);
            goto done;
        }
        if ((size_t)d > max_id) max_id = (size_t)d;
    }
    size_t n_added_json = (added && added->type == VITNA_JSON_ARRAY) ? added->u.array.count : 0;
    for (size_t i = 0; i < n_added_json; i++) {
        if (vitna_json_as_number(vitna_json_get(added->u.array.items[i], "id"), &d) && d >= 0 && d <= 16777216.0 && (size_t)d > max_id) {
            max_id = (size_t)d;
        }
    }
    t->vocab_size = max_id + 1;
    t->tokens = (tok_entry_t*)calloc(t->vocab_size, sizeof(tok_entry_t));
    t->is_special = (unsigned char*)calloc(t->vocab_size, 1);
    if (!t->tokens || !t->is_special) { errf(err, err_len, "out of memory", NULL); goto done; }

    /* Added tokens: their bytes are their content, as written. */
    t->added = (added_token_t*)calloc(n_added_json ? n_added_json : 1, sizeof(added_token_t));
    if (!t->added) { errf(err, err_len, "out of memory", NULL); goto done; }
    for (size_t i = 0; i < n_added_json; i++) {
        const vitna_json_value_t* a = added->u.array.items[i];
        const vitna_json_value_t* content = vitna_json_get(a, "content");
        const char* s = vitna_json_as_string(content);
        if (!s || !vitna_json_as_number(vitna_json_get(a, "id"), &d) || content->u.string.len == 0) {
            errf(err, err_len, "an added token lacks content or id", NULL);
            goto done;
        }
        if (!is_false_or_missing(vitna_json_get(a, "lstrip")) || !is_false_or_missing(vitna_json_get(a, "rstrip")) ||
            !is_false_or_missing(vitna_json_get(a, "single_word"))) {
            errf(err, err_len, "added tokens with lstrip, rstrip or single_word are not supported", s);
            goto done;
        }
        added_token_t* at = &t->added[t->n_added++];
        at->id = (int32_t)d;
        vitna_json_as_bool(vitna_json_get(a, "special"), &at->special);
        /* Unless the file says, a token is normalized when it is not special,
         * as the tokenizers library's AddedToken has it. */
        at->normalized = !at->special;
        vitna_json_as_bool(vitna_json_get(a, "normalized"), &at->normalized);
        if (at->special) t->is_special[at->id] = 1;
        /* A normalized token is matched in normalized text, so it is matched
         * as its own content normalized. */
        if (at->normalized && t->nfc) {
            at->bytes = vitna_uni_nfc((const unsigned char*)s, content->u.string.len, &at->len);
        } else {
            at->len = content->u.string.len;
            at->bytes = (unsigned char*)malloc(at->len);
            if (at->bytes) memcpy(at->bytes, s, at->len);
        }
        if (!at->bytes) { errf(err, err_len, "out of memory", NULL); goto done; }
        tok_entry_t* e = &t->tokens[at->id];
        if (!e->bytes) {
            e->bytes = (unsigned char*)malloc(at->len);
            if (!e->bytes) { errf(err, err_len, "out of memory", NULL); goto done; }
            memcpy(e->bytes, s, at->len);
            e->len = (uint32_t)at->len;
        }
    }

    /* Vocabulary entries: byte-level strings, stored as the bytes they stand for. */
    uint32_t byte_to_cp[256];
    int cp_to_byte[324];
    byte_level_maps(byte_to_cp, cp_to_byte);
    size_t longest = 0;
    for (size_t i = 0; i < vocab->u.object.count; i++) {
        if (vocab->u.object.members[i].key_len > longest) longest = vocab->u.object.members[i].key_len;
    }
    scratch = (unsigned char*)malloc(2 * longest + 2);
    if (!scratch) { errf(err, err_len, "out of memory", NULL); goto done; }
    for (size_t i = 0; i < vocab->u.object.count; i++) {
        const vitna_json_member_t* m = &vocab->u.object.members[i];
        vitna_json_as_number(m->value, &d);
        tok_entry_t* e = &t->tokens[(size_t)d];
        if (e->bytes) continue; /* an added token, or a repeated id */
        size_t n = 0;
        const unsigned char* src = scratch;
        if (!byte_level_decode(m->key, m->key_len, cp_to_byte, scratch, &n)) {
            /* Outside the byte-level alphabet: keep the text as written. */
            src = (const unsigned char*)m->key;
            n = m->key_len;
        }
        e->bytes = (unsigned char*)malloc(n ? n : 1);
        if (!e->bytes) { errf(err, err_len, "out of memory", NULL); goto done; }
        memcpy(e->bytes, src, n);
        e->len = (uint32_t)n;
    }

    t->map_cap = pow2_at_least(t->vocab_size * 2);
    t->map_ids = (int32_t*)malloc(t->map_cap * sizeof(int32_t));
    if (!t->map_ids) { errf(err, err_len, "out of memory", NULL); goto done; }
    for (size_t i = 0; i < t->map_cap; i++) t->map_ids[i] = -1;
    for (size_t id = 0; id < t->vocab_size; id++) {
        if (t->tokens[id].bytes && t->tokens[id].len > 0) map_insert(t, (int32_t)id);
    }
    for (int b = 0; b < 256; b++) {
        unsigned char one = (unsigned char)b;
        t->byte_token[b] = map_find(t, &one, 1);
    }

    /* Merges, ranked by their order in the list. */
    size_t n_merges = merges->u.array.count;
    t->merge_cap = pow2_at_least(n_merges * 2 + 1);
    t->merge_keys = (uint64_t*)malloc(t->merge_cap * sizeof(uint64_t));
    t->merge_ids = (int32_t*)malloc(t->merge_cap * sizeof(int32_t));
    t->merge_ranks = (uint32_t*)malloc(t->merge_cap * sizeof(uint32_t));
    t->merge_used = (unsigned char*)calloc(t->merge_cap, 1);
    if (!t->merge_keys || !t->merge_ids || !t->merge_ranks || !t->merge_used) { errf(err, err_len, "out of memory", NULL); goto done; }
    for (size_t r = 0; r < n_merges; r++) {
        const vitna_json_value_t* m = merges->u.array.items[r];
        const char *a, *b;
        size_t alen, blen;
        if (m->type == VITNA_JSON_STRING) {
            const char* sp = memchr(m->u.string.ptr, ' ', m->u.string.len);
            if (!sp) { errf(err, err_len, "a merge is not two tokens", m->u.string.ptr); goto done; }
            a = m->u.string.ptr;
            alen = (size_t)(sp - a);
            b = sp + 1;
            blen = m->u.string.len - alen - 1;
        } else if (m->type == VITNA_JSON_ARRAY && m->u.array.count == 2 &&
                   m->u.array.items[0]->type == VITNA_JSON_STRING && m->u.array.items[1]->type == VITNA_JSON_STRING) {
            a = m->u.array.items[0]->u.string.ptr;
            alen = m->u.array.items[0]->u.string.len;
            b = m->u.array.items[1]->u.string.ptr;
            blen = m->u.array.items[1]->u.string.len;
        } else {
            errf(err, err_len, "a merge is neither \"a b\" nor [a, b]", NULL);
            goto done;
        }
        unsigned char* buf = (unsigned char*)malloc(alen + blen + 1);
        size_t na = 0, nb = 0;
        if (!buf || !byte_level_decode(a, alen, cp_to_byte, buf, &na) || !byte_level_decode(b, blen, cp_to_byte, buf + na, &nb)) {
            free(buf);
            errf(err, err_len, "a merge uses characters outside the byte-level alphabet", NULL);
            goto done;
        }
        int32_t ia = map_find(t, buf, na);
        int32_t ib = map_find(t, buf + na, nb);
        int32_t iab = map_find(t, buf, na + nb);
        free(buf);
        if (ia < 0 || ib < 0 || iab < 0) {
            errf(err, err_len, "a merge names a token the vocabulary lacks", m->type == VITNA_JSON_STRING ? m->u.string.ptr : NULL);
            goto done;
        }
        merge_insert(t, ia, ib, iab, (uint32_t)r);
    }
    ok = true;

done:
    free(scratch);
    vitna_json_free(doc);
    if (!ok) {
        vitna_tokenizer_free(t);
        return NULL;
    }
    return t;
}

void vitna_tokenizer_free(vitna_tokenizer_t* t) {
    if (!t) return;
    vitna_wordpiece_free(t->wordpiece);
    if (t->tokens) {
        for (size_t i = 0; i < t->vocab_size; i++) free(t->tokens[i].bytes);
        free(t->tokens);
    }
    free(t->is_special);
    free(t->map_ids);
    free(t->merge_keys);
    free(t->merge_ids);
    free(t->merge_ranks);
    free(t->merge_used);
    if (t->added) {
        for (size_t i = 0; i < t->n_added; i++) free(t->added[i].bytes);
        free(t->added);
    }
    free(t);
}

size_t vitna_tokenizer_vocab_size(const vitna_tokenizer_t* t) {
    if (t && t->wordpiece) return vitna_wordpiece_vocab_size(t->wordpiece);
    return t ? t->vocab_size : 0;
}

const unsigned char* vitna_tokenizer_token_bytes(const vitna_tokenizer_t* t, int32_t id, size_t* len) {
    if (t && t->wordpiece) return vitna_wordpiece_token_bytes(t->wordpiece, id, len);
    if (!t || id < 0 || (size_t)id >= t->vocab_size || !t->tokens[id].bytes) return NULL;
    *len = t->tokens[id].len;
    return t->tokens[id].bytes;
}

bool vitna_tokenizer_is_special(const vitna_tokenizer_t* t, int32_t id) {
    if (t && t->wordpiece) return vitna_wordpiece_is_special(t->wordpiece, id);
    return t && id >= 0 && (size_t)id < t->vocab_size && t->is_special[id];
}

bool vitna_tokenizer_is_wordpiece(const vitna_tokenizer_t* t) {
    return t && t->wordpiece;
}

/* --- BPE on one piece --- */

typedef struct {
    uint32_t rank;
    uint32_t pos;
    int32_t merged;
} cand_t;

static bool cand_less(const cand_t* a, const cand_t* b) {
    return a->rank < b->rank || (a->rank == b->rank && a->pos < b->pos);
}

static void heap_push(cand_t* heap, size_t* n, cand_t c) {
    size_t i = (*n)++;
    heap[i] = c;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!cand_less(&heap[i], &heap[parent])) break;
        cand_t tmp = heap[i]; heap[i] = heap[parent]; heap[parent] = tmp;
        i = parent;
    }
}

static cand_t heap_pop(cand_t* heap, size_t* n) {
    cand_t top = heap[0];
    heap[0] = heap[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && cand_less(&heap[l], &heap[m])) m = l;
        if (r < *n && cand_less(&heap[r], &heap[m])) m = r;
        if (m == i) break;
        cand_t tmp = heap[i]; heap[i] = heap[m]; heap[m] = tmp;
        i = m;
    }
    return top;
}

static bool bpe_piece(const vitna_tokenizer_t* t, const unsigned char* p, size_t n, vitna_token_list_t* out) {
    if (n == 0) return true;
    if (t->ignore_merges) {
        int32_t whole = map_find(t, p, n);
        if (whole >= 0) return vitna_token_list_push(out, whole);
    }
    int32_t* id = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* prev = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* next = (int32_t*)malloc(n * sizeof(int32_t));
    unsigned char* alive = (unsigned char*)malloc(n);
    cand_t* heap = (cand_t*)malloc(3 * n * sizeof(cand_t));
    bool ok = id && prev && next && alive && heap;
    if (ok) {
        /* A byte the vocabulary lacks is dropped, as the tokenizers library
         * does when a BPE model has no unknown token. */
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            int32_t b = t->byte_token[p[i]];
            if (b >= 0) id[count++] = b;
        }
        for (size_t i = 0; i < count; i++) {
            prev[i] = (int32_t)i - 1;
            next[i] = (i + 1 < count) ? (int32_t)i + 1 : -1;
            alive[i] = 1;
        }
        size_t hn = 0;
        int32_t merged;
        uint32_t rank;
        for (size_t i = 0; i + 1 < count; i++) {
            if (merge_find(t, id[i], id[i + 1], &merged, &rank)) heap_push(heap, &hn, (cand_t){ rank, (uint32_t)i, merged });
        }
        while (hn > 0) {
            cand_t c = heap_pop(heap, &hn);
            int32_t i = (int32_t)c.pos;
            if (!alive[i] || next[i] < 0) continue;
            int32_t j = next[i];
            /* Skip a candidate a previous merge has made stale. */
            if (!merge_find(t, id[i], id[j], &merged, &rank) || merged != c.merged) continue;
            id[i] = merged;
            alive[j] = 0;
            next[i] = next[j];
            if (next[j] >= 0) prev[next[j]] = i;
            if (prev[i] >= 0 && merge_find(t, id[prev[i]], id[i], &merged, &rank)) {
                heap_push(heap, &hn, (cand_t){ rank, (uint32_t)prev[i], merged });
            }
            if (next[i] >= 0 && merge_find(t, id[i], id[next[i]], &merged, &rank)) {
                heap_push(heap, &hn, (cand_t){ rank, (uint32_t)i, merged });
            }
        }
        for (int32_t i = count > 0 ? 0 : -1; i >= 0 && ok; i = next[i]) {
            ok = vitna_token_list_push(out, id[i]);
        }
    }
    free(id);
    free(prev);
    free(next);
    free(alive);
    free(heap);
    return ok;
}

/* --- The GPT-2 split pattern on one piece --- */

typedef enum { CLS_LETTER, CLS_NUMBER, CLS_SPACE, CLS_OTHER } cls_t;

static cls_t classify(uint32_t cp) {
    if (vitna_uni_is_space(cp)) return CLS_SPACE;
    if (vitna_uni_is_letter(cp)) return CLS_LETTER;
    if (vitna_uni_is_number(cp)) return CLS_NUMBER;
    return CLS_OTHER;
}

/* The end of the run of class k starting at i. */
static size_t run_end(const unsigned char* s, size_t i, size_t e, cls_t k) {
    while (i < e) {
        uint32_t cp;
        size_t l = vitna_utf8_decode(s + i, e - i, &cp);
        if (classify(cp) != k) break;
        i += l;
    }
    return i;
}

static bool split_gpt2(const vitna_tokenizer_t* t, const unsigned char* s, size_t e, vitna_token_list_t* out) {
    size_t i = 0;
    while (i < e) {
        uint32_t cp;
        size_t l = vitna_utf8_decode(s + i, e - i, &cp);
        size_t end = 0;

        /* 's|'t|'re|'ve|'m|'ll|'d, lower case only */
        if (cp == '\'' && i + 1 < e) {
            unsigned char c1 = s[i + 1];
            unsigned char c2 = i + 2 < e ? s[i + 2] : 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') end = i + 2;
            else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) end = i + 3;
        }

        /* ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: an optional single
         * space, then a run of one class */
        if (!end) {
            size_t start = i;
            cls_t k = classify(cp);
            if (cp == ' ' && i + 1 < e) {
                uint32_t cp2;
                vitna_utf8_decode(s + i + 1, e - i - 1, &cp2);
                cls_t k2 = classify(cp2);
                if (k2 != CLS_SPACE) {
                    start = i + 1;
                    k = k2;
                }
            }
            if (k != CLS_SPACE) end = run_end(s, start, e, k);
        }

        /* `\s+(?!\S)` then `\s+`: a run of whitespace, less its last
         * character when a non-space follows and the run is longer than one */
        if (!end) {
            size_t r = i, last = i;
            while (r < e) {
                uint32_t c;
                size_t cl = vitna_utf8_decode(s + r, e - r, &c);
                if (!vitna_uni_is_space(c)) break;
                last = r;
                r += cl;
            }
            end = (r < e && last > i) ? last : r;
        }

        if (end <= i) end = i + l; /* not reached; a guard against looping */
        if (!bpe_piece(t, s + i, end - i, out)) return false;
        i = end;
    }
    return true;
}

/* A piece after the Digits step: the ByteLevel pre-tokenizer adds a leading
 * space when asked to, then splits. */
static bool encode_piece(const vitna_tokenizer_t* t, const unsigned char* s, size_t n, vitna_token_list_t* out) {
    if (n == 0) return true;
    if (t->add_prefix_space && s[0] != ' ') {
        unsigned char* buf = (unsigned char*)malloc(n + 1);
        if (!buf) return false;
        buf[0] = ' ';
        memcpy(buf + 1, s, n);
        bool ok = split_gpt2(t, buf, n + 1, out);
        free(buf);
        return ok;
    }
    return split_gpt2(t, s, n, out);
}

/* Text between added tokens: the Digits step, then each piece. */
static bool encode_segment(const vitna_tokenizer_t* t, const unsigned char* s, size_t n, vitna_token_list_t* out) {
    if (!t->digits) return encode_piece(t, s, n, out);
    size_t run = 0; /* start of the pending piece */
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        size_t l = vitna_utf8_decode(s + i, n - i, &cp);
        if (!vitna_uni_is_number(cp)) {
            i += l;
            continue;
        }
        if (!encode_piece(t, s + run, i - run, out)) return false;
        size_t end = i + l;
        if (t->digits == 2) {
            while (end < n) {
                uint32_t c;
                size_t cl = vitna_utf8_decode(s + end, n - end, &c);
                if (!vitna_uni_is_number(c)) break;
                end += cl;
            }
        }
        if (!encode_piece(t, s + i, end - i, out)) return false;
        i = run = end;
    }
    return encode_piece(t, s + run, n - run, out);
}

typedef bool (*segment_fn)(const vitna_tokenizer_t* t, const unsigned char* s, size_t n, vitna_token_list_t* out);

/* Match the added tokens that are normalized, or those that are not, in
 * s[0..len), leftmost and longest, and hand the text between them to each. */
static bool split_added(const vitna_tokenizer_t* t, const unsigned char* s, size_t len, bool normalized, segment_fn each,
                        vitna_token_list_t* out) {
    size_t seg = 0;
    size_t i = 0;
    while (i < len) {
        /* The longest added token starting here, if any. */
        const added_token_t* best = NULL;
        for (size_t k = 0; k < t->n_added; k++) {
            const added_token_t* a = &t->added[k];
            if (a->normalized == normalized && a->len > 0 && a->len <= len - i && a->bytes[0] == s[i] && memcmp(a->bytes, s + i, a->len) == 0 &&
                (!best || a->len > best->len)) {
                best = a;
            }
        }
        if (!best) {
            i++;
            continue;
        }
        if (!each(t, s + seg, i - seg, out)) return false;
        if (!vitna_token_list_push(out, best->id)) return false;
        i += best->len;
        seg = i;
    }
    return each(t, s + seg, len - seg, out);
}

/* Text between the added tokens matched as written: normalized, then split
 * on the added tokens matched in normalized text, then encoded. */
static bool encode_normalized(const vitna_tokenizer_t* t, const unsigned char* s, size_t n, vitna_token_list_t* out) {
    if (!t->nfc) return split_added(t, s, n, true, encode_segment, out);
    size_t m = 0;
    unsigned char* norm = vitna_uni_nfc(s, n, &m);
    if (!norm) return false;
    bool ok = split_added(t, norm, m, true, encode_segment, out);
    free(norm);
    return ok;
}

/* As the tokenizers library extracts added tokens (added_vocabulary.rs,
 * extract_and_normalize): those not normalized from the text as written;
 * then, in each piece between them, normalized, the normalized ones. */
bool vitna_tokenizer_encode(const vitna_tokenizer_t* t, const char* text, size_t len, vitna_token_list_t* out) {
    if (t->wordpiece) return vitna_wordpiece_encode(t->wordpiece, text, len, out);
    return split_added(t, (const unsigned char*)text, len, false, encode_normalized, out);
}

unsigned char* vitna_tokenizer_normalize(const vitna_tokenizer_t* t, const char* text, size_t len, size_t* out_len) {
    if (t->wordpiece) return vitna_wordpiece_normalize(t->wordpiece, text, len, out_len);
    if (t->nfc) return vitna_uni_nfc((const unsigned char*)text, len, out_len);
    unsigned char* copy = (unsigned char*)malloc(len ? len : 1);
    if (copy) memcpy(copy, text, len);
    *out_len = len;
    return copy;
}
