/**
 * wordpiece.c - A BERT tokenizer, read from a Hugging Face tokenizer.json.
 *
 * See wordpiece.h for the steps. Each follows the tokenizers library
 * (added_vocabulary.rs, normalizers/bert.rs, pre_tokenizers/bert.rs,
 * models/wordpiece/mod.rs, processors/template.rs and bert.rs).
 * tests/reference-embed.test.mjs compares the result with that library's ids
 * for the reference inputs and corpus, and engine/tools/check_wordpiece.py
 * with the library itself, over every code point and random strings.
 */

#include "wordpiece.h"
#include "unicode.h"
#include "wordpiece_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char* bytes;        /* the content, matched as written */
    size_t len;
    int32_t id;
} wp_added_t;

#define WP_EDGE_MAX 8            /* the most tokens the post-processor puts on either side */

struct vitna_wordpiece {
    size_t vocab_size;           /* one more than the largest id */
    unsigned char** text;        /* by id: the token, NUL-terminated; NULL where no token has this id */
    uint32_t* text_len;
    unsigned char* special;      /* by id */
    int32_t* map;                /* open addressing: token text -> id, -1 for an empty slot */
    size_t map_cap;              /* a power of two */
    int32_t unk;
    unsigned char prefix[16];    /* continuing_subword_prefix */
    size_t prefix_len;
    size_t max_word;             /* max_input_chars_per_word */
    bool clean_text, chinese, strip_accents, lowercase;
    wp_added_t* added;
    size_t n_added;
    int32_t before[WP_EDGE_MAX], after[WP_EDGE_MAX];
    size_t n_before, n_after;
};

/* --- Character data --- */

static bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

/* A byte that is not valid UTF-8 is held as 0x110000 + the byte, as
 * vitna_utf8_decode gives it: in no class, decomposing and lowercasing to
 * itself, so it stays part of whatever word it is in. */
static bool is_other(uint32_t cp) {
    return cp < 0x110000u && in_ranges(VITNA_WP_OTHER, VITNA_WP_OTHER_COUNT, cp);
}

static bool is_mark(uint32_t cp) {
    return cp >= 0x300 && cp < 0x110000u && in_ranges(VITNA_WP_MARK, VITNA_WP_MARK_COUNT, cp);
}

/* char::is_ascii_punctuation, or P* (pre_tokenizers/bert.rs, is_bert_punc). */
static bool is_punct(uint32_t cp) {
    if (cp < 0x80) return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
    return cp < 0x110000u && in_ranges(VITNA_WP_PUNCT, VITNA_WP_PUNCT_COUNT, cp);
}

/* normalizers/bert.rs, is_chinese_char. The ranges are the library's, which
 * start Extension E at U+2B920 where the Python BERT tokenizer has U+2B820. */
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

static uint8_t combining_class(uint32_t cp) {
    if (cp < 0x300 || cp >= 0x110000u) return 0;
    size_t lo = 0, hi = VITNA_WP_CCC_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < VITNA_WP_CCC[mid][0]) hi = mid;
        else if (cp > VITNA_WP_CCC[mid][1]) lo = mid + 1;
        else return (uint8_t)VITNA_WP_CCC[mid][2];
    }
    return 0;
}

static const uint32_t* find_row3(const uint32_t (*r)[3], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][0]) lo = mid + 1;
        else return r[mid];
    }
    return NULL;
}

/* --- Code point buffers --- */

typedef struct {
    uint32_t* cp;
    size_t n, cap;
} cps_t;

static bool cps_push(cps_t* b, uint32_t cp) {
    if (b->n == b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 64;
        uint32_t* grown = (uint32_t*)realloc(b->cp, cap * sizeof(uint32_t));
        if (!grown) return false;
        b->cp = grown;
        b->cap = cap;
    }
    b->cp[b->n++] = cp;
    return true;
}

/* Hangul syllables decompose by formula (Unicode, section 3.12). */
#define S_BASE 0xAC00u
#define L_BASE 0x1100u
#define V_BASE 0x1161u
#define T_BASE 0x11A7u
#define V_COUNT 21u
#define T_COUNT 28u
#define S_COUNT (19u * 21u * 28u)

/* Append cp's full canonical decomposition. */
static bool decompose(cps_t* b, uint32_t cp) {
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        const uint32_t s = cp - S_BASE;
        if (!cps_push(b, L_BASE + s / (V_COUNT * T_COUNT)) || !cps_push(b, V_BASE + (s % (V_COUNT * T_COUNT)) / T_COUNT)) return false;
        return s % T_COUNT == 0 || cps_push(b, T_BASE + s % T_COUNT);
    }
    const uint32_t* d = cp < 0x110000u ? find_row3(VITNA_WP_DECOMP, VITNA_WP_DECOMP_COUNT, cp) : NULL;
    if (!d) return cps_push(b, cp);
    return decompose(b, d[1]) && (d[2] == 0 || decompose(b, d[2]));
}

static size_t utf8_encode(uint32_t cp, unsigned char* out) {
    if (cp >= 0x110000u) { out[0] = (unsigned char)(cp - 0x110000u); return 1; }
    if (cp < 0x80) { out[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (unsigned char)(0xC0 | (cp >> 6));
        out[1] = (unsigned char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (unsigned char)(0xE0 | (cp >> 12));
        out[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (unsigned char)(0xF0 | (cp >> 18));
    out[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

/* --- The normalizer --- */

/* BertNormalizer (normalizers/bert.rs): clean_text, then handle_chinese_chars,
 * then strip_accents (NFD, then the nonspacing marks dropped), then
 * lowercase, each over the whole text in turn. */
static bool normalize_cps(const vitna_wordpiece_t* t, const unsigned char* s, size_t len, cps_t* out) {
    cps_t a = {0};
    bool ok = true;
    for (size_t i = 0; i < len && ok;) {
        uint32_t cp;
        i += vitna_utf8_decode(s + i, len - i, &cp);
        if (t->clean_text) {
            const bool keep_ws = cp == '\t' || cp == '\n' || cp == '\r';
            /* Dropping comes first, so a control character that is also
             * whitespace (vertical tab, form feed, next line) is dropped, not
             * made a space. */
            if (cp == 0 || cp == 0xFFFD || (!keep_ws && is_other(cp))) continue;
            if (keep_ws || vitna_uni_is_space(cp)) cp = ' ';
        }
        if (t->chinese && is_cjk(cp)) ok = cps_push(&a, ' ') && cps_push(&a, cp) && cps_push(&a, ' ');
        else ok = cps_push(&a, cp);
    }
    if (ok && t->strip_accents) {
        cps_t d = {0};
        for (size_t i = 0; i < a.n && ok; i++) ok = decompose(&d, a.cp[i]);
        if (ok) {
            /* Canonical ordering: each mark moves before the marks of a higher
             * class ahead of it, and never past a starter. */
            for (size_t i = 1; i < d.n; i++) {
                const uint8_t k = combining_class(d.cp[i]);
                if (k == 0) continue;
                for (size_t j = i; j > 0 && combining_class(d.cp[j - 1]) > k; j--) {
                    const uint32_t c = d.cp[j - 1];
                    d.cp[j - 1] = d.cp[j];
                    d.cp[j] = c;
                }
            }
            size_t n = 0;
            for (size_t i = 0; i < d.n; i++) {
                if (!is_mark(d.cp[i])) d.cp[n++] = d.cp[i];
            }
            d.n = n;
        }
        free(a.cp);
        a = d;
    }
    if (ok && t->lowercase) {
        for (size_t i = 0; i < a.n && ok; i++) {
            const uint32_t cp = a.cp[i];
            const uint32_t* low = cp < 0x110000u ? find_row3(VITNA_WP_LOWER, VITNA_WP_LOWER_COUNT, cp) : NULL;
            if (!low) ok = cps_push(out, cp);
            else ok = cps_push(out, low[1]) && (low[2] == 0 || cps_push(out, low[2]));
        }
        free(a.cp);
    } else if (ok) {
        free(out->cp);
        *out = a;
    } else {
        free(a.cp);
    }
    return ok;
}

unsigned char* vitna_wordpiece_normalize(const vitna_wordpiece_t* t, const char* text, size_t len, size_t* out_len) {
    cps_t n = {0};
    if (!normalize_cps(t, (const unsigned char*)text, len, &n)) {
        free(n.cp);
        return NULL;
    }
    unsigned char* out = (unsigned char*)malloc(n.n * 4 + 1);
    if (out) {
        size_t o = 0;
        for (size_t i = 0; i < n.n; i++) o += utf8_encode(n.cp[i], out + o);
        *out_len = o;
    }
    free(n.cp);
    return out;
}

/* --- The vocabulary --- */

static uint64_t fnv1a(const unsigned char* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int32_t map_find(const vitna_wordpiece_t* t, const unsigned char* p, size_t n) {
    for (size_t i = (size_t)fnv1a(p, n) & (t->map_cap - 1);; i = (i + 1) & (t->map_cap - 1)) {
        const int32_t id = t->map[i];
        if (id < 0) return -1;
        if (t->text_len[id] == n && memcmp(t->text[id], p, n) == 0) return id;
    }
}

/* --- Words --- */

/* The WordPiece model (models/wordpiece/mod.rs): the longest entries that
 * spell the word, left to right, each after the first with the prefix; the
 * unknown token alone when any part cannot be spelled, or when the word has
 * more than max_word characters. */
static bool encode_word(const vitna_wordpiece_t* t, const uint32_t* cps, size_t n, vitna_token_list_t* out) {
    if (n > t->max_word) return vitna_token_list_push(out, t->unk);
    size_t* off = (size_t*)malloc((n + 1) * sizeof(size_t));
    unsigned char* buf = (unsigned char*)malloc(t->prefix_len + n * 4 + 1);
    if (!off || !buf) {
        free(off);
        free(buf);
        return false;
    }
    /* The word's bytes after the prefix's place, with each character's offset. */
    unsigned char* word = buf + t->prefix_len;
    off[0] = 0;
    for (size_t i = 0; i < n; i++) off[i + 1] = off[i] + utf8_encode(cps[i], word + off[i]);
    memcpy(buf, t->prefix, t->prefix_len);

    const size_t mark = out->count;
    bool ok = true;
    for (size_t start = 0; start < n && ok;) {
        int32_t id = -1;
        size_t end = n;
        for (; end > start; end--) {
            /* After the first piece the candidate is the prefix and the piece,
             * which buf holds contiguously when the piece starts at word + off[start]:
             * the prefix is copied in front of it for the lookup. */
            if (start == 0) {
                id = map_find(t, word, off[end]);
            } else {
                unsigned char* cand = word + off[start] - t->prefix_len;
                unsigned char saved[16];
                memcpy(saved, cand, t->prefix_len);
                memcpy(cand, t->prefix, t->prefix_len);
                id = map_find(t, cand, t->prefix_len + off[end] - off[start]);
                memcpy(cand, saved, t->prefix_len);
            }
            if (id >= 0) break;
        }
        if (id < 0) {
            out->count = mark;
            ok = vitna_token_list_push(out, t->unk);
            break;
        }
        ok = vitna_token_list_push(out, id);
        start = end;
    }
    free(off);
    free(buf);
    return ok;
}

/* BertPreTokenizer (pre_tokenizers/bert.rs): split at whitespace, which is
 * removed, then at punctuation, each character of which is a word. */
static bool encode_segment(const vitna_wordpiece_t* t, const unsigned char* s, size_t len, vitna_token_list_t* out) {
    if (len == 0) return true;
    cps_t n = {0};
    bool ok = normalize_cps(t, s, len, &n);
    size_t start = 0;
    for (size_t i = 0; i <= n.n && ok; i++) {
        const bool end = i == n.n;
        const bool space = !end && vitna_uni_is_space(n.cp[i]);
        const bool punct = !end && !space && is_punct(n.cp[i]);
        if (!end && !space && !punct) continue;
        if (i > start) ok = encode_word(t, n.cp + start, i - start, out);
        if (ok && punct) ok = encode_word(t, n.cp + i, 1, out);
        start = i + 1;
    }
    free(n.cp);
    return ok;
}

/* Added tokens are matched as written, leftmost and longest
 * (added_vocabulary.rs), and the text between them is encoded. */
static bool encode_text(const vitna_wordpiece_t* t, const unsigned char* s, size_t len, vitna_token_list_t* out) {
    size_t seg = 0, i = 0;
    while (i < len) {
        const wp_added_t* best = NULL;
        for (size_t k = 0; k < t->n_added; k++) {
            const wp_added_t* a = &t->added[k];
            if (a->len > 0 && a->len <= len - i && a->bytes[0] == s[i] && memcmp(a->bytes, s + i, a->len) == 0 && (!best || a->len > best->len)) best = a;
        }
        if (!best) {
            i++;
            continue;
        }
        if (!encode_segment(t, s + seg, i - seg, out) || !vitna_token_list_push(out, best->id)) return false;
        i += best->len;
        seg = i;
    }
    return encode_segment(t, s + seg, len - seg, out);
}

bool vitna_wordpiece_encode(const vitna_wordpiece_t* t, const char* text, size_t len, vitna_token_list_t* out) {
    for (size_t i = 0; i < t->n_before; i++) {
        if (!vitna_token_list_push(out, t->before[i])) return false;
    }
    if (!encode_text(t, (const unsigned char*)text, len, out)) return false;
    for (size_t i = 0; i < t->n_after; i++) {
        if (!vitna_token_list_push(out, t->after[i])) return false;
    }
    return true;
}

/* --- Loading --- */

static bool errf(char* err, size_t err_len, const char* msg, const char* detail) {
    snprintf(err, err_len, "%s%s%s", msg, detail ? ": " : "", detail ? detail : "");
    return false;
}

static const char* type_of(const vitna_json_value_t* v) {
    return v && v->type == VITNA_JSON_OBJECT ? vitna_json_as_string(vitna_json_get(v, "type")) : NULL;
}

static bool absent(const vitna_json_value_t* v) {
    return v == NULL || vitna_json_is_null(v);
}

/* A flag that must be a boolean, or absent for its default. */
static bool flag(const vitna_json_value_t* obj, const char* key, bool dflt, bool* out) {
    const vitna_json_value_t* v = vitna_json_get(obj, key);
    if (absent(v)) {
        *out = dflt;
        return true;
    }
    return vitna_json_as_bool(v, out);
}

static bool small_id(const vitna_json_value_t* v, int32_t* out) {
    double d;
    if (!vitna_json_as_number(v, &d) || d < 0 || d > 16777216.0 || d != (double)(int32_t)d) return false;
    *out = (int32_t)d;
    return true;
}

bool vitna_wordpiece_wanted(const vitna_json_value_t* root) {
    const char* m = type_of(vitna_json_get(root, "model"));
    return m && strcmp(m, "WordPiece") == 0;
}

/* The ids a TemplateProcessing special token stands for, appended to dst. */
static bool template_ids(const vitna_json_value_t* specials, const char* name, int32_t* dst, size_t* n, char* err, size_t err_len) {
    const vitna_json_value_t* tok = specials ? vitna_json_get(specials, name) : NULL;
    const vitna_json_value_t* ids = tok ? vitna_json_get(tok, "ids") : NULL;
    if (!ids || ids->type != VITNA_JSON_ARRAY) return errf(err, err_len, "the post_processor names a special token it does not define", name);
    for (size_t i = 0; i < ids->u.array.count; i++) {
        int32_t id;
        if (*n == WP_EDGE_MAX || !small_id(ids->u.array.items[i], &id)) return errf(err, err_len, "the post_processor's special token ids are not supported", name);
        dst[(*n)++] = id;
    }
    return true;
}

static bool configure_post(vitna_wordpiece_t* t, const vitna_json_value_t* post, char* err, size_t err_len) {
    if (absent(post)) return true;
    const char* type = type_of(post);
    if (type && strcmp(type, "BertProcessing") == 0) {
        const vitna_json_value_t* cls = vitna_json_get(post, "cls");
        const vitna_json_value_t* sep = vitna_json_get(post, "sep");
        if (!cls || cls->type != VITNA_JSON_ARRAY || cls->u.array.count != 2 || !small_id(cls->u.array.items[1], &t->before[0]) || !sep ||
            sep->type != VITNA_JSON_ARRAY || sep->u.array.count != 2 || !small_id(sep->u.array.items[1], &t->after[0])) {
            return errf(err, err_len, "BertProcessing without a [CLS] and [SEP] id is not supported", NULL);
        }
        t->n_before = t->n_after = 1;
        return true;
    }
    if (!type || strcmp(type, "TemplateProcessing") != 0) return errf(err, err_len, "post_processors other than TemplateProcessing and BertProcessing are not supported", type);
    const vitna_json_value_t* single = vitna_json_get(post, "single");
    const vitna_json_value_t* specials = vitna_json_get(post, "special_tokens");
    if (!single || single->type != VITNA_JSON_ARRAY) return errf(err, err_len, "a TemplateProcessing without a single template is not supported", NULL);
    bool seen_a = false;
    for (size_t i = 0; i < single->u.array.count; i++) {
        const vitna_json_value_t* item = single->u.array.items[i];
        const vitna_json_value_t* seq = vitna_json_get(item, "Sequence");
        const vitna_json_value_t* sp = vitna_json_get(item, "SpecialToken");
        double type_id = 0;
        const vitna_json_value_t* part = seq ? seq : sp;
        if (!part || (vitna_json_get(part, "type_id") && (!vitna_json_as_number(vitna_json_get(part, "type_id"), &type_id) || type_id != 0))) {
            return errf(err, err_len, "a single template other than special tokens and sequence A, all of type 0, is not supported", NULL);
        }
        if (seq) {
            const char* id = vitna_json_as_string(vitna_json_get(seq, "id"));
            if (seen_a || !id || strcmp(id, "A") != 0) return errf(err, err_len, "a single template must hold sequence A once", NULL);
            seen_a = true;
            continue;
        }
        const char* name = vitna_json_as_string(vitna_json_get(sp, "id"));
        if (!name) return errf(err, err_len, "a special token in the template has no id", NULL);
        if (!template_ids(specials, name, seen_a ? t->after : t->before, seen_a ? &t->n_after : &t->n_before, err, err_len)) return false;
    }
    return seen_a || errf(err, err_len, "a single template must hold sequence A once", NULL);
}

vitna_wordpiece_t* vitna_wordpiece_from_json(const vitna_json_value_t* root, char* err, size_t err_len) {
    vitna_wordpiece_t* t = (vitna_wordpiece_t*)calloc(1, sizeof(vitna_wordpiece_t));
    if (!t) {
        errf(err, err_len, "out of memory", NULL);
        return NULL;
    }
    bool ok = false;
    const vitna_json_value_t* norm = vitna_json_get(root, "normalizer");
    const vitna_json_value_t* pre = vitna_json_get(root, "pre_tokenizer");
    const vitna_json_value_t* model = vitna_json_get(root, "model");
    const vitna_json_value_t* added = vitna_json_get(root, "added_tokens");

    if (!absent(norm)) {
        const char* type = type_of(norm);
        if (!type || strcmp(type, "BertNormalizer") != 0) { errf(err, err_len, "normalizers other than BertNormalizer are not supported with WordPiece", type); goto done; }
        bool strip_given = !absent(vitna_json_get(norm, "strip_accents"));
        if (!flag(norm, "clean_text", true, &t->clean_text) || !flag(norm, "handle_chinese_chars", true, &t->chinese) ||
            !flag(norm, "lowercase", true, &t->lowercase) || !flag(norm, "strip_accents", false, &t->strip_accents)) {
            errf(err, err_len, "a BertNormalizer setting is not a boolean", NULL);
            goto done;
        }
        /* strip_accents null follows lowercase, as in normalizers/bert.rs. */
        if (!strip_given) t->strip_accents = t->lowercase;
    }
    if (!type_of(pre) || strcmp(type_of(pre), "BertPreTokenizer") != 0) { errf(err, err_len, "pre_tokenizers other than BertPreTokenizer are not supported with WordPiece", type_of(pre)); goto done; }
    if (!absent(vitna_json_get(root, "truncation")) || !absent(vitna_json_get(root, "padding"))) {
        errf(err, err_len, "truncation and padding in tokenizer.json are not supported: a caller checks lengths itself", NULL);
        goto done;
    }
    if (!configure_post(t, vitna_json_get(root, "post_processor"), err, err_len)) goto done;

    const vitna_json_value_t* vocab = vitna_json_get(model, "vocab");
    const char* unk = vitna_json_as_string(vitna_json_get(model, "unk_token"));
    const char* prefix = vitna_json_as_string(vitna_json_get(model, "continuing_subword_prefix"));
    double max_word = 0;
    if (!vocab || vocab->type != VITNA_JSON_OBJECT || !unk || !prefix || strlen(prefix) >= sizeof(t->prefix) ||
        !vitna_json_as_number(vitna_json_get(model, "max_input_chars_per_word"), &max_word) || max_word < 1 || max_word > 1e6) {
        errf(err, err_len, "a WordPiece model needs a vocab, an unk_token, a continuing_subword_prefix and max_input_chars_per_word", NULL);
        goto done;
    }
    t->prefix_len = strlen(prefix);
    memcpy(t->prefix, prefix, t->prefix_len);
    t->max_word = (size_t)max_word;

    /* The vocabulary size is one more than the largest id anywhere. */
    size_t max_id = 0;
    for (size_t i = 0; i < vocab->u.object.count; i++) {
        int32_t id;
        if (!small_id(vocab->u.object.members[i].value, &id)) { errf(err, err_len, "a vocab id is not a small non-negative integer", vocab->u.object.members[i].key); goto done; }
        if ((size_t)id > max_id) max_id = (size_t)id;
    }
    const size_t n_added_json = added && added->type == VITNA_JSON_ARRAY ? added->u.array.count : 0;
    for (size_t i = 0; i < n_added_json; i++) {
        int32_t id;
        if (!small_id(vitna_json_get(added->u.array.items[i], "id"), &id)) { errf(err, err_len, "an added token's id is not a small non-negative integer", NULL); goto done; }
        if ((size_t)id > max_id) max_id = (size_t)id;
    }
    t->vocab_size = max_id + 1;
    t->text = (unsigned char**)calloc(t->vocab_size, sizeof(unsigned char*));
    t->text_len = (uint32_t*)calloc(t->vocab_size, sizeof(uint32_t));
    t->special = (unsigned char*)calloc(t->vocab_size, 1);
    t->map_cap = 64;
    while (t->map_cap < (vocab->u.object.count + n_added_json) * 2) t->map_cap *= 2;
    t->map = (int32_t*)malloc(t->map_cap * sizeof(int32_t));
    t->added = (wp_added_t*)calloc(n_added_json ? n_added_json : 1, sizeof(wp_added_t));
    if (!t->text || !t->text_len || !t->special || !t->map || !t->added) { errf(err, err_len, "out of memory", NULL); goto done; }
    for (size_t i = 0; i < t->map_cap; i++) t->map[i] = -1;

    for (size_t i = 0; i < vocab->u.object.count; i++) {
        const vitna_json_member_t* m = &vocab->u.object.members[i];
        int32_t id;
        small_id(m->value, &id);
        if (t->text[id]) { errf(err, err_len, "two vocab entries share an id", m->key); goto done; }
        t->text[id] = (unsigned char*)malloc(m->key_len + 1);
        if (!t->text[id]) { errf(err, err_len, "out of memory", NULL); goto done; }
        memcpy(t->text[id], m->key, m->key_len + 1);
        t->text_len[id] = (uint32_t)m->key_len;
        if (map_find(t, t->text[id], m->key_len) >= 0) { errf(err, err_len, "two vocab entries share a text", m->key); goto done; }
        size_t slot = (size_t)fnv1a(t->text[id], m->key_len) & (t->map_cap - 1);
        while (t->map[slot] >= 0) slot = (slot + 1) & (t->map_cap - 1);
        t->map[slot] = id;
    }
    t->unk = map_find(t, (const unsigned char*)unk, strlen(unk));
    if (t->unk < 0) { errf(err, err_len, "the unk_token is not in the vocab", unk); goto done; }

    for (size_t i = 0; i < n_added_json; i++) {
        const vitna_json_value_t* a = added->u.array.items[i];
        const vitna_json_value_t* content = vitna_json_get(a, "content");
        bool single_word = false, lstrip = false, rstrip = false, normalized = false, special = false;
        int32_t id;
        small_id(vitna_json_get(a, "id"), &id);
        if (!content || content->type != VITNA_JSON_STRING || content->u.string.len == 0 || !flag(a, "single_word", false, &single_word) ||
            !flag(a, "lstrip", false, &lstrip) || !flag(a, "rstrip", false, &rstrip) || !flag(a, "normalized", false, &normalized) ||
            !flag(a, "special", false, &special) || single_word || lstrip || rstrip || normalized) {
            errf(err, err_len, "an added token other than one matched as written, without single_word, lstrip or rstrip, is not supported",
                 content && content->type == VITNA_JSON_STRING ? content->u.string.ptr : NULL);
            goto done;
        }
        wp_added_t* w = &t->added[t->n_added++];
        w->bytes = (unsigned char*)malloc(content->u.string.len);
        if (!w->bytes) { errf(err, err_len, "out of memory", NULL); goto done; }
        memcpy(w->bytes, content->u.string.ptr, content->u.string.len);
        w->len = content->u.string.len;
        w->id = id;
        t->special[id] = special;
        if (!t->text[id]) {
            t->text[id] = (unsigned char*)malloc(w->len + 1);
            if (!t->text[id]) { errf(err, err_len, "out of memory", NULL); goto done; }
            memcpy(t->text[id], w->bytes, w->len);
            t->text[id][w->len] = '\0';
            t->text_len[id] = (uint32_t)w->len;
        }
    }
    for (size_t i = 0; i < t->n_before + t->n_after; i++) {
        const int32_t id = i < t->n_before ? t->before[i] : t->after[i - t->n_before];
        if ((size_t)id >= t->vocab_size) { errf(err, err_len, "the post_processor names an id outside the vocab", NULL); goto done; }
    }
    ok = true;

done:
    if (!ok) {
        vitna_wordpiece_free(t);
        return NULL;
    }
    return t;
}

void vitna_wordpiece_free(vitna_wordpiece_t* t) {
    if (!t) return;
    if (t->text) {
        for (size_t i = 0; i < t->vocab_size; i++) free(t->text[i]);
    }
    for (size_t i = 0; i < t->n_added; i++) free(t->added[i].bytes);
    free(t->added);
    free(t->text);
    free(t->text_len);
    free(t->special);
    free(t->map);
    free(t);
}

size_t vitna_wordpiece_vocab_size(const vitna_wordpiece_t* t) {
    return t->vocab_size;
}

const unsigned char* vitna_wordpiece_token_bytes(const vitna_wordpiece_t* t, int32_t id, size_t* len) {
    if (id < 0 || (size_t)id >= t->vocab_size || !t->text[id]) return NULL;
    *len = t->text_len[id];
    return t->text[id];
}

bool vitna_wordpiece_is_special(const vitna_wordpiece_t* t, int32_t id) {
    return id >= 0 && (size_t)id < t->vocab_size && t->special[id];
}
