/**
 * unit_tests.c - Tests for the engine's parts, each against an independent
 * statement of what it should compute: published test vectors, a formula
 * evaluated in double precision, or a small input worked by hand.
 *
 * The forward pass as a whole is tested against the pinned reference by
 * tests/reference.test.mjs. These tests cover the pieces, including those
 * the forward pass does not use yet.
 *
 * Exit status 0 when every check passes.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"
#include "crypto.h"
#include "expert_stream.h"
#include "gguf.h"
#include "json.h"
#include "jsonpfx.h"
#include "kernels.h"
#include "kv_cache.h"
#include "ops.h"
#include "quant.h"
#include "safetensors.h"
#include "sampler.h"
#include "strbuf.h"
#include "tokenizer.h"
#include "unicode.h"

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        g_checks++;                                                         \
        if (!(cond)) {                                                      \
            g_failures++;                                                   \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fputc('\n', stderr);                                            \
        }                                                                   \
    } while (0)

/* A deterministic generator for test data. */
static uint64_t g_rng = 0x2545F4914F6CDD1DULL;
static uint32_t rnd_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 16);
}
static float rnd_f(float lo, float hi) {
    return lo + (hi - lo) * (float)(rnd_u32() & 0xFFFFFF) / (float)0x1000000;
}

static bool write_file(const char* path, const void* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

/* --- JSON --- */

static void test_json(void) {
    const char* text = "\xEF\xBB\xBF {\"a\": [1, -2.5e2, true, false, null], \"s\": \"x\\u00e9\\ud83d\\ude80\\ud800q\\n\\\"\", \"o\": {}}";
    char err[160];
    vitna_json_doc_t* doc = vitna_json_parse(text, strlen(text), err, sizeof(err));
    CHECK(doc != NULL, "a valid document parses: %s", err);
    if (doc) {
        const vitna_json_value_t* root = vitna_json_root(doc);
        const vitna_json_value_t* a = vitna_json_get(root, "a");
        double d = 0;
        CHECK(a && a->type == VITNA_JSON_ARRAY && a->u.array.count == 5, "array of five");
        CHECK(a && vitna_json_as_number(a->u.array.items[1], &d) && d == -250.0, "-2.5e2 reads as -250");
        CHECK(a && a->u.array.items[2]->type == VITNA_JSON_TRUE && a->u.array.items[4]->type == VITNA_JSON_NULL, "literals");
        const char* s = vitna_json_as_string(vitna_json_get(root, "s"));
        /* x, e-acute, rocket (a surrogate pair), a lone surrogate as U+FFFD, q, newline, quote */
        const char want[] = "x\xC3\xA9\xF0\x9F\x9A\x80\xEF\xBF\xBDq\n\"";
        CHECK(s && strcmp(s, want) == 0, "escapes decode to UTF-8");
        CHECK(vitna_json_get(root, "o") && vitna_json_get(root, "o")->type == VITNA_JSON_OBJECT, "empty object");
        CHECK(vitna_json_get(root, "missing") == NULL, "a missing key is NULL");
        vitna_json_free(doc);
    }
    const char* bad[] = { "{\"a\":1,}", "[1,2", "\"abc", "{\"a\" 1}", "01", "[1] 2", "\"\x01\"", "tru" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        vitna_json_doc_t* d = vitna_json_parse(bad[i], strlen(bad[i]), err, sizeof(err));
        CHECK(d == NULL, "invalid JSON is refused: %s", bad[i]);
        vitna_json_free(d);
    }
    /* Nesting past the limit is refused rather than overflowing the stack. */
    size_t depth = VITNA_JSON_MAX_DEPTH + 10;
    char* deep = (char*)malloc(2 * depth + 1);
    for (size_t i = 0; i < depth; i++) { deep[i] = '['; deep[depth + i] = ']'; }
    deep[2 * depth] = '\0';
    vitna_json_doc_t* d = vitna_json_parse(deep, 2 * depth, err, sizeof(err));
    CHECK(d == NULL && strstr(err, "nested") != NULL, "deep nesting is refused");
    vitna_json_free(d);
    free(deep);
}

/* --- SafeTensors --- */

static size_t build_safetensors(unsigned char* buf, const char* header, const void* data, size_t data_len) {
    uint64_t n = strlen(header);
    for (int i = 0; i < 8; i++) buf[i] = (unsigned char)(n >> (8 * i));
    memcpy(buf + 8, header, n);
    memcpy(buf + 8 + n, data, data_len);
    return 8 + n + data_len;
}

static void test_safetensors(void) {
    const char* path = "vitna-unit-test.safetensors";
    unsigned char buf[1024];
    float data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    char err[256];
    vitna_safetensors_t st;

    size_t n = build_safetensors(buf,
        "{\"__metadata__\":{\"format\":\"pt\"},\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}", data, 16);
    CHECK(write_file(path, buf, n), "write a test file");
    bool ok = vitna_safetensors_open_ex(path, &st, err, sizeof(err));
    CHECK(ok, "a valid file opens: %s", err);
    if (ok) {
        CHECK(st.tensor_count == 1, "__metadata__ is not counted as a tensor (got %zu)", st.tensor_count);
        const vitna_tensor_desc_t* t = vitna_safetensors_find(&st, "w");
        CHECK(t && t->dtype == VITNA_DTYPE_F32 && t->ndim == 2 && vitna_tensor_numel(t) == 4, "the tensor's directory entry");
        CHECK(t && ((const float*)t->data_ptr)[3] == 4.0f, "the tensor's data, in place");
        CHECK(vitna_safetensors_find(&st, "__metadata__") == NULL, "no tensor named __metadata__");
        vitna_safetensors_close(&st);
    }

    struct { const char* header; size_t data_len; const char* why; } bad[] = {
        { "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,32]}}", 16, "outside the data section" },
        { "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,8]}}", 16, "does not match" },
        { "{\"w\":{\"dtype\":\"Q7\",\"shape\":[4],\"data_offsets\":[0,16]}}", 16, "unknown dtype" },
        { "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,0]}}", 16, "outside the data section" },
        { "{\"w\":{\"dtype\":\"F32\",\"shape\":[-4],\"data_offsets\":[0,16]}}", 16, "bad shape" },
        { "[1,2]", 0, "not a JSON object" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        n = build_safetensors(buf, bad[i].header, data, bad[i].data_len);
        write_file(path, buf, n);
        ok = vitna_safetensors_open_ex(path, &st, err, sizeof(err));
        CHECK(!ok && strstr(err, bad[i].why) != NULL, "refused (%s): got \"%s\"", bad[i].why, ok ? "opened" : err);
        if (ok) vitna_safetensors_close(&st);
    }
    /* A header length past the end of the file. */
    unsigned char short_file[12] = { 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, '{', '}', 0, 0 };
    write_file(path, short_file, sizeof(short_file));
    ok = vitna_safetensors_open_ex(path, &st, err, sizeof(err));
    CHECK(!ok, "a header length larger than the file is refused");
    if (ok) vitna_safetensors_close(&st);
    remove(path);
}

/* --- Unicode --- */

static void test_unicode(void) {
    CHECK(vitna_uni_is_letter('a') && vitna_uni_is_letter(0x00E9) && vitna_uni_is_letter(0x6771) && vitna_uni_is_letter(0x05E9), "letters");
    CHECK(!vitna_uni_is_letter('1') && !vitna_uni_is_letter(' ') && !vitna_uni_is_letter(0x1F680), "not letters");
    CHECK(vitna_uni_is_number('7') && vitna_uni_is_number(0x0663) && vitna_uni_is_number(0x216B) && vitna_uni_is_number(0x00BD), "numbers: Nd, Nl, No");
    CHECK(!vitna_uni_is_number('x'), "not a number");
    CHECK(vitna_uni_is_space(' ') && vitna_uni_is_space('\t') && vitna_uni_is_space(0x3000) && vitna_uni_is_space(0x85), "White_Space");
    CHECK(!vitna_uni_is_space(0x1D) && !vitna_uni_is_space(0x200B), "U+001D and U+200B are not White_Space");
    uint32_t cp;
    const unsigned char rocket[] = { 0xF0, 0x9F, 0x9A, 0x80 };
    CHECK(vitna_utf8_decode(rocket, 4, &cp) == 4 && cp == 0x1F680, "a four-byte sequence");
    const unsigned char overlong[] = { 0xC0, 0xAF };
    CHECK(vitna_utf8_decode(overlong, 2, &cp) == 1 && cp >= 0x110000, "an overlong sequence is one invalid byte");
    const unsigned char surrogate[] = { 0xED, 0xA0, 0x80 };
    CHECK(vitna_utf8_decode(surrogate, 3, &cp) == 1 && cp >= 0x110000, "an encoded surrogate is invalid");
}

static size_t put_utf8(uint32_t cp, unsigned char* out) {
    if (cp < 0x80) { out[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { out[0] = (unsigned char)(0xC0 | (cp >> 6)); out[1] = (unsigned char)(0x80 | (cp & 0x3F)); return 2; }
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

/* One case for each part of NFC, the expected forms from Python 3.12's
 * unicodedata.normalize, which has the Unicode version of the engine's
 * tables. engine/tools/check_nfc.py compares every code point, and a million
 * random strings of marks, composites and jamo, the same way. */
static void test_nfc(void) {
    static const struct {
        uint32_t in[4];
        uint32_t out[4];
        const char* what;
    } cases[] = {
        { { 0x65, 0x301 }, { 0xE9 }, "a mark composes with its starter" },
        { { 0x212B }, { 0xC5 }, "a singleton decomposes, and stays decomposed" },
        { { 0x1100, 0x1161, 0x11A8 }, { 0xAC01 }, "Hangul jamo compose into a syllable" },
        { { 0xAC00, 0x11A8 }, { 0xAC01 }, "a syllable takes a trailing consonant" },
        { { 0x958 }, { 0x915, 0x93C }, "a composition exclusion stays decomposed" },
        { { 0x344 }, { 0x308, 0x301 }, "a decomposition starting with a mark stays decomposed" },
        { { 0x65, 0x302, 0x323 }, { 0x1EC7 }, "marks reorder by class, then compose" },
        { { 0x3C9, 0x300, 0xE4B }, { 0x1F7C, 0xE4B }, "a mark composes past a mark of a lower class it moved behind" },
        { { 0x61, 0x300, 0x300 }, { 0xE0, 0x300 }, "a second mark of the same class is blocked" },
        { { 0xB47, 0xB3E }, { 0xB4B }, "two starters compose" },
        { { 0xFB01 }, { 0xFB01 }, "a compatibility ligature is left alone" },
        { { 0x301, 0x65 }, { 0x301, 0x65 }, "a leading mark has no starter to join" },
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        unsigned char in[16], want[16];
        size_t n_in = 0, n_want = 0;
        for (size_t i = 0; i < 4 && cases[c].in[i]; i++) n_in += put_utf8(cases[c].in[i], in + n_in);
        for (size_t i = 0; i < 4 && cases[c].out[i]; i++) n_want += put_utf8(cases[c].out[i], want + n_want);
        size_t n = 0;
        unsigned char* got = vitna_uni_nfc(in, n_in, &n);
        CHECK(got && n == n_want && memcmp(got, want, n) == 0, "NFC: %s", cases[c].what);
        free(got);
    }
    /* A byte that is not UTF-8 passes through as a starter that composes with nothing. */
    const unsigned char odd[] = { 'e', 0xFF, 0xCC, 0x81 };
    size_t n = 0;
    unsigned char* got = vitna_uni_nfc(odd, sizeof(odd), &n);
    CHECK(got && n == sizeof(odd) && memcmp(got, odd, n) == 0, "NFC: an invalid byte stays, and blocks composition across it");
    free(got);
    CHECK(vitna_uni_combining_class(0x301) == 230 && vitna_uni_combining_class(0x323) == 220 && vitna_uni_combining_class('a') == 0, "combining classes");
}

/* --- Tokenizer, on a vocabulary small enough to work by hand --- */

static const char* TINY_TOKENIZER =
    "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":11,\"content\":\"<s>\",\"single_word\":false,\"lstrip\":false,"
    "\"rstrip\":false,\"normalized\":false,\"special\":true}],\"normalizer\":null,"
    "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Digits\",\"individual_digits\":true},"
    "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true}]},"
    "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\"},"
    "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
    "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,"
    "\"vocab\":{\"a\":0,\"b\":1,\"c\":2,\"d\":3,\"\\u0120\":4,\"1\":5,\"2\":6,\"ab\":7,\"bc\":8,\"abc\":9,"
    "\"\\u0120a\":10,\"<s>\":11,\"aa\":12,\"!\":13,\"?\":14,\"!?\":15},"
    "\"merges\":[\"a b\",\"b c\",\"ab c\",\"\\u0120 a\",\"a a\",\"! ?\"]}}";

static bool ids_equal(const vitna_token_list_t* got, const int32_t* want, size_t n) {
    if (got->count != n) return false;
    for (size_t i = 0; i < n; i++) if (got->ids[i] != want[i]) return false;
    return true;
}

static void test_tokenizer(void) {
    const char* path = "vitna-unit-test-tokenizer.json";
    write_file(path, TINY_TOKENIZER, strlen(TINY_TOKENIZER));
    char err[256];
    vitna_tokenizer_t* tok = vitna_tokenizer_load(path, err, sizeof(err));
    CHECK(tok != NULL, "the tiny tokenizer loads: %s", err);
    if (tok) {
        struct { const char* text; int32_t ids[8]; size_t n; const char* why; } cases[] = {
            { "abc", { 9 }, 1, "merges apply lowest rank first: a+b, then ab+c" },
            { "bc", { 8 }, 1, "a single merge" },
            { "abab", { 7, 7 }, 2, "a merge applies at every position" },
            { "aaa", { 12, 0 }, 2, "equal ranks merge leftmost first" },
            { "a\x04" "b", { 0, 1 }, 2, "a byte with no token is dropped, and pieces do not merge across the split" },
            { "!\x04?", { 15 }, 1, "inside one piece, the neighbours of a dropped byte merge" },
            { "a12", { 0, 5, 6 }, 3, "each digit is its own piece" },
            { "<s>ab<s>", { 11, 7, 11 }, 3, "added tokens are matched before anything else" },
            { " a", { 10 }, 1, "a single leading space joins the letters" },
            { "  a", { 4, 10 }, 2, "a run of spaces before a letter leaves its last space to it" },
            { "", { 0 }, 0, "nothing in, nothing out" },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            vitna_token_list_t got = {0};
            vitna_tokenizer_encode(tok, cases[i].text, strlen(cases[i].text), &got);
            CHECK(ids_equal(&got, cases[i].ids, cases[i].n), "%s (%zu ids)", cases[i].why, got.count);
            vitna_token_list_free(&got);
        }
        size_t n = 0;
        const unsigned char* b = vitna_tokenizer_token_bytes(tok, 10, &n);
        CHECK(b && n == 2 && b[0] == ' ' && b[1] == 'a', "a byte-level token decodes to its bytes");
        CHECK(vitna_tokenizer_is_special(tok, 11) && !vitna_tokenizer_is_special(tok, 7), "special tokens");
        vitna_tokenizer_free(tok);
    }
    /* Anything the loader does not implement is refused, not approximated. */
    const char* unsupported = "{\"normalizer\":{\"type\":\"NFKC\"},\"pre_tokenizer\":{\"type\":\"ByteLevel\"},"
                              "\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]}}";
    write_file(path, unsupported, strlen(unsupported));
    tok = vitna_tokenizer_load(path, err, sizeof(err));
    CHECK(tok == NULL && strstr(err, "normalizer") != NULL, "a normalizer other than NFC is refused");
    vitna_tokenizer_free(tok);
    const char* adds_bos = "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"post_processor\":{\"type\":\"TemplateProcessing\","
                           "\"single\":[{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],"
                           "\"special_tokens\":{\"<s>\":{\"id\":\"<s>\",\"ids\":[0],\"tokens\":[\"<s>\"]}}},"
                           "\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]}}";
    write_file(path, adds_bos, strlen(adds_bos));
    tok = vitna_tokenizer_load(path, err, sizeof(err));
    CHECK(tok == NULL && strstr(err, "post_processor") != NULL, "a post-processor that adds a token is refused");
    vitna_tokenizer_free(tok);
    remove(path);
}

/* NFC, and added tokens matched as the tokenizers library matches them:
 * those not normalized in the text as written, then those that are, in
 * normalized text, as their own content normalized. Token 16 is written
 * decomposed, e and a combining acute; 17 overlaps the special <s>. The
 * post-processor is a template that adds nothing, as OLMoE's is. */
static const char* TINY_NFC_TOKENIZER =
    "{\"version\":\"1.0\",\"added_tokens\":["
    "{\"id\":11,\"content\":\"<s>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
    "{\"id\":16,\"content\":\"e\\u0301\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":true,\"special\":false},"
    "{\"id\":17,\"content\":\"a<\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":true,\"special\":false}],"
    "\"normalizer\":{\"type\":\"NFC\"},"
    "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true},"
    "\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],"
    "\"pair\":[{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"Sequence\":{\"id\":\"B\",\"type_id\":1}}],\"special_tokens\":{}},"
    "\"decoder\":{\"type\":\"ByteLevel\"},"
    "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
    "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,"
    "\"vocab\":{\"a\":0,\"b\":1,\"<s>\":11},\"merges\":[]}}";

static void test_tokenizer_nfc(void) {
    const char* path = "vitna-unit-test-tokenizer-nfc.json";
    write_file(path, TINY_NFC_TOKENIZER, strlen(TINY_NFC_TOKENIZER));
    char err[256];
    vitna_tokenizer_t* tok = vitna_tokenizer_load(path, err, sizeof(err));
    CHECK(tok != NULL, "the NFC tokenizer loads: %s", err);
    if (tok) {
        struct { const char* text; int32_t ids[4]; size_t n; const char* why; } cases[] = {
            { "\xC3\xA9", { 16 }, 1, "a normalized added token is matched as its content normalized" },
            { "e\xCC\x81", { 16 }, 1, "and in the text normalized" },
            { "a<s>", { 0, 11 }, 2, "a token not normalized is matched first, even where a normalized one starts earlier" },
            { "a<b", { 17, 1 }, 2, "a normalized token is matched where nothing else is" },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            vitna_token_list_t got = {0};
            vitna_tokenizer_encode(tok, cases[i].text, strlen(cases[i].text), &got);
            CHECK(ids_equal(&got, cases[i].ids, cases[i].n), "%s (%zu ids)", cases[i].why, got.count);
            vitna_token_list_free(&got);
        }
        size_t n = 0;
        unsigned char* norm = vitna_tokenizer_normalize(tok, "e\xCC\x81", 3, &n);
        CHECK(norm && n == 2 && memcmp(norm, "\xC3\xA9", 2) == 0, "the normalizer is NFC");
        free(norm);
        vitna_tokenizer_free(tok);
    }
    remove(path);
}

/* A BERT tokenizer, WordPiece, on a vocabulary small enough to work by hand:
 * BertNormalizer that cleans, spaces CJK, strips accents and lowercases, a
 * word limit of 10 characters, and [CLS] and [SEP] either side. */
static const char* TINY_WORDPIECE =
    "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":["
    "{\"id\":0,\"content\":\"[PAD]\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
    "{\"id\":1,\"content\":\"[UNK]\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
    "{\"id\":2,\"content\":\"[CLS]\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
    "{\"id\":3,\"content\":\"[SEP]\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}],"
    "\"normalizer\":{\"type\":\"BertNormalizer\",\"clean_text\":true,\"handle_chinese_chars\":true,\"strip_accents\":null,\"lowercase\":true},"
    "\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
    "\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"[CLS]\",\"type_id\":0}},"
    "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":0}}],"
    "\"special_tokens\":{\"[CLS]\":{\"id\":\"[CLS]\",\"ids\":[2],\"tokens\":[\"[CLS]\"]},\"[SEP]\":{\"id\":\"[SEP]\",\"ids\":[3],\"tokens\":[\"[SEP]\"]}}},"
    "\"decoder\":{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true},"
    "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":10,"
    "\"vocab\":{\"[PAD]\":0,\"[UNK]\":1,\"[CLS]\":2,\"[SEP]\":3,\"un\":4,\"##aff\":5,\"##able\":6,\"a\":7,\"##b\":8,\"cafe\":9,"
    "\",\":10,\"\\u4e2d\":11,\"!\":12,\"hello\":13,\"##s\":14,\"\\u03c3\":15,\"ab\":16}}}";

static void test_wordpiece(void) {
    const char* path = "vitna-unit-test-wordpiece.json";
    write_file(path, TINY_WORDPIECE, strlen(TINY_WORDPIECE));
    char err[256];
    vitna_tokenizer_t* tok = vitna_tokenizer_load(path, err, sizeof(err));
    CHECK(tok != NULL && vitna_tokenizer_is_wordpiece(tok), "the tiny WordPiece tokenizer loads: %s", err);
    if (tok) {
        struct { const char* text; int32_t ids[10]; size_t n; const char* why; } cases[] = {
            { "unaffable", { 2, 4, 5, 6, 3 }, 5, "the longest entries left to right, each after the first with ##" },
            { "UNAFFABLE", { 2, 4, 5, 6, 3 }, 5, "lowercased" },
            { "unaffableable", { 2, 1, 3 }, 3, "a word over the 10-character limit is the unknown token" },
            { "Caf\xC3\xA9, hellos!", { 2, 9, 10, 13, 14, 12, 3 }, 7, "accents stripped, and punctuation split off as words of its own" },
            { "a\xE4\xB8\xAD" "b", { 2, 7, 11, 1, 3 }, 5, "a CJK ideograph is a word, so b stands alone, where it is not spelled" },
            { "ab", { 2, 16, 3 }, 3, "the whole word where the vocabulary has it, not a then ##b" },
            { "abb", { 2, 16, 8, 3 }, 4, "ab, then ##b" },
            { "ac", { 2, 1, 3 }, 3, "a word with a part nothing spells is the unknown token, whole" },
            { "x[SEP]a", { 2, 1, 3, 7, 3 }, 5, "an added token is matched as written" },
            { "[sep]", { 2, 1, 1, 1, 3 }, 5, "and only as written: lowercase it is brackets around a word" },
            { "a\x07" "b\x0B" "b", { 2, 16, 8, 3 }, 4, "BEL and vertical tab are dropped, not made spaces, so the word stays whole" },
            { "a\tb", { 2, 7, 1, 3 }, 4, "a tab is a space" },
            { "\xCE\xA3", { 2, 15, 3 }, 3, "capital sigma lowercases to sigma, character by character" },
            { "", { 2, 3 }, 2, "an empty text is [CLS] and [SEP]" },
            { "a\xC3", { 2, 1, 3 }, 3, "a byte that is not UTF-8 stays in its word" },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            vitna_token_list_t got = {0};
            vitna_tokenizer_encode(tok, cases[i].text, strlen(cases[i].text), &got);
            CHECK(ids_equal(&got, cases[i].ids, cases[i].n), "%s (%zu ids)", cases[i].why, got.count);
            vitna_token_list_free(&got);
        }
        size_t n = 0;
        unsigned char* norm = vitna_tokenizer_normalize(tok, "Ab\x07 \xC3\xA9\xE4\xB8\xAD", 9, &n);
        const char want[] = "ab e \xE4\xB8\xAD ";
        CHECK(norm && n == sizeof(want) - 1 && memcmp(norm, want, n) == 0, "the normalizer: cleaned, CJK spaced, accents stripped, lowercased");
        free(norm);
        const unsigned char* b = vitna_tokenizer_token_bytes(tok, 13, &n);
        CHECK(b && n == 5 && memcmp(b, "hello", 5) == 0, "a token's text");
        CHECK(vitna_tokenizer_is_special(tok, 2) && !vitna_tokenizer_is_special(tok, 7), "special tokens");
        CHECK(vitna_tokenizer_vocab_size(tok) == 17, "the vocabulary size");
        vitna_tokenizer_free(tok);
    }
    /* What it does not implement is refused, not approximated. */
    const struct { const char* json; const char* why; } refused[] = {
        { "{\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
          "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"u\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":9,\"vocab\":{\"u\":0}}}",
          "BertNormalizer" },
        { "{\"pre_tokenizer\":{\"type\":\"Whitespace\"},"
          "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"u\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":9,\"vocab\":{\"u\":0}}}",
          "BertPreTokenizer" },
        { "{\"truncation\":{\"max_length\":8},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
          "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"u\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":9,\"vocab\":{\"u\":0}}}",
          "truncation" },
        { "{\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"added_tokens\":[{\"id\":1,\"content\":\"x\",\"normalized\":true}],"
          "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"u\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":9,\"vocab\":{\"u\":0}}}",
          "added token" },
        { "{\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
          "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"missing\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":9,\"vocab\":{\"u\":0}}}",
          "unk_token" },
    };
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); i++) {
        write_file(path, refused[i].json, strlen(refused[i].json));
        tok = vitna_tokenizer_load(path, err, sizeof(err));
        CHECK(tok == NULL && strstr(err, refused[i].why) != NULL, "refused (%s): got \"%s\"", refused[i].why, tok ? "loaded" : err);
        vitna_tokenizer_free(tok);
    }
    remove(path);
}

/* --- Float operations --- */

static void test_conversions(void) {
    CHECK(vitna_bf16_to_f32(0x3F80) == 1.0f && vitna_bf16_to_f32(0xC000) == -2.0f, "bf16");
    CHECK(vitna_f16_to_f32(0x3C00) == 1.0f && vitna_f16_to_f32(0xC000) == -2.0f, "f16 normal");
    CHECK(vitna_f16_to_f32(0x0001) == ldexpf(1.0f, -24), "f16 smallest subnormal");
    CHECK(vitna_f16_to_f32(0x03FF) == ldexpf(1023.0f, -24), "f16 largest subnormal");
    CHECK(vitna_f16_to_f32(0x7BFF) == 65504.0f && isinf(vitna_f16_to_f32(0x7C00)) && isnan(vitna_f16_to_f32(0x7E00)), "f16 max, inf, nan");
}

static double rel_err(double got, double want, double scale) {
    return fabs(got - want) / (scale > 1e-30 ? scale : 1e-30);
}

static void test_matvec(void) {
    const size_t shapes[][2] = { { 3, 5 }, { 67, 131 }, { 64, 576 }, { 17, 1536 } };
    for (size_t s = 0; s < 4; s++) {
        size_t rows = shapes[s][0], cols = shapes[s][1];
        uint16_t* wb = (uint16_t*)malloc(rows * cols * sizeof(uint16_t));
        float* wf = (float*)malloc(rows * cols * sizeof(float));
        float* x = (float*)malloc(cols * sizeof(float));
        float* y = (float*)malloc(rows * sizeof(float));
        float* ys = (float*)malloc(rows * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) {
            float v = rnd_f(-1.0f, 1.0f);
            uint32_t bits;
            memcpy(&bits, &v, 4);
            wb[i] = (uint16_t)(bits >> 16);
            wf[i] = vitna_bf16_to_f32(wb[i]);
        }
        for (size_t c = 0; c < cols; c++) x[c] = rnd_f(-2.0f, 2.0f);
        double worst_b = 0, worst_f = 0, worst_s = 0;
        vitna_matvec(wb, VITNA_DTYPE_BF16, x, y, rows, cols);
        vitna_matvec_scalar(wb, VITNA_DTYPE_BF16, x, ys, rows, cols);
        for (size_t r = 0; r < rows; r++) {
            double ref = 0, mag = 0;
            for (size_t c = 0; c < cols; c++) {
                ref += (double)wf[r * cols + c] * x[c];
                mag += fabs((double)wf[r * cols + c] * x[c]);
            }
            double e1 = rel_err(y[r], ref, mag), e2 = rel_err(ys[r], ref, mag);
            if (e1 > worst_b) worst_b = e1;
            if (e2 > worst_s) worst_s = e2;
        }
        vitna_matvec(wf, VITNA_DTYPE_F32, x, y, rows, cols);
        for (size_t r = 0; r < rows; r++) {
            double ref = 0, mag = 0;
            for (size_t c = 0; c < cols; c++) {
                ref += (double)wf[r * cols + c] * x[c];
                mag += fabs((double)wf[r * cols + c] * x[c]);
            }
            double e = rel_err(y[r], ref, mag);
            if (e > worst_f) worst_f = e;
        }
        /* float32 accumulation of up to 1536 products: within a few 1e-6 of
         * the double-precision sum, relative to the sum of magnitudes. */
        CHECK(worst_b < 2e-5 && worst_s < 2e-5 && worst_f < 2e-5,
              "matvec %zux%zu (%s path): bf16 %.2e, scalar %.2e, f32 %.2e", rows, cols, vitna_matvec_path(), worst_b, worst_s, worst_f);
        free(wb); free(wf); free(x); free(y); free(ys);
    }
}

static void test_rope(void) {
    const size_t hd = 64;
    float v[64], w[64], cs[32], sn[32];
    const double theta = 100000.0;
    for (size_t pos = 0; pos < 300; pos += 37) {
        for (size_t i = 0; i < hd; i++) v[i] = w[i] = rnd_f(-3.0f, 3.0f);
        for (size_t j = 0; j < hd / 2; j++) {
            double a = (double)pos * pow(theta, -(double)(2 * j) / (double)hd);
            cs[j] = (float)cos(a);
            sn[j] = (float)sin(a);
        }
        vitna_rope_half(v, hd, cs, sn);
        double worst = 0;
        for (size_t j = 0; j < hd / 2; j++) {
            double a = (double)pos * pow(theta, -(double)(2 * j) / (double)hd);
            double e0 = w[j] * cos(a) - w[j + hd / 2] * sin(a);
            double e1 = w[j + hd / 2] * cos(a) + w[j] * sin(a);
            worst = fmax(worst, fmax(fabs(v[j] - e0), fabs(v[j + hd / 2] - e1)));
        }
        CHECK(worst < 1e-5, "half-split rotation at position %zu: %.2e", pos, worst);

        /* kv_cache.c's interleaved rotation, which the Llama forward pass
         * does not use: pairs (2j, 2j + 1) turned by pos / theta^(2j / d). */
        for (size_t i = 0; i < hd; i++) v[i] = w[i];
        vitna_rope_apply(v, hd, pos, (float)theta);
        worst = 0;
        for (size_t j = 0; j < hd / 2; j++) {
            double a = (double)pos / pow(theta, (double)(2 * j) / (double)hd);
            double e0 = w[2 * j] * cos(a) - w[2 * j + 1] * sin(a);
            double e1 = w[2 * j] * sin(a) + w[2 * j + 1] * cos(a);
            worst = fmax(worst, fmax(fabs(v[2 * j] - e0), fabs(v[2 * j + 1] - e1)));
        }
        /* A float32 angle of up to 300 radians is good to about 3e-5. */
        CHECK(worst < 1e-3, "interleaved rotation at position %zu: %.2e", pos, worst);
    }
}

static void test_rmsnorm_and_swiglu(void) {
    const size_t dims[] = { 7, 17, 576, 1537 };
    for (size_t k = 0; k < 4; k++) {
        size_t n = dims[k];
        float* x = (float*)malloc(n * sizeof(float));
        float* w = (float*)malloc(n * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        double ss = 0;
        for (size_t i = 0; i < n; i++) {
            x[i] = rnd_f(-4.0f, 4.0f);
            w[i] = rnd_f(0.5f, 1.5f);
            ss += (double)x[i] * x[i];
        }
        vitna_rmsnorm(x, w, y, n, 1e-5f);
        double inv = 1.0 / sqrt(ss / (double)n + 1e-5);
        double worst = 0, largest = 0;
        for (size_t i = 0; i < n; i++) {
            double e = x[i] * inv * w[i];
            worst = fmax(worst, fabs(y[i] - e));
            largest = fmax(largest, fabs(e));
        }
        CHECK(worst / largest < 1e-6, "rmsnorm over %zu: error %.2e of the largest output", n, worst / largest);
        free(x); free(w); free(y);
    }
    float gu[16], out[8], a[8], b[8], c[8];
    for (int i = 0; i < 8; i++) { gu[i] = a[i] = rnd_f(-6.0f, 6.0f); gu[8 + i] = b[i] = rnd_f(-2.0f, 2.0f); }
    vitna_swiglu(gu, out, 8);
    vitna_silu_mul(a, b, c, 8);
    double worst = 0;
    for (int i = 0; i < 8; i++) {
        double e = a[i] / (1.0 + exp(-a[i])) * b[i];
        worst = fmax(worst, fmax(fabs(out[i] - e), fabs(c[i] - e)));
    }
    CHECK(worst < 1e-5, "silu(gate) * up: %.2e", worst);
}

/* The quantized matrix-vector products, each against a decoding of the
 * format its header describes, in double precision. */
static void test_quantized_gemv(void) {
    const size_t rows = 9, cols = 256, group = 64;
    const size_t groups = rows * cols / group;
    float* scales = (float*)malloc(groups * sizeof(float));
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));
    for (size_t i = 0; i < groups; i++) scales[i] = rnd_f(0.01f, 0.1f);
    for (size_t c = 0; c < cols; c++) x[c] = rnd_f(-1.0f, 1.0f);

    /* int8: one signed byte per weight. */
    int8_t* w8 = (int8_t*)malloc(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) w8[i] = (int8_t)(rnd_u32() & 0xFF);
    vitna_gemv_int8(w8, scales, group, x, y, rows, cols);
    double worst8 = 0;
    for (size_t r = 0; r < rows; r++) {
        double ref = 0;
        for (size_t c = 0; c < cols; c++) ref += w8[r * cols + c] * (double)scales[r * cols / group + c / group] * x[c];
        worst8 = fmax(worst8, fabs(y[r] - ref));
    }
    CHECK(worst8 < 1e-3, "int8 GEMV against its format: %.2e", worst8);

    /* int4: two per byte, low nibble first, stored with an offset of 8. */
    uint8_t* w4 = (uint8_t*)malloc(rows * cols / 2);
    for (size_t i = 0; i < rows * cols / 2; i++) w4[i] = (uint8_t)rnd_u32();
    vitna_gemv_int4(w4, scales, group, x, y, rows, cols);
    double worst4 = 0;
    for (size_t r = 0; r < rows; r++) {
        double ref = 0;
        for (size_t c = 0; c < cols; c++) {
            uint8_t byte = w4[(r * cols + c) / 2];
            int q = ((c % 2) ? (byte >> 4) : (byte & 0x0F)) - 8;
            ref += q * (double)scales[r * cols / group + c / group] * x[c];
        }
        worst4 = fmax(worst4, fabs(y[r] - ref));
    }
    CHECK(worst4 < 1e-3, "int4 GEMV against its format: %.2e", worst4);

    /* int3: eight per three bytes, least significant bits first, offset 4. */
    uint8_t* w3 = (uint8_t*)malloc(rows * cols / 8 * 3);
    for (size_t i = 0; i < rows * cols / 8 * 3; i++) w3[i] = (uint8_t)rnd_u32();
    vitna_gemv_int3(w3, scales, group, x, y, rows, cols);
    double worst3 = 0;
    for (size_t r = 0; r < rows; r++) {
        double ref = 0;
        for (size_t c = 0; c < cols; c++) {
            const uint8_t* p = w3 + r * (cols / 8 * 3) + (c / 8) * 3;
            uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            int q = (int)((v >> (3 * (c % 8))) & 7) - 4;
            ref += q * (double)scales[r * cols / group + c / group] * x[c];
        }
        worst3 = fmax(worst3, fabs(y[r] - ref));
    }
    CHECK(worst3 < 1e-3, "int3 GEMV against its format: %.2e", worst3);

    /* int2: four per byte, least significant bits first, offset 2. */
    uint8_t* w2 = (uint8_t*)malloc(rows * cols / 4);
    for (size_t i = 0; i < rows * cols / 4; i++) w2[i] = (uint8_t)rnd_u32();
    vitna_gemv_int2(w2, scales, group, x, y, rows, cols);
    double worst2 = 0;
    for (size_t r = 0; r < rows; r++) {
        double ref = 0;
        for (size_t c = 0; c < cols; c++) {
            int q = (int)((w2[(r * cols + c) / 4] >> (2 * (c % 4))) & 3) - 2;
            ref += q * (double)scales[r * cols / group + c / group] * x[c];
        }
        worst2 = fmax(worst2, fabs(y[r] - ref));
    }
    CHECK(worst2 < 1e-3, "int2 GEMV against its format: %.2e", worst2);

    free(scales); free(x); free(y); free(w8); free(w4); free(w3); free(w2);
}

/* --- SHA-256, against FIPS 180-2's examples --- */

static void sha_hex(const void* data, size_t n, size_t chunk, char out[65]) {
    vitna_sha256_ctx_t ctx;
    vitna_sha256_init(&ctx);
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i += chunk) vitna_sha256_update(&ctx, p + i, (n - i < chunk) ? n - i : chunk);
    vitna_sha256_final_hex(&ctx, out);
}

static void test_sha256(void) {
    char hex[65];
    sha_hex("", 0, 1, hex);
    CHECK(strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0, "SHA-256 of nothing: %s", hex);
    sha_hex("abc", 3, 3, hex);
    CHECK(strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0, "SHA-256 of abc: %s", hex);
    const char* two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    sha_hex(two, strlen(two), 7, hex);
    CHECK(strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0, "SHA-256 of the two-block example, fed 7 bytes at a time: %s", hex);
    char* million = (char*)malloc(1000000);
    memset(million, 'a', 1000000);
    sha_hex(million, 1000000, 4093, hex);
    CHECK(strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") == 0, "SHA-256 of a million a's: %s", hex);
    free(million);
}

/* --- Clock --- */

static void test_clock(void) {
    /* Windows' performance counter usually runs at 10 MHz. ticks * 1e9 used
     * to overflow 64 bits past 2^64 / 1e9 ticks, 31 minutes after boot. */
    const uint64_t f = 10000000ULL;
    const uint64_t wrap = 18446744073ULL; /* 2^64 / 1e9, rounded down */
    CHECK(vitna_ticks_to_nanos(0, f) == 0, "no ticks, no time");
    CHECK(vitna_ticks_to_nanos(3, f) == 300, "a 10 MHz tick is 100 ns");
    CHECK(vitna_ticks_to_nanos(f, f) == 1000000000ULL, "10 million ticks are a second");
    const uint64_t before = vitna_ticks_to_nanos(wrap - 1, f);
    const uint64_t after = vitna_ticks_to_nanos(wrap + 1, f);
    CHECK(before == 1844674407200ULL && after == 1844674407400ULL, "past the old wrap the clock runs on: %llu, then %llu",
          (unsigned long long)before, (unsigned long long)after);
    CHECK(vitna_ticks_to_nanos(86400ULL * f, f) == 86400ULL * 1000000000ULL, "a day of ticks at 10 MHz is exact");
    const uint64_t g = 3000000000ULL;
    CHECK(vitna_ticks_to_nanos(86400ULL * g + g / 2, g) == 86400500000000ULL, "and a day and half a second at 3 GHz");
    const uint64_t t0 = vitna_time_nanos();
    const uint64_t t1 = vitna_time_nanos();
    CHECK(t1 >= t0, "the clock does not run backwards");
}

/* --- Sampling --- */

static void test_sampler(void) {
    float tied[5] = { 1.0f, 3.0f, 2.0f, 3.0f, -1.0f };
    CHECK(vitna_argmax(tied, 5) == 1, "argmax takes the lowest id among ties");

    float logits[3] = { logf(0.7f), logf(0.2f), logf(0.1f) };
    vitna_sampler_t s;
    vitna_sampler_init(&s, 3, 42);
    vitna_sampling_t greedy = { 0.0f, 0, 1.0f, 0 };
    CHECK(vitna_sample(&s, logits, &greedy) == 0, "temperature 0 is greedy");
    vitna_sampling_t k1 = { 1.0f, 1, 1.0f, 0 };
    CHECK(vitna_sample(&s, logits, &k1) == 0, "top-k 1 is greedy");
    vitna_sampling_t p_small = { 1.0f, 0, 0.5f, 0 };
    CHECK(vitna_sample(&s, logits, &p_small) == 0, "top-p 0.5 keeps only the 0.7 token");

    vitna_sampling_t plain = { 1.0f, 0, 1.0f, 0 };
    int counts[3] = { 0, 0, 0 };
    const int n = 40000;
    for (int i = 0; i < n; i++) counts[vitna_sample(&s, logits, &plain)]++;
    CHECK(fabs(counts[0] / (double)n - 0.7) < 0.015 && fabs(counts[1] / (double)n - 0.2) < 0.015 && fabs(counts[2] / (double)n - 0.1) < 0.015,
          "temperature 1 draws in proportion: %d %d %d of %d", counts[0], counts[1], counts[2], n);

    vitna_sampling_t p9 = { 1.0f, 0, 0.85f, 0 };
    counts[0] = counts[1] = counts[2] = 0;
    for (int i = 0; i < n; i++) counts[vitna_sample(&s, logits, &p9)]++;
    CHECK(counts[2] == 0 && fabs(counts[0] / (double)n - 0.7 / 0.9) < 0.015, "top-p 0.85 keeps the 0.7 and 0.2 tokens, renormalized: %d %d %d", counts[0], counts[1], counts[2]);

    /* The same seed gives the same draws. */
    vitna_sampler_t a, b;
    vitna_sampler_init(&a, 3, 7);
    vitna_sampler_init(&b, 3, 7);
    bool same = true;
    for (int i = 0; i < 100; i++) same = same && vitna_sample(&a, logits, &plain) == vitna_sample(&b, logits, &plain);
    CHECK(same, "a seed reproduces its draws");
    vitna_sampler_free(&a);
    vitna_sampler_free(&b);
    vitna_sampler_free(&s);
}

/* The sampler as it was before its radix sort, kept as the statement of what
 * it must draw: the same arithmetic, with qsort ordering the ids by
 * probability, largest first, lower id first on ties. u is the uniform draw
 * the sampler would take from its generator. */
static const float* g_ref_probs;
static int ref_by_prob_desc(const void* a, const void* b) {
    int32_t ia = *(const int32_t*)a, ib = *(const int32_t*)b;
    if (g_ref_probs[ia] > g_ref_probs[ib]) return -1;
    if (g_ref_probs[ia] < g_ref_probs[ib]) return 1;
    return (ia > ib) - (ia < ib);
}

static int32_t ref_sample(float* probs, int32_t* order, size_t n, const float* logits, const vitna_sampling_t* cfg, double u01) {
    float max = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - max) / cfg->temperature);
        sum += probs[i];
        order[i] = (int32_t)i;
    }
    for (size_t i = 0; i < n; i++) probs[i] = (float)(probs[i] / sum);
    g_ref_probs = probs;
    qsort(order, n, sizeof(int32_t), ref_by_prob_desc);
    size_t keep = (cfg->top_k > 0 && cfg->top_k < n) ? cfg->top_k : n;
    double kept_mass = 0.0;
    for (size_t i = 0; i < keep; i++) kept_mass += probs[order[i]];
    if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = keep;
        for (size_t i = 0; i < keep; i++) {
            cum += probs[order[i]] / kept_mass;
            if (cum >= cfg->top_p) {
                cut = i + 1;
                break;
            }
        }
        keep = cut;
        kept_mass = 0.0;
        for (size_t i = 0; i < keep; i++) kept_mass += probs[order[i]];
    }
    double u = u01 * kept_mass;
    double cum = 0.0;
    for (size_t i = 0; i < keep; i++) {
        cum += probs[order[i]];
        if (u < cum) return order[i];
    }
    return order[keep - 1];
}

/* Draws from vitna_sample against the reference, the same generator state
 * each time; returns how many differed. */
static int sample_mismatches(const float* logits, size_t n, const vitna_sampling_t* cfg, int draws, uint64_t seed) {
    vitna_sampler_t s, rng;
    float* probs = (float*)malloc(n * sizeof(float));
    int32_t* order = (int32_t*)malloc(n * sizeof(int32_t));
    if (!probs || !order || !vitna_sampler_init(&s, n, seed) || !vitna_sampler_init(&rng, 1, seed)) {
        free(probs);
        free(order);
        return draws;
    }
    int bad = 0;
    for (int d = 0; d < draws; d++) {
        const int32_t got = vitna_sample(&s, logits, cfg);
        const int32_t want = ref_sample(probs, order, n, logits, cfg, vitna_sampler_uniform(&rng));
        bad += got != want;
    }
    vitna_sampler_free(&s);
    vitna_sampler_free(&rng);
    free(probs);
    free(order);
    return bad;
}

static void test_sampler_matches_reference(void) {
    const float temps[] = { 0.7f, 1.0f, 1.5f };
    const size_t ks[] = { 0, 1, 5, 50 };
    const float ps[] = { 1.0f, 0.99f, 0.9f, 0.5f };

    /* A vocabulary of SmolLM2's size, logits spread as a model's are. */
    const size_t big = 49152;
    float* logits = (float*)malloc(big * sizeof(float));
    for (size_t i = 0; i < big; i++) logits[i] = rnd_f(-12.0f, 8.0f);
    for (int t = 0; t < 3; t++) {
        const vitna_sampling_t cfg = { temps[t], t == 1 ? 50u : 0u, t == 2 ? 0.9f : 1.0f, 0 };
        const int bad = sample_mismatches(logits, big, &cfg, 12, 100 + t);
        CHECK(bad == 0, "49,152 logits, temperature %.1f: %d of 12 draws differ from the qsort sampler", temps[t], bad);
    }

    /* Many exact ties: logits from a handful of values, so equal
     * probabilities are everywhere and the lower id must come first. */
    const size_t mid = 3000;
    for (size_t i = 0; i < mid; i++) logits[i] = (float)(rnd_u32() % 7) * 0.5f;
    /* Probabilities that underflow to zero, and subnormal ones. */
    float* extreme = (float*)malloc(mid * sizeof(float));
    for (size_t i = 0; i < mid; i++) {
        const uint32_t r = rnd_u32() % 5;
        extreme[i] = r == 0 ? -1.0e4f : r == 1 ? rnd_f(-104.0f, -86.0f) : r == 2 ? 0.0f : rnd_f(-20.0f, 2.0f);
    }
    int configs = 0, bad_ties = 0, bad_extreme = 0;
    for (int t = 0; t < 3; t++) {
        for (int k = 0; k < 4; k++) {
            for (int p = 0; p < 4; p++) {
                const vitna_sampling_t cfg = { temps[t], ks[k], ps[p], 0 };
                bad_ties += sample_mismatches(logits, mid, &cfg, 40, 1000 + configs);
                bad_extreme += sample_mismatches(extreme, mid, &cfg, 40, 2000 + configs);
                configs++;
            }
        }
    }
    CHECK(bad_ties == 0, "logits with many ties: %d of %d draws differ from the qsort sampler", bad_ties, configs * 40);
    CHECK(bad_extreme == 0, "zero and subnormal probabilities: %d of %d draws differ from the qsort sampler", bad_extreme, configs * 40);

    /* All equal: every setting that keeps one token keeps the lowest id. */
    for (size_t i = 0; i < 10; i++) logits[i] = 1.25f;
    vitna_sampler_t s;
    vitna_sampler_init(&s, 10, 5);
    const vitna_sampling_t top1 = { 1.0f, 1, 1.0f, 0 };
    bool lowest = true;
    for (int d = 0; d < 20; d++) lowest = lowest && vitna_sample(&s, logits, &top1) == 0;
    CHECK(lowest, "equal logits, top-k 1: the lowest id every time");
    vitna_sampler_free(&s);
    free(extreme);
    free(logits);
}

/* --- JSON output, UTF-8 boundaries and ChatML, for the HTTP API --- */

/* --- Experts read from the drive --- */

/* The byte at offset o of test file f: a pattern no two nearby offsets share. */
static unsigned char pattern_byte(size_t f, uint64_t o) {
    return (unsigned char)((o * 131u + f * 17u + (o >> 8)) & 0xFF);
}

static bool part_is(const void* data, size_t f, uint64_t offset, uint64_t length) {
    const unsigned char* p = (const unsigned char*)data;
    for (uint64_t i = 0; i < length; i++) {
        if (p[i] != pattern_byte(f, offset + i)) return false;
    }
    return true;
}

static void test_expert_stream(void) {
    /* Two files whose sizes are not sector multiples. */
    const char* paths[2] = { "vitna-unit-test-experts-a.bin", "vitna-unit-test-experts-b.bin" };
    const size_t sizes[2] = { 3 * 4096 + 1000, 2 * 4096 + 77 };
    for (size_t f = 0; f < 2; f++) {
        unsigned char* bytes = (unsigned char*)malloc(sizes[f]);
        for (size_t i = 0; i < sizes[f]; i++) bytes[i] = pattern_byte(f, i);
        CHECK(write_file(paths[f], bytes, sizes[f]), "write test file %zu", f);
        free(bytes);
    }
    /* p0: one part, starting mid-sector and crossing one. p1: three parts that
     * follow one another, given out of order: one run. p2: parts in both
     * files. p3: a part ending at the end of its file. p4: past it. */
    const vitna_expert_place_t places[5] = {
        { { { 0, 100, 5000 } }, 1 },
        { { { 0, 5400, 200 }, { 0, 5100, 300 }, { 0, 5600, 1000 } }, 3 },
        { { { 1, 10, 50 }, { 0, 9000, 3000 } }, 2 },
        { { { 1, 8100, 169 } }, 1 },
        { { { 0, 13000, 1000 } }, 1 },
    };
    CHECK(vitna_expert_stream_slot_bytes_for(places, 4, 2) == 8192, "the widest place, in whole sectors: 8192 bytes");
    const vitna_extent_t empty = { 0, 0, 0 };
    const vitna_expert_place_t bad = { { empty }, 1 };
    CHECK(vitna_expert_stream_slot_bytes_for(&bad, 1, 2) == 0, "an empty part is not a place");

    char err[256] = "";
    vitna_expert_stream_t* s = vitna_expert_stream_open(paths, 2, places, 5, 4, 2, err, sizeof(err));
    CHECK(s != NULL, "the stream opens its files for direct I/O: %s", err);
    if (s) {
        vitna_expert_data_t d[2];
        uint32_t ids[2];
        bool ok = true;
        for (uint32_t p = 0; p < 4; p++) {
            ids[0] = p;
            ok = vitna_expert_stream_acquire(s, ids, 1, d) && ok;
            const vitna_expert_place_t* pl = &places[p];
            for (size_t j = 0; j < pl->n_parts; j++) {
                CHECK(part_is(d[0].part[j], pl->part[j].file, pl->part[j].offset, pl->part[j].length), "place %u part %zu holds its bytes", p, j);
            }
            vitna_expert_stream_release(s, ids, 1);
        }
        CHECK(ok, "four places are read");
        vitna_expert_stream_stats_t st = vitna_expert_stream_stats(s);
        CHECK(st.acquired == 4 && st.misses == 4 && st.hits == 0 && st.reads == 5, "four misses, read in five runs (p2's two files)");

        ids[0] = 3;
        ids[1] = 0;
        CHECK(vitna_expert_stream_acquire(s, ids, 2, d) && part_is(d[1].part[0], 0, 100, 5000), "held twice over, still there");
        vitna_expert_stream_release(s, ids, 2);
        st = vitna_expert_stream_stats(s);
        CHECK(st.hits == 2 && st.misses == 4, "what the cache holds is not read again");

        ids[0] = 4;
        CHECK(!vitna_expert_stream_acquire(s, ids, 1, d), "a part past the end of its file is a failed read");
        ids[0] = 0;
        CHECK(vitna_expert_stream_acquire(s, ids, 1, d) && part_is(d[0].part[0], 0, 100, 5000), "and the stream goes on");
        vitna_expert_stream_release(s, ids, 1);
        vitna_expert_stream_close(s);
    }

    /* A prefetch is read without anyone waiting, and counted when used. */
    s = vitna_expert_stream_open(paths, 2, places, 5, 4, 2, err, sizeof(err));
    if (s) {
        uint32_t ids[1] = { 2 };
        vitna_expert_data_t d[1];
        vitna_expert_stream_prefetch(s, ids, 1);
        CHECK(vitna_expert_stream_acquire(s, ids, 1, d) && part_is(d[0].part[1], 0, 9000, 3000), "a prefetched place holds its bytes");
        vitna_expert_stream_release(s, ids, 1);
        const vitna_expert_stream_stats_t st = vitna_expert_stream_stats(s);
        CHECK(st.prefetched == 1 && st.prefetch_used == 1 && st.misses == 0 && st.hits + st.in_flight == 1, "the prefetch served the acquisition");
        vitna_expert_stream_close(s);
    }
    /* hold_ready, which the GPU's guess copies from: it holds only a place
     * already read, starts no read and counts no acquisition, and what it
     * holds keeps its slot. Two slots, so p0 held leaves one for the rest. */
    s = vitna_expert_stream_open(paths, 2, places, 5, 2, 1, err, sizeof(err));
    if (s) {
        vitna_expert_data_t d[1];
        const uint32_t p0 = 0, others[2] = { 1, 3 };
        CHECK(!vitna_expert_stream_hold_ready(s, p0, d), "a place not read is not held");
        vitna_expert_stream_stats_t st = vitna_expert_stream_stats(s);
        CHECK(st.reads == 0 && st.acquired == 0 && st.misses == 0, "and nothing is read for it");
        bool ok = vitna_expert_stream_acquire(s, &p0, 1, d);
        vitna_expert_stream_release(s, &p0, 1);
        CHECK(ok && vitna_expert_stream_hold_ready(s, p0, d) && part_is(d[0].part[0], 0, 100, 5000), "a place read is held, with its bytes");
        for (size_t i = 0; i < 2; i++) {
            ok = vitna_expert_stream_acquire(s, &others[i], 1, d) && ok;
            vitna_expert_stream_release(s, &others[i], 1);
        }
        CHECK(ok && vitna_expert_stream_hold_ready(s, p0, d) && part_is(d[0].part[0], 0, 100, 5000),
              "held, it keeps its slot while two other places pass through the other");
        st = vitna_expert_stream_stats(s);
        CHECK(st.acquired == 3 && st.misses == 3, "only the acquisitions count: %llu acquired", (unsigned long long)st.acquired);
        vitna_expert_stream_release(s, &p0, 1);
        vitna_expert_stream_release(s, &p0, 1);
        size_t bytes = 0;
        const void* mem = vitna_expert_stream_memory(s, &bytes);
        CHECK(mem != NULL && bytes == 2 * 8192 && vitna_expert_stream_slots(s) == 2, "the slots are one block of 2 x 8192 bytes");
        vitna_expert_stream_close(s);
    }
    /* Two slots. The least used is given up first: p0, used twice, outlasts
     * p1, used once and more recently, which the least recently used would
     * have kept instead. */
    s = vitna_expert_stream_open(paths, 2, places, 5, 2, 1, err, sizeof(err));
    if (s) {
        const uint32_t sequence[5] = { 0, 0, 1, 3, 0 };
        vitna_expert_data_t d[1];
        bool ok = true;
        for (size_t i = 0; i < 5; i++) {
            ok = vitna_expert_stream_acquire(s, &sequence[i], 1, d) && ok;
            vitna_expert_stream_release(s, &sequence[i], 1);
        }
        const vitna_expert_stream_stats_t st = vitna_expert_stream_stats(s);
        CHECK(ok && st.misses == 3 && st.hits == 2, "p3 takes p1's slot, not p0's: %llu misses, %llu hits",
              (unsigned long long)st.misses, (unsigned long long)st.hits);
        vitna_expert_stream_close(s);
    }
    const char* missing[1] = { "vitna-unit-test-no-such-file.bin" };
    CHECK(vitna_expert_stream_open(missing, 1, places, 1, 4, 1, err, sizeof(err)) == NULL && strstr(err, "direct I/O") != NULL,
          "a file that cannot be opened is refused with the reason: %s", err);
    remove(paths[0]);
    remove(paths[1]);
}

static void test_api_helpers(void) {
    vitna_strbuf_t sb;
    vitna_sb_init(&sb);
    const unsigned char in[] = "a\"b\\c\n\x01\xC3\xA9\xFF\xE6\x9D";
    vitna_sb_json_string(&sb, in, sizeof(in) - 1);
    /* quote, backslash and newline escaped, U+0001 as \u0001, e-acute kept,
     * a stray byte and a cut-off three-byte sequence each as U+FFFD */
    const char* want = "\"a\\\"b\\\\c\\n\\u0001\xC3\xA9\\ufffd\\ufffd\\ufffd\"";
    CHECK(sb.ok && strcmp(sb.data, want) == 0, "JSON string escaping: got %s", sb.data ? sb.data : "(null)");
    vitna_sb_free(&sb);

    const unsigned char euro[] = { 'x', 0xE2, 0x82, 0xAC };
    CHECK(vitna_utf8_complete_prefix(euro, 4) == 4, "a whole character is complete");
    CHECK(vitna_utf8_complete_prefix(euro, 3) == 1 && vitna_utf8_complete_prefix(euro, 2) == 1, "a character cut short is held back");
    const unsigned char rocket[] = { 0xF0, 0x9F, 0x9A, 0x80 };
    CHECK(vitna_utf8_complete_prefix(rocket, 3) == 0 && vitna_utf8_complete_prefix(rocket, 4) == 4, "four-byte characters");

    const char* roles[] = { "system", "user" };
    const char* contents[] = { "Be brief.", "Hi" };
    char* out = NULL;
    size_t n = 0;
    CHECK(vitna_chatml_format(roles, contents, 2, &out, &n) &&
          strcmp(out, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n") == 0,
          "ChatML formatting");
    free(out);
}

/* --- The JSON-object prefix check behind JSON mode --- */

/* Whether every byte of s is accepted, and whether the object is then complete. */
static void pfx_run(const char* s, size_t n, bool* all, bool* complete) {
    vitna_jsonpfx_t p;
    vitna_jsonpfx_init(&p);
    *all = vitna_jsonpfx_feed(&p, (const unsigned char*)s, n);
    *complete = *all && vitna_jsonpfx_complete(&p);
}

/* A random JSON value into sb, nested at most depth deep, with some whitespace. */
static void rand_ws(vitna_strbuf_t* sb) {
    static const char* ws[] = { "", "", "", " ", "\n", "  ", "\t", "\r\n" };
    vitna_sb_puts(sb, ws[rnd_u32() % 8]);
}

static void rand_string(vitna_strbuf_t* sb) {
    static const char* parts[] = { "a", "key", " ", "\\\"", "\\\\", "\\n", "\\u00e9", "\\uD83D\\uDE80", "\xC3\xA9", "\xE6\x9D\xB1", "\xF0\x9F\x9A\x80", "}", "{", "]", ",", ":", "1" };
    vitna_sb_puts(sb, "\"");
    size_t n = rnd_u32() % 5;
    for (size_t i = 0; i < n; i++) vitna_sb_puts(sb, parts[rnd_u32() % 17]);
    vitna_sb_puts(sb, "\"");
}

static void rand_value(vitna_strbuf_t* sb, int depth, bool object) {
    int kind = object ? 0 : (int)(rnd_u32() % (depth > 0 ? 7 : 5));
    static const char* numbers[] = { "0", "-0", "12", "-7.25", "3e8", "1E-5", "0.5e+2", "-10.0" };
    static const char* literals[] = { "true", "false", "null" };
    switch (kind) {
        case 0: case 5: {
            vitna_sb_puts(sb, "{");
            size_t n = rnd_u32() % 4;
            for (size_t i = 0; i < n; i++) {
                if (i) vitna_sb_puts(sb, ",");
                rand_ws(sb);
                rand_string(sb);
                rand_ws(sb);
                vitna_sb_puts(sb, ":");
                rand_ws(sb);
                rand_value(sb, depth - 1, false);
                rand_ws(sb);
            }
            if (n == 0) rand_ws(sb);
            vitna_sb_puts(sb, "}");
            break;
        }
        case 6: {
            vitna_sb_puts(sb, "[");
            size_t n = rnd_u32() % 4;
            for (size_t i = 0; i < n; i++) {
                if (i) vitna_sb_puts(sb, ",");
                rand_ws(sb);
                rand_value(sb, depth - 1, false);
                rand_ws(sb);
            }
            vitna_sb_puts(sb, "]");
            break;
        }
        case 1: rand_string(sb); break;
        case 2: vitna_sb_puts(sb, numbers[rnd_u32() % 8]); break;
        default: vitna_sb_puts(sb, literals[rnd_u32() % 3]); break;
    }
}

static void test_jsonpfx(void) {
    bool all, complete;
    struct { const char* s; bool all; bool complete; } cases[] = {
        { "{}", true, true },
        { " {\"a\": 1}", true, true },
        { "{\"key\": 1}", true, true },            /* the old grammar.c rejected this at byte 2 */
        { "{\"a\": null, \"b\": [true, false, -0.5e3, \"}\"]}", true, true },
        { "{\"a\": \"}\"}", true, true },           /* a brace inside a string is text */
        { "{\"a\": {\"b\": [[], {}]}}", true, true },
        { "{\"a\": 1", true, false },               /* a prefix: accepted, not complete */
        { "{\"a\": tr", true, false },
        { "{\"a\": \"\\u00", true, false },
        { "{\"a\": 01}", false, false },           /* no leading zeros */
        { "{\"a\": tru}", false, false },
        { "{abc}", false, false },                 /* the old grammar.c accepted this */
        { "{\"a\" 1}", false, false },
        { "{\"a\": 1,}", false, false },           /* no trailing comma */
        { "[1, 2]", false, false },                /* the top level must be an object */
        { "{\"a\": \"x\ny\"}", false, false },     /* a raw newline in a string */
        { "{\"a\": \"\\q\"}", false, false },      /* not an escape */
        { "{\"a\": \"\xC0\xAF\"}", false, false }, /* overlong UTF-8 */
        { "{\"a\": \"\xED\xA0\x80\"}", false, false }, /* an encoded surrogate */
        { "{} x", false, false },                  /* nothing but whitespace after the object */
        { "{}   ", true, true },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        pfx_run(cases[i].s, strlen(cases[i].s), &all, &complete);
        CHECK(all == cases[i].all && complete == cases[i].complete, "jsonpfx on %s: accepted %d complete %d", cases[i].s, all, complete);
    }

    /* A run of whitespace is capped, so padding cannot go on forever. */
    char pad[64] = "{";
    memset(pad + 1, ' ', VITNA_JSONPFX_MAX_WS + 1);
    pfx_run(pad, 1 + VITNA_JSONPFX_MAX_WS, &all, &complete);
    CHECK(all, "whitespace up to the cap is accepted");
    pfx_run(pad, 2 + VITNA_JSONPFX_MAX_WS, &all, &complete);
    CHECK(!all, "whitespace past the cap is refused");

    /* The bytes allowed after `{"a":` are exactly those that can start a value, or whitespace. */
    vitna_jsonpfx_t p;
    vitna_jsonpfx_init(&p);
    vitna_jsonpfx_feed(&p, (const unsigned char*)"{\"a\":", 5);
    uint8_t allowed[32];
    vitna_jsonpfx_next_bytes(&p, allowed);
    const char* starts = " \t\n\r{[\"-0123456789tfn";
    int count = 0;
    for (int b = 0; b < 256; b++) count += (allowed[b >> 3] >> (b & 7)) & 1;
    bool all_starts = true;
    for (const char* c = starts; *c; c++) all_starts = all_starts && ((allowed[(unsigned char)*c >> 3] >> (*c & 7)) & 1);
    CHECK(all_starts && count == (int)strlen(starts), "next bytes after a colon: %d allowed", count);

    /* Random objects: every one is accepted byte by byte, and complete only at its end. */
    int bad = 0;
    for (int i = 0; i < 400; i++) {
        vitna_strbuf_t sb;
        vitna_sb_init(&sb);
        rand_value(&sb, 4, true);
        vitna_jsonpfx_t q;
        vitna_jsonpfx_init(&q);
        for (size_t k = 0; k < sb.len; k++) {
            if (!vitna_jsonpfx_feed(&q, (const unsigned char*)sb.data + k, 1) || (vitna_jsonpfx_complete(&q) != (k + 1 == sb.len))) {
                bad++;
                break;
            }
        }
        vitna_sb_free(&sb);
    }
    CHECK(bad == 0, "400 random objects accepted byte by byte, complete exactly at the end (%d failed)", bad);

    /* Random objects mutated by one ASCII byte: the check agrees with the
     * engine's JSON parser about which are complete objects. */
    int disagree = 0;
    static const char alphabet[] = "{}[]\":,-0123456789.eEtrufalsn \\u";
    for (int i = 0; i < 3000; i++) {
        vitna_strbuf_t sb;
        vitna_sb_init(&sb);
        rand_value(&sb, 3, true);
        size_t at = rnd_u32() % (sb.len + 1);
        char c = alphabet[rnd_u32() % (sizeof(alphabet) - 1)];
        int op = (int)(rnd_u32() % 3);
        vitna_strbuf_t m;
        vitna_sb_init(&m);
        vitna_sb_append(&m, sb.data, at);
        if (op != 2) vitna_sb_append(&m, &c, 1);                       /* insert, or replace */
        size_t skip = (op == 0) ? at : at + 1;                          /* op 1 and 2 drop the byte at `at` */
        if (skip < sb.len) vitna_sb_append(&m, sb.data + skip, sb.len - skip);
        pfx_run(m.data ? m.data : "", m.len, &all, &complete);
        char err[160];
        vitna_json_doc_t* doc = vitna_json_parse(m.data ? m.data : "", m.len, err, sizeof(err));
        bool parser_object = doc && vitna_json_root(doc)->type == VITNA_JSON_OBJECT;
        vitna_json_free(doc);
        /* A mutation can cut a multi-byte character in two. JSON text must
         * be UTF-8 (RFC 8259 section 8.1), and the prefix check holds to
         * that, while the engine's parser copies string bytes unchecked; so
         * where the text is not UTF-8 the check must refuse it, whatever the
         * parser says. */
        bool utf8 = vitna_utf8_complete_prefix((const unsigned char*)m.data, m.len) == m.len;
        for (size_t k = 0; utf8 && k < m.len;) {
            uint32_t cp;
            size_t l = vitna_utf8_decode((const unsigned char*)m.data + k, m.len - k, &cp);
            utf8 = cp < 0x110000;
            k += l;
        }
        bool expected = utf8 && parser_object;
        if (complete != expected) {
            if (disagree < 3) {
                fprintf(stderr, "  disagreement (check %d, parser %d, utf8 %d) on:", complete, parser_object, utf8);
                for (size_t k = 0; k < m.len; k++) {
                    unsigned char ch = (unsigned char)m.data[k];
                    if (ch >= 0x20 && ch < 0x7F) fputc(ch, stderr);
                    else fprintf(stderr, "\\x%02X", ch);
                }
                fputc('\n', stderr);
            }
            disagree++;
        }
        vitna_sb_free(&sb);
        vitna_sb_free(&m);
    }
    CHECK(disagree == 0, "3000 mutated objects: the prefix check and the parser agree (%d did not)", disagree);
}

/* --- ggml's block formats (quant.h) ---
 *
 * Each block is built here from the layout in quant.h's comment, written
 * out independently of quant.c: chosen scales and integers packed into
 * bytes, and the weights expected computed from them by the formula. */

static void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static void test_quant(void) {
    /* float16 values exact in float32: 0.5, 0.25, 0.03125, -0.125 */
    const uint16_t H_HALF = 0x3800, H_QUARTER = 0x3400, H_32TH = 0x2800, H_NEG_8TH = 0xB000;

    /* Q8_0: d, then 32 int8. */
    uint8_t q8[34];
    put_u16(q8, H_NEG_8TH);
    int8_t q8v[32];
    for (int i = 0; i < 32; i++) { q8v[i] = (int8_t)(i * 9 - 128 + (i == 31 ? 127 - 151 : 0)); q8[2 + i] = (uint8_t)q8v[i]; }
    float out[256];
    vitna_dequant_q8_0(q8, out);
    int bad = 0;
    for (int i = 0; i < 32; i++) bad += out[i] != (float)q8v[i] * -0.125f;
    CHECK(bad == 0, "Q8_0: w = q * d (%d of 32 differ)", bad);

    /* Q4_K: d, dmin, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles. */
    uint8_t q4[144];
    memset(q4, 0, sizeof(q4));
    put_u16(q4, H_HALF);
    put_u16(q4 + 2, H_QUARTER);
    uint8_t sc[8], mn[8], q4v[256];
    for (int j = 0; j < 8; j++) { sc[j] = (uint8_t)(7 * j + 5); mn[j] = (uint8_t)(63 - 6 * j); }
    uint8_t* s = q4 + 4;
    for (int j = 0; j < 4; j++) {
        s[j] = (uint8_t)((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
        s[j + 4] = (uint8_t)((mn[j] & 63) | ((mn[j + 4] >> 4) << 6));
        s[j + 8] = (uint8_t)((sc[j + 4] & 0x0F) | ((mn[j + 4] & 0x0F) << 4));
    }
    for (int i = 0; i < 256; i++) {
        q4v[i] = (uint8_t)((i * 7 + 3) & 0x0F);
        const int g = i / 32, l = i % 32;
        q4[16 + 32 * (g / 2) + l] |= (uint8_t)(q4v[i] << ((g & 1) ? 4 : 0));
    }
    vitna_dequant_q4_k(q4, out);
    bad = 0;
    for (int i = 0; i < 256; i++) {
        const float d1 = 0.5f * (float)sc[i / 32], m1 = 0.25f * (float)mn[i / 32];
        const float p = d1 * (float)q4v[i];
        bad += out[i] != p - m1;
    }
    CHECK(bad == 0, "Q4_K: w = (d * scale) * q - (dmin * min), group by group (%d of 256 differ)", bad);

    /* Q6_K: 128 bytes of low nibbles, 64 of high pairs, 16 int8 scales, d. */
    uint8_t q6[210];
    memset(q6, 0, sizeof(q6));
    int8_t sc6[16];
    int q6v[256];
    for (int k = 0; k < 16; k++) { sc6[k] = (int8_t)(k * 11 - 80); q6[192 + k] = (uint8_t)sc6[k]; }
    put_u16(q6 + 208, H_32TH);
    for (int i = 0; i < 256; i++) {
        const int q = (i * 37 + 11) % 64; /* 0..63 */
        q6v[i] = q - 32;
        const int h = i / 128, r = (i % 128) / 32, l = i % 32;
        q6[64 * h + l + 32 * (r & 1)] |= (uint8_t)((q & 0x0F) << (4 * (r >> 1)));
        q6[128 + 32 * h + l] |= (uint8_t)(((q >> 4) & 0x03) << (2 * r));
    }
    vitna_dequant_q6_k(q6, out);
    bad = 0;
    for (int i = 0; i < 256; i++) {
        const float ds = 0.03125f * (float)sc6[i / 16];
        bad += out[i] != ds * (float)q6v[i];
    }
    CHECK(bad == 0, "Q6_K: w = (d * scale) * (q - 32), sixteen at a time (%d of 256 differ)", bad);

    /* Sizes, and blocks run together. */
    CHECK(vitna_row_bytes(VITNA_DTYPE_Q8_0, 64) == 68 && vitna_row_bytes(VITNA_DTYPE_Q4_K, 512) == 288 &&
          vitna_row_bytes(VITNA_DTYPE_Q6_K, 256) == 210 && vitna_row_bytes(VITNA_DTYPE_BF16, 3) == 6,
          "row sizes are whole blocks");
    CHECK(vitna_row_bytes(VITNA_DTYPE_Q4_K, 100) == 0 && vitna_row_bytes(VITNA_DTYPE_Q8_0, 33) == 0, "a row of part of a block has no size");
    CHECK(vitna_dtype_size(VITNA_DTYPE_Q4_K) == 0 && strcmp(vitna_dtype_name(VITNA_DTYPE_Q6_K), "Q6_K") == 0, "a block format has no element size");
    uint8_t two[288];
    memcpy(two, q4, 144);
    memcpy(two + 144, q4, 144);
    float both[512];
    vitna_to_f32(two, VITNA_DTYPE_Q4_K, both, 512);
    vitna_dequant_q4_k(q4, out);
    CHECK(memcmp(both, out, sizeof(out)) == 0 && memcmp(both + 256, out, sizeof(out)) == 0, "vitna_to_f32 widens block after block");

    /* A matrix in a block format multiplies as its widened float32 rows do,
     * bit for bit: the same products, added in the same order. */
    enum { ROWS = 3, COLS = 512 };
    uint8_t w[ROWS * 288];
    float wf[ROWS * COLS], x[COLS], y_block[ROWS], y_f32[ROWS];
    for (int r = 0; r < ROWS; r++) {
        for (int b = 0; b < 2; b++) {
            uint8_t* blk = w + r * 288 + b * 144;
            memcpy(blk, q4, 144);
            for (int i = 16; i < 144; i++) blk[i] = (uint8_t)rnd_u32();
        }
    }
    vitna_to_f32(w, VITNA_DTYPE_Q4_K, wf, ROWS * COLS);
    for (int i = 0; i < COLS; i++) x[i] = rnd_f(-1.0f, 1.0f);
    vitna_matvec(w, VITNA_DTYPE_Q4_K, x, y_block, ROWS, COLS);
    if (strcmp(vitna_matvec_path(), "neon") == 0) {
        vitna_matvec_scalar(wf, VITNA_DTYPE_F32, x, y_f32, ROWS, COLS);
    } else {
        vitna_matvec(wf, VITNA_DTYPE_F32, x, y_f32, ROWS, COLS);
    }
    CHECK(memcmp(y_block, y_f32, sizeof(y_f32)) == 0, "Q4_K rows multiply as their widened float32 rows, bit for bit");
    vitna_matvec_scalar(w, VITNA_DTYPE_Q4_K, x, y_block, ROWS, COLS);
    vitna_matvec_scalar(wf, VITNA_DTYPE_F32, x, y_f32, ROWS, COLS);
    CHECK(memcmp(y_block, y_f32, sizeof(y_f32)) == 0, "and the scalar loop too");
}

/* --- GGUF (gguf.h) --- */

typedef struct {
    uint8_t* p;
    size_t n, cap;
} gbuf_t;

static void g_put(gbuf_t* b, const void* src, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2 + 64;
        b->p = (uint8_t*)realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void g_u32(gbuf_t* b, uint32_t v) { uint8_t t[4]; for (int i = 0; i < 4; i++) t[i] = (uint8_t)(v >> (8 * i)); g_put(b, t, 4); }
static void g_u64(gbuf_t* b, uint64_t v) { uint8_t t[8]; for (int i = 0; i < 8; i++) t[i] = (uint8_t)(v >> (8 * i)); g_put(b, t, 8); }
static void g_str(gbuf_t* b, const char* s) { g_u64(b, strlen(s)); g_put(b, s, strlen(s)); }
static void g_kv_str(gbuf_t* b, const char* k, const char* v) { g_str(b, k); g_u32(b, 8); g_str(b, v); }
static void g_kv_u32(gbuf_t* b, const char* k, uint32_t v) { g_str(b, k); g_u32(b, 4); g_u32(b, v); }
static void g_tensor(gbuf_t* b, const char* name, uint32_t n_dims, const uint64_t* ne, uint32_t type, uint64_t offset) {
    g_str(b, name);
    g_u32(b, n_dims);
    for (uint32_t d = 0; d < n_dims; d++) g_u64(b, ne[d]);
    g_u32(b, type);
    g_u64(b, offset);
}

/* A small GGUF file: an embedding of 3 rows of Q8_0, a stack of 2 experts of
 * 2 rows each, and a norm in F32, each at a multiple of 32 in the data. The
 * first five arguments break it in one way each. */
static size_t build_gguf(gbuf_t* b, const char* arch, uint32_t version, const char* extra_name, uint32_t emb_type, uint64_t norm_offset) {
    b->n = 0;
    g_u32(b, 0x46554747u);
    g_u32(b, version);
    g_u64(b, extra_name ? 4 : 3);
    g_u64(b, 4);
    g_kv_str(b, "general.architecture", arch);
    g_kv_u32(b, "general.alignment", 32);
    char key[64];
    snprintf(key, sizeof(key), "%s.block_count", arch);
    g_kv_u32(b, key, 1);
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    g_kv_u32(b, key, 2);
    const uint64_t emb[2] = { 32, 3 }, exps[3] = { 32, 2, 2 }, norm[1] = { 32 };
    g_tensor(b, "token_embd.weight", 2, emb, emb_type, 0);
    g_tensor(b, "blk.0.ffn_gate_exps.weight", 3, exps, 8, 128);
    g_tensor(b, "blk.0.attn_norm.weight", 1, norm, 0, norm_offset);
    if (extra_name) g_tensor(b, extra_name, 1, norm, 0, 288);
    while (b->n % 32) g_put(b, "", 1);
    const size_t data_start = b->n;
    uint8_t data[448];
    for (size_t i = 0; i < sizeof(data); i++) data[i] = (uint8_t)(i * 7);
    g_put(b, data, sizeof(data));
    return data_start;
}

static void test_gguf(void) {
    const char* path = "vitna-unit-test.gguf";
    gbuf_t b = { NULL, 0, 0 };
    vitna_safetensors_t st;
    vitna_gguf_info_t info;
    char err[512];

    const size_t data_start = build_gguf(&b, "olmoe", 3, NULL, 8, 288);
    write_file(path, b.p, b.n);
    bool ok = vitna_gguf_open(path, &st, &info, err, sizeof(err));
    CHECK(ok, "a small GGUF file opens: %s", err);
    if (ok) {
        CHECK(strcmp(info.arch, "olmoe") == 0 && info.block_count == 1 && info.expert_count == 2 && info.alignment == 32,
              "its metadata is read");
        CHECK(st.tensor_count == 4 && info.file_tensors == 3, "a stack of 2 experts is listed as 2 tensors (%zu listed)", st.tensor_count);
        const vitna_tensor_desc_t* e = vitna_safetensors_find(&st, "model.embed_tokens.weight");
        CHECK(e && e->dtype == VITNA_DTYPE_Q8_0 && e->ndim == 2 && e->shape[0] == 3 && e->shape[1] == 32 &&
              e->data_ptr == (const uint8_t*)st.mmap.data + data_start, "token_embd is the embedding, rows then columns, in place");
        const vitna_tensor_desc_t* x1 = vitna_safetensors_find(&st, "model.layers.0.mlp.experts.1.gate_proj.weight");
        CHECK(x1 && x1->shape[0] == 2 && x1->shape[1] == 32 &&
              x1->data_ptr == (const uint8_t*)st.mmap.data + data_start + 128 + 2 * 34,
              "expert 1 is its slice of the stack");
        const vitna_tensor_desc_t* n = vitna_safetensors_find(&st, "model.layers.0.input_layernorm.weight");
        CHECK(n && n->dtype == VITNA_DTYPE_F32 && n->ndim == 1 && n->shape[0] == 32, "attn_norm is the input norm");
        vitna_safetensors_close(&st);
    }

    struct { const char* arch; uint32_t version; const char* extra; uint32_t emb_type; uint64_t norm_offset; const char* why; } bad[] = {
        { "llama", 3, NULL, 8, 288, "permuted" },
        { "olmoe", 1, NULL, 8, 288, "version" },
        { "olmoe", 3, "blk.0.ffn_gate_shexp.weight", 8, 288, "knows where to put" },
        { "olmoe", 3, NULL, 2, 288, "Q4_0" },
        { "olmoe", 3, NULL, 8, 300, "alignment" },
        { "olmoe", 3, NULL, 8, 448, "outside the data section" },
        { "olmoe", 3, "blk.0.attn_norm.weight", 8, 288, "listed twice" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        build_gguf(&b, bad[i].arch, bad[i].version, bad[i].extra, bad[i].emb_type, bad[i].norm_offset);
        write_file(path, b.p, b.n);
        ok = vitna_gguf_open(path, &st, &info, err, sizeof(err));
        CHECK(!ok && strstr(err, bad[i].why) != NULL, "refused (%s): got \"%s\"", bad[i].why, ok ? "opened" : err);
        if (ok) vitna_safetensors_close(&st);
    }
    /* Cut short in the metadata, and not a GGUF file at all. */
    build_gguf(&b, "olmoe", 3, NULL, 8, 288);
    write_file(path, b.p, 100); /* inside the second metadata entry */
    ok = vitna_gguf_open(path, &st, &info, err, sizeof(err));
    CHECK(!ok && strstr(err, "runs past") != NULL, "a file cut short in its metadata is refused: got \"%s\"", ok ? "opened" : err);
    if (ok) vitna_safetensors_close(&st);
    write_file(path, "GGML not this", 13);
    ok = vitna_gguf_open(path, &st, &info, err, sizeof(err));
    CHECK(!ok && strstr(err, "not a GGUF") != NULL, "another file is refused: got \"%s\"", ok ? "opened" : err);
    if (ok) vitna_safetensors_close(&st);
    CHECK(vitna_gguf_path("a/b.GGUF") && !vitna_gguf_path("b.safetensors") && !vitna_gguf_path("gguf"), "a .gguf path is told apart");
    free(b.p);
    remove(path);
}

int main(void) {
    test_quant();
    test_gguf();
    test_expert_stream();
    test_api_helpers();
    test_jsonpfx();
    test_json();
    test_safetensors();
    test_unicode();
    test_nfc();
    test_tokenizer();
    test_tokenizer_nfc();
    test_wordpiece();
    test_conversions();
    test_matvec();
    test_rope();
    test_rmsnorm_and_swiglu();
    test_quantized_gemv();
    test_sha256();
    test_clock();
    test_sampler();
    test_sampler_matches_reference();
    printf("%d checks, %d failed (matvec path: %s, Unicode %s)\n", g_checks, g_failures, vitna_matvec_path(), vitna_uni_version());
    return g_failures == 0 ? 0 : 1;
}
