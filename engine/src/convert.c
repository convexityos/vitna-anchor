/**
 * convert.c - Requests of three shapes rewritten as OpenAI's chat shape (convert.h).
 *
 * Text in parts or blocks is joined with a newline, as vLLM and llama.cpp
 * join a message's text parts, so two pieces never run together.
 */

#include "convert.h"
#include "chat.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void vitna_chat_request_init(vitna_chat_request_t* cr) {
    memset(cr, 0, sizeof(*cr));
    vitna_sb_init(&cr->messages);
    vitna_sb_init(&cr->tools);
    cr->enable_thinking = -1;
}

void vitna_chat_request_free(vitna_chat_request_t* cr) {
    vitna_sb_free(&cr->messages);
    vitna_sb_free(&cr->tools);
}

static bool refuse(vitna_chat_request_t* cr, bool unsupported, const char* param, const char* fmt, ...) {
    cr->unsupported = unsupported;
    snprintf(cr->param, sizeof(cr->param), "%s", param ? param : "");
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cr->message, sizeof(cr->message), fmt, ap);
    va_end(ap);
    return false;
}

static bool is_str(const vitna_json_value_t* v) { return v && v->type == VITNA_JSON_STRING; }
static bool is_arr(const vitna_json_value_t* v) { return v && v->type == VITNA_JSON_ARRAY; }
static bool is_obj(const vitna_json_value_t* v) { return v && v->type == VITNA_JSON_OBJECT; }
static bool given(const vitna_json_value_t* v) { return v && v->type != VITNA_JSON_NULL; }
static bool str_eq(const vitna_json_value_t* v, const char* s) { return is_str(v) && strcmp(v->u.string.ptr, s) == 0; }
static const char* type_name(const vitna_json_value_t* v) { return is_str(v) ? v->u.string.ptr : "unknown"; }

static void put_str(vitna_strbuf_t* sb, const char* s, size_t n) { vitna_sb_json_string(sb, (const unsigned char*)(s ? s : ""), s ? n : 0); }

/* Append s to out, after a newline if out already holds text. */
static void join(vitna_strbuf_t* out, bool* any, const char* s, size_t n) {
    if (*any) vitna_sb_puts(out, "\n");
    vitna_sb_append(out, s, n);
    *any = true;
}

/* Text from a string, or from an array of parts whose type is one of types
 * (NULL-terminated), each with its text in "text", joined as join does. False
 * for any other part. */
static bool text_of(const vitna_json_value_t* v, const char* const* types, vitna_strbuf_t* out, bool* any) {
    if (is_str(v)) {
        join(out, any, v->u.string.ptr, v->u.string.len);
        return true;
    }
    if (!is_arr(v)) return false;
    for (size_t i = 0; i < v->u.array.count; i++) {
        const vitna_json_value_t* part = v->u.array.items[i];
        const vitna_json_value_t* type = is_obj(part) ? vitna_json_get(part, "type") : NULL;
        const vitna_json_value_t* text = is_obj(part) ? vitna_json_get(part, "text") : NULL;
        bool known = false;
        for (size_t t = 0; types[t] && !known; t++) known = str_eq(type, types[t]);
        if (!known || !is_str(text)) return false;
        join(out, any, text->u.string.ptr, text->u.string.len);
    }
    return true;
}

/* --- The conversation, a message at a time --- */

/* An assistant turn gathered across blocks or items (its reasoning, its text
 * and its tool calls) and written once whole, as the template reads one. */
typedef struct {
    vitna_strbuf_t* out;
    size_t count;
    bool open;
    vitna_strbuf_t content, reasoning, calls;
    bool any_content, has_reasoning, any_reasoning;
    size_t n_calls;
} conv_t;

static void conv_init(conv_t* c, vitna_strbuf_t* out) {
    memset(c, 0, sizeof(*c));
    c->out = out;
    vitna_sb_init(&c->content);
    vitna_sb_init(&c->reasoning);
    vitna_sb_init(&c->calls);
    vitna_sb_puts(out, "[");
}

static void conv_comma(conv_t* c) {
    if (c->count++) vitna_sb_puts(c->out, ", ");
}

static void conv_flush(conv_t* c) {
    if (!c->open) return;
    conv_comma(c);
    vitna_sb_puts(c->out, "{\"role\": \"assistant\", \"content\": ");
    put_str(c->out, c->content.data, c->content.len);
    if (c->has_reasoning) {
        vitna_sb_puts(c->out, ", \"reasoning_content\": ");
        put_str(c->out, c->reasoning.data, c->reasoning.len);
    }
    if (c->n_calls) {
        vitna_sb_puts(c->out, ", \"tool_calls\": [");
        vitna_sb_append(c->out, c->calls.data, c->calls.len);
        vitna_sb_puts(c->out, "]");
    }
    vitna_sb_puts(c->out, "}");
    vitna_sb_clear(&c->content);
    vitna_sb_clear(&c->reasoning);
    vitna_sb_clear(&c->calls);
    c->any_content = c->has_reasoning = c->any_reasoning = false;
    c->n_calls = 0;
    c->open = false;
}

/* A message of its own: system, user or tool, the last with the id of the call it answers. */
static void conv_message(conv_t* c, const char* role, const char* content, size_t n, const char* call_id, size_t id_len) {
    conv_flush(c);
    conv_comma(c);
    vitna_sb_puts(c->out, "{\"role\": ");
    put_str(c->out, role, strlen(role));
    vitna_sb_puts(c->out, ", \"content\": ");
    put_str(c->out, content, n);
    if (call_id) {
        vitna_sb_puts(c->out, ", \"tool_call_id\": ");
        put_str(c->out, call_id, id_len);
    }
    vitna_sb_puts(c->out, "}");
}

static void conv_text(conv_t* c, const char* s, size_t n) {
    c->open = true;
    join(&c->content, &c->any_content, s, n);
}

static void conv_reasoning(conv_t* c, const char* s, size_t n) {
    c->open = true;
    c->has_reasoning = true;
    join(&c->reasoning, &c->any_reasoning, s, n);
}

/* A call's arguments as the template reads them. An object is written as the
 * template's tojson writes it. So is a string holding a JSON object, as vLLM
 * parses one before applying a template: the model was trained on its
 * arguments written that way, however compactly a client wrote them. Any other
 * string stays a string, which the template writes as it is; none at all, or
 * an empty string, is an empty object. */
static void put_arguments(vitna_strbuf_t* out, const vitna_json_value_t* args) {
    if (!given(args) || (is_str(args) && args->u.string.len == 0)) {
        vitna_sb_puts(out, "{}");
        return;
    }
    if (is_str(args)) {
        char err[64];
        vitna_json_doc_t* doc = vitna_json_parse(args->u.string.ptr, args->u.string.len, err, sizeof(err));
        const vitna_json_value_t* root = doc ? vitna_json_root(doc) : NULL;
        if (is_obj(root)) {
            vitna_json_write_py(out, root);
            vitna_json_free(doc);
            return;
        }
        vitna_json_free(doc);
    }
    vitna_json_write_py(out, args);
}

static void conv_call(conv_t* c, const vitna_json_value_t* id, const char* name, size_t name_len, const vitna_json_value_t* args) {
    c->open = true;
    if (c->n_calls++) vitna_sb_puts(&c->calls, ", ");
    vitna_sb_puts(&c->calls, "{\"id\": ");
    put_str(&c->calls, is_str(id) ? id->u.string.ptr : "", is_str(id) ? id->u.string.len : 0);
    vitna_sb_puts(&c->calls, ", \"type\": \"function\", \"function\": {\"name\": ");
    put_str(&c->calls, name, name_len);
    vitna_sb_puts(&c->calls, ", \"arguments\": ");
    put_arguments(&c->calls, args);
    vitna_sb_puts(&c->calls, "}}");
}

static void conv_end(conv_t* c) {
    conv_flush(c);
    vitna_sb_puts(c->out, "]");
    vitna_sb_free(&c->content);
    vitna_sb_free(&c->reasoning);
    vitna_sb_free(&c->calls);
}

/* --- Tools --- */

/* A tool in OpenAI's chat shape, from its name, description and JSON schema:
 * {"type": "function", "function": {"name", "description", "parameters"}}, in
 * that order whatever order the client gave, and without fields the model was
 * not trained to read, such as strict. */
static void put_tool(vitna_strbuf_t* out, bool first, const vitna_json_value_t* name, const vitna_json_value_t* description,
                     const vitna_json_value_t* parameters) {
    vitna_sb_puts(out, first ? "[" : ", ");
    vitna_sb_puts(out, "{\"type\": \"function\", \"function\": {\"name\": ");
    put_str(out, name->u.string.ptr, name->u.string.len);
    if (is_str(description)) {
        vitna_sb_puts(out, ", \"description\": ");
        put_str(out, description->u.string.ptr, description->u.string.len);
    }
    if (given(parameters)) {
        vitna_sb_puts(out, ", \"parameters\": ");
        vitna_json_write_py(out, parameters);
    }
    vitna_sb_puts(out, "}}");
}

/* A tool's name in each shape: OpenAI chat's under function, the others' at the top. */
static const vitna_json_value_t* chat_tool_name(const vitna_json_value_t* t) {
    const vitna_json_value_t* fn = is_obj(t) ? vitna_json_get(t, "function") : NULL;
    return is_obj(fn) ? vitna_json_get(fn, "name") : NULL;
}

static const vitna_json_value_t* top_tool_name(const vitna_json_value_t* t) {
    return is_obj(t) ? vitna_json_get(t, "name") : NULL;
}

/* After the tools and tool_choice are read: a choice that needs a tool must
 * have one, and a named one must be among them, as the hosted APIs require. */
static bool check_choice(vitna_chat_request_t* cr, const vitna_json_value_t* tools, const vitna_json_value_t* (*name_of)(const vitna_json_value_t*)) {
    if (cr->tool_choice != VITNA_TOOLS_REQUIRED && cr->tool_choice != VITNA_TOOLS_NAMED) return true;
    if (!is_arr(tools) || tools->u.array.count == 0) return refuse(cr, false, "tool_choice", "a tool_choice that calls a tool needs tools to call");
    if (cr->tool_choice == VITNA_TOOLS_REQUIRED) return true;
    for (size_t i = 0; i < tools->u.array.count; i++) {
        if (str_eq(name_of(tools->u.array.items[i]), cr->tool_name)) return true;
    }
    return refuse(cr, false, "tool_choice", "tool_choice names the tool \"%s\", which is not among the tools", cr->tool_name);
}

static bool set_named(vitna_chat_request_t* cr, const vitna_json_value_t* name) {
    if (name->u.string.len >= sizeof(cr->tool_name) || memchr(name->u.string.ptr, '\0', name->u.string.len)) {
        return refuse(cr, false, "tool_choice", "tool_choice names a tool whose name is longer than %zu bytes", sizeof(cr->tool_name) - 1);
    }
    cr->tool_choice = VITNA_TOOLS_NAMED;
    memcpy(cr->tool_name, name->u.string.ptr, name->u.string.len);
    cr->tool_name[name->u.string.len] = '\0';
    return true;
}

static bool done(vitna_chat_request_t* cr) {
    if (!cr->messages.ok || !cr->tools.ok) return refuse(cr, false, NULL, "out of memory");
    return true;
}

/* --- OpenAI's chat completions --- */

bool vitna_convert_chat(const vitna_json_value_t* req, vitna_chat_request_t* cr) {
    static const char* const text_parts[] = { "text", NULL };
    if (given(vitna_json_get(req, "functions"))) return refuse(cr, true, "functions", "`functions` is the old shape of tools: send `tools`");
    const vitna_json_value_t* fc = vitna_json_get(req, "function_call");
    if (given(fc) && !str_eq(fc, "none") && !str_eq(fc, "auto")) {
        return refuse(cr, true, "function_call", "`function_call` is the old shape of tool_choice: send `tool_choice`");
    }
    const vitna_json_value_t* messages = vitna_json_get(req, "messages");
    if (!is_arr(messages) || messages->u.array.count == 0) return refuse(cr, false, "messages", "`messages` must be a non-empty array");
    conv_t c;
    conv_init(&c, &cr->messages);
    char param[64];
    for (size_t i = 0; i < messages->u.array.count; i++) {
        const vitna_json_value_t* m = messages->u.array.items[i];
        snprintf(param, sizeof(param), "messages[%zu]", i);
        const vitna_json_value_t* role = is_obj(m) ? vitna_json_get(m, "role") : NULL;
        if (!is_str(role)) {
            conv_end(&c);
            return refuse(cr, false, param, "each message needs a role");
        }
        const char* r = role->u.string.ptr;
        const bool assistant = strcmp(r, "assistant") == 0, tool = strcmp(r, "tool") == 0;
        const bool system = strcmp(r, "system") == 0 || strcmp(r, "developer") == 0, user = strcmp(r, "user") == 0;
        if (!assistant && !tool && !system && !user) {
            conv_end(&c);
            if (strcmp(r, "function") == 0) return refuse(cr, true, param, "a `function` message is the old shape of a tool result: send a `tool` message");
            return refuse(cr, false, param, "unknown role \"%s\"", r);
        }
        if (given(vitna_json_get(m, "name"))) {
            conv_end(&c);
            return refuse(cr, true, param, "a message's `name` has no place in this model's chat format");
        }
        if (given(vitna_json_get(m, "function_call"))) {
            conv_end(&c);
            return refuse(cr, true, param, "an assistant's `function_call` is the old shape of tool_calls: send `tool_calls`");
        }
        const vitna_json_value_t* content = vitna_json_get(m, "content");
        vitna_strbuf_t text;
        vitna_sb_init(&text);
        bool any = false;
        if (given(content) && !text_of(content, text_parts, &text, &any)) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, true, param, "the model reads text only");
        }
        if (!given(content) && !assistant) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, false, param, tool ? "a tool message needs content" : "each message needs text content");
        }
        if (assistant) {
            conv_flush(&c);
            c.open = true;
            if (text.len) conv_text(&c, text.data, text.len);
            const vitna_json_value_t* rc = vitna_json_get(m, "reasoning_content");
            if (!is_str(rc)) rc = vitna_json_get(m, "reasoning");
            if (is_str(rc)) conv_reasoning(&c, rc->u.string.ptr, rc->u.string.len);
            const vitna_json_value_t* calls = vitna_json_get(m, "tool_calls");
            if (given(calls) && !is_arr(calls)) {
                vitna_sb_free(&text);
                conv_end(&c);
                return refuse(cr, false, param, "`tool_calls` must be an array");
            }
            for (size_t j = 0; is_arr(calls) && j < calls->u.array.count; j++) {
                const vitna_json_value_t* call = calls->u.array.items[j];
                const vitna_json_value_t* fn = is_obj(call) ? vitna_json_get(call, "function") : NULL;
                const vitna_json_value_t* fname = is_obj(fn) ? vitna_json_get(fn, "name") : NULL;
                if (!is_str(fname)) {
                    vitna_sb_free(&text);
                    conv_end(&c);
                    return refuse(cr, false, param, "each tool call needs a function with a name");
                }
                conv_call(&c, vitna_json_get(call, "id"), fname->u.string.ptr, fname->u.string.len, vitna_json_get(fn, "arguments"));
            }
            conv_flush(&c);
        } else if (tool) {
            const vitna_json_value_t* id = vitna_json_get(m, "tool_call_id");
            conv_message(&c, "tool", text.data, text.len, is_str(id) ? id->u.string.ptr : "", is_str(id) ? id->u.string.len : 0);
        } else {
            conv_message(&c, user ? "user" : "system", text.data, text.len, NULL, 0);
        }
        vitna_sb_free(&text);
    }
    conv_end(&c);

    const vitna_json_value_t* tools = vitna_json_get(req, "tools");
    if (given(tools)) {
        if (!is_arr(tools)) return refuse(cr, false, "tools", "`tools` must be an array");
        for (size_t i = 0; i < tools->u.array.count; i++) {
            const vitna_json_value_t* t = tools->u.array.items[i];
            const vitna_json_value_t* fname = chat_tool_name(t);
            snprintf(param, sizeof(param), "tools[%zu]", i);
            if (!str_eq(vitna_json_get(t, "type"), "function")) {
                return refuse(cr, true, param, "a tool here is of type function, not of type %s", type_name(is_obj(t) ? vitna_json_get(t, "type") : NULL));
            }
            if (!is_str(fname)) return refuse(cr, false, param, "a tool needs a function with a name");
            const vitna_json_value_t* fn = vitna_json_get(t, "function");
            put_tool(&cr->tools, i == 0, fname, vitna_json_get(fn, "description"), vitna_json_get(fn, "parameters"));
        }
        if (tools->u.array.count) vitna_sb_puts(&cr->tools, "]");
    }
    const vitna_json_value_t* choice = vitna_json_get(req, "tool_choice");
    if (str_eq(choice, "none")) {
        cr->tool_choice = VITNA_TOOLS_NONE;
    } else if (str_eq(choice, "required")) {
        cr->tool_choice = VITNA_TOOLS_REQUIRED;
    } else if (is_obj(choice)) {
        const vitna_json_value_t* fn = vitna_json_get(choice, "function");
        const vitna_json_value_t* fname = is_obj(fn) ? vitna_json_get(fn, "name") : NULL;
        if (!str_eq(vitna_json_get(choice, "type"), "function") || !is_str(fname)) {
            return refuse(cr, false, "tool_choice", "`tool_choice` names a function as {\"type\": \"function\", \"function\": {\"name\": ...}}");
        }
        if (!set_named(cr, fname)) return false;
    } else if (given(choice) && !str_eq(choice, "auto")) {
        return refuse(cr, false, "tool_choice", "`tool_choice` is none, auto, required or a function");
    }
    if (!check_choice(cr, tools, chat_tool_name)) return false;

    /* The template's switch as vLLM takes it, and OpenAI's effort. */
    const vitna_json_value_t* kwargs = vitna_json_get(req, "chat_template_kwargs");
    bool b;
    if (is_obj(kwargs) && vitna_json_as_bool(vitna_json_get(kwargs, "enable_thinking"), &b)) cr->enable_thinking = b ? 1 : 0;
    const vitna_json_value_t* effort = vitna_json_get(req, "reasoning_effort");
    if (str_eq(effort, "none") || str_eq(effort, "minimal")) cr->enable_thinking = 0;
    else if (is_str(effort)) cr->enable_thinking = 1;
    return done(cr);
}

/* --- Anthropic's Messages --- */

bool vitna_convert_messages(const vitna_json_value_t* req, vitna_chat_request_t* cr) {
    static const char* const text_blocks[] = { "text", NULL };
    conv_t c;
    conv_init(&c, &cr->messages);
    char param[64];
    const vitna_json_value_t* system = vitna_json_get(req, "system");
    if (given(system)) {
        vitna_strbuf_t text;
        vitna_sb_init(&text);
        bool any = false;
        if (!text_of(system, text_blocks, &text, &any)) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, false, "system", "`system` must be a string or an array of text blocks");
        }
        conv_message(&c, "system", text.data, text.len, NULL, 0);
        vitna_sb_free(&text);
    }
    const vitna_json_value_t* messages = vitna_json_get(req, "messages");
    if (!is_arr(messages) || messages->u.array.count == 0) {
        conv_end(&c);
        return refuse(cr, false, "messages", "`messages` must be a non-empty array");
    }
    /* Consecutive messages of one role are one turn, as Anthropic's API
     * combines them. A user turn is its tool results, each a tool message,
     * then its text as one user message; an assistant turn is its thinking,
     * text and tool uses, as one assistant message. A system message among
     * them (Anthropic's mid-conversation system beta, which Claude Code
     * sends) is a system message there, as the template writes one. */
    vitna_strbuf_t text;
    vitna_sb_init(&text);
    bool any_text = false, user_open = false, any_result = false;
    const size_t n = messages->u.array.count;
    for (size_t i = 0; i < n; i++) {
        const vitna_json_value_t* m = messages->u.array.items[i];
        snprintf(param, sizeof(param), "messages.%zu", i);
        const vitna_json_value_t* role = is_obj(m) ? vitna_json_get(m, "role") : NULL;
        const vitna_json_value_t* content = is_obj(m) ? vitna_json_get(m, "content") : NULL;
        const bool user = str_eq(role, "user"), assistant = str_eq(role, "assistant"), sys = str_eq(role, "system");
        if (!user && !assistant && !sys) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, false, param, "a message's role is user, assistant or system");
        }
        if (!is_str(content) && !is_arr(content)) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, false, param, "a message's content is a string or an array of content blocks");
        }
        if (user && !user_open) {
            conv_flush(&c);
            user_open = true;
            any_text = any_result = false;
            vitna_sb_clear(&text);
        }
        if (!user && user_open) {
            /* A turn of tool results alone has no user message after them. */
            if (any_text || !any_result) conv_message(&c, "user", text.data, text.len, NULL, 0);
            user_open = false;
        }
        if (sys) {
            vitna_strbuf_t s;
            vitna_sb_init(&s);
            bool any = false;
            if (!text_of(content, text_blocks, &s, &any)) {
                vitna_sb_free(&s);
                vitna_sb_free(&text);
                conv_end(&c);
                return refuse(cr, true, param, "a system message holds text only");
            }
            conv_message(&c, "system", s.data, s.len, NULL, 0);
            vitna_sb_free(&s);
            continue;
        }
        if (is_str(content)) {
            if (user) join(&text, &any_text, content->u.string.ptr, content->u.string.len);
            else conv_text(&c, content->u.string.ptr, content->u.string.len);
            continue;
        }
        if (assistant) c.open = true;
        for (size_t j = 0; j < content->u.array.count; j++) {
            const vitna_json_value_t* b = content->u.array.items[j];
            const vitna_json_value_t* type = is_obj(b) ? vitna_json_get(b, "type") : NULL;
            char bparam[64];
            snprintf(bparam, sizeof(bparam), "messages.%zu.content.%zu", i, j);
            const vitna_json_value_t* t = is_obj(b) ? vitna_json_get(b, "text") : NULL;
            if (str_eq(type, "text") && is_str(t)) {
                if (user) join(&text, &any_text, t->u.string.ptr, t->u.string.len);
                else conv_text(&c, t->u.string.ptr, t->u.string.len);
            } else if (user && str_eq(type, "tool_result")) {
                const vitna_json_value_t* id = vitna_json_get(b, "tool_use_id");
                const vitna_json_value_t* rc = vitna_json_get(b, "content");
                vitna_strbuf_t result;
                vitna_sb_init(&result);
                bool any = false;
                if (given(rc) && !text_of(rc, text_blocks, &result, &any)) {
                    vitna_sb_free(&result);
                    vitna_sb_free(&text);
                    conv_end(&c);
                    return refuse(cr, true, bparam, "the model reads text only, so a tool result's content must be text");
                }
                conv_message(&c, "tool", result.data, result.len, is_str(id) ? id->u.string.ptr : "", is_str(id) ? id->u.string.len : 0);
                vitna_sb_free(&result);
                any_result = true;
            } else if (assistant && str_eq(type, "tool_use")) {
                const vitna_json_value_t* name = vitna_json_get(b, "name");
                if (!is_str(name)) {
                    vitna_sb_free(&text);
                    conv_end(&c);
                    return refuse(cr, false, bparam, "a tool_use block needs a name");
                }
                conv_call(&c, vitna_json_get(b, "id"), name->u.string.ptr, name->u.string.len, vitna_json_get(b, "input"));
            } else if (assistant && str_eq(type, "thinking")) {
                const vitna_json_value_t* th = vitna_json_get(b, "thinking");
                conv_reasoning(&c, is_str(th) ? th->u.string.ptr : "", is_str(th) ? th->u.string.len : 0);
            } else if (assistant && str_eq(type, "redacted_thinking")) {
                /* Another model's sealed reasoning: nothing this one can read. */
            } else {
                vitna_sb_free(&text);
                conv_end(&c);
                return refuse(cr, true, bparam, "the model reads text, tool uses, tool results and thinking, not a block of type %s", type_name(type));
            }
        }
    }
    /* A conversation that ends with the assistant asks for that reply to be
     * continued, as a prefill: written without its end, and continued. */
    const vitna_json_value_t* last = messages->u.array.items[n - 1];
    if (str_eq(vitna_json_get(last, "role"), "assistant")) {
        if (c.n_calls) {
            vitna_sb_free(&text);
            conv_end(&c);
            return refuse(cr, false, "messages", "the last message is the assistant's, to be continued, and a reply cannot be continued after a tool use");
        }
        cr->continue_final = true;
    }
    if (user_open && (any_text || !any_result)) conv_message(&c, "user", text.data, text.len, NULL, 0);
    vitna_sb_free(&text);
    conv_end(&c);

    const vitna_json_value_t* tools = vitna_json_get(req, "tools");
    if (given(tools)) {
        if (!is_arr(tools)) return refuse(cr, false, "tools", "`tools` must be an array");
        for (size_t i = 0; i < tools->u.array.count; i++) {
            const vitna_json_value_t* t = tools->u.array.items[i];
            const vitna_json_value_t* type = is_obj(t) ? vitna_json_get(t, "type") : NULL;
            const vitna_json_value_t* name = top_tool_name(t);
            snprintf(param, sizeof(param), "tools.%zu", i);
            if (given(type) && !str_eq(type, "custom")) {
                return refuse(cr, true, param, "a tool here is one the client runs, with a name and an input_schema, not Anthropic's own tool of type %s",
                              type_name(type));
            }
            if (!is_str(name)) return refuse(cr, false, param, "a tool needs a name");
            put_tool(&cr->tools, i == 0, name, vitna_json_get(t, "description"), vitna_json_get(t, "input_schema"));
        }
        if (tools->u.array.count) vitna_sb_puts(&cr->tools, "]");
    }
    const vitna_json_value_t* choice = vitna_json_get(req, "tool_choice");
    if (is_obj(choice)) {
        const vitna_json_value_t* type = vitna_json_get(choice, "type");
        if (str_eq(type, "none")) {
            cr->tool_choice = VITNA_TOOLS_NONE;
        } else if (str_eq(type, "any")) {
            cr->tool_choice = VITNA_TOOLS_REQUIRED;
        } else if (str_eq(type, "tool")) {
            const vitna_json_value_t* name = vitna_json_get(choice, "name");
            if (!is_str(name)) return refuse(cr, false, "tool_choice", "a tool_choice of type tool needs the tool's name");
            if (!set_named(cr, name)) return false;
        } else if (!str_eq(type, "auto")) {
            return refuse(cr, false, "tool_choice", "a tool_choice's type is auto, any, tool or none");
        }
    } else if (given(choice)) {
        return refuse(cr, false, "tool_choice", "`tool_choice` must be an object");
    }
    if (!check_choice(cr, tools, top_tool_name)) return false;

    /* Anthropic's API thinks only when asked to. */
    const vitna_json_value_t* thinking = vitna_json_get(req, "thinking");
    const vitna_json_value_t* tt = is_obj(thinking) ? vitna_json_get(thinking, "type") : NULL;
    if (given(thinking) && !str_eq(tt, "enabled") && !str_eq(tt, "disabled") && !str_eq(tt, "adaptive")) {
        return refuse(cr, false, "thinking", "a thinking's type is enabled, disabled or adaptive");
    }
    cr->enable_thinking = str_eq(tt, "enabled") || str_eq(tt, "adaptive") ? 1 : 0;
    return done(cr);
}

/* --- OpenAI's Responses --- */

/* The tools OpenAI runs on its own side, which a local server cannot. */
static bool hosted_tool(const vitna_json_value_t* type) {
    static const char* const hosted[] = { "file_search", "code_interpreter", "image_generation", "mcp", "computer_use_preview", "computer", NULL };
    if (!is_str(type)) return false;
    if (strncmp(type->u.string.ptr, "web_search", 10) == 0) return true;
    for (size_t i = 0; hosted[i]; i++) {
        if (strcmp(type->u.string.ptr, hosted[i]) == 0) return true;
    }
    return false;
}

/* Name something left out in cr->ignored, after what is there. */
static void note_ignored(vitna_chat_request_t* cr, const char* fmt, ...) {
    size_t at = strlen(cr->ignored);
    if (at && at + 2 < sizeof(cr->ignored)) {
        memcpy(cr->ignored + at, ", ", 3);
        at += 2;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cr->ignored + at, sizeof(cr->ignored) - at, fmt, ap);
    va_end(ap);
}

/* Whether a Responses request's tools have a function of this name, at the top or in a namespace. */
static bool responses_has_tool(const vitna_json_value_t* tools, const char* name) {
    for (size_t i = 0; is_arr(tools) && i < tools->u.array.count; i++) {
        const vitna_json_value_t* t = tools->u.array.items[i];
        if (str_eq(vitna_json_get(t, "type"), "function") && str_eq(top_tool_name(t), name)) return true;
        if (str_eq(vitna_json_get(t, "type"), "namespace") && vitna_responses_namespace(tools, name, strlen(name))) return true;
    }
    return false;
}

const vitna_json_value_t* vitna_responses_namespace(const vitna_json_value_t* tools, const char* name, size_t len) {
    for (size_t i = 0; is_arr(tools) && i < tools->u.array.count; i++) {
        const vitna_json_value_t* t = tools->u.array.items[i];
        if (!is_obj(t) || !str_eq(vitna_json_get(t, "type"), "namespace")) continue;
        const vitna_json_value_t* inner = vitna_json_get(t, "tools");
        for (size_t k = 0; is_arr(inner) && k < inner->u.array.count; k++) {
            const vitna_json_value_t* fname = top_tool_name(inner->u.array.items[k]);
            if (is_str(fname) && fname->u.string.len == len && memcmp(fname->u.string.ptr, name, len) == 0) {
                const vitna_json_value_t* ns = vitna_json_get(t, "name");
                return is_str(ns) ? ns : NULL;
            }
        }
    }
    return NULL;
}

bool vitna_convert_responses(const vitna_json_value_t* req, vitna_chat_request_t* cr) {
    static const char* const text_parts[] = { "input_text", "output_text", "text", NULL };
    static const char* const reasoning_parts[] = { "reasoning_text", "summary_text", NULL };
    conv_t c;
    conv_init(&c, &cr->messages);
    char param[64];
    const vitna_json_value_t* instructions = vitna_json_get(req, "instructions");
    if (given(instructions) && !is_str(instructions)) {
        conv_end(&c);
        return refuse(cr, false, "instructions", "`instructions` must be a string");
    }
    if (is_str(instructions)) conv_message(&c, "system", instructions->u.string.ptr, instructions->u.string.len, NULL, 0);
    const vitna_json_value_t* input = vitna_json_get(req, "input");
    if (is_str(input)) {
        conv_message(&c, "user", input->u.string.ptr, input->u.string.len, NULL, 0);
    } else if (is_arr(input) && input->u.array.count > 0) {
        /* Reasoning goes with the assistant turn that follows it, and
         * function calls with the assistant message before them. */
        vitna_strbuf_t reasoning;
        vitna_sb_init(&reasoning);
        bool pending = false, any_reasoning = false;
        for (size_t i = 0; i < input->u.array.count; i++) {
            const vitna_json_value_t* it = input->u.array.items[i];
            snprintf(param, sizeof(param), "input[%zu]", i);
            const vitna_json_value_t* type = is_obj(it) ? vitna_json_get(it, "type") : NULL;
            const vitna_json_value_t* role = is_obj(it) ? vitna_json_get(it, "role") : NULL;
            const char* why = NULL;
            bool unsupported = false;
            if ((!given(type) || str_eq(type, "message")) && is_str(role)) {
                vitna_strbuf_t text;
                vitna_sb_init(&text);
                bool any = false;
                const char* r = role->u.string.ptr;
                if (!text_of(vitna_json_get(it, "content"), text_parts, &text, &any)) {
                    why = "the model reads text only";
                    unsupported = true;
                } else if (strcmp(r, "assistant") == 0) {
                    conv_flush(&c);
                    c.open = true;
                    if (text.len) conv_text(&c, text.data, text.len);
                    if (pending) {
                        conv_reasoning(&c, reasoning.data, reasoning.len);
                        vitna_sb_clear(&reasoning);
                        pending = any_reasoning = false;
                    }
                } else if (strcmp(r, "user") == 0 || strcmp(r, "system") == 0 || strcmp(r, "developer") == 0) {
                    conv_message(&c, strcmp(r, "user") == 0 ? "user" : "system", text.data, text.len, NULL, 0);
                } else {
                    why = "a message's role is user, system, developer or assistant";
                }
                vitna_sb_free(&text);
            } else if (str_eq(type, "function_call")) {
                const vitna_json_value_t* name = vitna_json_get(it, "name");
                if (!is_str(name)) {
                    why = "a function_call needs a name";
                } else {
                    if (!c.open || pending) {
                        conv_flush(&c);
                        c.open = true;
                    }
                    if (pending) {
                        conv_reasoning(&c, reasoning.data, reasoning.len);
                        vitna_sb_clear(&reasoning);
                        pending = any_reasoning = false;
                    }
                    conv_call(&c, vitna_json_get(it, "call_id"), name->u.string.ptr, name->u.string.len, vitna_json_get(it, "arguments"));
                }
            } else if (str_eq(type, "function_call_output")) {
                const vitna_json_value_t* id = vitna_json_get(it, "call_id");
                vitna_strbuf_t out;
                vitna_sb_init(&out);
                bool any = false;
                if (!text_of(vitna_json_get(it, "output"), text_parts, &out, &any)) {
                    why = "the model reads text only, so a function_call_output must be text";
                    unsupported = true;
                } else {
                    conv_message(&c, "tool", out.data, out.len, is_str(id) ? id->u.string.ptr : "", is_str(id) ? id->u.string.len : 0);
                }
                vitna_sb_free(&out);
            } else if (str_eq(type, "reasoning")) {
                /* The model's own reasoning, in content, else its summary. */
                const vitna_json_value_t* rc = vitna_json_get(it, "content");
                if (!is_arr(rc) || rc->u.array.count == 0) rc = vitna_json_get(it, "summary");
                if (is_arr(rc) && !text_of(rc, reasoning_parts, &reasoning, &any_reasoning)) {
                    why = "a reasoning item's content is reasoning_text, and its summary summary_text";
                } else {
                    pending = true;
                }
            } else {
                why = "the server reads messages, function calls, their outputs and reasoning";
                unsupported = true;
            }
            if (why) {
                vitna_sb_free(&reasoning);
                conv_end(&c);
                if (unsupported && is_str(type) && !str_eq(type, "message") && !str_eq(type, "function_call_output")) {
                    return refuse(cr, true, param, "the server reads messages, function calls, their outputs and reasoning, not an item of type %s",
                                  type->u.string.ptr);
                }
                return refuse(cr, unsupported, param, "%s", why);
            }
        }
        vitna_sb_free(&reasoning);
    } else {
        conv_end(&c);
        return refuse(cr, false, "input", "`input` must be a string or a non-empty array of items");
    }
    conv_end(&c);

    const vitna_json_value_t* tools = vitna_json_get(req, "tools");
    if (given(tools)) {
        if (!is_arr(tools)) return refuse(cr, false, "tools", "`tools` must be an array");
        for (size_t i = 0; i < tools->u.array.count; i++) {
            const vitna_json_value_t* t = tools->u.array.items[i];
            const vitna_json_value_t* type = is_obj(t) ? vitna_json_get(t, "type") : NULL;
            snprintf(param, sizeof(param), "tools[%zu]", i);
            if (str_eq(type, "function")) {
                const vitna_json_value_t* name = top_tool_name(t);
                if (!is_str(name)) return refuse(cr, false, param, "a function tool needs a name");
                put_tool(&cr->tools, cr->tools.len == 0, name, vitna_json_get(t, "description"), vitna_json_get(t, "parameters"));
            } else if (str_eq(type, "namespace")) {
                /* A namespace's functions are offered as functions, each by
                 * its own name, and a call to one is returned with its
                 * namespace beside its name (vitna_responses_namespace). */
                const vitna_json_value_t* inner = vitna_json_get(t, "tools");
                for (size_t k = 0; is_arr(inner) && k < inner->u.array.count; k++) {
                    const vitna_json_value_t* f = inner->u.array.items[k];
                    const vitna_json_value_t* fname = top_tool_name(f);
                    if (!str_eq(vitna_json_get(f, "type"), "function") || !is_str(fname)) {
                        return refuse(cr, true, param, "the tools of a namespace here are functions with names");
                    }
                    put_tool(&cr->tools, cr->tools.len == 0, fname, vitna_json_get(f, "description"), vitna_json_get(f, "parameters"));
                }
            } else if (hosted_tool(type)) {
                /* A tool OpenAI runs on its own side, which a local server
                 * cannot: left out of what the model is offered, and named
                 * among what was ignored, in characters a header can carry. */
                char safe[48];
                size_t k = 0;
                for (; k + 1 < sizeof(safe) && k < type->u.string.len; k++) {
                    const char ch = type->u.string.ptr[k];
                    safe[k] = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' ? ch : '_';
                }
                safe[k] = '\0';
                note_ignored(cr, "tools.%zu.%s", i, safe);
            } else {
                return refuse(cr, true, param, "a tool here is a function, or a namespace of functions, not of type %s", type_name(type));
            }
        }
        if (cr->tools.len) vitna_sb_puts(&cr->tools, "]");
    }
    const vitna_json_value_t* choice = vitna_json_get(req, "tool_choice");
    if (str_eq(choice, "none")) {
        cr->tool_choice = VITNA_TOOLS_NONE;
    } else if (str_eq(choice, "required")) {
        cr->tool_choice = VITNA_TOOLS_REQUIRED;
    } else if (is_obj(choice)) {
        const vitna_json_value_t* name = vitna_json_get(choice, "name");
        if (!str_eq(vitna_json_get(choice, "type"), "function") || !is_str(name)) {
            return refuse(cr, false, "tool_choice", "`tool_choice` names a function as {\"type\": \"function\", \"name\": ...}");
        }
        if (!set_named(cr, name)) return false;
    } else if (given(choice) && !str_eq(choice, "auto")) {
        return refuse(cr, false, "tool_choice", "`tool_choice` is none, auto, required or a function");
    }
    if ((cr->tool_choice == VITNA_TOOLS_REQUIRED || cr->tool_choice == VITNA_TOOLS_NAMED) && cr->tools.len == 0) {
        return refuse(cr, false, "tool_choice", "a tool_choice that calls a tool needs tools to call");
    }
    if (cr->tool_choice == VITNA_TOOLS_NAMED && !responses_has_tool(tools, cr->tool_name)) {
        return refuse(cr, false, "tool_choice", "tool_choice names the tool \"%s\", which is not among the tools", cr->tool_name);
    }

    const vitna_json_value_t* reasoning = vitna_json_get(req, "reasoning");
    const vitna_json_value_t* effort = is_obj(reasoning) ? vitna_json_get(reasoning, "effort") : NULL;
    if (str_eq(effort, "none") || str_eq(effort, "minimal")) cr->enable_thinking = 0;
    else if (is_str(effort)) cr->enable_thinking = 1;
    return done(cr);
}
