"""Gate A11 by hand: the official Anthropic and OpenAI SDKs against a running server.

Each API is driven the way an agent drives it: a question that needs a tool,
the model's call parsed by the SDK, the tool's result sent back, and the
model's answer, whole and streamed. The SDKs parse every response and every
stream event into their own types, so a field missing or misshapen fails
here as it would fail a client. Greedy throughout, so a run repeats.

    pip install anthropic openai
    vitna-anchor serve --model models/qwen3-30b-a3b --weights <file.gguf> --model-id qwen3-30b-a3b --device cuda --ctx 32768 --port 8771
    python scripts/check-api-clients.py --base http://127.0.0.1:8771 --model qwen3-30b-a3b

The key is a placeholder: the server has no authentication, and the SDKs
want one to send.
"""

import argparse
import json
import sys

import anthropic
import openai

WEATHER = {"Paris": "Sunny, 21 C", "Oslo": "Rain, 9 C"}


def get_weather(city):
    return WEATHER.get(city, "No report")


ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "The current weather in a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name"}}, "required": ["city"]},
    }
]
OPENAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in ANTHROPIC_TOOLS]
RESPONSES_TOOLS = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in ANTHROPIC_TOOLS]
QUESTION = "What is the weather in Paris right now? Use the tool."

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def anthropic_checks(base, model):
    client = anthropic.Anthropic(base_url=base, api_key="placeholder", max_retries=0, timeout=600)

    # A tool call, whole, then its result sent back.
    first = client.messages.create(model=model, max_tokens=512, temperature=0, tools=ANTHROPIC_TOOLS, messages=[{"role": "user", "content": QUESTION}])
    uses = [b for b in first.content if b.type == "tool_use"]
    check("anthropic: the model calls the tool", first.stop_reason == "tool_use" and len(uses) == 1, f"{first.stop_reason} {[b.type for b in first.content]}")
    if not uses:
        return
    use = uses[0]
    check("anthropic: the call's input is an object the SDK parsed", isinstance(use.input, dict) and use.input.get("city") == "Paris", json.dumps(use.input))
    second = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        tools=ANTHROPIC_TOOLS,
        messages=[
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in first.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": get_weather(use.input.get("city"))}]},
        ],
    )
    answer = "".join(b.text for b in second.content if b.type == "text")
    check("anthropic: the answer uses the tool's result", second.stop_reason == "end_turn" and "21" in answer, repr(answer[:200]))

    # The same call streamed, accumulated by the SDK from the events.
    with client.messages.stream(model=model, max_tokens=512, temperature=0, tools=ANTHROPIC_TOOLS, messages=[{"role": "user", "content": QUESTION}]) as s:
        for _ in s:
            pass
        streamed = s.get_final_message()
    same = [(b.type, getattr(b, "text", None), getattr(b, "name", None), getattr(b, "input", None)) for b in streamed.content] == [
        (b.type, getattr(b, "text", None), getattr(b, "name", None), getattr(b, "input", None)) for b in first.content
    ]
    check("anthropic: the stream's message is the whole one's", same and streamed.stop_reason == first.stop_reason, f"{[b.type for b in streamed.content]}")
    check(
        "anthropic: usage alike, whole and streamed",
        streamed.usage.input_tokens + (streamed.usage.cache_read_input_tokens or 0) == first.usage.input_tokens + (first.usage.cache_read_input_tokens or 0)
        and streamed.usage.output_tokens == first.usage.output_tokens,
        f"{streamed.usage} / {first.usage}",
    )

    # Thinking, when asked for, comes back as a thinking block before the text.
    thought = client.messages.create(
        model=model, max_tokens=2048, thinking={"type": "enabled", "budget_tokens": 1024}, messages=[{"role": "user", "content": "Is 391 a prime number? Answer yes or no."}]
    )
    kinds = [b.type for b in thought.content]
    check("anthropic: thinking, then text", kinds[:1] == ["thinking"] and "text" in kinds, f"{kinds} {thought.stop_reason}")

    # The token counter counts what the message reads.
    counted = client.messages.count_tokens(model=model, tools=ANTHROPIC_TOOLS, messages=[{"role": "user", "content": QUESTION}])
    check("anthropic: count_tokens is the prompt's size", counted.input_tokens == first.usage.input_tokens + (first.usage.cache_read_input_tokens or 0), f"{counted.input_tokens}")


def chat_checks(base, model):
    client = openai.OpenAI(base_url=base + "/v1", api_key="placeholder", max_retries=0, timeout=600)
    messages = [{"role": "user", "content": QUESTION}]
    first = client.chat.completions.create(model=model, messages=messages, tools=OPENAI_TOOLS, temperature=0, max_tokens=512)
    choice = first.choices[0]
    calls = choice.message.tool_calls or []
    check("chat: the model calls the tool", choice.finish_reason == "tool_calls" and len(calls) == 1, f"{choice.finish_reason} {choice.message}")
    if not calls:
        return
    args = json.loads(calls[0].function.arguments)
    check("chat: the call's arguments are JSON", args.get("city") == "Paris", calls[0].function.arguments)
    follow = messages + [choice.message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": calls[0].id, "content": get_weather(args.get("city"))}]
    second = client.chat.completions.create(model=model, messages=follow, tools=OPENAI_TOOLS, temperature=0, max_tokens=512)
    answer = second.choices[0].message.content or ""
    check("chat: the answer uses the tool's result", second.choices[0].finish_reason == "stop" and "21" in answer, repr(answer[:200]))

    # Streamed, through the SDK's own accumulator.
    with client.chat.completions.stream(model=model, messages=messages, tools=OPENAI_TOOLS, temperature=0, max_tokens=512) as s:
        for _ in s:
            pass
        streamed = s.get_final_completion()
    sc = streamed.choices[0]
    same = sc.finish_reason == "tool_calls" and [(c.function.name, c.function.arguments) for c in sc.message.tool_calls or []] == [
        (c.function.name, c.function.arguments) for c in calls
    ]
    # A stream opens with "content": "", as OpenAI's does, before anyone knows
    # whether text will follow, so the SDK accumulates "" where a whole reply
    # of calls alone says null: both are no text.
    check(
        "chat: the stream's completion is the whole one's",
        same and (sc.message.content or "") == (choice.message.content or ""),
        f"{sc.finish_reason} {sc.message.tool_calls}",
    )


def responses_checks(base, model):
    client = openai.OpenAI(base_url=base + "/v1", api_key="placeholder", max_retries=0, timeout=600)
    first = client.responses.create(model=model, input=QUESTION, tools=RESPONSES_TOOLS, temperature=0, max_output_tokens=512)
    calls = [o for o in first.output if o.type == "function_call"]
    check("responses: the model calls the tool", first.status == "completed" and len(calls) == 1, f"{first.status} {[o.type for o in first.output]}")
    if not calls:
        return
    args = json.loads(calls[0].arguments)
    check("responses: the call's arguments are JSON", args.get("city") == "Paris", calls[0].arguments)
    follow = [{"role": "user", "content": QUESTION}] + [o.model_dump(exclude_none=True) for o in first.output]
    follow.append({"type": "function_call_output", "call_id": calls[0].call_id, "output": get_weather(args.get("city"))})
    second = client.responses.create(model=model, input=follow, tools=RESPONSES_TOOLS, temperature=0, max_output_tokens=512)
    check("responses: the answer uses the tool's result", second.status == "completed" and "21" in second.output_text, repr(second.output_text[:200]))

    # Streamed, through the SDK's own accumulator, which checks the events' order.
    with client.responses.stream(model=model, input=QUESTION, tools=RESPONSES_TOOLS, temperature=0, max_output_tokens=512) as s:
        for _ in s:
            pass
        streamed = s.get_final_response()
    same = [(o.type, getattr(o, "name", None), getattr(o, "arguments", None)) for o in streamed.output] == [
        (o.type, getattr(o, "name", None), getattr(o, "arguments", None)) for o in first.output
    ]
    check("responses: the stream's response is the whole one's", same and streamed.status == "completed", f"{[o.type for o in streamed.output]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8771")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    a = ap.parse_args()
    print(f"anthropic {anthropic.__version__}, openai {openai.__version__}, against {a.base}", flush=True)
    for run in (anthropic_checks, chat_checks, responses_checks):
        try:
            run(a.base, a.model)
        except Exception as e:  # an SDK that cannot parse a response raises: that is a failure to report, not to hide
            check(f"{run.__name__} ran", False, f"{type(e).__name__}: {e}")
    print(f"{len(failures)} failed" if failures else "all passed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
