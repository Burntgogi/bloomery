"""Reference renders of the MiniMax-M3.1-Flash chat template for crates/serve/tests/fixtures/minimaxm31-renders.json.

usage (transformers 5.12.1 and jinja2 3.1.x; the template is the checkpoint folder's `chat_template.jinja`, byte for
byte, which the folder's header carries as `tokenizer.chat_template`):
    python3 render_minimaxm31.py minimaxm31-chat-template.jinja > minimaxm31-renders.json

The renderer is Hugging Face's own, `transformers.utils.chat_template_utils.render_jinja_template`, the function
`apply_chat_template` calls: an ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks, the loopcontrols
extension and the `generation` tag, `tojson` as json.dumps with ensure_ascii=False, and the globals raise_exception and
strftime_now.

The variables are the ones bloomery-serve's render_chat (crates/serve/src/api.rs) passes for a body: messages (the role
`developer` sent as `system`, a content array flattened to its text parts joined by "\n"), add_generation_prompt (the
body's boolean, else true), bos_token and eos_token (the mock engine's texts), tools and reasoning_effort when present
and not null, and chat_template_kwargs merged on top. The template reads an effort knob and no thinking switch, so the
server sends the effort "none" as "low" (`ThinkingOff::LowEffort`).

Every object in a body is written with its keys sorted, at every level, as the checked-in fixture was recorded; the key
order does not decide the comparison (bloomery's template engine keeps a body's insertion order, as jinja2 does).
"""
import json
import platform
import sys

import jinja2
import transformers
from transformers.utils.chat_template_utils import render_jinja_template

BOS = "<s>"
EOS = "</s>"

WEATHER = {
    "function": {
        "description": "Current weather for a city.",
        "name": "get_weather",
        "parameters": {
            "properties": {
                "days": {"description": "How many days ahead.", "type": "integer"},
                "location": {"description": "City name, e.g. Seoul.", "type": "string"},
            },
            "required": ["location"],
            "type": "object",
        },
    },
    "type": "function",
}
TIME = {
    "function": {
        "description": "The time now.",
        "name": "get_time",
        "parameters": {"properties": {}, "type": "object"},
    },
    "type": "function",
}
NOTE = {
    "function": {
        "description": "Save a note.",
        "name": "save_note",
        "parameters": {
            "properties": {
                "filter": {"type": "object"},
                "text": {"type": "string"},
            },
            "type": "object",
        },
    },
    "type": "function",
}


def call(id_, name, arguments):
    return {"function": {"arguments": arguments, "name": name}, "id": id_, "type": "function"}


CASES = [
    ("system and user", {"messages": [
        {"content": "You are terse.", "role": "system"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("no system message, the default identity", {"messages": [
        {"content": "Hi", "role": "user"},
    ]}),
    ("two turns with a system prompt, the reasoning kept apart", {"messages": [
        {"content": "You are terse.", "role": "system"},
        {"content": "What is the capital of France?", "role": "user"},
        {"content": "Paris.", "reasoning_content": "The user asks a geography question.", "role": "assistant"},
        {"content": "And of Italy?", "role": "user"},
    ]}),
    ("a turn that wrote no reasoning closes the span empty", {"messages": [
        {"content": "What is the capital of France?", "role": "user"},
        {"content": "Paris.", "role": "assistant"},
        {"content": "And of Italy?", "role": "user"},
    ]}),
    ("think span in the answer, split at its tags", {"add_generation_prompt": False, "messages": [
        {"content": "2+2?", "role": "user"},
        {"content": "<mm:think>\nAdd them.\n</mm:think>\n\n4", "role": "assistant"},
    ]}),
    ("reasoning_content given as an empty string is not split out of the answer", {"add_generation_prompt": False, "messages": [
        {"content": "2+2?", "role": "user"},
        {"content": "<mm:think>Add them.</mm:think>4", "reasoning_content": "", "role": "assistant"},
    ]}),
    ("a developer turn first, sent as system", {"messages": [
        {"content": "Answer in Korean.", "role": "developer"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("a system turn in the middle renders as a developer turn", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "role": "assistant"},
        {"content": "From now on answer in French.", "role": "system"},
        {"content": "Again?", "role": "user"},
    ]}),
    ("a root turn carries the date and the location", {"messages": [
        {"content": "You are the house assistant.", "role": "root",
         "current_date": "2026-10-23", "current_location": "Seoul"},
        {"content": "You are terse.", "role": "system"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("a root turn with no content falls to the default identity", {"messages": [
        {"content": "", "role": "root", "current_date": "2026-10-23"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("reasoning_effort high", {"messages": [{"content": "Hi", "role": "user"}], "reasoning_effort": "high"}),
    ("reasoning_effort none is sent as low", {"messages": [{"content": "Hi", "role": "user"}],
                                              "reasoning_effort": "none", "_vars": {"reasoning_effort": "low"}}),
    ("the effort knob as a template kwarg", {"messages": [{"content": "Hi", "role": "user"}],
                                             "chat_template_kwargs": {"effort": "max"}}),
    ("content parts flattened to their text", {"messages": [
        {"content": [{"text": "Describe", "type": "text"}, {"text": "this.", "type": "text"}], "role": "user"},
    ]}),
    ("tools and a user turn", {"messages": [{"content": "Weather in Seoul?", "role": "user"}], "tools": [WEATHER, TIME]}),
    ("two calls and their results in one tool turn", {"messages": [
        {"content": "Weather and time in Seoul?", "role": "user"},
        {"content": "Checking both.", "role": "assistant", "tool_calls": [
            call("call_a", "get_weather", {"days": 3, "location": "Seoul"}),
            call("call_b", "get_time", {}),
        ]},
        {"content": "{\"temp\": 21}", "role": "tool", "tool_call_id": "call_a"},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_b"},
    ], "tools": [WEATHER, TIME]}),
    ("a call with no text, then a user turn", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": None, "role": "assistant", "tool_calls": [call("call_c", "get_time", {})]},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_c"},
        {"content": "Thanks.", "role": "user"},
    ], "tools": [TIME]}),
    ("non-ASCII, nested and null arguments", {"add_generation_prompt": False, "messages": [
        {"content": "Note it.", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [
            call("call_d", "save_note", {
                "filter": {"flag": True, "n": [1, 2.5], "none": None, "q": "서울 ☃"},
                "skipped": None,
                "text": "a<b & \"c\"",
            }),
        ]},
    ], "tools": [NOTE]}),
    ("string arguments", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [call("call_e", "get_time", "{}")]},
    ], "tools": [TIME]}),
    ("a tool result with no call before it is refused", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "12:00", "role": "tool"},
    ]}),
]


def variables(body):
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if m["role"] == "developer":
            m["role"] = "system"
        if isinstance(m.get("content"), list):
            m["content"] = "\n".join(
                p["text"] for p in m["content"] if p.get("type") == "text" and isinstance(p.get("text"), str)
            )
        msgs.append(m)
    agp = body.get("add_generation_prompt")
    v = {
        "messages": msgs,
        "add_generation_prompt": agp if isinstance(agp, bool) else True,
        "bos_token": BOS,
        "eos_token": EOS,
    }
    for k in ("tools", "reasoning_effort"):
        if body.get(k) is not None:
            v[k] = body[k]
    v.update(body.get("_vars", {}))
    if isinstance(body.get("chat_template_kwargs"), dict):
        v.update(body["chat_template_kwargs"])
    return v


def render(source, v):
    v = dict(v)
    messages = v.pop("messages")
    tools = v.pop("tools", None)
    add_generation_prompt = v.pop("add_generation_prompt")
    rendered, _ = render_jinja_template(
        [messages], tools=tools, chat_template=source, add_generation_prompt=add_generation_prompt, **v
    )
    return rendered[0]


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    cases = []
    for name, body in CASES:
        # The body as the test sends it: keys sorted at every level, the script's own `_vars` left out.
        body = json.loads(json.dumps(body, sort_keys=True))
        sent = {k: v for k, v in body.items() if k != "_vars"}
        case = {"name": name, "body": sent}
        try:
            case["prompt"] = render(source, variables(body))
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": "minimaxm31-chat-template.jinja, the checkpoint folder's chat_template.jinja (and the header's "
        "tokenizer.chat_template), byte for byte",
        "reference": f"transformers {transformers.__version__} render_jinja_template (jinja2 {jinja2.__version__}, "
        f"Python {platform.python_version()}), the Hugging Face chat-template environment; each case rendered with the "
        "variables bloomery-serve's /apply-template passes (see render_minimaxm31.py); a case jinja2 refused carries "
        "its error in place of a prompt. Script: crates/serve/tests/fixtures/render_minimaxm31.py, run as "
        "`python3 render_minimaxm31.py minimaxm31-chat-template.jinja > minimaxm31-renders.json`",
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
