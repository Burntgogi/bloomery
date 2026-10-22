"""Reference renders of the Raon-1-mini chat template for raon-renders.json.

usage (jinja2 3.1.x; the chat template is the file's own, `tokenizer.chat_template`, byte for byte the
`chat_template.jinja` of the checkpoint):
    python3 render_raon.py raon-chat-template.jinja > raon-renders.json

The environment is Hugging Face's `apply_chat_template` one: an immutable sandboxed environment with
trim_blocks, lstrip_blocks, the loopcontrols extension and the `generation` tag (transformers'
AssistantTracker: the block renders its body); `tojson` is json.dumps with ensure_ascii=False, indent=None,
separators=None and sort_keys=False as defaults; the global raise_exception.

Each case's `vars` are the variables the template renders with. A case jinja2 refused carries its error in
place of a prompt.
"""
import json
import sys

import jinja2
import jinja2.ext as jinja2_ext
from jinja2 import nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Generation(jinja2_ext.Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_generation", []), [], [], body).set_lineno(lineno)

    def _generation(self, caller):
        return caller()


env = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[jinja2_ext.loopcontrols, Generation],
)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
)
env.globals["raise_exception"] = raise_exception

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 서울."},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {"name": "get_time", "description": "The time now.", "parameters": {"type": "object", "properties": {}}},
}
HI = [{"role": "user", "content": "Hi"}]


def call(name, arguments):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}


THINKING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!", "reasoning_content": "The user greets.\n"},
    {"role": "user", "content": "Again?"},
    {"role": "assistant", "content": "Hello again.", "reasoning_content": "The user repeats."},
    {"role": "user", "content": "One more."},
]

CASES = [
    ("a user turn", {"messages": HI, "add_generation_prompt": True}),
    ("system and user, thinking off", {"messages": [
        {"role": "system", "content": "  You are terse.\n"},
        {"role": "user", "content": "Hi"},
    ], "add_generation_prompt": True, "enable_thinking": False}),
    ("no generation prompt", {"messages": HI, "add_generation_prompt": False}),
    ("thinking kept in every turn", {"messages": THINKING, "add_generation_prompt": True}),
    ("history thinking truncated", {"messages": THINKING, "add_generation_prompt": True, "truncate_history_thinking": True}),
    ("thinking inside the content", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nGreet back.\n</think>\n\nHello!"},
        {"role": "user", "content": "Bye"},
    ], "add_generation_prompt": True}),
    ("an assistant turn with no loss", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet.", "loss": False},
    ], "add_generation_prompt": False}),
    ("content parts", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "No lists."}]},
        {"role": "user", "content": [{"type": "text", "text": "서울 날씨? ☃ \"quoted\" <b>"}]},
    ], "add_generation_prompt": True}),
    ("tools and a user turn", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Seoul?"},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": True}),
    ("a tool call and its answer", {"messages": [
        {"role": "user", "content": "Weather in Seoul?"},
        {"role": "assistant", "content": "Checking.", "reasoning_content": "Look it up.", "tool_calls": [
            call("get_weather", {"location": "서울", "days": 3, "detail": {"hourly": True, "n": [1, 2.5, None]}}),
        ]},
        {"role": "tool", "content": "{\"temp\": 21}"},
        {"role": "assistant", "content": "21 degrees."},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": False}),
    ("two calls, two answers, then a user turn", {"messages": [
        {"role": "user", "content": "Weather and time in Seoul?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"location": "Seoul"}),
            {"type": "function", "function": {"name": "get_time", "arguments": {}}},
        ]},
        {"role": "tool", "content": "21"},
        {"role": "tool", "content": "12:00"},
        {"role": "user", "content": "Thanks."},
    ], "tools": [WEATHER, TIME], "add_generation_prompt": True}),
    ("a call with no arguments key", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_time"}}]},
    ], "tools": [TIME], "add_generation_prompt": False}),
    ("a string argument value", {"messages": [
        {"role": "user", "content": "Note."},
        {"role": "assistant", "content": "", "tool_calls": [call("save", {"text": "a\nb", "n": 1, "ok": False})]},
    ], "add_generation_prompt": False}),
    ("an image part is refused", {"messages": [
        {"role": "user", "content": [{"type": "image", "image": "x"}]},
    ], "add_generation_prompt": True}),
    ("no messages is refused", {"messages": [], "add_generation_prompt": True}),
    ("a late system message is refused", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Late."},
    ], "add_generation_prompt": True}),
    ("only tool responses is refused", {"messages": [
        {"role": "tool", "content": "<tool_response>x</tool_response>"},
    ], "add_generation_prompt": True}),
    ("an unknown role is refused", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "narrator", "content": "..."},
    ], "add_generation_prompt": True}),
]


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    template = env.from_string(source)
    cases = []
    for name, vars in CASES:
        case = {"name": name, "vars": vars}
        try:
            case["prompt"] = template.render(**vars)
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": "raon-chat-template.jinja",
        "reference": (
            f"jinja2 {jinja2.__version__} (Python {sys.version.split()[0]}), the Hugging Face chat-template "
            "environment (ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks, the loopcontrols "
            "extension and the generation tag; tojson = json.dumps with ensure_ascii=False, indent=None, "
            "separators=None, sort_keys=False as defaults; the global raise_exception); a case jinja2 refused "
            "carries its error in place of a prompt. Script: crates/jinja/tests/fixtures/render_raon.py, run as "
            "`python3 render_raon.py raon-chat-template.jinja > raon-renders.json`"
        ),
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
