"""Parsing and rendering of Muse Glimmer's ATEM channel format.

The model speaks in channel-scoped messages. A turn looks like:

    <|start|>assistant to=self<|message|>thinking...<|eom|>
    <|start|>assistant to=molbench.dock_and_score<|message|>
    <atem:function_calls>
    <atem:invoke name="molbench.dock_and_score">
    <atem:parameter name="smiles">CCO</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eom|>
    <|start|>assistant to=user<|message|>final answer<|eot|>

and a tool result comes back as:

    <|start|>tool NAME<|message|><tool_output name="NAME">...</tool_output><|eot|>

This module deliberately reimplements the format instead of importing vLLM's
parser, so the environment runs anywhere, including a laptop with no CUDA. The
tests pin it against the shapes the shipped chat template produces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

START = "<|start|>"
MESSAGE = "<|message|>"
END_TURN = "<|eot|>"
END_MESSAGE = "<|eom|>"

_SEGMENT = re.compile(
    r"^(?P<role>assistant|tool)"
    r"(?:\s+to=(?P<recipient>[^\s<]+))?"
    r"(?P<tool_name>\s+[^\s<]+)?"
    r"\s*" + re.escape(MESSAGE) + r"(?P<body>.*)$",
    re.DOTALL,
)
_INVOKE = re.compile(
    r"<atem:invoke\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<params>.*?)</atem:invoke>",
    re.DOTALL,
)
_PARAM = re.compile(
    r"<atem:parameter\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


@dataclass
class ToolCall:
    """One `<atem:invoke>` block the model emitted."""

    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ParsedTurn:
    """Everything one assistant turn produced, split by channel."""

    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str | None = None
    malformed: list[str] = field(default_factory=list)
    ended_turn: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def _coerce(value: str) -> Any:
    """ATEM parameters arrive as text; recover JSON values where they are one.

    Booleans, numbers, lists and objects all travel as their JSON spelling, so
    a successful parse is the honest reading. Anything else stays a string,
    which is what SMILES and SMARTS need.
    """
    text = value.strip()
    if text == "":
        return ""
    if text in ("true", "false"):
        return text == "true"
    if text == "null":
        return None
    if text[0] in "[{" or re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_tool_calls(body: str) -> tuple[list[ToolCall], list[str]]:
    """Extract every `<atem:invoke>` in one message body."""
    calls: list[ToolCall] = []
    problems: list[str] = []
    for invoke in _INVOKE.finditer(body):
        arguments = {
            param.group("name"): _coerce(param.group("value"))
            for param in _PARAM.finditer(invoke.group("params"))
        }
        calls.append(
            ToolCall(
                name=invoke.group("name"), arguments=arguments, raw=invoke.group(0)
            )
        )
    if not calls and "<atem:invoke" in body:
        problems.append("an <atem:invoke> block was opened but never closed")
    return calls, problems


def parse_assistant_text(text: str) -> ParsedTurn:
    """Split raw generated text into reasoning, tool calls and a final answer.

    The generation prompt already ends with `<|start|>assistant`, so the first
    channel header usually arrives without it; that case is normalized here.
    """
    turn = ParsedTurn()
    stream = text if text.lstrip().startswith(START) else START + "assistant" + text

    segments = [chunk for chunk in stream.split(START) if chunk.strip()]
    for segment in segments:
        match = _SEGMENT.match(segment.strip())
        if match is None:
            turn.malformed.append(segment.strip()[:200])
            continue

        body = match.group("body")
        for terminator in (END_TURN, END_MESSAGE):
            if terminator in body:
                if terminator == END_TURN:
                    turn.ended_turn = True
                body = body.split(terminator)[0]
                break

        recipient = (match.group("recipient") or "user").strip()
        if match.group("role") == "tool":
            continue  # tool results are environment-authored, not model output

        if recipient == "self":
            turn.reasoning = (turn.reasoning + "\n" + body.strip()).strip()
        elif recipient == "user":
            turn.final_text = body.strip()
            turn.ended_turn = True
        else:
            calls, problems = parse_tool_calls(body)
            turn.tool_calls.extend(calls)
            turn.malformed.extend(problems)
            if not calls and not problems:
                turn.malformed.append(
                    f"message addressed to {recipient} carried no tool call"
                )
    return turn


def render_tool_result(name: str, content: str, *, max_chars: int | None = None) -> str:
    """The environment's reply to one tool call, in the format the model expects."""
    text = content
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated, {len(content)} chars total]"
    return f'{START}tool {name}{MESSAGE}<tool_output name="{name}">\n{text}\n</tool_output>{END_TURN}'


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a call the way the model would, for building SFT traces."""
    lines = [
        f"{START}assistant to={name}{MESSAGE}<atem:function_calls>",
        f'<atem:invoke name="{name}">',
    ]
    for key, value in arguments.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f'<atem:parameter name="{key}">{rendered}</atem:parameter>')
    lines.append("</atem:invoke>")
    lines.append("</atem:function_calls>")
    return "\n".join(lines) + END_MESSAGE


def render_reasoning(text: str) -> str:
    return f"{START}assistant to=self{MESSAGE}{text}{END_MESSAGE}"


def render_final(text: str) -> str:
    return f"{START}assistant to=user{MESSAGE}{text}{END_TURN}"


def render_continuation(
    reasoning: str | None = None,
    tool_call: tuple[str, dict[str, Any]] | None = None,
    final: str | None = None,
) -> str:
    """One assistant turn in the form a policy actually emits.

    The generation prompt already ends with `<|start|>assistant`, so the first
    channel header arrives without it. Scripted policies and SFT traces have to
    match that, or the leading header is counted as a malformed segment.
    """
    parts: list[str] = []
    if reasoning is not None:
        parts.append(render_reasoning(reasoning))
    if tool_call is not None:
        parts.append(render_tool_call(tool_call[0], tool_call[1]))
    if final is not None:
        parts.append(render_final(final))
    text = "".join(parts)
    prefix = START + "assistant"
    return text.removeprefix(prefix)
