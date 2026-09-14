"""One-shot planner for `omarchy-voice say`.

The daemon itself is speech-to-speech over the Realtime API. This module is
the typed equivalent: the same tools, the same policy gate, no microphone.
It talks to Chat Completions over HTTPS so a command can be tried without
opening a websocket.

Which endpoint answers is decided by `routing.planner`: the providers are
tried in order and the first one that returns a completion is kept for the
rest of the turn. A provider that cannot be reached, has no key, or answers
with an error hands the turn to the next one. See providers.py.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import capabilities, providers
from .config import Config
from .persona import PERSONA
from .providers import Provider
from .tools import TOOL_SCHEMAS, Executor, tools_for

CHAT_URL = providers.OPENAI_CHAT_URL


@dataclass
class Turn:
    """One request and everything that came of it."""
    text: str
    reply: str = ""
    actions: list[str] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0
    tokens: dict = field(default_factory=dict)
    # Which rung of the ladder answered, and what it cost to get there.
    provider: str = ""
    model: str = ""
    failovers: list[str] = field(default_factory=list)


def to_chat_tools(schemas: list[dict] | None = None) -> list[dict]:
    converted = []
    for schema in schemas if schemas is not None else TOOL_SCHEMAS:
        converted.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        })
    return converted


def _system_prompt(config=None) -> str:
    from .tasks import ROUTING
    from .vision import ROUTING as VISION_ROUTING
    return "\n\n".join([
        PERSONA,
        ROUTING if config is not None and config.tasks_enabled else "",
        VISION_ROUTING if config is not None and config.vision_enabled else "",
        capabilities.manifest(),
        "# The desktop right now\n\n" + capabilities.live_state(),
    ])


class PlannerUnavailable(RuntimeError):
    """Something the one-shot planner needs is missing."""


class Planner:
    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.executor = executor

    def think(self, text: str) -> Turn:
        turn = Turn(text=text)
        started = time.monotonic()
        try:
            turn.reply = self._loop(text, turn)
        except PlannerUnavailable as exc:
            turn.error = str(exc)
            turn.reply = "My planner isn't configured yet."
        except Exception as exc:  # a voice tool must not die on one bad turn
            turn.error = f"{type(exc).__name__}: {exc}"
            turn.reply = "Something went wrong with that."
        turn.elapsed = time.monotonic() - started
        return turn

    def _ladder(self) -> list[Provider]:
        try:
            ladder = providers.chat_ladder(self.config, openai_chat_url=CHAT_URL)
        except ValueError as exc:
            raise PlannerUnavailable(f"provider config: {exc}") from exc
        usable = [p for p in ladder if p.has_key()]
        if not usable:
            raise PlannerUnavailable(
                "no planner provider has a key — put one of "
                + ", ".join(p.api_key_env for p in ladder)
                + " in ~/.config/omarchy-voice/env")
        return usable

    def _loop(self, text: str, turn: Turn) -> str:
        ladder = self._ladder()
        messages: list[dict] = [
            {"role": "system", "content": _system_prompt(self.config)},
            {"role": "user", "content": text},
        ]
        tools = to_chat_tools(tools_for(self.config))
        reply = ""
        rung = 0

        for _ in range(self.config.max_turns):
            data, rung = _ask(messages, tools, ladder, rung, turn)
            usage = data.get("usage") or {}
            if usage:
                turn.tokens = {
                    "in": usage.get("prompt_tokens", 0),
                    "out": usage.get("completion_tokens", 0),
                }
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            said = (message.get("content") or "").strip()
            if said:
                reply = said
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return reply or "Done."

            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    outcome_text = f"ERROR: could not parse arguments: {exc}"
                else:
                    outcome = self.executor.call(name, args)
                    turn.actions.append(self.executor.describe(name, args))
                    outcome_text = outcome.as_tool_result()
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": outcome_text,
                })
            if self.executor.pending:
                return reply or "That needs confirmation."

        return reply or "Ran out of steps on that one."


def _ask(messages: list[dict], tools: list[dict], ladder: list[Provider],
         rung: int, turn: Turn) -> tuple[dict, int]:
    """One completion from the first provider at or after `rung` that answers.

    The rung is sticky within a turn: a provider that has been answering keeps
    the conversation, and only a failure moves it down the ladder. Moving down
    is safe mid-turn because every rung speaks the same wire format, so the
    tool calls already in `messages` mean the same thing to the next one.
    """
    failures: list[str] = []
    for index in range(rung, len(ladder)):
        provider = ladder[index]
        try:
            data = _chat(messages, tools, provider)
        except PlannerUnavailable as exc:
            failures.append(str(exc))
            following = ladder[index + 1].name if index + 1 < len(ladder) else ""
            if following:
                turn.failovers.append(f"{exc} — trying {following}")
            continue
        turn.provider, turn.model = provider.name, provider.chat_model
        return data, index
    raise PlannerUnavailable("every planner provider failed: " + "; ".join(failures))


def _chat(messages: list[dict], tools: list[dict], provider: Provider) -> dict:
    body = json.dumps({
        "model": provider.chat_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }).encode()
    request = urllib.request.Request(
        provider.chat_url,
        data=body,
        headers=provider.chat_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=provider.timeout_seconds) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise PlannerUnavailable(f"{provider.name}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PlannerUnavailable(f"{provider.name}: could not reach it: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PlannerUnavailable(f"{provider.name}: no answer within {provider.timeout_seconds:g}s") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlannerUnavailable(f"{provider.name}: answered with something that is not JSON") from exc
