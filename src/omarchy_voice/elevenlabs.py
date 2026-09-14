"""ElevenLabs Agents as a rung of `routing.realtime`.

The Agents websocket does the listening, the thinking and the talking on
ElevenLabs' side: their transcriber, their turn taking, an LLM of your choice
behind their prompt, their voice. What stays here is the same as with OpenAI
Realtime: the microphone, the speaker, and every desktop tool, which run
locally through `tools.Executor` and its policy. To the agent they are client
tools; a `client_tool_call` comes down the socket, the executor runs it, and a
`client_tool_result` goes back.

That needs an agent to exist on the ElevenLabs side with our tool schemas on
it, so there is a one-off step:

    omarchy-voice elevenlabs sync

which creates (or updates) an agent named AGENT_NAME with one client tool per
schema, records its id in AGENTS_FILE, and enables the per-session prompt
override the daemon relies on to send the capability manifest and the desktop
snapshot when a conversation opens. Run it again after upgrading; `doctor`
says when the agent is behind the installed tools.

This module owns the wire shapes and the REST calls. The session behaviour
around them (which event does what) lives in realtime.py next to the OpenAI
handling, because it shares all of the state.

Billing differs from OpenAI in one way that shapes the daemon: an open agent
conversation is charged per minute, silence included, where an idle Realtime
socket costs nothing. So the daemon opens the agent conversation when
listening is toggled on and closes it when toggled off, rather than holding a
socket for the life of the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .config import CONFIG_DIR
from .providers import Provider, USER_AGENT

AGENTS_FILE = CONFIG_DIR / "elevenlabs-agents.json"
AGENT_NAME = "OMA (omarchy-voice)"
# English agents must use the v2 turbo or flash model; v2.5 is refused for "en".
TTS_MODEL = "eleven_flash_v2"
# Client tools may wait at most this long for their result. compose_windows
# can take 32 s; a shell command longer than this is a task worker's job.
TOOL_TIMEOUT_SECONDS = 120
# ElevenLabs closes a conversation at this age. Rotation in realtime.py
# retires the socket between turns well before it, same as for OpenAI.
MAX_CONVERSATION_SECONDS = 3600
# The agent speaks up on its own after this much user silence, and 30 is the
# most the platform allows. The daemon sends `user_activity` while listening
# so the timer never fires; this is the backstop if it stops doing so.
TURN_TIMEOUT_SECONDS = 30
ACTIVITY_INTERVAL_SECONDS = 10.0
# Events the agent sends the client. Anything not listed is never sent, so
# the ones the session handles must be here.
CLIENT_EVENTS = [
    "conversation_initiation_metadata", "ping", "audio", "interruption",
    "user_transcript", "tentative_user_transcript", "agent_response",
    "agent_response_correction", "agent_response_complete", "client_tool_call",
    "client_error",
]
REQUEST_TIMEOUT_SECONDS = 30.0

# Added to the agent's stored prompt. The session override carries the full
# persona; this is what the agent says if the override is ever refused.
STORED_PROMPT = (
    "You are OMA, the voice control layer of an Omarchy Linux desktop. Act "
    "through the tools you have been given; say in one short sentence what "
    "happened. If the user has gone quiet, stay quiet: never fill silence with "
    "a check-in question."
)


class ApiError(RuntimeError):
    """A REST call that came back with an error status."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


# --- wire shapes --------------------------------------------------------------

def hello(provider: Provider, instructions: str) -> dict:
    """The first message on a conversation: prompt and voice for this session."""
    override: dict = {
        "agent": {"prompt": {"prompt": instructions}, "first_message": ""},
    }
    if provider.realtime_voice:
        override["tts"] = {"voice_id": provider.realtime_voice}
    return {
        "type": "conversation_initiation_client_data",
        "conversation_config_override": override,
        "source_info": {"source": "unknown", "version": USER_AGENT},
    }


def audio_chunk(b64: str) -> dict:
    return {"user_audio_chunk": b64}


def user_message(text: str) -> dict:
    return {"type": "user_message", "text": text}


def contextual_update(text: str) -> dict:
    return {"type": "contextual_update", "text": text}


def user_activity() -> dict:
    return {"type": "user_activity"}


def pong(event_id: int) -> dict:
    return {"type": "pong", "event_id": event_id}


def tool_result(call_id: str, output: str) -> dict:
    return {
        "type": "client_tool_result",
        "tool_call_id": call_id,
        "result": output,
        "is_error": output.startswith("ERROR:"),
    }


def audio_rate(fmt: str) -> int | None:
    """`pcm_24000` -> 24000. Anything that is not raw PCM is None."""
    prefix, _, rate = (fmt or "").partition("_")
    if prefix != "pcm" or not rate.isdigit():
        return None
    return int(rate)


# --- tool schemas -------------------------------------------------------------

def _property(schema: dict, name: str = "") -> dict:
    """One JSON-schema property in the subset the tools API accepts.

    Bounds and `additionalProperties` have no equivalent there; the bounds go
    into the description so the model still sees them. The API refuses a
    property with no description at all, so a bare one is named after itself.
    """
    kind = schema.get("type", "string")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    description = schema.get("description", "") or f"The {name.replace('_', ' ') or 'value'}."
    bounds = [f"{word} {schema[key]}" for key, word in (("minimum", "at least"), ("maximum", "at most"))
              if key in schema]
    if bounds:
        description = f"{description} ({', '.join(bounds)})".strip()
    out: dict = {"type": kind, "description": description}
    if kind == "object":
        out["properties"] = {key: _property(sub, key) for key, sub in (schema.get("properties") or {}).items()}
        out["required"] = list(schema.get("required") or [])
    elif kind == "array":
        out["items"] = _property(schema.get("items") or {"type": "string"}, f"{name} item")
    elif schema.get("enum"):
        if kind == "string":
            out["enum"] = [str(v) for v in schema["enum"]]
        else:
            out["description"] = f"{description} (one of: {', '.join(str(v) for v in schema['enum'])})".strip()
    return out


def parameters(schema: dict) -> dict:
    top = _property({**schema, "type": "object"})
    top.pop("description", None)
    return top


def tool_configs(schemas: list[dict]) -> list[dict]:
    """Realtime-format function schemas -> client tool configs."""
    return [{
        "type": "client",
        "name": schema["name"],
        "description": schema["description"],
        "expects_response": True,
        "response_timeout_secs": TOOL_TIMEOUT_SECONDS,
        "parameters": parameters(schema.get("parameters") or {"type": "object", "properties": {}}),
    } for schema in schemas]


def agent_config(*, rate: int, llm: str, voice_id: str, tool_ids: list[str]) -> dict:
    """The agent as `sync` wants it. The prompt here is the fallback only."""
    tts: dict = {"model_id": TTS_MODEL, "agent_output_audio_format": f"pcm_{rate}"}
    if voice_id:
        tts["voice_id"] = voice_id
    return {
        "name": AGENT_NAME,
        "conversation_config": {
            "asr": {"user_input_audio_format": f"pcm_{rate}"},
            "turn": {"turn_timeout": TURN_TIMEOUT_SECONDS},
            "tts": tts,
            "conversation": {
                "max_duration_seconds": MAX_CONVERSATION_SECONDS,
                "client_events": list(CLIENT_EVENTS),
            },
            "agent": {
                "first_message": "",
                "language": "en",
                "prompt": {"prompt": STORED_PROMPT, "llm": llm, "tool_ids": list(tool_ids)},
            },
        },
        "platform_settings": {
            "overrides": {
                "conversation_config_override": {
                    "agent": {"first_message": True, "prompt": {"prompt": True}},
                    "tts": {"voice_id": True},
                },
            },
        },
    }


def fingerprint(*, rate: int, llm: str, voice_id: str, tools: list[dict]) -> str:
    """What `sync` would produce; changes when the agent needs syncing again."""
    payload = json.dumps({"rate": rate, "llm": llm, "voice": voice_id, "tools": tools}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --- what sync recorded ---------------------------------------------------------

def load_state(path: Path | None = None) -> dict:
    try:
        data = json.loads((path or AGENTS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict, path: Path | None = None) -> None:
    path = path or AGENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    tmp.replace(path)


def agent_id_for(provider: Provider, state: dict | None = None) -> str:
    """The agent this rung talks to: the profile's own, else what sync made."""
    if provider.agent_id:
        return provider.agent_id
    record = (state if state is not None else load_state()).get(provider.name) or {}
    return str(record.get("agent_id") or "")


def stale(provider: Provider, current: str, state: dict | None = None) -> bool:
    """True when the installed tools differ from what the agent was synced with."""
    if provider.agent_id:
        return False  # somebody else's agent; nothing to compare against
    record = (state if state is not None else load_state()).get(provider.name) or {}
    return bool(record.get("agent_id")) and record.get("fingerprint") != current


# --- REST ---------------------------------------------------------------------

class Api:
    """The few calls sync and the daemon need, on urllib so nothing is added."""

    def __init__(self, provider: Provider, timeout: float = REQUEST_TIMEOUT_SECONDS):
        self.base = provider.api_url.rstrip("/")
        self.key = provider.key()
        self.timeout = timeout
        self.extra = dict(provider.headers)

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"xi-api-key": self.key, "User-Agent": USER_AGENT, **self.extra}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ApiError(exc.code, detail or exc.reason) from None
        except urllib.error.URLError as exc:
            raise ApiError(0, str(exc.reason)) from None
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            raise ApiError(0, "response was not JSON") from None

    def signed_url(self, agent_id: str) -> str:
        data = self.request("GET", f"/v1/convai/conversation/get-signed-url?agent_id={agent_id}")
        url = data.get("signed_url")
        if not isinstance(url, str) or not url.startswith("wss://"):
            raise ApiError(0, "no signed_url in the response")
        return url

    def find_tools(self, names: set[str]) -> dict[str, str]:
        """Name -> id for existing client tools whose names we are about to use."""
        found: dict[str, str] = {}
        cursor = ""
        while True:
            query = "?types=client&page_size=100" + (f"&cursor={cursor}" if cursor else "")
            data = self.request("GET", "/v1/convai/tools" + query)
            for tool in data.get("tools") or []:
                name = (tool.get("tool_config") or {}).get("name")
                if name in names and tool.get("id"):
                    found[name] = tool["id"]
            cursor = data.get("next_cursor") or ""
            if not data.get("has_more") or not cursor:
                return found

    def upsert_tool(self, config: dict, tool_id: str) -> str:
        if tool_id:
            try:
                self.request("PATCH", f"/v1/convai/tools/{tool_id}", {"tool_config": config})
                return tool_id
            except ApiError as exc:
                if exc.status != 404:
                    raise
        return str(self.request("POST", "/v1/convai/tools", {"tool_config": config})["id"])

    def upsert_agent(self, config: dict, agent_id: str) -> str:
        if agent_id:
            try:
                self.request("PATCH", f"/v1/convai/agents/{agent_id}", config)
                return agent_id
            except ApiError as exc:
                if exc.status != 404:
                    raise
        return str(self.request("POST", "/v1/convai/agents/create", config)["agent_id"])


def signed_url(provider: Provider, agent_id: str) -> str:
    return Api(provider).signed_url(agent_id)


def sync(provider: Provider, *, rate: int, schemas: list[dict], api: Api | None = None,
         state_path: Path | None = None) -> dict:
    """Create or update the agent and its client tools; record what was made.

    Idempotent: tool ids from the last sync are updated in place, tools that
    exist by name from an earlier install are adopted, and the agent id is
    kept. Returns the record written for this provider.
    """
    api = api or Api(provider)
    state = load_state(state_path)
    record = dict(state.get(provider.name) or {})
    known: dict[str, str] = dict(record.get("tools") or {})
    configs = tool_configs(schemas)
    wanted = {c["name"] for c in configs}
    missing = wanted - set(known)
    if missing:
        known.update(api.find_tools(missing))

    tools: dict[str, str] = {}
    for config in configs:
        tools[config["name"]] = api.upsert_tool(config, known.get(config["name"], ""))
    ordered = [tools[c["name"]] for c in configs]

    agent = agent_config(rate=rate, llm=provider.realtime_model,
                         voice_id=provider.realtime_voice, tool_ids=ordered)
    agent_id = api.upsert_agent(agent, provider.agent_id or str(record.get("agent_id") or ""))

    record = {
        "agent_id": agent_id,
        "tools": tools,
        "fingerprint": fingerprint(rate=rate, llm=provider.realtime_model,
                                   voice_id=provider.realtime_voice, tools=configs),
    }
    state[provider.name] = record
    save_state(state, state_path)
    return record
