"""Provider profiles and the routing ladders that pick between them.

One profile per vendor. A profile can carry a Chat Completions endpoint (used
by the typed planner behind `omarchy-voice say`), a Realtime websocket endpoint
(used by the listening daemon), or both. `[routing]` in config.toml names which
profiles to try and in what order: the first one that answers is used, the rest
are failover.

    [routing]
    planner  = ["openai", "openrouter"]
    realtime = ["openai"]

    [providers.openrouter]
    api_key_env = "OPENROUTER_API_KEY"
    chat_url    = "https://openrouter.ai/api/v1/chat/completions"
    chat_model  = "openai/gpt-4.1-mini"
    cost        = "see openrouter.ai/models"

The built-in `openai` profile is assembled from the classic `[openai]` and
`[realtime]` settings, so a config file that predates this module keeps
working unchanged. Every endpoint must speak the OpenAI wire format: Chat
Completions with function tools for the planner, the Realtime GA protocol
for the daemon. Proxies such as LiteLLM and OpenRouter qualify; a vendor with
its own protocol does not, however good its models are.

The one exception is `protocol = "elevenlabs"`: the daemon then talks the
ElevenLabs Agents websocket instead, with the desktop tools registered on the
agent as client tools. A built-in `elevenlabs` profile carries the defaults;
`omarchy-voice elevenlabs sync` creates the agent. See elevenlabs.py.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from . import __version__
from .config import Config

# Groq's edge refuses the stock Python-urllib agent outright (HTTP 403, error
# code 1010), so every request names this program instead. A profile's own
# headers still win.
USER_AGENT = f"omarchy-voice/{__version__}"

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
ELEVENLABS_REALTIME_URL = "wss://api.elevenlabs.io/v1/convai/conversation"
ELEVENLABS_API_URL = "https://api.elevenlabs.io"
# The LLM ElevenLabs runs behind the agent. Same default as the typed planner.
ELEVENLABS_LLM = "gpt-4.1"

# How a realtime endpoint is spoken to. `openai` is the Realtime GA protocol;
# `elevenlabs` is the Agents websocket, which needs an agent created first.
PROTOCOLS = ("openai", "elevenlabs")

# Everything a `[providers.<name>]` table may contain. Anything else is a typo,
# and a typo in a provider table is worth stopping on: a misspelt `chat_url`
# silently leaves the built-in endpoint in place and bills the wrong account.
PROFILE_KEYS = {
    "api_key_env": str,
    "protocol": str,
    # ElevenLabs only: the agent to converse with, and the REST base that
    # signs the websocket URL and hosts the agent. Empty agent_id means "the
    # one `omarchy-voice elevenlabs sync` created", recorded in CONFIG_DIR.
    "agent_id": str,
    "api_url": str,
    "chat_url": str,
    "chat_model": str,
    "realtime_url": str,
    "realtime_model": str,
    "realtime_voice": str,
    "realtime_transcribe_model": str,
    "headers": dict,
    "timeout_seconds": (int, float),
    # Descriptive only. Nothing routes on these; they are printed by `doctor`
    # so the person choosing the ladder can see what each rung costs.
    "duplex": str,
    "latency": str,
    "cost": str,
    "notes": str,
}

# Hostname suffixes that count as private network, where a plain ws:// or
# http:// endpoint is acceptable: the traffic never leaves the LAN or tailnet.
PRIVATE_SUFFIXES = (".local", ".lan", ".home.arpa", ".ts.net", ".internal")


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str = ""
    protocol: str = "openai"
    agent_id: str = ""
    api_url: str = ""
    chat_url: str = ""
    chat_model: str = ""
    realtime_url: str = ""
    realtime_model: str = ""
    realtime_voice: str = ""
    # None means "inherit": the OpenAI transcriber for the openai profile and
    # nothing for everyone else, since the model id is OpenAI's.
    realtime_transcribe_model: str | None = None
    headers: dict = field(default_factory=dict)
    timeout_seconds: float = 60.0
    duplex: str = ""
    latency: str = ""
    cost: str = ""
    notes: str = ""

    @property
    def is_openai(self) -> bool:
        return any(_host(url).endswith("openai.com")
                   for url in (self.chat_url, self.realtime_url) if url)

    @property
    def is_elevenlabs(self) -> bool:
        return self.protocol == "elevenlabs"

    def key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    def has_key(self) -> bool:
        """True when the profile can authenticate, or needs no key at all."""
        return not self.api_key_env or bool(os.environ.get(self.api_key_env))

    def chat_headers(self) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT, **self.headers}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {self.key()}"
        return headers

    def realtime_headers(self, safety_identifier: str = "") -> dict:
        headers = {"User-Agent": USER_AGENT, **self.headers}
        # ElevenLabs authenticates the websocket with a token inside a signed
        # URL fetched over REST; the key itself never goes on the socket.
        if self.api_key_env and not self.is_elevenlabs:
            headers["Authorization"] = f"Bearer {self.key()}"
        # The safety identifier is an OpenAI account feature. Sending it to a
        # third party would only tell them something about this install.
        if safety_identifier and _host(self.realtime_url).endswith("openai.com"):
            headers["OpenAI-Safety-Identifier"] = safety_identifier
        return headers

    def summary(self, which: str) -> str:
        """One line for `doctor`: model, host, and whatever metadata was given."""
        if which == "chat":
            model, url = self.chat_model, self.chat_url
        else:
            model, url = self.realtime_model, self.realtime_url
        parts = [f"{model} via {_host(url)}"]
        if which != "chat" and self.is_elevenlabs:
            parts[0] = f"ElevenLabs agent, {model} behind it, via {_host(url)}"
        for label in ("duplex", "latency", "cost", "notes"):
            value = getattr(self, label)
            if value:
                parts.append(f"{label} {value}")
        return "; ".join(parts)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_private_host(host: str) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        # Tailscale hands out 100.64/10, which the stdlib does not count as
        # private, and a tailnet is exactly where a plain ws:// proxy lives.
        return (address.is_private or address.is_loopback
                or address in ipaddress.ip_network("100.64.0.0/10"))
    if host in ("localhost",) or "." not in host:
        return True
    return host.endswith(PRIVATE_SUFFIXES)


def _check_url(name: str, key: str, url: str, secure: str, plain: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in (secure, plain) or not parts.hostname:
        raise ValueError(f"providers.{name}.{key} must be a {secure}:// URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(f"providers.{name}.{key} cannot carry credentials, a query, or a fragment")
    if any(c.isspace() for c in url):
        raise ValueError(f"providers.{name}.{key} cannot contain whitespace")
    if parts.scheme == plain and not _is_private_host(parts.hostname):
        raise ValueError(
            f"providers.{name}.{key} uses {plain}:// to a public host; "
            f"use {secure}:// unless the endpoint is on the LAN or tailnet")


def _builtin_openai(config: Config, chat_url: str, realtime_url: str) -> dict:
    return {
        "api_key_env": config.api_key_env,
        "chat_url": chat_url,
        "chat_model": config.planner_model,
        "realtime_url": realtime_url,
        "realtime_model": config.realtime_model,
        "realtime_voice": config.realtime_voice,
        "realtime_transcribe_model": config.realtime_transcribe_model,
    }


def _builtin_elevenlabs() -> dict:
    return {
        "api_key_env": "ELEVENLABS_API_KEY",
        "protocol": "elevenlabs",
        "api_url": ELEVENLABS_API_URL,
        "realtime_url": ELEVENLABS_REALTIME_URL,
        "realtime_model": ELEVENLABS_LLM,
        "duplex": "full, server-side turn taking",
        "cost": "per minute plus the LLM behind the agent, see elevenlabs.io/pricing",
    }


def profiles(config: Config, *, openai_chat_url: str = OPENAI_CHAT_URL,
             openai_realtime_url: str = OPENAI_REALTIME_URL) -> dict[str, Provider]:
    """Every provider this config knows about, built-in openai included.

    Raises ValueError for a malformed table. The message names the key, never
    the value, so it is safe to print.
    """
    tables: dict[str, dict] = {
        "openai": _builtin_openai(config, openai_chat_url, openai_realtime_url),
        "elevenlabs": _builtin_elevenlabs(),
    }
    for name, table in (config.providers or {}).items():
        if not isinstance(table, dict):
            raise ValueError(f"providers.{name} must be a table")
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"provider name {name!r} may only contain letters, digits, - and _")
        unknown = sorted(set(table) - set(PROFILE_KEYS))
        if unknown:
            raise ValueError(f"providers.{name} has unknown keys: {', '.join(unknown)}")
        for key, kind in PROFILE_KEYS.items():
            if key in table and not isinstance(table[key], kind):
                raise ValueError(f"providers.{name}.{key} has the wrong type")
        base = tables.get(name, {})
        tables[name] = {**base, **table}

    result: dict[str, Provider] = {}
    for name, table in tables.items():
        provider = Provider(name=name, **table)
        if provider.chat_url:
            _check_url(name, "chat_url", provider.chat_url, "https", "http")
        if provider.realtime_url:
            _check_url(name, "realtime_url", provider.realtime_url, "wss", "ws")
        if provider.api_key_env and not provider.api_key_env.replace("_", "").isalnum():
            raise ValueError(f"providers.{name}.api_key_env must name an environment variable")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in provider.headers.items()):
            raise ValueError(f"providers.{name}.headers must map strings to strings")
        if provider.protocol not in PROTOCOLS:
            raise ValueError(f"providers.{name}.protocol must be one of: {', '.join(PROTOCOLS)}")
        if provider.agent_id and not provider.agent_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"providers.{name}.agent_id does not look like an agent id")
        if provider.api_url:
            _check_url(name, "api_url", provider.api_url, "https", "http")
        if provider.is_elevenlabs:
            if not provider.api_url:
                provider = Provider(**{**provider.__dict__, "api_url": ELEVENLABS_API_URL})
            if not provider.realtime_url:
                provider = Provider(**{**provider.__dict__, "realtime_url": ELEVENLABS_REALTIME_URL})
            if not provider.realtime_model:
                provider = Provider(**{**provider.__dict__, "realtime_model": ELEVENLABS_LLM})
            # The voice is an ElevenLabs voice id, not an OpenAI voice name, so
            # the [realtime] voice must not leak in; empty means the agent's own.
            # The transcriber is ElevenLabs' too, so nothing to inherit there.
            provider = Provider(**{**provider.__dict__, "realtime_transcribe_model": ""})
        elif provider.agent_id or provider.api_url:
            raise ValueError(f"providers.{name} has agent_id or api_url but protocol is not elevenlabs")
        if provider.realtime_url and not provider.realtime_voice and not provider.is_elevenlabs:
            provider = Provider(**{**provider.__dict__, "realtime_voice": config.realtime_voice})
        if provider.realtime_url and provider.realtime_transcribe_model is None:
            inherited = config.realtime_transcribe_model if name == "openai" else ""
            provider = Provider(**{**provider.__dict__, "realtime_transcribe_model": inherited})
        result[name] = provider
    return result


def _ladder(config: Config, names: list[str], which: str, known: dict[str, Provider]) -> list[Provider]:
    if not names:
        raise ValueError(f"routing.{which} is empty; name at least one provider")
    seen: list[Provider] = []
    for name in names:
        if not isinstance(name, str) or name not in known:
            raise ValueError(
                f"routing.{which} names unknown provider {name!r}; "
                f"known: {', '.join(sorted(known))}")
        provider = known[name]
        url = provider.chat_url if which == "planner" else provider.realtime_url
        model = provider.chat_model if which == "planner" else provider.realtime_model
        kind = "chat" if which == "planner" else "realtime"
        if not url:
            raise ValueError(f"providers.{name} has no {kind}_url, so it cannot be in routing.{which}")
        if not model:
            raise ValueError(f"providers.{name} needs a {kind}_model to be in routing.{which}")
        if provider not in seen:
            seen.append(provider)
    return seen


def chat_ladder(config: Config, *, openai_chat_url: str = OPENAI_CHAT_URL) -> list[Provider]:
    """Providers to try for the typed planner, in order."""
    known = profiles(config, openai_chat_url=openai_chat_url)
    return _ladder(config, list(config.routing_planner), "planner", known)


def realtime_ladder(config: Config, *, openai_realtime_url: str = OPENAI_REALTIME_URL) -> list[Provider]:
    """Providers to try for the speech daemon, in order."""
    known = profiles(config, openai_realtime_url=openai_realtime_url)
    return _ladder(config, list(config.routing_realtime), "realtime", known)


def missing_keys(ladder: list[Provider]) -> list[str]:
    """Problems, one per rung, when no rung on the ladder can authenticate."""
    if any(p.has_key() for p in ladder):
        return []
    return [f"{p.api_key_env} is not set (provider {p.name})" for p in ladder]
