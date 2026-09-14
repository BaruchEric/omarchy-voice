"""Provider profiles and routing: defaults, validation, planner failover.

Run with: python3 -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import cli, config as cfg, planner, providers
from omarchy_voice.config import Config


def write(test, text: str) -> Path:
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    path = Path(tmp.name) / "config.toml"
    path.write_text(text)
    return path


class ProfileTests(unittest.TestCase):
    def test_builtin_openai_mirrors_the_classic_settings(self):
        config = Config(planner_model="gpt-x", realtime_model="rt-x", realtime_voice="v",
                        realtime_transcribe_model="t", api_key_env="K")
        openai = providers.profiles(config)["openai"]
        self.assertEqual(openai.chat_url, providers.OPENAI_CHAT_URL)
        self.assertEqual(openai.chat_model, "gpt-x")
        self.assertEqual(openai.realtime_url, providers.OPENAI_REALTIME_URL)
        self.assertEqual(openai.realtime_model, "rt-x")
        self.assertEqual(openai.realtime_voice, "v")
        self.assertEqual(openai.realtime_transcribe_model, "t")
        self.assertEqual(openai.api_key_env, "K")

    def test_a_config_without_routing_keeps_the_old_behaviour(self):
        self.assertEqual([p.name for p in providers.chat_ladder(Config())], ["openai"])
        self.assertEqual([p.name for p in providers.realtime_ladder(Config())], ["openai"])

    def test_provider_tables_load_from_toml(self):
        path = write(self, '''
[routing]
planner = ["openrouter", "openai"]
realtime = ["proxy", "openai"]

[providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"
chat_url = "https://openrouter.ai/api/v1/chat/completions"
chat_model = "openai/gpt-4.1-mini"
headers = { "HTTP-Referer" = "https://example.invalid" }
cost = "cheap"

[providers.proxy]
api_key_env = ""
realtime_url = "ws://127.0.0.1:4000/v1/realtime"
realtime_model = "gpt-realtime-2.1"
''')
        config = cfg.load(path)
        self.assertEqual(config.unknown_keys, [])
        self.assertEqual(config.routing_planner, ["openrouter", "openai"])
        chat = providers.chat_ladder(config)
        self.assertEqual([p.name for p in chat], ["openrouter", "openai"])
        self.assertEqual(chat[0].chat_model, "openai/gpt-4.1-mini")
        self.assertEqual(chat[0].headers, {"HTTP-Referer": "https://example.invalid"})
        self.assertEqual(chat[0].cost, "cheap")
        realtime = providers.realtime_ladder(config)
        self.assertEqual([p.name for p in realtime], ["proxy", "openai"])
        # A third party inherits the voice but not OpenAI's transcriber model.
        self.assertEqual(realtime[0].realtime_voice, config.realtime_voice)
        self.assertEqual(realtime[0].realtime_transcribe_model, "")
        self.assertEqual(realtime[1].realtime_transcribe_model, config.realtime_transcribe_model)

    def test_the_openai_profile_can_be_overridden_piecemeal(self):
        config = Config(providers={"openai": {"realtime_model": "gpt-realtime-mini"}})
        openai = providers.profiles(config)["openai"]
        self.assertEqual(openai.realtime_model, "gpt-realtime-mini")
        self.assertEqual(openai.chat_model, config.planner_model)

    def test_unknown_profile_keys_are_an_error(self):
        config = Config(providers={"x": {"chat_urll": "https://x.example/v1/chat/completions"}})
        with self.assertRaisesRegex(ValueError, "chat_urll"):
            providers.profiles(config)

    def test_a_ladder_rejects_names_it_does_not_know(self):
        with self.assertRaisesRegex(ValueError, "unknown provider 'nope'"):
            providers.chat_ladder(Config(routing_planner=["nope"]))

    def test_a_rung_needs_the_endpoint_the_ladder_is_for(self):
        config = Config(providers={"x": {"chat_url": "https://x.example/v1/chat/completions",
                                         "chat_model": "m"}},
                        routing_realtime=["x"])
        with self.assertRaisesRegex(ValueError, "no realtime_url"):
            providers.realtime_ladder(config)
        config = Config(providers={"y": {"chat_url": "https://y.example/v1/chat/completions"}},
                        routing_planner=["y"])
        with self.assertRaisesRegex(ValueError, "chat_model"):
            providers.chat_ladder(config)

    def test_an_empty_ladder_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            providers.chat_ladder(Config(routing_planner=[]))

    def test_plain_transports_are_only_for_private_hosts(self):
        for url, ok in (("ws://example.com/v1/realtime", False),
                        ("ws://100.88.84.71:4000/v1/realtime", True),
                        ("ws://192.168.1.10:4000/v1/realtime", True),
                        ("ws://litellm.lan:4000/v1/realtime", True),
                        ("ws://mini/v1/realtime", True),
                        ("wss://example.com/v1/realtime", True)):
            config = Config(providers={"p": {"realtime_url": url, "realtime_model": "m"}})
            if ok:
                providers.profiles(config)
            else:
                with self.assertRaisesRegex(ValueError, "public host"):
                    providers.profiles(config)
        config = Config(providers={"p": {"chat_url": "http://api.example.com/v1/chat/completions"}})
        with self.assertRaisesRegex(ValueError, "public host"):
            providers.profiles(config)

    def test_urls_cannot_smuggle_credentials_or_queries(self):
        # Assembled so the publication scanner does not read a fake credential
        # URL out of a test fixture.
        with_userinfo = "https://" + "user:pw" + "@x.example/v1/chat/completions"
        for url in (with_userinfo,
                    "https://x.example/v1/chat/completions?key=1",
                    "ftp://x.example/v1/chat/completions"):
            config = Config(providers={"p": {"chat_url": url}})
            with self.assertRaises(ValueError):
                providers.profiles(config)

    def test_the_safety_identifier_only_goes_to_openai(self):
        config = Config(providers={"other": {"realtime_url": "wss://voice.example/v1/realtime",
                                             "realtime_model": "m", "api_key_env": "OTHER_KEY"}})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k1", "OTHER_KEY": "k2"}):
            known = providers.profiles(config)
            self.assertIn("OpenAI-Safety-Identifier", known["openai"].realtime_headers("id"))
            headers = known["other"].realtime_headers("id")
        self.assertNotIn("OpenAI-Safety-Identifier", headers)
        self.assertEqual(headers["Authorization"], "Bearer k2")

    def test_a_profile_without_a_key_env_needs_no_key(self):
        config = Config(providers={"local": {"api_key_env": "",
                                             "chat_url": "http://127.0.0.1:11434/v1/chat/completions",
                                             "chat_model": "m"}})
        local = providers.profiles(config)["local"]
        self.assertTrue(local.has_key())
        self.assertNotIn("Authorization", local.chat_headers())

    def test_missing_keys_only_matter_when_no_rung_has_one(self):
        config = Config(providers={"b": {"api_key_env": "B_KEY", "chat_url": "https://b.example/v1/chat/completions",
                                         "chat_model": "m", "realtime_url": "wss://b.example/v1/realtime",
                                         "realtime_model": "m"}},
                        routing_realtime=["openai", "b"])
        with mock.patch.dict(os.environ, {"B_KEY": "k"}, clear=True):
            self.assertEqual(providers.missing_keys(providers.realtime_ladder(config)), [])
        with mock.patch.dict(os.environ, {}, clear=True):
            problems = providers.missing_keys(providers.realtime_ladder(config))
        self.assertEqual(problems, ["OPENAI_API_KEY is not set (provider openai)",
                                    "B_KEY is not set (provider b)"])

    def test_the_cli_can_pin_one_rung(self):
        args = cli.build_parser().parse_args(["say", "--provider", "x", "hello"])
        self.assertEqual(args.provider, "x")
        args = cli.build_parser().parse_args(["run", "--provider", "y"])
        self.assertEqual(args.provider, "y")
        path = write(self, '[routing]\nplanner = ["openai", "z"]\n')
        self.assertEqual(cfg.load(path, routing_planner=["z"]).routing_planner, ["z"])


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answer(text: str = "Done.", tool_calls=None) -> dict:
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "down", {}, io.BytesIO(b"down"))


class FakeHTTP:
    """urlopen, keyed by host: an exception to raise, or a payload per call."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, request, timeout=None):
        host = urlsplit(request.full_url).hostname
        self.calls.append((host, json.loads(request.data), dict(request.header_items())))
        action = self.routes[host]
        if callable(action):
            action = action(sum(1 for c in self.calls if c[0] == host))
        if isinstance(action, BaseException):
            raise action
        return _Response(action)


class _Executor:
    def __init__(self):
        self.pending = None
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return mock.Mock(as_tool_result=lambda: "ok")

    def describe(self, name, args):
        return f"{name} {args}"


class UserAgentTests(unittest.TestCase):
    def test_requests_name_the_program_unless_a_profile_says_otherwise(self):
        config = Config(providers={"groq": {"api_key_env": "", "chat_url": "https://api.groq.com/openai/v1/chat/completions",
                                            "chat_model": "m"},
                                   "custom": {"api_key_env": "", "chat_url": "https://x.example/v1/chat/completions",
                                              "chat_model": "m", "headers": {"User-Agent": "mine/1"}}})
        table = providers.profiles(config)
        self.assertEqual(table["groq"].chat_headers()["User-Agent"], providers.USER_AGENT)
        self.assertEqual(table["openai"].realtime_headers()["User-Agent"], providers.USER_AGENT)
        self.assertEqual(table["custom"].chat_headers()["User-Agent"], "mine/1")


class PlannerFailoverTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(planner, "_system_prompt", lambda config=None: "system")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k-openai", "SECOND_KEY": "k-second"},
                                  clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = Config(
            providers={"second": {"api_key_env": "SECOND_KEY",
                                  "chat_url": "https://second.example/v1/chat/completions",
                                  "chat_model": "second-model"}},
            routing_planner=["openai", "second"], notify=False)

    def think(self, routes: dict, config: Config | None = None):
        http = FakeHTTP(routes)
        with mock.patch.object(planner.urllib.request, "urlopen", http):
            turn = planner.Planner(config or self.config, _Executor()).think("hello")
        return turn, http

    def test_the_first_rung_answers_when_it_can(self):
        turn, http = self.think({"api.openai.com": _answer("Hi.")})
        self.assertEqual(turn.reply, "Hi.")
        self.assertEqual((turn.provider, turn.model), ("openai", self.config.planner_model))
        self.assertEqual(turn.failovers, [])
        host, body, headers = http.calls[0]
        self.assertEqual(body["model"], self.config.planner_model)
        self.assertEqual(headers["Authorization"], "Bearer k-openai")

    def test_a_server_error_hands_the_turn_to_the_next_rung(self):
        turn, http = self.think({"api.openai.com": _http_error(503),
                                 "second.example": _answer("From second.")})
        self.assertEqual(turn.reply, "From second.")
        self.assertEqual((turn.provider, turn.model), ("second", "second-model"))
        self.assertEqual(len(turn.failovers), 1)
        self.assertIn("openai: HTTP 503", turn.failovers[0])
        self.assertIn("trying second", turn.failovers[0])
        self.assertEqual(turn.error, "")
        _, body, headers = http.calls[-1]
        self.assertEqual(body["model"], "second-model")
        self.assertEqual(headers["Authorization"], "Bearer k-second")

    def test_an_unreachable_host_and_a_timeout_also_fail_over(self):
        for exc in (urllib.error.URLError("refused"), TimeoutError()):
            turn, _ = self.think({"api.openai.com": exc, "second.example": _answer("ok")})
            self.assertEqual(turn.provider, "second", exc)

    def test_a_rung_without_a_key_is_skipped_quietly(self):
        with mock.patch.dict(os.environ, {"SECOND_KEY": "k-second"}, clear=True):
            turn, http = self.think({"second.example": _answer("ok")})
        self.assertEqual(turn.provider, "second")
        self.assertEqual(turn.failovers, [])
        self.assertEqual([c[0] for c in http.calls], ["second.example"])

    def test_no_keys_at_all_is_reported_with_every_env_name(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            turn, http = self.think({})
        self.assertIn("OPENAI_API_KEY", turn.error)
        self.assertIn("SECOND_KEY", turn.error)
        self.assertEqual(http.calls, [])

    def test_every_rung_failing_names_each_failure(self):
        turn, _ = self.think({"api.openai.com": _http_error(500), "second.example": _http_error(429)})
        self.assertTrue(turn.error.startswith("every planner provider failed"), turn.error)
        self.assertIn("openai: HTTP 500", turn.error)
        self.assertIn("second: HTTP 429", turn.error)
        self.assertEqual(turn.provider, "")

    def test_the_rung_is_sticky_across_tool_rounds(self):
        call = {"id": "c1", "function": {"name": "hypr_dispatch", "arguments": "{}"}}

        def second(n):
            return _answer("", [call]) if n == 1 else _answer("Done twice.")

        turn, http = self.think({"api.openai.com": _http_error(502), "second.example": second})
        self.assertEqual(turn.reply, "Done twice.")
        self.assertEqual([c[0] for c in http.calls], ["api.openai.com", "second.example", "second.example"])
        self.assertEqual(len(turn.failovers), 1)

    def test_a_broken_ladder_is_a_planner_problem_not_a_crash(self):
        turn, http = self.think({}, Config(routing_planner=["ghost"]))
        self.assertIn("unknown provider 'ghost'", turn.error)
        self.assertEqual(http.calls, [])


if __name__ == "__main__":
    unittest.main()
