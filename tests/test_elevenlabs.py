"""The ElevenLabs Agents rung: profile, agent sync, and the session's wire.

Everything here is offline. The REST client is replaced by a fake that records
what sync would have sent, and the session tests feed events straight into
`_on_agent_event` over a recording socket. The end-to-end pass against a fake
agent server needs python-websockets and skips without it.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - depends on the machine
    serve = None

from omarchy_voice import elevenlabs, feedback, providers, realtime, session as session_mod
from omarchy_voice.config import Config
from omarchy_voice.tools import tools_for


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def events(self, kind):
        return [e for e in self.sent if e.get("type") == kind]


def _patch_dirs(test, root: Path) -> None:
    for name, value in (("LOG_FILE", root / "session.log"), ("STATE_FILE", root / "state.json"),
                        ("STATE_DIR", root), ("RUNTIME_DIR", root)):
        patcher = mock.patch.object(feedback, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    patcher = mock.patch.object(elevenlabs, "AGENTS_FILE", root / "elevenlabs-agents.json")
    patcher.start()
    test.addCleanup(patcher.stop)


# --- the profile ----------------------------------------------------------------

class ProfileTests(unittest.TestCase):
    def test_elevenlabs_is_built_in_with_its_own_protocol(self):
        provider = providers.profiles(Config())["elevenlabs"]
        self.assertTrue(provider.is_elevenlabs)
        self.assertEqual(provider.api_key_env, "ELEVENLABS_API_KEY")
        self.assertEqual(provider.realtime_url, providers.ELEVENLABS_REALTIME_URL)
        self.assertEqual(provider.api_url, providers.ELEVENLABS_API_URL)
        self.assertEqual(provider.realtime_model, providers.ELEVENLABS_LLM)

    def test_the_openai_voice_name_does_not_leak_into_the_agent(self):
        provider = providers.profiles(Config(realtime_voice="marin"))["elevenlabs"]
        self.assertEqual(provider.realtime_voice, "")
        self.assertEqual(provider.realtime_transcribe_model, "")

    def test_the_profile_can_be_overridden_piecemeal(self):
        config = Config(providers={"elevenlabs": {"realtime_voice": "voice123", "realtime_model": "claude-sonnet-4-5",
                                                  "agent_id": "agent_abc"}})
        provider = providers.profiles(config)["elevenlabs"]
        self.assertEqual((provider.realtime_voice, provider.realtime_model, provider.agent_id),
                         ("voice123", "claude-sonnet-4-5", "agent_abc"))
        self.assertTrue(provider.is_elevenlabs)

    def test_a_second_elevenlabs_profile_gets_the_defaults(self):
        config = Config(providers={"eleven-eu": {"protocol": "elevenlabs", "api_key_env": "EL_EU",
                                                 "api_url": "https://api.eu.residency.elevenlabs.io"}},
                        routing_realtime=["eleven-eu"])
        provider = providers.realtime_ladder(config)[0]
        self.assertEqual(provider.realtime_url, providers.ELEVENLABS_REALTIME_URL)
        self.assertEqual(provider.realtime_model, providers.ELEVENLABS_LLM)

    def test_bad_protocol_and_misplaced_agent_keys_are_errors(self):
        with self.assertRaisesRegex(ValueError, "protocol must be one of"):
            providers.profiles(Config(providers={"x": {"protocol": "deepgram"}}))
        with self.assertRaisesRegex(ValueError, "protocol is not elevenlabs"):
            providers.profiles(Config(providers={"x": {"agent_id": "agent_1", "chat_url": "https://x.example/v1"}}))
        with self.assertRaisesRegex(ValueError, "agent_id"):
            providers.profiles(Config(providers={"elevenlabs": {"agent_id": "agent 1; drop"}}))

    def test_the_key_never_goes_on_the_websocket(self):
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "xi-secret"}):
            headers = providers.profiles(Config())["elevenlabs"].realtime_headers("safety")
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("OpenAI-Safety-Identifier", headers)
        self.assertNotIn("xi-secret", json.dumps(headers))

    def test_doctor_summary_says_what_it_is(self):
        summary = providers.profiles(Config())["elevenlabs"].summary("realtime")
        self.assertIn("ElevenLabs agent", summary)
        self.assertIn("api.elevenlabs.io", summary)


# --- schemas and the agent ------------------------------------------------------

class ToolSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schemas = realtime.to_realtime_tools(tools_for(Config(allow_shell=True)))
        self.configs = elevenlabs.tool_configs(self.schemas)
        self.by_name = {c["name"]: c for c in self.configs}

    def test_every_tool_becomes_a_blocking_client_tool(self):
        self.assertEqual({c["name"] for c in self.configs}, {s["name"] for s in self.schemas})
        for config in self.configs:
            self.assertEqual(config["type"], "client")
            self.assertTrue(config["expects_response"])
            self.assertEqual(config["response_timeout_secs"], elevenlabs.TOOL_TIMEOUT_SECONDS)
            self.assertEqual(config["parameters"]["type"], "object")
        self.assertIn("confirm_last", self.by_name)
        self.assertIn("cancel_last", self.by_name)

    def test_enums_survive_and_bounds_move_into_the_description(self):
        kind = self.by_name["hypr_query"]["parameters"]["properties"]["kind"]
        self.assertIn("clients", kind["enum"])
        bounded = [p for c in self.configs for p in c["parameters"]["properties"].values()
                   if "at least" in p.get("description", "") or "at most" in p.get("description", "")]
        self.assertTrue(bounded)

    def test_nothing_the_tools_api_rejects_is_sent(self):
        def walk(node, top=False):
            self.assertNotIn("additionalProperties", node)
            self.assertNotIn("minimum", node)
            self.assertNotIn("maximum", node)
            if not top:
                self.assertTrue(node.get("description"), node)  # the API refuses a bare property
            for sub in (node.get("properties") or {}).values():
                walk(sub)
            if "items" in node:
                walk(node["items"])
        for config in self.configs:
            walk(config["parameters"], top=True)

    def test_nested_pane_objects_keep_their_shape(self):
        panes = self.by_name["compose_windows"]["parameters"]["properties"]["panes"]
        self.assertEqual(panes["type"], "array")
        self.assertEqual(panes["items"]["type"], "object")
        self.assertIn("kind", panes["items"]["properties"])
        self.assertIn("kind", panes["items"]["required"])


class AgentConfigTests(unittest.TestCase):
    def test_the_agent_speaks_and_hears_pcm_at_the_daemon_rate(self):
        agent = elevenlabs.agent_config(rate=24000, llm="gpt-4.1", voice_id="", tool_ids=["t1", "t2"])
        conv = agent["conversation_config"]
        self.assertEqual(conv["asr"]["user_input_audio_format"], "pcm_24000")
        self.assertEqual(conv["tts"]["agent_output_audio_format"], "pcm_24000")
        self.assertNotIn("voice_id", conv["tts"])
        self.assertEqual(conv["agent"]["prompt"]["tool_ids"], ["t1", "t2"])
        self.assertEqual(conv["agent"]["prompt"]["llm"], "gpt-4.1")
        self.assertEqual(conv["agent"]["first_message"], "")
        self.assertEqual(conv["turn"]["turn_timeout"], elevenlabs.TURN_TIMEOUT_SECONDS)
        for event in ("client_tool_call", "ping", "audio", "user_transcript", "agent_response"):
            self.assertIn(event, conv["conversation"]["client_events"])

    def test_the_session_prompt_override_is_enabled(self):
        agent = elevenlabs.agent_config(rate=16000, llm="gpt-4.1", voice_id="v", tool_ids=[])
        override = agent["platform_settings"]["overrides"]["conversation_config_override"]
        self.assertTrue(override["agent"]["prompt"]["prompt"])
        self.assertTrue(override["agent"]["first_message"])
        self.assertEqual(agent["conversation_config"]["tts"]["voice_id"], "v")

    def test_hello_carries_the_instructions_and_no_greeting(self):
        provider = providers.profiles(Config(providers={"elevenlabs": {"realtime_voice": "v9"}}))["elevenlabs"]
        hello = elevenlabs.hello(provider, "be brief")
        self.assertEqual(hello["type"], "conversation_initiation_client_data")
        override = hello["conversation_config_override"]
        self.assertEqual(override["agent"]["prompt"]["prompt"], "be brief")
        self.assertEqual(override["agent"]["first_message"], "")
        self.assertEqual(override["tts"]["voice_id"], "v9")
        self.assertNotIn("tts", elevenlabs.hello(providers.profiles(Config())["elevenlabs"], "x")
                         ["conversation_config_override"])

    def test_audio_rate_reads_pcm_formats_only(self):
        self.assertEqual(elevenlabs.audio_rate("pcm_16000"), 16000)
        self.assertIsNone(elevenlabs.audio_rate("ulaw_8000"))
        self.assertIsNone(elevenlabs.audio_rate(""))

    def test_tool_results_flag_errors(self):
        self.assertFalse(elevenlabs.tool_result("c1", "done")["is_error"])
        self.assertTrue(elevenlabs.tool_result("c1", "ERROR: no")["is_error"])


class FakeApi:
    """Records calls; hands out ids like the real service would."""

    def __init__(self, existing: dict[str, str] | None = None):
        self.existing = dict(existing or {})
        self.created_tools: list[dict] = []
        self.updated_tools: list[tuple[str, dict]] = []
        self.agents: list[tuple[str, dict]] = []
        self.searched: set[str] = set()

    def find_tools(self, names):
        self.searched |= set(names)
        return {n: i for n, i in self.existing.items() if n in names}

    def upsert_tool(self, config, tool_id):
        if tool_id:
            self.updated_tools.append((tool_id, config))
            return tool_id
        self.created_tools.append(config)
        return f"tool_{len(self.created_tools)}"

    def upsert_agent(self, config, agent_id):
        self.agents.append((agent_id, config))
        return agent_id or "agent_new"


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "agents.json"
        self.provider = providers.profiles(Config())["elevenlabs"]
        self.schemas = realtime.to_realtime_tools(tools_for(Config()))

    def sync(self, api):
        return elevenlabs.sync(self.provider, rate=24000, schemas=self.schemas, api=api, state_path=self.state)

    def test_a_first_sync_creates_every_tool_and_the_agent(self):
        api = FakeApi()
        record = self.sync(api)
        self.assertEqual(len(api.created_tools), len(self.schemas))
        self.assertEqual(api.updated_tools, [])
        self.assertEqual(record["agent_id"], "agent_new")
        agent_id, config = api.agents[0]
        self.assertEqual(agent_id, "")
        self.assertEqual(config["conversation_config"]["agent"]["prompt"]["tool_ids"],
                         [record["tools"][s["name"]] for s in self.schemas])
        self.assertEqual(oct(self.state.stat().st_mode & 0o777), "0o600")

    def test_a_second_sync_updates_in_place(self):
        first = self.sync(FakeApi())
        api = FakeApi()
        second = self.sync(api)
        self.assertEqual(api.created_tools, [])
        self.assertEqual(len(api.updated_tools), len(self.schemas))
        self.assertEqual(api.agents[0][0], "agent_new")
        self.assertEqual(second, first)
        self.assertEqual(api.searched, set())

    def test_tools_left_by_an_earlier_install_are_adopted_by_name(self):
        api = FakeApi(existing={"hypr_query": "tool_old"})
        record = self.sync(api)
        self.assertEqual(record["tools"]["hypr_query"], "tool_old")
        self.assertIn(("tool_old", next(c for c in elevenlabs.tool_configs(self.schemas)
                                        if c["name"] == "hypr_query")), api.updated_tools)

    def test_a_configured_agent_id_is_used_and_never_reported_stale(self):
        provider = providers.profiles(Config(providers={"elevenlabs": {"agent_id": "agent_mine"}}))["elevenlabs"]
        self.assertEqual(elevenlabs.agent_id_for(provider, {}), "agent_mine")
        self.assertFalse(elevenlabs.stale(provider, "anything", {"elevenlabs": {"fingerprint": "old"}}))

    def test_stale_means_the_tools_changed_since_the_sync(self):
        record = self.sync(FakeApi())
        state = elevenlabs.load_state(self.state)
        self.assertEqual(elevenlabs.agent_id_for(self.provider, state), record["agent_id"])
        self.assertFalse(elevenlabs.stale(self.provider, record["fingerprint"], state))
        self.assertTrue(elevenlabs.stale(self.provider, "different", state))
        self.assertFalse(elevenlabs.stale(self.provider, "different", {}))


# --- the session --------------------------------------------------------------

class AgentSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        _patch_dirs(self, Path(self.tmp.name))
        self.log = Path(self.tmp.name) / "session.log"
        self.config = Config(dry_run=True, notify=False, routing_realtime=["elevenlabs"], max_turns=3)
        self.session = realtime.RealtimeSession(self.config)
        self.session.provider = providers.profiles(self.config)["elevenlabs"]
        self.session.speaker.write = mock.AsyncMock()
        self.session.speaker.interrupt = mock.AsyncMock()
        self.socket = FakeSocket()
        self.session.ws = self.socket

    async def tool_call(self, name, params, call_id="call_1"):
        await self.session._on_agent_event({"type": "client_tool_call", "client_tool_call": {
            "tool_name": name, "tool_call_id": call_id, "parameters": params,
            "event_id": 1, "expects_response": True}})
        await asyncio.gather(*self.session._tool_tasks)
        return self.socket.events("client_tool_result")

    async def test_metadata_establishes_the_session_and_adopts_the_rates(self):
        await self.session._on_agent_event({"type": "conversation_initiation_metadata",
                                            "conversation_initiation_metadata_event": {
                                                "conversation_id": "conv_1",
                                                "user_input_audio_format": "pcm_16000",
                                                "agent_output_audio_format": "pcm_22050"}})
        self.assertTrue(self.session._established)
        self.assertEqual(self.session._rate, 16000)
        self.assertEqual(self.session.speaker.rate, 22050)
        self.assertIn("agent conversation conv_1", self.log.read_text())

    async def test_a_non_pcm_agent_is_named_in_the_log(self):
        await self.session._on_agent_event({"type": "conversation_initiation_metadata",
                                            "conversation_initiation_metadata_event": {
                                                "user_input_audio_format": "ulaw_8000",
                                                "agent_output_audio_format": "ulaw_8000"}})
        self.assertIn("elevenlabs sync", self.log.read_text())
        self.assertEqual(self.session._rate, 24000)

    async def test_pings_are_answered(self):
        await self.session._on_agent_event({"type": "ping", "ping_event": {"event_id": 7, "ping_ms": 0}})
        self.assertEqual(self.socket.events("pong"), [{"type": "pong", "event_id": 7}])

    async def test_audio_goes_to_the_speaker_and_interruptions_stop_it(self):
        import base64
        pcm = b"\x01\x02" * 50
        await self.session._on_agent_event({"type": "audio", "audio_event": {
            "audio_base_64": base64.b64encode(pcm).decode(), "event_id": 1}})
        self.session.speaker.write.assert_awaited_once_with(pcm)
        await self.session._on_agent_event({"type": "interruption", "interruption_event": {"event_id": 2}})
        self.session.speaker.interrupt.assert_awaited_once()

    async def test_a_tool_call_runs_locally_and_answers_by_id(self):
        results = await self.tool_call("hypr_dispatch", {"lua": 'hl.dsp.focus({ workspace = "3" })'}, "call_9")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_call_id"], "call_9")
        self.assertFalse(results[0]["is_error"])
        self.assertIn("action  ", self.log.read_text())

    async def test_an_unknown_tool_is_an_error_result_not_a_crash(self):
        results = await self.tool_call("no_such_tool", {})
        self.assertTrue(results[0]["is_error"])
        self.assertTrue(results[0]["result"].startswith("ERROR:"))

    async def test_confirm_needs_a_new_user_turn(self):
        held = self.session.executor.call("omarchy_cli", {"command": "reboot"})
        self.assertFalse(held.ok)
        self.session._user_turn_since_hold = False
        results = await self.tool_call("confirm_last", {"heard_phrase": "confirm"})
        self.assertTrue(results[0]["is_error"])
        self.assertIsNotNone(self.session.executor.pending)
        await self.session._on_agent_event({"type": "user_transcript",
                                            "user_transcription_event": {"user_transcript": "confirm"}})
        results = await self.tool_call("confirm_last", {"heard_phrase": "confirm"}, "call_2")
        self.assertFalse(results[1]["is_error"])
        self.assertIsNone(self.session.executor.pending)
        self.assertIn("heard   'confirm'", self.log.read_text())

    async def test_too_many_tool_calls_without_a_user_turn_are_refused(self):
        lua = 'hl.dsp.focus({ workspace = "2" })'
        for index in range(3):
            results = await self.tool_call("hypr_dispatch", {"lua": lua}, f"call_{index}")
            self.assertFalse(results[-1]["is_error"])
        results = await self.tool_call("hypr_dispatch", {"lua": lua}, "call_late")
        self.assertTrue(results[-1]["is_error"])
        self.assertIn("guard   refused hypr_dispatch", self.log.read_text())
        await self.session._on_agent_event({"type": "user_transcript",
                                            "user_transcription_event": {"user_transcript": "again"}})
        results = await self.tool_call("hypr_dispatch", {"lua": lua}, "call_fresh")
        self.assertFalse(results[-1]["is_error"])

    async def test_replies_are_logged_and_settle_the_bar(self):
        await self.session._on_agent_event({"type": "agent_response",
                                            "agent_response_event": {"agent_response": "Switched to three."}})
        self.assertIn("reply   'Switched to three.'", self.log.read_text())
        self.assertFalse(self.session._response_running)

    async def test_a_typed_turn_is_a_user_message(self):
        with mock.patch.object(realtime.capabilities, "live_state", return_value="Workspaces: 1"):
            self.assertEqual(await self.session._inject("which workspace"), "sent")
        self.assertEqual(self.socket.events("user_message")[0]["text"], "which workspace")
        self.assertEqual(len(self.socket.events("contextual_update")), 1)

    async def test_a_typed_turn_while_muted_is_queued_for_the_next_conversation(self):
        self.session.ws = None
        self.assertIn("opening", await self.session._inject("hello"))
        self.assertTrue(self.session._wake.is_set())
        self.session.ws = self.socket
        await self.session._on_agent_event({"type": "conversation_initiation_metadata",
                                            "conversation_initiation_metadata_event": {
                                                "user_input_audio_format": "pcm_24000",
                                                "agent_output_audio_format": "pcm_24000"}})
        self.assertEqual(self.socket.events("user_message")[0]["text"], "hello")
        self.assertEqual(self.session._typed, [])

    async def test_an_unchanged_desktop_is_not_sent_twice(self):
        with mock.patch.object(realtime.capabilities, "live_state", return_value="Workspaces: 1"):
            for _ in range(2):
                self.session._state_refreshed = 0.0
                await self.session._refresh_state()
        self.assertEqual(len(self.socket.events("contextual_update")), 1)
        self.assertIn("Workspaces: 1", self.socket.events("contextual_update")[0]["text"])
        with mock.patch.object(realtime.capabilities, "live_state", return_value="Workspaces: 2"):
            self.session._state_refreshed = 0.0
            await self.session._refresh_state()
        self.assertEqual(len(self.socket.events("contextual_update")), 2)
        self.assertEqual(self.socket.events("conversation.item.create"), [])

    async def test_a_finished_watch_is_announced_as_a_notice(self):
        self.session.active = True
        with mock.patch.object(realtime, "WATCH_MIN_GAP_SECONDS", 0):
            await self.session._announce({"vanished": False, "timed_out": False, "label": "make",
                                          "seconds": 3.0, "target": "Work:1.0", "tail": "ok"})
        text = self.socket.events("user_message")[0]["text"]
        self.assertIn("system notice", text)
        self.assertIn("make finished", text)
        self.assertEqual(self.socket.events("response.create"), [])

    async def test_muting_hangs_up_the_conversation(self):
        self.session.active = True
        with mock.patch.object(self.session.executor.vision, "stop_owned"):
            await self.session._set_active(False)
        self.assertTrue(self.session._parked)
        self.assertTrue(self.session._stop.is_set())

    async def test_client_errors_reach_the_bar(self):
        await self.session._on_agent_event({"type": "client_error", "error_event": {
            "code": 1008, "error_name": "override_not_allowed", "message": "prompt override is off"}})
        self.assertIn("override_not_allowed", self.log.read_text())

    async def test_the_hello_is_the_initiation_message(self):
        with mock.patch.object(realtime.capabilities, "manifest", return_value="manifest"), \
                mock.patch.object(realtime.capabilities, "live_state", return_value="live"):
            hello = await self.session._hello()
        self.assertEqual(hello["type"], "conversation_initiation_client_data")
        prompt = hello["conversation_config_override"]["agent"]["prompt"]["prompt"]
        self.assertIn("Your name is OMA", prompt)
        self.assertIn("manifest", prompt)
        self.assertIn("live", prompt)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        _patch_dirs(self, Path(self.tmp.name))

    def test_an_agentless_elevenlabs_only_ladder_is_unconfigured(self):
        config = Config(routing_realtime=["elevenlabs"])
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            problems = realtime.check_ready(config)
        self.assertTrue(any("has no agent yet" in p for p in problems), problems)

    def test_a_synced_agent_clears_it(self):
        elevenlabs.save_state({"elevenlabs": {"agent_id": "agent_ok", "tools": {}, "fingerprint": "f"}})
        config = Config(routing_realtime=["elevenlabs"])
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            problems = realtime.check_ready(config)
        self.assertFalse(any("agent" in p for p in problems), problems)

    def test_the_agent_gap_does_not_block_a_ladder_with_another_rung(self):
        config = Config(routing_realtime=["elevenlabs", "openai"])
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k", "OPENAI_API_KEY": "k"}):
            problems = realtime.check_ready(config)
        self.assertFalse(any("agent" in p for p in problems), problems)


# --- end to end against a fake agent -------------------------------------------

class FakeAgentServer:
    """The ElevenLabs Agents websocket from the client's side, in miniature."""

    def __init__(self):
        self.received: list[dict] = []
        self.connections = 0
        self.closed = asyncio.Event()
        self.tool_answered = asyncio.Event()
        self.port = 0
        self._server = None

    async def start(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/v1/convai/conversation?agent_id=agent_fake&token=t"

    async def _handle(self, ws):
        self.connections += 1
        self.closed.clear()
        try:
            async for raw in ws:
                event = json.loads(raw)
                self.received.append(event)
                if event.get("type") == "conversation_initiation_client_data":
                    await ws.send(json.dumps({"type": "conversation_initiation_metadata",
                                              "conversation_initiation_metadata_event": {
                                                  "conversation_id": "conv_fake",
                                                  "user_input_audio_format": "pcm_24000",
                                                  "agent_output_audio_format": "pcm_24000"}}))
                    await ws.send(json.dumps({"type": "ping", "ping_event": {"event_id": 1, "ping_ms": 0}}))
                    await ws.send(json.dumps({"type": "client_tool_call", "client_tool_call": {
                        "tool_name": "hypr_dispatch", "tool_call_id": "call_fake",
                        "parameters": {"lua": 'hl.dsp.focus({ workspace = "3" })'},
                        "event_id": 2, "expects_response": True}}))
                elif event.get("type") == "client_tool_result":
                    await ws.send(json.dumps({"type": "agent_response",
                                              "agent_response_event": {"agent_response": "Done."}}))
                    self.tool_answered.set()
        finally:
            self.closed.set()


@unittest.skipIf(serve is None, "websockets is not installed")
class AgentWireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        _patch_dirs(self, root)
        self.log = root / "session.log"
        for module, names in ((session_mod, ("SOCKET_PATH", "RUNTIME_DIR")),
                              (realtime, ("SAFETY_ID_FILE", "CONFIG_DIR"))):
            for name in names:
                value = root / ("control.sock" if name == "SOCKET_PATH" else
                                "safety-id" if name == "SAFETY_ID_FILE" else "")
                patcher = mock.patch.object(module, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
        patcher = mock.patch('omarchy_voice.network.STATE_DIR', root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "xi-test", "OPENAI_API_KEY": "sk-test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = FakeAgentServer()
        await self.server.start()
        patcher = mock.patch.object(elevenlabs, "signed_url", lambda provider, agent_id: self.server.url)
        patcher.start()
        self.addCleanup(patcher.stop)
        elevenlabs.save_state({"elevenlabs": {"agent_id": "agent_fake", "tools": {}, "fingerprint": "f"}})
        # No real microphone: the recorder is a process that produces nothing.
        patcher = mock.patch.object(realtime.RealtimeSession, "_mic_loop", mock.AsyncMock())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_a_conversation_opens_on_listen_runs_tools_and_hangs_up_on_mute(self):
        config = Config(dry_run=True, notify=False, routing_realtime=["elevenlabs"])
        session = realtime.RealtimeSession(config)
        runner = asyncio.create_task(session.run())
        await asyncio.sleep(0.3)
        self.assertEqual(self.server.connections, 0, "muted must not open a billed conversation")

        with mock.patch.object(session.executor.vision, "stop_owned"):
            await session._set_active(True)
            await asyncio.wait_for(self.server.tool_answered.wait(), timeout=20)
            hello = self.server.received[0]
            self.assertEqual(hello["type"], "conversation_initiation_client_data")
            self.assertIn("Your name is OMA", hello["conversation_config_override"]["agent"]["prompt"]["prompt"])
            kinds = [e.get("type") for e in self.server.received]
            self.assertIn("pong", kinds)
            result = next(e for e in self.server.received if e.get("type") == "client_tool_result")
            self.assertEqual(result["tool_call_id"], "call_fake")
            self.assertFalse(result["is_error"])

            await session._set_active(False)
            await asyncio.wait_for(self.server.closed.wait(), timeout=10)
            await asyncio.sleep(0.2)
            self.assertEqual(self.server.connections, 1)

            await session._set_active(True)
            await asyncio.sleep(0.5)
            self.assertEqual(self.server.connections, 2, "listening again opens a fresh conversation")

        session._user_quit = True
        session._stop.set()
        self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)
        log = self.log.read_text()
        self.assertIn("protocol=elevenlabs", log)
        self.assertIn("agent conversation conv_fake", log)
        self.assertIn("reply   'Done.'", log)
        self.assertNotIn("failing over", log)
        self.assertNotIn("reconnecting", log)

    async def test_a_rung_without_an_agent_hands_over(self):
        elevenlabs.save_state({})
        good = None

        class Good:
            def __init__(self):
                self.answered = asyncio.Event()
                self.port = 0

            async def start(self):
                self._server = await serve(self._handle, "127.0.0.1", 0)
                self.port = self._server.sockets[0].getsockname()[1]

            async def stop(self):
                self._server.close()
                await self._server.wait_closed()

            async def _handle(self, ws):
                await ws.send(json.dumps({"type": "session.created", "session": {"id": "s"}}))
                async for raw in ws:
                    if json.loads(raw).get("type") == "session.update":
                        self.answered.set()

        good = Good()
        await good.start()
        self.addAsyncCleanup(good.stop)
        config = Config(dry_run=True, notify=False, routing_realtime=["elevenlabs", "openai"])
        with mock.patch.object(realtime, "REALTIME_URL", f"ws://127.0.0.1:{good.port}/v1/realtime"):
            session = realtime.RealtimeSession(config)
            runner = asyncio.create_task(session.run())
            await asyncio.wait_for(good.answered.wait(), timeout=20)
            session._user_quit = True
            session._stop.set()
            self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)
        log = self.log.read_text()
        self.assertIn("elevenlabs has no agent yet", log)
        self.assertIn("failing over to openai", log)
        self.assertEqual(self.server.connections, 0)


if __name__ == "__main__":
    unittest.main()
