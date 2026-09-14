"""The daemon walks `routing.realtime` when a rung never answers.

Two fake endpoints: one that is dead or refuses the handshake, and one that
behaves like the Realtime API. The session must end up talking to the second,
and the log must say why. Needs python-websockets, like test_realtime_wire.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import json
import os
import socket
import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - depends on the machine
    serve = None

from omarchy_voice import feedback, realtime, session as session_mod
from omarchy_voice.config import Config


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class GoodServer:
    def __init__(self):
        self.received: list[dict] = []
        self.answered = asyncio.Event()
        self.port = 0
        self._server = None

    async def start(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_good"}}))
        async for raw in ws:
            event = json.loads(raw)
            self.received.append(event)
            if event.get("type") == "session.update":
                await ws.send(json.dumps({"type": "response.done",
                                          "response": {"status": "completed", "output": []}}))
                self.answered.set()


class RejectingServer:
    """Answers the handshake with 401, the way a wrong key looks."""

    def __init__(self):
        self.attempts = 0
        self.port = 0
        self._server = None

    async def start(self):
        self._server = await serve(self._handle, "127.0.0.1", 0, process_request=self._reject)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def _reject(self, connection, request):
        self.attempts += 1
        return connection.respond(HTTPStatus.UNAUTHORIZED, "no\n")

    async def _handle(self, ws):  # pragma: no cover - never reached
        pass


@unittest.skipIf(serve is None, "websockets is not installed")
class FailoverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.log = root / "session.log"
        patcher = mock.patch('omarchy_voice.network.STATE_DIR', root)
        patcher.start()
        self.addCleanup(patcher.stop)
        for module, names in ((feedback, ("LOG_FILE", "STATE_FILE", "STATE_DIR", "RUNTIME_DIR")),
                              (session_mod, ("SOCKET_PATH", "RUNTIME_DIR")),
                              (realtime, ("SAFETY_ID_FILE", "CONFIG_DIR"))):
            for name in names:
                value = root / ("session.log" if name == "LOG_FILE" else
                                "state.json" if name == "STATE_FILE" else
                                "control.sock" if name == "SOCKET_PATH" else
                                "safety-id" if name == "SAFETY_ID_FILE" else "")
                patcher = mock.patch.object(module, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
        self.good = GoodServer()
        await self.good.start()
        for name, value in (("REALTIME_URL", f"ws://127.0.0.1:{self.good.port}/v1/realtime"),
                            ("RECONNECT_BASE_DELAY", 0.01)):
            patcher = mock.patch.object(realtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OTHER_KEY": "other"})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.good.stop()

    def config(self, first_url: str) -> Config:
        return Config(dry_run=True, notify=False,
                      providers={"first": {"api_key_env": "OTHER_KEY", "realtime_url": first_url,
                                           "realtime_model": "first-model", "realtime_voice": "alloy"}},
                      routing_realtime=["first", "openai"])

    async def run_session(self, config: Config) -> int:
        session = realtime.RealtimeSession(config)
        runner = asyncio.create_task(session.run())
        await asyncio.wait_for(self.good.answered.wait(), timeout=30)
        session._user_quit = True
        session._stop.set()
        return await asyncio.wait_for(runner, timeout=10)

    async def test_a_dead_rung_hands_over_after_two_tries(self):
        config = self.config(f"ws://127.0.0.1:{_free_port()}/v1/realtime")
        self.assertEqual(await self.run_session(config), 0)
        update = [e for e in self.good.received if e["type"] == "session.update"][0]["session"]
        self.assertEqual(update["model"], config.realtime_model)
        self.assertEqual(update["audio"]["output"]["voice"], config.realtime_voice)
        log = self.log.read_text()
        self.assertIn("provider=first", log)
        self.assertIn("first never answered — failing over to openai", log)
        self.assertIn("provider=openai", log)

    async def test_a_rejected_handshake_hands_over_at_once(self):
        rejecting = RejectingServer()
        await rejecting.start()
        self.addAsyncCleanup(rejecting.stop)
        config = self.config(f"ws://127.0.0.1:{rejecting.port}/v1/realtime")
        self.assertEqual(await self.run_session(config), 0)
        self.assertEqual(rejecting.attempts, 1)
        self.assertIn("failing over to openai", self.log.read_text())

    async def test_the_last_rung_keeps_the_reconnect_budget(self):
        # Only one rung, and it is dead: no failover, the usual give-up path.
        config = Config(dry_run=True, notify=False,
                        providers={"only": {"api_key_env": "OTHER_KEY",
                                            "realtime_url": f"ws://127.0.0.1:{_free_port()}/v1/realtime",
                                            "realtime_model": "m"}},
                        routing_realtime=["only"])
        with mock.patch.object(realtime, "RECONNECT_ATTEMPTS", 1):
            session = realtime.RealtimeSession(config)
            self.assertEqual(await asyncio.wait_for(session.run(), timeout=20), 1)
        log = self.log.read_text()
        self.assertNotIn("failing over", log)
        self.assertIn("gave up after 1 reconnects", log)

    async def test_no_key_on_any_rung_is_reported_before_connecting(self):
        config = self.config(f"ws://127.0.0.1:{_free_port()}/v1/realtime")
        with mock.patch.dict(os.environ, {}, clear=True):
            problems = realtime.check_ready(config)
            with self.assertRaisesRegex(realtime.RealtimeUnavailable, "OTHER_KEY, OPENAI_API_KEY"):
                await realtime.RealtimeSession(config).run()
        self.assertIn("OTHER_KEY is not set (provider first)", problems)
        self.assertIn("OPENAI_API_KEY is not set (provider openai)", problems)


if __name__ == "__main__":
    unittest.main()
