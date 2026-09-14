"""The daemon retires a socket before OpenAI's 60-minute cap does.

Reuses the fake Realtime server and the patched paths from
test_realtime_failover. Needs python-websockets, like that file.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_realtime_failover import FailoverTests, serve  # noqa: E402

from omarchy_voice import realtime  # noqa: E402
from omarchy_voice.config import Config  # noqa: E402


@unittest.skipIf(serve is None, "websockets is not installed")
class RotationTests(FailoverTests):
    async def test_a_muted_session_is_replaced_ahead_of_the_cap(self):
        config = Config(dry_run=True, notify=False, routing_realtime=["openai"])
        with mock.patch.object(realtime, "SESSION_ROTATE_SECONDS", 0.3):
            session = realtime.RealtimeSession(config)
            runner = asyncio.create_task(session.run())
            for _ in range(200):
                await asyncio.sleep(0.05)
                if sum(1 for e in self.good.received if e["type"] == "session.update") >= 3:
                    break
            session._user_quit = True
            session._stop.set()
            self.assertEqual(await asyncio.wait_for(runner, timeout=10), 0)
        updates = sum(1 for e in self.good.received if e["type"] == "session.update")
        self.assertGreaterEqual(updates, 3, "each rotation opens a fresh session")
        log = self.log.read_text()
        self.assertIn("ahead of the 60-minute cap", log)
        self.assertNotIn("retry", log)
        self.assertNotIn("error", log)

    async def test_an_active_session_waits_for_a_quiet_moment(self):
        config = Config(dry_run=True, notify=False, routing_realtime=["openai"])

        async def no_mic(session):   # no pw-record on the test machine
            await session._stop.wait()

        def updates():
            return sum(1 for e in self.good.received if e["type"] == "session.update")

        with mock.patch.object(realtime, "SESSION_ROTATE_SECONDS", 0.2), \
                mock.patch.object(realtime, "SESSION_ROTATE_POLL_SECONDS", 0.05), \
                mock.patch.object(realtime.RealtimeSession, "_mic_loop", no_mic):
            session = realtime.RealtimeSession(config)
            session.active = True
            session._active_event.set()
            runner = asyncio.create_task(session.run())
            for _ in range(100):
                await asyncio.sleep(0.05)
                if updates():
                    break
            session._user_speaking = True   # mid-sentence; the fake server never clears this
            await asyncio.sleep(0.6)
            self.assertEqual(updates(), 1, "no rotation while the user is speaking")
            session._user_speaking = False
            for _ in range(100):
                await asyncio.sleep(0.05)
                if updates() >= 2:
                    break
            self.assertTrue(session.active, "rotation keeps the gate open")
            session._user_quit = True
            session._stop.set()
            await asyncio.wait_for(runner, timeout=10)
        self.assertGreaterEqual(updates(), 2)

    async def test_session_expired_is_a_note_not_an_error(self):
        session = realtime.RealtimeSession(Config(dry_run=True, notify=False))
        with mock.patch.object(session.feedback, "notify") as notify:
            await session._on_event({"type": "error", "error": {
                "code": "session_expired",
                "message": "Your session hit the maximum duration of 60 minutes."}})
        notify.assert_not_called()
        log = self.log.read_text()
        self.assertIn("note    session_expired", log)
        self.assertNotIn("error   session_expired", log)


# The inherited tests belong to test_realtime_failover; do not run them twice.
for _name in list(vars(FailoverTests)):
    if _name.startswith("test_"):
        setattr(RotationTests, _name, None)

if __name__ == "__main__":
    unittest.main()
