"""Recorder-driven daily Kaggle upload: schedule, isolation from the recorder, shutdown."""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.recorder import autoupload
from backend.recorder.autoupload import AutoUpload, next_daily

LOG = logging.getLogger("test")


class FakeWriter:
    def __init__(self):
        self.rows = []

    def write(self, stream, src, kind, data, recv_ns=None):
        self.rows.append((kind, data))


def fake_upload(monkeypatch, script):
    """Make the child process run `script` (python code) instead of the real uploader."""
    monkeypatch.setattr(autoupload, "command", lambda env, data, summary: [
        sys.executable, "-c", script.replace("SUMMARY", str(summary))])


def test_next_daily_local_time():
    tz = ZoneInfo("America/Chicago")
    now = datetime(2027, 1, 14, 2, 0, tzinfo=tz)
    assert next_daily(now, "03:30") == datetime(2027, 1, 14, 3, 30, tzinfo=tz)
    assert next_daily(datetime(2027, 1, 14, 3, 30, tzinfo=tz), "03:30") == datetime(2027, 1, 15, 3, 30, tzinfo=tz)


def test_enabled_needs_dataset_and_token(monkeypatch):
    for k in ("KAGGLE_DATASET", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "RECORDER_UPLOAD"):
        monkeypatch.delenv(k, raising=False)
    assert not autoupload.enabled()
    monkeypatch.setenv("KAGGLE_DATASET", "me/ticks")
    assert not autoupload.enabled()
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    assert autoupload.enabled()
    monkeypatch.setenv("RECORDER_UPLOAD", "off")
    assert not autoupload.enabled()


def test_success_and_failure_are_recorded(tmp_path, monkeypatch):
    w = FakeWriter()
    up = AutoUpload(tmp_path, None, w, LOG)
    fake_upload(monkeypatch, "import json; json.dump({'files': 3, 'bytes': 2**30, 'days': ['a','b']}, "
                             "open(r'SUMMARY','w')); print('uploaded')")
    assert asyncio.run(up.attempt(asyncio.Event())) is True
    assert up.state()["last_result"] == "ok" and up.state()["files"] == 3
    assert "uploaded" in up.log_path.read_text()
    fake_upload(monkeypatch, "import sys; print('boom', file=sys.stderr); sys.exit(2)")
    assert asyncio.run(up.attempt(asyncio.Event())) is False
    assert up.state()["last_result"] == "exit 2" and "boom" in up.log_path.read_text()
    kinds = [k for k, _ in w.rows]
    assert kinds == ["upload_started", "upload_finished", "upload_started", "upload_finished"]
    assert w.rows[1][1]["ok"] and not w.rows[3][1]["ok"]


def test_recorder_shutdown_stops_a_running_upload(tmp_path, monkeypatch):
    fake_upload(monkeypatch, "import time; time.sleep(60)")
    up = AutoUpload(tmp_path, None, FakeWriter(), LOG)

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(up.attempt(stop))
        await asyncio.sleep(0.5)
        t0 = time.time()
        stop.set()
        result = await task
        return result, time.time() - t0

    result, took = asyncio.run(main())
    assert result is None and took < 10 and up.proc is None
    assert "last_failure" not in up.state()  # a stop is not counted as a failure


def test_upload_runs_in_its_own_session(tmp_path, monkeypatch):
    """Ctrl-C goes to the terminal's foreground process group; the upload is not in it."""
    fake_upload(monkeypatch, "import time; time.sleep(1)")
    up = AutoUpload(tmp_path, None, FakeWriter(), LOG)

    async def main():
        task = asyncio.create_task(up.attempt(asyncio.Event()))
        await asyncio.sleep(0.3)
        pid = up.proc.pid
        isolated = os.getsid(pid) == pid and os.getpgid(pid) == pid and os.getpgid(pid) != os.getpgid(0)
        return isolated, await task

    assert asyncio.run(main()) == (True, True)


def test_failed_upload_retries_then_waits_for_next_day(tmp_path, monkeypatch):
    fake_upload(monkeypatch, "import sys; sys.exit(1)")
    up = AutoUpload(tmp_path, None, FakeWriter(), LOG)
    sleeps = []

    async def main():
        stop = asyncio.Event()

        async def fake_sleep(stop_, seconds):
            sleeps.append(seconds)
            if len(sleeps) > 4:
                stop.set()

        await up.run(stop, fake_sleep)

    asyncio.run(main())
    # stale state -> start-up run after 15 min, then 3 hourly retries, then the daily wait
    assert sleeps[0] == 15 * 60 and sleeps[1:4] == [3600, 3600, 3600] and sleeps[4] > 0
    assert len([k for k, _ in up.writer.rows if k == "upload_started"]) == 4
