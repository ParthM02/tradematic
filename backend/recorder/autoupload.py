"""Daily Kaggle upload, started by the recorder, isolated from recording.

The upload runs `python -m backend.recorder.upload` as a separate child process, so it
cannot block or crash the recorder:
  * lowest CPU priority (nice 19) and, on Linux, idle-only disk I/O (ionice -c3);
  * its own session, so Ctrl-C in the recorder's terminal does not reach it directly;
    the recorder stops it on shutdown;
  * no extra disk copies: files are hard-linked, then read once for upload;
  * output goes to <data>/logs/upload.log; the recorder logs one line per attempt.
Schedule: daily at RECORDER_UPLOAD_TIME (local time of this machine, default 03:30). A
start-up run happens 15 min after launch if the last success is over 25 h old. A failed
attempt is retried hourly, up to 3 times, then it waits for the next daily slot. At most
one upload runs at a time, with a 4 h limit. Last results are kept in
<data>/.upload_state.json.
"""
import asyncio
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import REPO_ROOT

STARTUP_DELAY_S, RETRY_S, MAX_RETRIES, TIMEOUT_S, STALE_S = 15 * 60, 3600, 3, 4 * 3600, 25 * 3600


def enabled():
    has_token = os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    off = os.environ.get("RECORDER_UPLOAD", "").strip().lower() in ("0", "off", "false", "no")
    return bool(os.environ.get("KAGGLE_DATASET", "").strip() and has_token and not off)


def next_daily(now, hhmm):
    """Next local datetime at hh:mm strictly after `now` (an aware local datetime)."""
    h, m = (int(x) for x in hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def command(env_file, data_dir, summary):
    cmd = [sys.executable, "-m", "backend.recorder.upload", "--env-file", str(env_file),
           "--data", str(data_dir), "--summary-file", str(summary)]
    prefix = []
    if shutil.which("ionice"):  # Linux: only use the disk when nothing else wants it
        prefix += ["ionice", "-c3"]
    if shutil.which("nice"):
        prefix += ["nice", "-n", "19"]
    return prefix + cmd


class AutoUpload:
    def __init__(self, data_dir, env_file, writer, log, upload_time="03:30"):
        self.data_dir, self.env_file, self.writer, self.log = Path(data_dir), env_file, writer, log
        self.upload_time = upload_time
        self.state_path = self.data_dir / ".upload_state.json"
        self.log_path = self.data_dir / "logs" / "upload.log"
        self.proc = None

    def state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def save(self, **kv):
        s = self.state() | kv
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=2))
        os.replace(tmp, self.state_path)

    async def attempt(self, stop):
        """One upload in a child process. True = success, False = failed, None = stopped."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.data_dir / "logs" / "upload_summary.json"
        summary.unlink(missing_ok=True)
        started = time.time()
        self.writer.write("health", "recorder", "upload_started", {})
        self.log.info("Kaggle upload started (low priority; output in %s)", self.log_path)
        with open(self.log_path, "ab") as out:  # noqa: ASYNC230 - one small append per day
            out.write(f"\n===== {datetime.now().astimezone().isoformat()} upload\n".encode())
            out.flush()
            self.proc = await asyncio.create_subprocess_exec(
                *command(self.env_file, self.data_dir, summary), cwd=str(REPO_ROOT),
                stdout=out, stderr=out, stdin=asyncio.subprocess.DEVNULL, start_new_session=True)
            code = await self._wait(stop)
        self.proc = None
        secs = time.time() - started
        info = {}
        try:
            info = json.loads(summary.read_text())
        except (OSError, ValueError):
            pass
        ok = code == 0
        self.writer.write("health", "recorder", "upload_finished",
                          {"ok": ok, "exit_code": code, "seconds": round(secs), **info})
        what = (f"{info['files']} files, {info['bytes']/2**30:.2f} GB" if "files" in info else "nothing new")
        if ok:
            self.log.info("Kaggle upload finished in %dm%02ds: %s", secs // 60, secs % 60, what)
            self.save(last_success=time.time(), last_result="ok", last_seconds=round(secs), **info)
        elif code is not None:
            self.log.warning("Kaggle upload failed (exit %s) after %ds; see %s", code, secs, self.log_path)
            self.save(last_failure=time.time(), last_result=f"exit {code}")
        return None if code is None else ok  # None: stopped by recorder shutdown

    async def _wait(self, stop):
        """Wait for the child; stop it on recorder shutdown or after TIMEOUT_S."""
        done = asyncio.create_task(self.proc.wait())
        stopping = asyncio.create_task(stop.wait())
        await asyncio.wait({done, stopping}, timeout=TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        stopping.cancel()
        if done.done():
            return done.result()
        reason = "recorder stopping" if stop.is_set() else f"over {TIMEOUT_S // 3600} h"
        self.log.warning("stopping Kaggle upload (%s); it will run again later", reason)
        self.terminate()
        try:
            await asyncio.wait_for(done, 15)
        except TimeoutError:
            self.proc.kill()
            await done
        return None if stop.is_set() else -1

    def terminate(self):
        if self.proc is not None and self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)  # the child and anything it started
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()

    async def run(self, stop, sleep):
        last = self.state().get("last_success", 0)
        if time.time() - last > STALE_S:
            await sleep(stop, STARTUP_DELAY_S)
            await self._with_retries(stop, sleep)
        while not stop.is_set():
            now = datetime.now().astimezone()
            nxt = next_daily(now, self.upload_time)
            self.log.debug("next Kaggle upload at %s", nxt.isoformat())
            await sleep(stop, (nxt - now).total_seconds())
            await self._with_retries(stop, sleep)

    async def _with_retries(self, stop, sleep):
        for i in range(1 + MAX_RETRIES):
            if stop.is_set():
                return
            if await self.attempt(stop):
                return
            if i < MAX_RETRIES and not stop.is_set():
                self.log.info("retrying Kaggle upload in %d min", RETRY_S // 60)
                await sleep(stop, RETRY_S)
