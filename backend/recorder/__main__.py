"""Run the live recorder.

    uv run --group recorder python -m backend.recorder                   # Webull + Coinbase
    uv run --group recorder python -m backend.recorder --no-webull       # Coinbase + discovery only
    uv run --group recorder python -m backend.recorder --seconds 300     # bounded run

Streams written under RECORDER_DATA_DIR (default backend/data/recorder/):
Files are zstd-compressed JSONL, one per stream per UTC hour (read them with
backend.recorder.records.iter_records). It runs until stopped. If Webull's connection
dies or sends nothing for RECORDER_STALL_SECONDS, only Webull is reconnected (after 60 s,
doubling up to 15 min while it keeps failing); everything else keeps recording. During
Kalshi maintenance (Thursday 3-5 AM ET, announced windows, or a reported pause) the
reconnect waits until maintenance ends; see exchange.py.
  webull/       event-quote, event-snapshot, event-tick, notice + recorder events
  coinbase/     ticker, match, heartbeat, gap, connected/disconnected
  markets/      Kalshi metadata for each newly selected contract (strike, open/close)
  settlements/  official result and expiration_value after close
  health/       periodic counters, subscription state, queue depth, maintenance events
"""
import argparse
import asyncio
import logging
import os
import signal
import time

from . import autoupload, coinbase, exchange
from .config import REPO_ROOT, from_env, load_env_file
from .discovery import Universe
from .settlements import SettlementTracker
from .writer import RecordWriter

log = logging.getLogger("recorder")


class ErrorLog:
    """Log the first failure of a streak and the recovery, not every retry."""

    def __init__(self, what):
        self.what, self.failing = what, False

    def failed(self, e):
        if not self.failing:
            log.warning("%s failed: %r (further failures logged on recovery)", self.what, e)
        self.failing = True

    def ok(self):
        if self.failing:
            log.info("%s recovered", self.what)
        self.failing = False


class WebullSlot:
    """Holds the current Webull streamer so it can be replaced after a maintenance outage."""

    def __init__(self, cfg, writer, loop, enabled):
        self.cfg, self.writer, self.loop, self.enabled = cfg, writer, loop, enabled
        self.streamer = None
        self.dead = asyncio.Event()
        self.reason = None
        self.restarts = 0

    def start(self, desired=()):
        if not self.enabled:
            return
        from .webull_stream import WebullStreamer

        s = WebullStreamer(self.cfg, self.writer)
        s.on_fatal = lambda: self.loop.call_soon_threadsafe(self.kill, "streaming stopped")
        s.desired = set(desired)
        self.reason = None
        self.dead.clear()
        self.streamer = s
        s.start()

    def kill(self, reason):
        """Ask the supervisor to replace the current streamer (call on the event loop)."""
        if not self.dead.is_set():
            self.reason = reason
            self.dead.set()

    def got_data(self):
        return self.streamer is not None and self.streamer.last_message_ns > 0

    def stop(self):
        if self.streamer is not None:
            self.streamer.stop()

    def set_desired(self, symbols):
        if self.streamer is not None:
            self.streamer.set_desired(symbols)

    def health(self):
        return self.streamer.health() if self.streamer is not None else None


async def discovery_loop(cfg, universe, writer, webull, tracker, stop):
    errors = ErrorLog("contract discovery")
    while not stop.is_set():
        try:
            by_series = await asyncio.to_thread(universe.fetch)
            added, removed, new_records = universe.refresh(by_series)
            for m in new_records:
                writer.write("markets", "kalshi", "market", m)
                tracker.track(m)
            if added or removed:
                writer.write("health", "recorder", "universe",
                             {"added": added, "removed": removed, "size": len(universe.markets)})
                log.debug("universe +%d -%d = %d", len(added), len(removed), len(universe.markets))
            await asyncio.to_thread(webull.set_desired, set(universe.markets))
            errors.ok()
        except Exception as e:  # noqa: BLE001 - keep recording through API hiccups
            writer.write("health", "recorder", "discovery_error", {"error": repr(e)[:1000]})
            errors.failed(e)
        await _sleep(stop, cfg.discovery_seconds)


async def settlement_loop(tracker, stop):
    while not stop.is_set():
        await asyncio.to_thread(tracker.poll)
        await _sleep(stop, 30)


def next_backoff(current, got_data, first=60, cap=900):
    """Seconds to wait before reconnecting Webull.

    At least 60 s, because Webull keeps an old session for about a minute and allows 5 per
    App Key. Doubles on each failure without data (e.g. bad credentials), up to 15 min;
    back to 60 s once a connection delivered data.
    """
    return first if got_data or current is None else min(current * 2, cap)


async def webull_supervisor(slot, maint, universe, writer, stop, first=60, cap=900):
    """Restart Webull streaming in-process whenever it dies or stalls; never exits.

    Coinbase, discovery and settlements keep running throughout. During Kalshi maintenance
    the restart waits until the quiet period ends.
    """
    backoff = None
    while not stop.is_set():
        dead = asyncio.create_task(slot.dead.wait())
        stopping = asyncio.create_task(stop.wait())
        await asyncio.wait({dead, stopping}, return_when=asyncio.FIRST_COMPLETED)
        dead.cancel()
        stopping.cancel()
        if stop.is_set():
            return
        got_data = slot.got_data()
        reason = slot.reason or "streaming stopped"
        slot.stop()
        if maint.quiet():
            log.info("Webull %s during Kalshi maintenance (%s); reconnecting when it ends",
                     reason, ", ".join(maint.reasons()))
            writer.write("health", "recorder", "webull_waiting", {"reason": reason, "reasons": maint.reasons()})
            while maint.quiet() and not stop.is_set():
                await _sleep(stop, 30)
            wait = first
        else:
            backoff = next_backoff(backoff, got_data, first, cap)
            wait = backoff
            log.warning("Webull %s; reconnecting in %ds", reason, wait)
            writer.write("health", "recorder", "webull_waiting", {"reason": reason, "retry_in": wait})
        await _sleep(stop, wait)
        if stop.is_set():
            return
        slot.restarts += 1
        writer.write("health", "recorder", "webull_restart", {"restarts": slot.restarts})
        log.info("restarting Webull streaming (restart %d)", slot.restarts)
        slot.start(universe.markets)


def stalled(webull_health, stall_seconds, started_ns, now_ns):
    """True when Webull should be streaming but nothing arrived for `stall_seconds`.

    `started_ns` is when silence could first count: process start, or the end of the most
    recent maintenance quiet period.
    """
    if not stall_seconds or webull_health is None or not webull_health["desired"]:
        return False
    age = webull_health["last_message_age_s"]
    since_start = (now_ns - started_ns) / 1e9
    age = since_start if age is None else min(age, since_start)
    return age > stall_seconds


def status_line(totals, minutes, h):
    """One human-readable summary of the last `minutes` (records by source, health now)."""
    by_src = {}
    for key, n in totals.items():
        stream, src, _kind = key.split("/")
        if stream in ("webull", "coinbase") and src != "recorder":
            by_src[stream] = by_src.get(stream, 0) + n
    w = h["webull"]
    webull = ("off" if w is None else
              f"{'connected' if w['connected'] else 'DISCONNECTED'} {w['subscribed']}/{w['desired']} subscribed")
    line = (f"last {minutes:.0f} min: webull {by_src.get('webull', 0):,} msgs, coinbase "
            f"{by_src.get('coinbase', 0):,} msgs | webull {webull} | contracts {h['universe']}, "
            f"pending settlements {h['pending_settlements']}, queue {h['queue']}, dropped {h['dropped']}")
    if h.get("maintenance"):
        line += " | Kalshi maintenance: " + ", ".join(h["maintenance"])
    return line


async def health_loop(cfg, writer, universe, webull, tracker, maint, stop):
    last = {}
    since_log = {}
    last_log = time.monotonic()
    silence_from = time.time_ns()  # silence before this moment never counts as a stall
    while not stop.is_set():
        await _sleep(stop, cfg.health_seconds)
        counts = {"/".join(k): v for k, v in writer.counts.items()}
        delta = {k: v - last.get(k, 0) for k, v in counts.items() if v != last.get(k, 0)}
        last = counts
        reasons = maint.reasons()
        h = {"interval_s": cfg.health_seconds, "new_records": delta, "queue": writer.q.qsize(),
             "dropped": writer.dropped, "universe": len(universe.markets),
             "pending_settlements": len(tracker.pending), "webull": webull.health(),
             "maintenance": reasons}
        writer.write("health", "recorder", "health", h)
        for k, v in delta.items():
            since_log[k] = since_log.get(k, 0) + v
        if cfg.log_seconds and time.monotonic() - last_log >= cfg.log_seconds:
            log.info("%s", status_line(since_log, (time.monotonic() - last_log) / 60, h))
            since_log, last_log = {}, time.monotonic()
        if reasons:
            silence_from = time.time_ns()
            continue
        if not webull.dead.is_set() and stalled(h["webull"], cfg.stall_seconds, silence_from, time.time_ns()):
            writer.write("health", "recorder", "stalled", {"stall_seconds": cfg.stall_seconds})
            webull.kill(f"sent nothing for {cfg.stall_seconds}s")
            silence_from = time.time_ns()  # the restarted connection gets a fresh clock


async def _sleep(stop, seconds):
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def main_async(cfg, use_webull, seconds, env_file=None):
    """Runs until Ctrl-C / SIGTERM (or --seconds). Webull failures are handled in-process."""
    writer = RecordWriter(cfg.data_dir, level=cfg.zstd_level).start()
    writer.write("health", "recorder", "start", cfg.public_summary())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    universe = Universe(cfg.series, cfg.hourly_strikes, cfg.max_symbols)
    tracker = SettlementTracker(writer)
    maint = exchange.Maintenance(cfg.maintenance_buffer_minutes)
    webull = WebullSlot(cfg, writer, loop, use_webull)
    webull.start()
    tasks = [
        asyncio.create_task(coinbase.run(cfg.products, writer, stop)),
        asyncio.create_task(exchange.run(maint, writer, stop, _sleep, log)),
        asyncio.create_task(discovery_loop(cfg, universe, writer, webull, tracker, stop)),
        asyncio.create_task(settlement_loop(tracker, stop)),
        asyncio.create_task(health_loop(cfg, writer, universe, webull, tracker, maint, stop)),
        asyncio.create_task(webull_supervisor(webull, maint, universe, writer, stop)),
    ]
    uploader = None
    if env_file is not None and autoupload.enabled():
        uploader = autoupload.AutoUpload(cfg.data_dir, env_file, writer, log, cfg.upload_time)
        tasks.append(asyncio.create_task(uploader.run(stop, _sleep)))
        log.info("daily Kaggle upload at %s local time -> %s", cfg.upload_time,
                 os.environ.get("KAGGLE_DATASET"))
    if seconds:
        loop.call_later(seconds, stop.set)
    await stop.wait()
    webull.stop()
    if uploader is not None:
        uploader.terminate()
    await asyncio.gather(*tasks, return_exceptions=True)
    writer.write("health", "recorder", "stop", {"pending_settlements": len(tracker.pending),
                                                "webull_restarts": webull.restarts})
    writer.close()
    log.info("stopped; records written: %d", sum(writer.counts.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    p.add_argument("--no-webull", action="store_true", help="skip Webull (no keys needed)")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl-C)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # The Webull SDK logs whole failed requests (headers include the app key). The recorder
    # logs a one-line summary of every failure itself, so keep the SDK's own logger silent.
    logging.getLogger("webull").setLevel(logging.CRITICAL)
    # ...except token setup, which says when to approve the token in the Webull app (2FA).
    logging.getLogger("webull.core.http.initializer.token").setLevel(logging.INFO)
    load_env_file(args.env_file)
    cfg = from_env()
    use_webull = not args.no_webull
    if use_webull and not cfg.webull_enabled:
        p.error(f"WEBULL_APP_KEY / WEBULL_APP_SECRET not set (looked in {args.env_file}); "
                "use --no-webull to record Coinbase only")
    log.info("config %s -> %s", cfg.public_summary(), cfg.data_dir)
    started = time.time()
    asyncio.run(main_async(cfg, use_webull, args.seconds, args.env_file))
    log.info("ran %.0fs", time.time() - started)


if __name__ == "__main__":
    main()
