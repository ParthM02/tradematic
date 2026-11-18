"""Kalshi maintenance awareness (Webull routes event-contract orders to Kalshi).

Kalshi pauses trading every Thursday 3:00-5:00 AM ET and may disconnect sessions then;
rarely it pauses the whole exchange, inside that window or during an unscheduled outage.
Docs: https://docs.kalshi.com/getting_started/maintenance_and_pauses

A "quiet period" is any of:
  * the weekly Thursday window, widened by `buffer_minutes` on each side (DST-aware),
  * an extra window announced in GET /exchange/schedule `maintenance_windows`, widened the same way,
  * GET /exchange/status (HTTP 200 only) reporting trading or the exchange inactive, for
    the default index or the "Crypto & Commodities" index. Per Kalshi's docs, non-200
    responses are not pauses, and a single false reading for a non-default index is
    re-read before it counts.
During a quiet period the recorder keeps recording whatever arrives, but it does not treat
silence or Webull errors as failures (no stall restarts), and it logs errors once instead
of every cycle. Start/end of every quiet period is written to the `health` stream as
`maintenance` events, so analysis can mask those windows (see `quiet_intervals`).
"""
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .discovery import kalshi_get

ET = ZoneInfo("America/New_York")
WEEKLY = (3, 3, 5)  # Thursday (weekday 3), 03:00 to 05:00 ET
CRYPTO_INDEX = "Crypto & Commodities"


def weekly_window(now_ts, buffer_minutes=15):
    """(start_ts, end_ts) of this week's Thursday window if `now_ts` is inside it (with buffer)."""
    now = datetime.fromtimestamp(now_ts, ET)
    buf = timedelta(minutes=buffer_minutes)
    for day_offset in (0, -1, 1):  # the buffer can cross midnight
        d = (now + timedelta(days=day_offset)).date()
        if d.weekday() != WEEKLY[0]:
            continue
        start = datetime(d.year, d.month, d.day, WEEKLY[1], tzinfo=ET) - buf
        end = datetime(d.year, d.month, d.day, WEEKLY[2], tzinfo=ET) + buf
        if start <= now < end:
            return start.timestamp(), end.timestamp()
    return None


def parse_announced(schedule_payload):
    """Extra maintenance windows from GET /exchange/schedule, as (start_ts, end_ts)."""
    out = []
    for w in (schedule_payload or {}).get("schedule", {}).get("maintenance_windows", []) or []:
        try:
            s = datetime.fromisoformat(w["start_datetime"]).timestamp()
            e = datetime.fromisoformat(w["end_datetime"]).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append((s, e))
    return out


def paused_indexes(status_payload):
    """Names of exchange indexes reported paused in a 200 /exchange/status body."""
    paused = []
    if not status_payload.get("exchange_active", True) or not status_payload.get("trading_active", True):
        paused.append("Default")
    for ix in status_payload.get("exchange_index_statuses", []) or []:
        if ix.get("description") == CRYPTO_INDEX and not (
                ix.get("exchange_active", True) and ix.get("trading_active", True)):
            paused.append(CRYPTO_INDEX)
    return paused


class Maintenance:
    def __init__(self, buffer_minutes=15):
        self.buffer = buffer_minutes
        self.announced = []
        self.paused = []         # confirmed paused indexes from the latest status reads
        self._suspect = []       # first false reading for the crypto index, awaiting re-read
        self.last_status_ok = None
        self.was_quiet = None

    def reasons(self, now_ts=None):
        now_ts = time.time() if now_ts is None else now_ts
        r = []
        if weekly_window(now_ts, self.buffer):
            r.append("weekly")
        b = self.buffer * 60
        if any(s - b <= now_ts < e + b for s, e in self.announced):
            r.append("announced")
        if self.paused:
            r.append("paused:" + ",".join(self.paused))
        return r

    def quiet(self, now_ts=None):
        return bool(self.reasons(now_ts))

    def update_status(self, payload):
        """Feed one 200 /exchange/status body. A crypto-index-only pause needs two reads."""
        now_paused = paused_indexes(payload)
        if "Default" in now_paused:
            self.paused = now_paused
        elif now_paused:
            self.paused = now_paused if self._suspect == now_paused else self.paused
        else:
            self.paused = []
        self._suspect = now_paused
        self.last_status_ok = time.time()

    def update_schedule(self, payload):
        self.announced = parse_announced(payload)


def quiet_intervals(health_records):
    """[(start_ns, end_ns)] of recorded quiet periods, for masking data in analysis.

    `health_records` are records from the `health` stream (records.iter_records). An
    interval still open at the end is closed at the last record's time.
    """
    out, start, last = [], None, None
    for r in health_records:
        last = r["recv_ns"]
        if r.get("type") != "maintenance":
            continue
        if r["data"].get("quiet") and start is None:
            start = r["recv_ns"]
        elif not r["data"].get("quiet") and start is not None:
            out.append((start, r["recv_ns"]))
            start = None
    if start is not None:
        out.append((start, last))
    return out


async def run(state, writer, stop, sleep, log, status_seconds=60, schedule_seconds=6 * 3600):
    """Poll Kalshi status/schedule and write a `maintenance` event whenever quiet changes."""
    import asyncio

    next_schedule = 0.0
    while not stop.is_set():
        now = time.time()
        if now >= next_schedule:
            try:
                sched = await asyncio.to_thread(kalshi_get, "/exchange/schedule")
                before = state.announced
                state.update_schedule(sched)
                if state.announced != before:
                    writer.write("health", "recorder", "maintenance_schedule",
                                 {"announced": state.announced})
                next_schedule = now + schedule_seconds
            except Exception as e:  # noqa: BLE001 - try again next cycle
                next_schedule = now + 600
                log.debug("exchange schedule unavailable: %r", e)
        try:
            state.update_status(await asyncio.to_thread(kalshi_get, "/exchange/status"))
        except Exception as e:  # noqa: BLE001 - non-200 is not a pause (Kalshi docs)
            log.debug("exchange status unavailable: %r", e)
        reasons = state.reasons()
        quiet = bool(reasons)
        if quiet != state.was_quiet:
            writer.write("health", "recorder", "maintenance", {"quiet": quiet, "reasons": reasons})
            if state.was_quiet is not None or quiet:
                log.info("maintenance quiet period %s (%s)", "started" if quiet else "ended",
                         ", ".join(reasons) or "trading active")
            state.was_quiet = quiet
        await sleep(stop, status_seconds)
