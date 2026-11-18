"""Kalshi maintenance handling: weekly window, announced windows, status pauses, recorder behavior."""
import asyncio
import logging
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

from backend.recorder import exchange
from backend.recorder.__main__ import WebullSlot, stalled, webull_supervisor
from backend.recorder.exchange import (
    Maintenance,
    parse_announced,
    paused_indexes,
    quiet_intervals,
    weekly_window,
)

ET = ZoneInfo("America/New_York")


def ts(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET).timestamp()


def test_weekly_window_thursday_3_to_5_et_with_buffer_and_dst():
    # 2027-01-14 is a Thursday in EST; 2027-07-15 is a Thursday in EDT
    for y, mo, d in [(2027, 1, 14), (2027, 7, 15)]:
        assert weekly_window(ts(y, mo, d, 2, 44), 15) is None
        w = weekly_window(ts(y, mo, d, 2, 45), 15)
        assert w == (ts(y, mo, d, 2, 45), ts(y, mo, d, 5, 15))
        assert weekly_window(ts(y, mo, d, 4, 0), 0) is not None
        assert weekly_window(ts(y, mo, d, 5, 14), 15) is not None
        assert weekly_window(ts(y, mo, d, 5, 15), 15) is None
    assert weekly_window(ts(2027, 1, 13, 4, 0), 15) is None  # Wednesday
    assert weekly_window(ts(2027, 1, 15, 4, 0), 15) is None  # Friday
    # a large buffer crosses midnight into Wednesday
    assert weekly_window(ts(2027, 1, 13, 23, 30), 240) is not None


def test_announced_windows_and_status_parsing():
    sched = {"schedule": {"maintenance_windows": [
        {"start_datetime": "2027-01-20T14:00:00Z", "end_datetime": "2027-01-20T15:00:00Z"},
        {"start_datetime": "bad"}]}}
    windows = parse_announced(sched)
    assert len(windows) == 1
    m = Maintenance(buffer_minutes=10)
    m.update_schedule(sched)
    s, e = windows[0]
    assert m.reasons(s - 601) == [] and m.reasons(s - 599) == ["announced"]
    assert m.reasons(e + 599) == ["announced"] and not m.quiet(e + 601)
    ok = {"exchange_active": True, "trading_active": True, "exchange_index_statuses": [
        {"description": "Crypto & Commodities", "exchange_active": True, "trading_active": True}]}
    assert paused_indexes(ok) == []
    crypto_off = {**ok, "exchange_index_statuses": [
        {"description": "Crypto & Commodities", "exchange_active": True, "trading_active": False}]}
    assert paused_indexes(crypto_off) == ["Crypto & Commodities"]
    assert paused_indexes({**ok, "trading_active": False}) == ["Default"]


def test_status_pause_needs_two_reads_for_crypto_index_only():
    m = Maintenance(0)
    off = {"exchange_active": True, "trading_active": True, "exchange_index_statuses": [
        {"description": "Crypto & Commodities", "exchange_active": False, "trading_active": False}]}
    on = {"exchange_active": True, "trading_active": True, "exchange_index_statuses": []}
    now = ts(2027, 1, 18, 12)  # a Monday noon: not the weekly window
    m.update_status(off)
    assert not m.quiet(now)  # one false reading for a non-default index is re-read first
    m.update_status(off)
    assert m.reasons(now) == ["paused:Crypto & Commodities"]
    m.update_status(on)
    assert not m.quiet(now)
    m.update_status({"exchange_active": False, "trading_active": False})
    assert m.quiet(now)  # default index pause counts at once


def test_quiet_intervals_from_health_records():
    recs = [{"recv_ns": 1, "type": "health", "data": {}},
            {"recv_ns": 2, "type": "maintenance", "data": {"quiet": True}},
            {"recv_ns": 5, "type": "maintenance", "data": {"quiet": False}},
            {"recv_ns": 8, "type": "maintenance", "data": {"quiet": True}},
            {"recv_ns": 9, "type": "health", "data": {}}]
    assert quiet_intervals(recs) == [(2, 5), (8, 9)]


def test_stall_clock_restarts_after_maintenance():
    h = {"desired": 5, "last_message_age_s": 7200}  # silent through a 2 h window
    end_of_quiet = 10_000 * 10**9
    assert not stalled(h, 600, end_of_quiet, end_of_quiet + 300 * 10**9)  # grace after the window
    assert stalled(h, 600, end_of_quiet, end_of_quiet + 601 * 10**9)


class FakeWriter:
    def __init__(self):
        self.rows = []

    def write(self, stream, src, kind, data, recv_ns=None):
        self.rows.append((stream, kind, data))


class FakeStreamer:
    started: ClassVar[list] = []

    def __init__(self, cfg, writer):
        self.desired, self.on_fatal, self.stopped = set(), None, False

    def start(self):
        FakeStreamer.started.append(self)

    def stop(self):
        self.stopped = True

    def set_desired(self, s):
        self.desired = set(s)

    def health(self):
        return {"connected": True, "desired": len(self.desired), "subscribed": 0, "last_message_age_s": None}


class FakeUniverse:
    markets: ClassVar[dict] = {"KXBTC15M-A": {}, "KXBTCD-B": {}}


def run_supervisor(maint_quiet_sequence, monkeypatch, deaths=1, data=False):
    """Kill Webull `deaths` times and report what the supervisor did (it must never exit)."""
    import backend.recorder.webull_stream as ws

    monkeypatch.setattr(ws, "WebullStreamer", FakeStreamer)
    FakeStreamer.started = []
    seq = list(maint_quiet_sequence)
    waits = []

    class M:
        def quiet(self):
            return seq.pop(0) if len(seq) > 1 else seq[0]

        def reasons(self):
            return ["weekly"]

    async def fast_sleep(stop, seconds):
        waits.append(seconds)
        await asyncio.sleep(0)

    async def main():
        monkeypatch.setattr("backend.recorder.__main__._sleep", fast_sleep)
        loop = asyncio.get_running_loop()
        stop, w = asyncio.Event(), FakeWriter()
        slot = WebullSlot(None, w, loop, True)
        slot.start(FakeUniverse.markets)
        task = asyncio.create_task(webull_supervisor(slot, M(), FakeUniverse(), w, stop))
        for n in range(deaths):
            await asyncio.sleep(0)
            FakeStreamer.started[-1].last_message_ns = 1 if data else 0
            FakeStreamer.started[-1].on_fatal()  # Webull's loop dies (from its thread)
            for _ in range(100):
                await asyncio.sleep(0)
                if len(FakeStreamer.started) > n + 1:
                    break
        assert not task.done()  # never exits on its own
        stop.set()
        await task
        return w, slot, waits

    return asyncio.run(main())


def test_supervisor_waits_out_maintenance_then_restarts(monkeypatch):
    w, slot, _ = run_supervisor([True, True, False], monkeypatch)
    assert slot.restarts == 1
    assert len(FakeStreamer.started) == 2 and FakeStreamer.started[0].stopped
    assert FakeStreamer.started[1].desired == set(FakeUniverse.markets)  # resubscribes everything
    assert [k for _, k, _ in w.rows] == ["webull_waiting", "webull_restart"]


def test_supervisor_restarts_outside_maintenance_with_backoff(monkeypatch):
    _, slot, waits = run_supervisor([False], monkeypatch, deaths=4)
    assert slot.restarts == 4 and waits == [60, 120, 240, 480]  # failing without data: doubles
    _, slot, waits = run_supervisor([False], monkeypatch, deaths=3, data=True)
    assert waits == [60, 60, 60]  # each connection delivered data: back to 60 s


def test_next_backoff_caps_at_15_minutes():
    from backend.recorder.__main__ import next_backoff

    b = None
    for _ in range(10):
        b = next_backoff(b, got_data=False)
    assert b == 900 and next_backoff(900, got_data=True) == 60


def test_exchange_run_writes_quiet_transitions(monkeypatch):
    calls = {"n": 0}
    statuses = [{"exchange_active": False, "trading_active": False},
                {"exchange_active": True, "trading_active": True}]

    def fake_get(path):
        if path == "/exchange/schedule":
            return {"schedule": {"maintenance_windows": []}}
        calls["n"] += 1
        return statuses[min(calls["n"] - 1, 1)]

    monkeypatch.setattr(exchange, "kalshi_get", fake_get)
    monkeypatch.setattr(exchange, "weekly_window", lambda now, buf: None)

    async def main():
        stop, w = asyncio.Event(), FakeWriter()

        async def sleep(stop_, seconds):
            if calls["n"] >= 3:
                stop.set()

        await exchange.run(Maintenance(0), w, stop, sleep, logging.getLogger("t"))
        return w

    w = asyncio.run(main())
    events = [(k, d["quiet"]) for _, k, d in w.rows if k == "maintenance"]
    assert events == [("maintenance", True), ("maintenance", False)]
