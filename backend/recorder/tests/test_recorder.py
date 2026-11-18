import json
import time

import pytest

from backend.recorder.coinbase import TradeGapDetector
from backend.recorder.config import from_env
from backend.recorder.discovery import Universe, select
from backend.recorder.records import iter_records
from backend.recorder.settlements import SettlementTracker
from backend.recorder.writer import RecordWriter, hour_path

NOW = 1_800_000_000  # 2027-01-15T08:00:00Z


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def mkt(ticker, open_ts, close_ts, bid=None, ask=None, event=None):
    m = {"ticker": ticker, "event_ticker": event or ticker.rsplit("-", 1)[0],
         "open_time": iso(open_ts), "close_time": iso(close_ts)}
    if bid is not None:
        m["yes_bid_dollars"], m["yes_ask_dollars"] = f"{bid:.4f}", f"{ask:.4f}"
    return m


def ladder(event, close_ts, mids):
    return [mkt(f"{event}-T{i}", close_ts - 3600, close_ts, m - 0.01, m + 0.01, event) for i, m in enumerate(mids)]


def test_select_15m_takes_live_and_hourly_takes_nearest_50c_of_first_event():
    fifteen = [mkt("KXBTC15M-A-00", NOW - 60, NOW + 840), mkt("KXBTC15M-B-15", NOW + 840, NOW + 1740)]
    hourly = ladder("KXBTCD-H1", NOW + 1800, [0.05, 0.30, 0.48, 0.55, 0.90]) + \
        ladder("KXBTCD-H2", NOW + 5400, [0.5, 0.5])
    got = select({"KXBTC15M": fifteen, "KXBTCD": hourly}, hourly_strikes=2, now=NOW)
    assert set(got) == {"KXBTC15M-A-00", "KXBTCD-H1-T2", "KXBTCD-H1-T3"}


def test_select_skips_unpriced_and_one_sided_quotes():
    ms = ladder("KXETHD-H1", NOW + 600, [0.5]) + [mkt("KXETHD-H1-TX", NOW - 60, NOW + 600)]
    ms.append(mkt("KXETHD-H1-TY", NOW - 60, NOW + 600, 0.6, 0.6))  # crossed/locked: no mid
    assert set(select({"KXETHD": ms}, 5, now=NOW)) == {"KXETHD-H1-T0"}


def test_universe_is_sticky_until_close_and_capped():
    u = Universe(("KXSOLD",), hourly_strikes=1, max_symbols=2)
    added, removed, new = u.refresh({"KXSOLD": ladder("KXSOLD-H1", NOW + 600, [0.2, 0.5])}, now=NOW)
    assert added == ["KXSOLD-H1-T1"] and not removed and len(new) == 1
    # price moves: T0 is now nearest 50c; T1 is kept (sticky), T0 added
    added, removed, new = u.refresh({"KXSOLD": ladder("KXSOLD-H1", NOW + 600, [0.5, 0.9])}, now=NOW + 60)
    assert added == ["KXSOLD-H1-T0"] and not removed
    # cap of 2 blocks a third symbol
    more = ladder("KXSOLD-H1", NOW + 600, [0.1, 0.1, 0.5])
    added, _, _ = u.refresh({"KXSOLD": more}, now=NOW + 120)
    assert added == [] and len(u.markets) == 2
    # after close, everything is released
    added, removed, _ = u.refresh({"KXSOLD": []}, now=NOW + 601)
    assert sorted(removed) == ["KXSOLD-H1-T0", "KXSOLD-H1-T1"] and not u.markets


def test_writer_rotates_by_utc_hour_and_round_trips(tmp_path):
    w = RecordWriter(tmp_path, flush_seconds=0.05).start()
    t0 = NOW * 10**9
    w.write("webull", "webull", "event-tick", {"yes_price": "0.42"}, t0)
    w.write("webull", "webull", "event-tick", {"yes_price": "0.43"}, t0 + 3600 * 10**9)
    w.close()
    p0, p1 = hour_path(tmp_path, "webull", t0), hour_path(tmp_path, "webull", t0 + 3600 * 10**9)
    assert p0 != p1 and p0.name == "webull_20270115T08.jsonl.zst"
    assert list(iter_records(p0)) == [
        {"recv_ns": t0, "src": "webull", "type": "event-tick", "data": {"yes_price": "0.42"}}]
    # reopening appends a second zstd frame that still reads as one stream
    w2 = RecordWriter(tmp_path).start()
    w2.write("webull", "webull", "event-tick", {"yes_price": "0.44"}, t0 + 1)
    w2.close()
    assert [r["data"]["yes_price"] for r in iter_records(p0)] == ["0.42", "0.44"]


def test_trade_gap_detector():
    g = TradeGapDetector()
    assert g.check("BTC-USD", 10) == 0
    assert g.check("BTC-USD", 11) == 0
    assert g.check("BTC-USD", 15) == 3
    assert g.check("BTC-USD", 14) == 0  # out of order: ignored
    assert g.check("ETH-USD", 1) == 0


def test_event_protobuf_decodes_every_field():
    pb = pytest.importorskip("webull.data.quotes.subscribe.message_pb2")
    from backend.recorder.webull_stream import ProtoDictDecoder

    q = pb.EventQuote()
    q.basic.symbol, q.basic.timestamp = "KXBTC15M-26NOV162230-30", "1794885000123"
    lvl = q.yes_bids.add()
    lvl.price, lvl.size = "0.41", "120"
    q.no_bids.add(price="0.57", size="80")
    d = ProtoDictDecoder(pb.EventQuote).parse(q.SerializeToString())
    assert d == {"basic": {"symbol": "KXBTC15M-26NOV162230-30", "timestamp": "1794885000123"},
                 "yes_bids": [{"price": "0.41", "size": "120"}], "no_bids": [{"price": "0.57", "size": "80"}]}
    t = pb.EventTick(yes_price="0.42", no_price="0.58", volume="3", side="BUY", trade_id="x1", time="1")
    assert ProtoDictDecoder(pb.EventTick).parse(t.SerializeToString())["trade_id"] == "x1"


class FakeWriter:
    def __init__(self):
        self.rows = []

    def write(self, stream, src, kind, data, recv_ns=None):
        self.rows.append((stream, kind, data))


def test_settlement_tracker_waits_then_records(monkeypatch):
    calls = []
    payload = {"markets": [{"ticker": "E-T1", "result": "yes", "expiration_value": "1.5"},
                           {"ticker": "E-T2", "result": ""}, {"ticker": "OTHER", "result": "no"}]}
    monkeypatch.setattr("backend.recorder.settlements.kalshi_get", lambda path: calls.append(path) or payload)
    w = FakeWriter()
    tr = SettlementTracker(w)
    for t in ("E-T1", "E-T2"):
        tr.track(mkt(t, NOW - 3600, NOW, event="E"))
    tr.poll(now=NOW + 60)
    assert calls == []  # not due yet
    tr.poll(now=NOW + 130)
    assert [r[2]["ticker"] for r in w.rows if r[1] == "settlement"] == ["E-T1"]
    assert tr.pending["E"]["tickers"] == {"E-T2"}
    tr.poll(now=NOW + 6 * 3600 + 1)
    assert w.rows[-1][1] == "settlement_timeout" and not tr.pending


def test_config_rejects_bad_env_and_hides_secret(monkeypatch):
    monkeypatch.setenv("WEBULL_APP_SECRET", "s3cret")
    monkeypatch.setenv("WEBULL_APP_KEY", "k")
    cfg = from_env()
    assert cfg.webull_enabled and "s3cret" not in repr(cfg) and "s3cret" not in json.dumps(cfg.public_summary())
    monkeypatch.setenv("WEBULL_TRANSPORT", "udp")
    with pytest.raises(ValueError):
        from_env()
