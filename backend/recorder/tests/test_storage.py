"""Storage formats: zstd writer, crash recovery, multi-format reader, compaction, upload staging."""
import gzip
import json
import os
import time

import pytest
import zstandard

from backend.recorder import upload
from backend.recorder.__main__ import stalled
from backend.recorder.compact import candidates, compact_file, fold_coinbase
from backend.recorder.records import hour_files, iter_raw, iter_records
from backend.recorder.writer import _ZstdFile, hour_path

NOW_NS = 1_800_000_000 * 10**9  # 2027-01-15T08:00:00Z


def rec(i, kind="event-tick", **data):
    return {"recv_ns": NOW_NS + i, "src": "webull", "type": kind, "data": data or {"i": i}}


def write_gz(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ crash recovery ----
def test_unclosed_zstd_file_reads_up_to_last_flush(tmp_path):
    p = tmp_path / "x.jsonl.zst"
    f = _ZstdFile(p, level=9)
    for i in range(3):
        f.write((json.dumps(rec(i)) + "\n").encode())
    f.flush()
    f.write((json.dumps(rec(3)) + "\n").encode())  # buffered, never flushed: lost in a crash
    f._fh.flush()
    # simulate a crash: no close(), the frame never ends
    assert [r["recv_ns"] - NOW_NS for r in iter_records(p)] == [0, 1, 2]


def test_truncated_and_damaged_tails_are_skipped(tmp_path):
    p = tmp_path / "x.jsonl.zst"
    f = _ZstdFile(p, level=9)
    for i in range(50):
        f.write((json.dumps(rec(i)) + "\n").encode())
    f.flush()
    f.close()
    good = p.read_bytes()
    p.write_bytes(good + b"\x28\xb5\x2f\xfd garbage")  # a broken frame after a power cut
    assert len(list(iter_records(p))) == 50
    p.write_bytes(good[: len(good) // 2])  # cut mid-block
    assert len(list(iter_records(p))) < 50  # no exception; partial data only
    g = tmp_path / "y.jsonl.gz"
    write_gz(g, [rec(i) for i in range(20)])
    g.write_bytes(g.read_bytes()[:-15])
    assert 0 < len(list(iter_records(g))) <= 20
    (tmp_path / "empty.jsonl.zst").write_bytes(b"")
    assert list(iter_records(tmp_path / "empty.jsonl.zst")) == []


def test_reader_handles_large_multiframe_files(tmp_path):
    p = tmp_path / "big.jsonl.zst"
    for session in range(3):  # three recorder sessions append three frames
        f = _ZstdFile(p, level=3)
        for i in range(20_000):
            f.write((json.dumps(rec(session * 20_000 + i, pad="x" * 50)) + "\n").encode())
            if i % 5000 == 0:
                f.flush()
        f.close()
    assert sum(1 for _ in iter_records(p)) == 60_000


def test_hour_files_prefers_compact_then_zstd_then_gzip(tmp_path):
    d = tmp_path / "webull" / "20270115"
    write_gz(d / "webull_20270115T08.jsonl.gz", [rec(0)])
    write_gz(d / "webull_20270115T09.jsonl.gz", [rec(0)])
    (d / "webull_20270115T09.jsonl.zst").write_bytes(zstandard.compress(b"{}\n"))
    (d / "webull_20270115T10.jsonl.zst").write_bytes(zstandard.compress(b"{}\n"))
    (d / "webull_20270115T10.compact.jsonl.zst").write_bytes(zstandard.compress(b"{}\n"))
    (d / "webull_20270115T11.jsonl.zst.partial").write_bytes(b"")
    assert [p.name for p in hour_files(tmp_path, "webull")] == [
        "webull_20270115T08.jsonl.gz", "webull_20270115T09.jsonl.zst", "webull_20270115T10.compact.jsonl.zst"]


# ------------------------------------------------------------------ compaction ----
def cb_trade(i, trade_id, recv, side="buy"):
    t = {"recv_ns": recv, "src": "coinbase", "type": "ticker", "data": {
        "type": "ticker", "sequence": 100 + trade_id, "product_id": "BTC-USD", "price": "65000.01",
        "best_bid": "65000", "side": side, "time": "2027-01-15T08:00:00.1Z", "trade_id": trade_id,
        "last_size": "0.5"}}
    m = {"recv_ns": recv + 7, "src": "coinbase", "type": "match", "data": {
        "type": "match", "trade_id": trade_id, "maker_order_id": f"mk{i}", "taker_order_id": f"tk{i}",
        "side": {"buy": "sell", "sell": "buy"}[side], "size": "0.5", "price": "65000.01",
        "product_id": "BTC-USD", "sequence": 100 + trade_id, "time": "2027-01-15T08:00:00.1Z"}}
    return t, m


def coinbase_hour():
    t1, m1 = cb_trade(1, 1, NOW_NS)
    t2, m2 = cb_trade(2, 2, NOW_NS + 10, side="sell")
    t3, m3 = cb_trade(3, 3, NOW_NS + 20)
    m3["data"]["size"] = "0.6"  # does not match its ticker: must stay a separate record
    t4, m4 = cb_trade(4, 4, NOW_NS + 30)
    hb = {"recv_ns": NOW_NS + 40, "src": "coinbase", "type": "heartbeat", "data": {"x": 1}}
    # ticker before its match, with another trade between; match-before-ticker; heartbeat
    return [t1, t2, m1, m2, t3, m3, m4, t4, hb]


def test_fold_coinbase_round_trips_exactly_in_order(tmp_path):
    records = coinbase_hour()
    folded = fold_coinbase(json.loads(json.dumps(records)))
    assert len(folded) == len(records) - 2  # m1 and m2 folded; m3 (differs) and m4 (before) kept
    p = tmp_path / "f.jsonl.zst"
    p.write_bytes(zstandard.compress("".join(json.dumps(r) + "\n" for r in folded).encode()))
    assert list(iter_records(p)) == records


def test_compact_file_replaces_source_and_is_lossless(tmp_path):
    src = tmp_path / "coinbase" / "20270115" / "coinbase_20270115T08.jsonl.gz"
    write_gz(src, coinbase_hour())
    n, _, _ = compact_file(src, level=19)
    out = src.with_name("coinbase_20270115T08.compact.jsonl.zst")
    assert n == 9 and out.exists() and not src.exists()
    assert list(iter_records(out)) == coinbase_hour()
    assert sum(1 for _ in iter_raw(out)) == 7  # stored compactly
    with pytest.raises(FileExistsError):
        write_gz(src, coinbase_hour())
        compact_file(src)


def test_compact_candidates_skip_recent_and_already_compacted(tmp_path):
    d = tmp_path / "webull" / "20270115"
    write_gz(d / "webull_20270115T08.jsonl.gz", [rec(0)])
    write_gz(d / "webull_20270115T09.jsonl.gz", [rec(0)])
    old = time.time() - 3 * 3600
    os.utime(d / "webull_20270115T08.jsonl.gz", (old, old))
    assert [p.name for p in candidates(tmp_path, 90)] == ["webull_20270115T08.jsonl.gz"]
    compact_file(d / "webull_20270115T08.jsonl.gz")
    assert list(candidates(tmp_path, 90)) == []


# ------------------------------------------------------------------ watchdog ----
def test_stall_detection():
    h = {"desired": 5, "last_message_age_s": 30}
    assert not stalled(h, 600, 0, 10**12)
    assert stalled({**h, "last_message_age_s": 601}, 600, 0, 10**12)
    assert stalled({**h, "last_message_age_s": None}, 600, 0, 601 * 10**9)  # never received
    assert not stalled({**h, "last_message_age_s": None}, 600, 0, 300 * 10**9)
    assert not stalled({**h, "desired": 0, "last_message_age_s": 9999}, 600, 0, 10**12)
    assert not stalled(None, 600, 0, 10**12) and not stalled(h, 0, 0, 10**15)


# ------------------------------------------------------------------ upload staging ----
def _hour_file(root, stream, recv_ns, age_s=3 * 3600, body=b"{}\n"):
    p = hour_path(root, stream, recv_ns)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(zstandard.compress(body))
    old = time.time() - age_s
    os.utime(p, (old, old))
    return p


def test_upload_stages_only_finished_recordings_in_day_folders(tmp_path):
    keep = [_hour_file(tmp_path, s, NOW_NS) for s in ("webull", "coinbase", "settlements")]
    keep.append(_hour_file(tmp_path, "webull", NOW_NS + 86400 * 10**9))  # next day
    _hour_file(tmp_path, "webull", NOW_NS + 2 * 86400 * 10**9, age_s=0)  # being written now
    for secret in (tmp_path / ".webull_token" / "token.txt", tmp_path / "parquet" / "wb_tick.parquet",
                   tmp_path / "logs" / "upload.log"):
        secret.parent.mkdir(parents=True)
        secret.write_text("x")
    files = upload.finished_files(tmp_path, 70)
    assert sorted(files) == sorted(keep)
    root = upload.stage(tmp_path, files, "me/ticks", "Tradematic ticks")
    staged = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert staged == sorted(["dataset-metadata.json", "_about/README.md"]
                            + [f"{p.parent.name}/{p.name}" for p in keep])
    assert sorted(x.name for x in root.iterdir()) == ["20270115", "20270116", "_about", "dataset-metadata.json"]
    assert all((root / p.parent.name / p.name).stat().st_ino == p.stat().st_ino for p in keep)  # links
    meta = json.loads((root / "dataset-metadata.json").read_text())
    assert meta == {"title": "Tradematic ticks", "id": "me/ticks", "licenses": [{"name": "other"}]}
    upload.stage(tmp_path, files, "me/ticks", "Tradematic ticks")  # idempotent
    assert all(p.exists() for p in keep)  # sources untouched


def test_top_level_limit_is_enforced(tmp_path):
    files = [_hour_file(tmp_path, "health", NOW_NS + d * 86400 * 10**9) for d in range(49)]
    with pytest.raises(RuntimeError, match="50 top-level"):
        upload.stage(tmp_path, files, "me/x", "Title x")
    root = upload.stage(tmp_path, files[:48], "me/x", "Title x")  # 48 days + _about + metadata
    assert len(list(root.iterdir())) == 50


def test_a_single_day_still_stages_two_folders(tmp_path):
    """Kaggle flattens an upload with exactly one folder; _about prevents that."""
    root = upload.stage(tmp_path, [_hour_file(tmp_path, "webull", NOW_NS)], "me/x", "Title x")
    folders = sorted(x.name for x in root.iterdir() if x.is_dir())
    assert folders == ["20270115", "_about"]


def test_archives_go_to_ram_disk_when_it_fits(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDER_UPLOAD_TMPDIR", str(tmp_path))
    d, where = upload.scratch_dir(1000)
    assert d == tmp_path / "tradematic-upload" and d.is_dir() and where == str(tmp_path)
    d, where = upload.scratch_dir(10**18)  # does not fit anywhere -> system temp, said so
    assert "not enough" in where


def test_status_line_summarizes_interval():
    from backend.recorder.__main__ import status_line

    totals = {"webull/webull/event-tick": 900, "webull/webull/event-quote": 100,
              "webull/recorder/subscribe_ok": 3, "coinbase/coinbase/ticker": 50, "health/recorder/health": 30}
    h = {"webull": {"connected": True, "subscribed": 26, "desired": 26, "last_message_age_s": 0.1},
         "universe": 26, "pending_settlements": 4, "queue": 0, "dropped": 0}
    line = status_line(totals, 15, h)
    assert line.startswith("last 15 min: webull 1,000 msgs, coinbase 50 msgs | webull connected 26/26")
    assert "OFF" not in line and "off" in status_line({}, 15, {**h, "webull": None})
