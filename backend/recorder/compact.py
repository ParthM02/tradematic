"""Optional: squeeze finished hour files further, losslessly (slow; not needed day to day).

    uv run --group recorder python -m backend.recorder.compact [--data DIR] [--level 19]
                                                               [--keep-source] [--min-age-minutes 90]

For each finished hour file (.jsonl.zst or older .jsonl.gz) without a compact version:
  1. read every record (a damaged tail from an unclean shutdown is skipped),
  2. for coinbase, fold each `match` into the earlier `ticker` for the same trade: the
     match keeps only the fields the ticker lacks (maker/taker order ids, its own recv_ns)
     and its line offset, so both records and the exact line order come back on read,
  3. write high-level zstd to a temp file, read that file back and check it reproduces the
     source records exactly, in order,
  4. only then rename it to <name>.compact.jsonl.zst and delete the source.
Measured on the 2026-11-17 overnight recording: about 40% of the gzip size for Webull and
70% for Coinbase, at 2-4 minutes of laptop CPU per busy hour at level 19.
"""
import argparse
import json
import os
import time
from pathlib import Path

import zstandard

from .config import REPO_ROOT
from .records import hour_key, iter_raw, iter_records, match_from_ticker

MATCH_KEYS = ["type", "trade_id", "maker_order_id", "taker_order_id", "side", "size", "price",
              "product_id", "sequence", "time"]
STREAMS = ("webull", "coinbase", "markets", "settlements", "health")


def _dumps(r):
    return json.dumps(r, separators=(",", ":"), ensure_ascii=False)


def fold_coinbase(records):
    """Fold each match into the first earlier ticker for the same trade, when exact."""
    first_ticker = {}
    for i, r in enumerate(records):
        d = r.get("data")
        if r.get("type") == "ticker" and isinstance(d, dict):
            first_ticker.setdefault((d.get("product_id"), d.get("trade_id")), i)
    extras, absorbed = {}, set()
    for j, r in enumerate(records):
        if r.get("type") != "match" or set(r) != {"recv_ns", "src", "type", "data"}:
            continue
        d = r["data"]
        if not isinstance(d, dict) or list(d) != MATCH_KEYS:
            continue
        i = first_ticker.get((d["product_id"], d["trade_id"]))
        if i is None or i > j or i in extras:
            continue
        m = {"recv_ns": r["recv_ns"], "maker_order_id": d["maker_order_id"],
             "taker_order_id": d["taker_order_id"], "d": j - i}
        try:
            same = match_from_ticker(records[i], m) == r
        except (KeyError, TypeError):
            same = False
        if same:
            extras[i] = m
            absorbed.add(j)
    return [dict(r, _m=extras[i]) if i in extras else r
            for i, r in enumerate(records) if i not in absorbed]


def compact_path(src):
    return Path(src).with_name(hour_key(src) + ".compact.jsonl.zst")


def compact_file(src, level=19, keep_source=False):
    """Returns (records, source bytes, compact bytes). Raises and keeps the source on any mismatch."""
    src = Path(src)
    dest = compact_path(src)
    if dest.exists():
        raise FileExistsError(dest)
    records = list(iter_records(src))
    canonical = [_dumps(r) for r in records]
    body = fold_coinbase(records) if src.parent.parent.name == "coinbase" else records
    tmp = dest.with_name(dest.name + ".partial")
    payload = "".join(_dumps(r) + "\n" for r in body).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=level, write_checksum=True).compress(payload))
        f.flush()
        os.fsync(f.fileno())
    restored = [_dumps(r) for r in iter_records(tmp)]
    if restored != canonical or sum(1 for _ in iter_raw(tmp)) != len(body):
        tmp.unlink()
        raise RuntimeError(f"round-trip mismatch for {src}; source kept")
    os.replace(tmp, dest)
    size0 = src.stat().st_size
    if not keep_source:
        src.unlink()
    return len(records), size0, dest.stat().st_size


def candidates(root, min_age_minutes):
    """Finished, not-yet-compacted hour files (the recorder has not touched them recently)."""
    now = time.time()
    for stream in STREAMS:
        for p in sorted(Path(root, stream).glob("*/*")):
            name = p.name
            if name.endswith((".compact.jsonl.zst", ".partial")):
                continue
            if not name.endswith((".jsonl.zst", ".jsonl.gz")):
                continue
            if compact_path(p).exists() or now - p.stat().st_mtime < min_age_minutes * 60:
                continue
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "backend/data/recorder"))
    ap.add_argument("--level", type=int, default=19)
    ap.add_argument("--keep-source", action="store_true")
    ap.add_argument("--min-age-minutes", type=float, default=90,
                    help="only files untouched for this long (the hour is over and closed)")
    a = ap.parse_args()
    before = after = 0
    for p in candidates(a.data, a.min_age_minutes):
        t = time.time()
        n, size0, size1 = compact_file(p, a.level, a.keep_source)
        before, after = before + size0, after + size1
        print(f"{Path(p).relative_to(a.data)}: {n:,} records  {size0/2**20:.1f} -> {size1/2**20:.1f} MB "
              f"({size1/max(size0, 1):.0%}) in {time.time()-t:.0f}s", flush=True)
    if before:
        print(f"total {before/2**20:.1f} -> {after/2**20:.1f} MB ({after/before:.0%})")


if __name__ == "__main__":
    main()
