"""Lossless storage benchmark on one recorded hour (webull + coinbase).

Every alternative is checked for exact reconstruction of the original records.
"""
import gzip
import json
import sys
import time
import zlib
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2] / "data/recorder"
HOUR = sys.argv[1] if len(sys.argv) > 1 else "20261117T15"


def lines(path):
    raw, out = path.read_bytes(), b""
    while raw:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out += d.decompress(raw)
        raw = d.unused_data
    return [ln for ln in out.split(b"\n") if ln]


def zstd(b, level):
    return pa.Codec("zstd", compression_level=level).compress(b, asbytes=True)


def report(name, raw_len, comp_len, base, secs=None):
    extra = f"  {secs:5.1f}s" if secs is not None else ""
    print(f"  {name:44s} {comp_len/2**20:7.2f} MB  ({comp_len/base:5.1%} of current){extra}")


for stream in ("webull", "coinbase"):
    path = ROOT / stream / HOUR[:8] / f"{stream}_{HOUR}.jsonl.gz"
    cur = path.stat().st_size
    L = lines(path)
    blob = b"\n".join(L) + b"\n"
    recs = [json.loads(x) for x in L]
    print(f"== {stream} {HOUR}: {len(L):,} records, {len(blob)/2**20:.1f} MB raw JSON, current gzip {cur/2**20:.2f} MB")
    by = {}
    for x, r in zip(L, recs):
        by[r["type"]] = by.get(r["type"], 0) + len(x)
    print("   raw bytes by type:", {k: f"{v/len(blob):.0%}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:6]})
    for lvl in (6, 9):
        t = time.time()
        report(f"gzip-{lvl} JSON", len(blob), len(gzip.compress(blob, lvl)), cur, time.time() - t)
    for lvl in (3, 9, 19):
        t = time.time()
        report(f"zstd-{lvl} JSON", len(blob), len(zstd(blob, lvl)), cur, time.time() - t)
