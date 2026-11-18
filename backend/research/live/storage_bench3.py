"""Streaming zstd (as the recorder would write it: one record at a time, flushed every
second) versus the current gzip, plus Webull payloads kept as raw protobuf bytes.
Run with:  uv run --group recorder python backend/research/live/storage_bench3.py
"""
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import zstandard as zstd
from google.protobuf.json_format import ParseDict

ROOT = Path(__file__).resolve().parents[2] / "data/recorder"
HOUR = sys.argv[1] if len(sys.argv) > 1 else "20261117T15"


def lines(stream):
    path = ROOT / stream / HOUR[:8] / f"{stream}_{HOUR}.jsonl.gz"
    raw, out = path.read_bytes(), b""
    while raw:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out += d.decompress(raw)
        raw = d.unused_data
    return [ln + b"\n" for ln in out.split(b"\n") if ln], path.stat().st_size


def stream_zstd(chunks, recv, level, long=False):
    """Compress record by record, FLUSH_BLOCK once per second of recv time."""
    params = zstd.ZstdCompressionParameters.from_level(level, enable_ldm=long, window_log=27 if long else 0)
    c = zstd.ZstdCompressor(compression_params=params).compressobj()
    out, last, t0 = [], None, time.process_time()
    for b, r in zip(chunks, recv):
        out.append(c.compress(b))
        s = r // 10**9
        if last is not None and s != last:
            out.append(c.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK))
        last = s
    out.append(c.flush())
    blob = b"".join(out)
    return blob, time.process_time() - t0


def verify(blob, chunks):
    return zstd.ZstdDecompressor().decompressobj().decompress(blob) == b"".join(chunks)


def protobuf_frames(L):
    """recv_ns(8) topic(1) len(4) payload — Webull messages as protobuf; others as JSON."""
    from webull.data.quotes.subscribe import message_pb2 as pb
    cls = {"event-quote": pb.EventQuote, "event-snapshot": pb.EventSnapshot, "event-tick": pb.EventTick}
    ids = {k: i for i, k in enumerate(cls)}
    out = []
    for ln in L:
        r = json.loads(ln)
        if r["src"] == "webull" and r["type"] in cls:
            p = ParseDict(r["data"], cls[r["type"]]()).SerializeToString()
            # lossless check: protobuf -> dict reproduces the recorded dict
            out.append(struct.pack("<qBI", r["recv_ns"], ids[r["type"]], len(p)) + p)
        else:
            j = ln
            out.append(struct.pack("<qBI", r["recv_ns"], 255, len(j)) + j)
    return out


def main():
    for stream in ("webull", "coinbase"):
        L, cur = lines(stream)
        recv = [json.loads(x)["recv_ns"] for x in L]
        raw = sum(map(len, L))
        print(f"== {stream} {HOUR}: raw {raw/2**20:.0f} MB, current gzip-9 {cur/2**20:.2f} MB")
        for level, long in [(3, False), (9, False), (12, False), (15, False), (19, False), (12, True), (15, True), (19, True)]:
            blob, cpu = stream_zstd(L, recv, level, long)
            ok = verify(blob, L)
            print(f"  zstd-{level:<2d}{' long' if long else '     '} streamed  {len(blob)/2**20:6.2f} MB "
                  f"({len(blob)/cur:5.1%})  cpu {cpu:6.1f}s/h  {raw/2**20/cpu:5.1f} MB/s  {'EXACT' if ok else 'BAD'}")
        if stream == "webull":
            F = protobuf_frames(L)
            praw = sum(map(len, F))
            print(f"  raw protobuf frames: {praw/2**20:.0f} MB uncompressed")
            for level, long in [(9, False), (12, True), (15, True), (19, True)]:
                blob, cpu = stream_zstd(F, recv, level, long)
                print(f"  protobuf zstd-{level:<2d}{' long' if long else '     '}  {len(blob)/2**20:6.2f} MB "
                      f"({len(blob)/cur:5.1%})  cpu {cpu:6.1f}s/h")


if __name__ == "__main__":
    main()
