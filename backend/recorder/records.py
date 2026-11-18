"""Read recorded hour files, whatever their format.

Formats, detected from the file's first bytes (never its name):
  *.jsonl.zst          live recorder output: zstd frames, one new frame per writer session
  *.compact.jsonl.zst  output of `compact` (high-level zstd, Coinbase matches folded)
  *.jsonl.gz           older recordings (gzip members)

Reading streams in 1 MB chunks, so memory stays small regardless of file size. A file
cut off by a crash or power loss yields every record up to the last complete
flush; the damaged tail is skipped.
"""
import heapq
import json
import zlib
from pathlib import Path

import zstandard

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
GZIP_MAGIC = b"\x1f\x8b"
SUFFIXES = (".compact.jsonl.zst", ".jsonl.zst", ".jsonl.gz")  # preference order per hour
FLIP = {"buy": "sell", "sell": "buy"}


def file_format(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head == ZSTD_MAGIC:
        return "zstd"
    if head[:2] == GZIP_MAGIC:
        return "gzip"
    return "empty" if not head else "unknown"


def _decoders(fmt):
    if fmt == "zstd":
        return lambda: zstandard.ZstdDecompressor().decompressobj(), zstandard.ZstdError
    return lambda: zlib.decompressobj(16 + zlib.MAX_WBITS), zlib.error


def iter_chunks(path, size=1 << 20):
    """Decoded bytes across all frames/members; stops quietly at a damaged tail."""
    fmt = file_format(path)
    if fmt == "empty":
        return
    if fmt == "unknown":
        raise ValueError(f"{path}: not a gzip or zstd file")
    new, error = _decoders(fmt)
    d = new()
    with open(path, "rb") as f:
        buf = f.read(size)
        while buf:
            try:
                out = d.decompress(buf)
            except error:
                return
            if out:
                yield out
            if d.eof:  # frame/member finished; the rest belongs to the next one
                d, rest = new(), d.unused_data
                buf = rest or f.read(size)
            else:
                buf = f.read(size)


def iter_lines(path):
    tail = b""
    for chunk in iter_chunks(path):
        tail += chunk
        *lines, tail = tail.split(b"\n")
        yield from lines
    if tail:
        yield tail


def iter_raw(path):
    """Stored records as written (Coinbase matches may be folded into tickers)."""
    for line in iter_lines(path):
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue  # a partial last line


def match_from_ticker(ticker, m):
    """Rebuild a folded Coinbase `match` record from its ticker plus the stored extras."""
    t = ticker["data"]
    return {"recv_ns": m["recv_ns"], "src": ticker["src"], "type": "match", "data": {
        "type": "match", "trade_id": t["trade_id"], "maker_order_id": m["maker_order_id"],
        "taker_order_id": m["taker_order_id"], "side": FLIP[t["side"]], "size": t["last_size"],
        "price": t["price"], "product_id": t["product_id"], "sequence": t["sequence"], "time": t["time"]}}


def iter_records(path):
    """The original records, in their original order, from any format."""
    pending = []  # (original position, tiebreak, record) for folded matches
    pos = 0
    for n, r in enumerate(iter_raw(path)):
        while pending and pending[0][0] == pos:
            yield heapq.heappop(pending)[2]
            pos += 1
        m = r.pop("_m", None)
        if m is not None:
            heapq.heappush(pending, (pos + m["d"], n, match_from_ticker(r, m)))
        yield r
        pos += 1
    while pending:
        yield heapq.heappop(pending)[2]


def hour_key(path):
    """'webull_20261117T15' for any of the supported suffixes."""
    name = Path(path).name
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return None


def hour_files(root, stream):
    """One file per recorded hour, preferring compact > live zstd > gzip. Sorted by hour."""
    best = {}
    for p in Path(root, stream).glob("*/*"):
        key = hour_key(p)
        if key is None:
            continue
        rank = next(i for i, s in enumerate(SUFFIXES) if p.name.endswith(s))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, p)
    return [best[k][1] for k in sorted(best)]
