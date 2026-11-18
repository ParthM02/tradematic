"""Thread-safe append-only writer: one zstd JSONL file per stream per UTC hour.

Every record is {"recv_ns": local receive time, "src": source, "type": message type,
"data": payload}. Producers (MQTT thread, asyncio loop) only enqueue; a single writer
thread owns all file handles.

Compression is streaming zstd (default level 9). About once a second the writer ends
a zstd block and hands it to the OS, so a crash loses at most roughly the last second;
`records.iter_records` reads everything up to that point. Reopening an hour's file after
a restart appends a new zstd frame, which the reader handles. Nothing is fsync'd: the
OS writes to the SD card on its own schedule, which keeps card wear low; a sudden power
cut can lose the last few seconds the OS had not written yet.
"""
import json
import queue
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import zstandard

_STOP = object()
SUFFIX = ".jsonl.zst"


def hour_path(root, stream, recv_ns):
    t = datetime.fromtimestamp(recv_ns / 1e9, UTC)
    return Path(root) / stream / t.strftime("%Y%m%d") / f"{stream}_{t.strftime('%Y%m%dT%H')}{SUFFIX}"


def _default(o):
    # Decimal and other numeric-like values from SDK objects are stored as strings.
    return str(o)


class _ZstdFile:
    """Append-mode zstd frame writer with block flushes."""

    def __init__(self, path, level):
        self.path = path
        self._fh = open(path, "ab")  # noqa: SIM115 - long-lived, closed in close()
        self._c = zstandard.ZstdCompressor(level=level, write_checksum=True).compressobj()

    def write(self, data: bytes):
        out = self._c.compress(data)
        if out:
            self._fh.write(out)

    def flush(self):
        self._fh.write(self._c.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
        self._fh.flush()

    def close(self):
        self._fh.write(self._c.flush())  # ends the frame (with checksum)
        self._fh.close()


class RecordWriter:
    def __init__(self, root, level=9, flush_seconds=1.0, max_queue=1_000_000):
        self.root = Path(root)
        self.level = level
        self.flush_seconds = flush_seconds
        self.q = queue.Queue(maxsize=max_queue)
        self.counts = Counter()
        self.dropped = 0
        self._files = {}
        self._thread = threading.Thread(target=self._run, name="record-writer", daemon=True)

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def write(self, stream, src, kind, data, recv_ns=None):
        rec = {"recv_ns": recv_ns or time.time_ns(), "src": src, "type": kind, "data": data}
        try:
            self.q.put_nowait((stream, rec))
        except queue.Full:
            self.dropped += 1  # reported in health; never block the network threads

    def close(self, timeout=10):
        self.q.put(_STOP)
        self._thread.join(timeout)

    def _handle(self, stream, recv_ns):
        path = hour_path(self.root, stream, recv_ns)
        f = self._files.get(stream)
        if f is not None and f.path == path:
            return f
        if f is not None:
            f.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        f = self._files[stream] = _ZstdFile(path, self.level)
        return f

    def _flush(self):
        """Flush every open file; close files for hours that have ended."""
        now_ns = time.time_ns()
        for stream, f in list(self._files.items()):
            if f.path != hour_path(self.root, stream, now_ns):
                f.close()
                del self._files[stream]
            else:
                f.flush()

    def _run(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self.q.get(timeout=self.flush_seconds)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if item is not None:
                stream, rec = item
                line = json.dumps(rec, separators=(",", ":"), default=_default) + "\n"
                self._handle(stream, rec["recv_ns"]).write(line.encode("utf-8"))
                self.counts[(stream, rec["src"], rec["type"])] += 1
            if time.monotonic() - last_flush >= self.flush_seconds:
                self._flush()
                last_flush = time.monotonic()
        for f in self._files.values():
            f.close()
        self._files.clear()
