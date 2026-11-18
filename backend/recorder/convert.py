"""Convert recorded JSONL.gz streams into Parquet tables for analysis.

    uv run --group recorder python -m backend.recorder.convert [--data DIR] [--out DIR]

Tables (times are integer milliseconds since epoch, UTC; prices in dollars):
  wb_quote     symbol, ts (Webull), recv, yes_bid, yes_bid_sz, no_bid, no_bid_sz,
               yes_ask (=1-no_bid), no_ask (=1-yes_bid), yes_depth5, no_depth5,
               yes_px/yes_sz/no_px/no_sz (full ladders as lists)
  wb_snapshot  every snapshot field, numeric
  wb_tick      symbol, ts, recv, time, yes_price, no_price, volume, side, trade_id
  cb_ticker    product, ts (Coinbase), recv, price, bid, ask, bid_sz, ask_sz, last_size, side, trade_id
  cb_match     product, ts, recv, trade_id, price, size, side (maker side, Coinbase convention)
  markets      Kalshi metadata for each recorded contract
  settlements  official result / expiration_value
  events       recorder events (connects, subscribes, gaps, notices, health)
Reads .jsonl.zst, .compact.jsonl.zst and .jsonl.gz hour files (one per hour, see records.hour_files).
Hour files are converted in parallel; a damaged tail after an unclean shutdown is skipped.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT
from .records import hour_files, iter_records


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _iso_ms(s):
    return int(datetime.fromisoformat(s).timestamp() * 1000) if s else None


def convert_webull(path):
    quote, snap, tick, events = [], [], [], []
    for r in iter_records(path):
        recv, kind, d = r["recv_ns"] // 1_000_000, r["type"], r["data"]
        if r["src"] != "webull" or kind == "notice":
            events.append({"recv": recv, "src": r["src"], "type": kind, "data": json.dumps(d)})
            continue
        b = d.get("basic", {})
        sym, ts = b.get("symbol"), int(b["timestamp"]) if b.get("timestamp") else None
        if kind == "event-quote":
            yp = [_f(x["price"]) for x in d.get("yes_bids", [])]
            ys = [_f(x["size"]) for x in d.get("yes_bids", [])]
            npx = [_f(x["price"]) for x in d.get("no_bids", [])]
            ns = [_f(x["size"]) for x in d.get("no_bids", [])]
            quote.append({"symbol": sym, "ts": ts, "recv": recv,
                          "yes_bid": yp[0] if yp else None, "yes_bid_sz": ys[0] if ys else None,
                          "no_bid": npx[0] if npx else None, "no_bid_sz": ns[0] if ns else None,
                          "yes_depth5": sum(ys[:5]), "no_depth5": sum(ns[:5]),
                          "yes_px": yp, "yes_sz": ys, "no_px": npx, "no_sz": ns})
        elif kind == "event-snapshot":
            row = {"symbol": sym, "ts": ts, "recv": recv}
            row.update({k: _f(v) for k, v in d.items() if k != "basic"})
            snap.append(row)
        elif kind == "event-tick":
            tick.append({"symbol": sym, "ts": ts, "recv": recv, "time": int(d["time"]) if d.get("time") else None,
                         "yes_price": _f(d.get("yes_price")), "no_price": _f(d.get("no_price")),
                         "volume": _f(d.get("volume")), "side": d.get("side"), "trade_id": d.get("trade_id")})
    q = pd.DataFrame(quote)
    if len(q):
        q["yes_ask"] = 1 - q.no_bid
        q["no_ask"] = 1 - q.yes_bid
    return {"wb_quote": q, "wb_snapshot": pd.DataFrame(snap), "wb_tick": pd.DataFrame(tick),
            "events": pd.DataFrame(events)}


def convert_coinbase(path):
    tk, mt, events = [], [], []
    for r in iter_records(path):
        recv, kind, d = r["recv_ns"] // 1_000_000, r["type"], r["data"]
        if kind == "ticker":
            tk.append({"product": d["product_id"], "ts": _iso_ms(d.get("time")), "recv": recv,
                       "price": _f(d.get("price")), "bid": _f(d.get("best_bid")), "ask": _f(d.get("best_ask")),
                       "bid_sz": _f(d.get("best_bid_size")), "ask_sz": _f(d.get("best_ask_size")),
                       "last_size": _f(d.get("last_size")), "side": d.get("side"), "trade_id": d.get("trade_id")})
        elif kind == "match":
            mt.append({"product": d["product_id"], "ts": _iso_ms(d.get("time")), "recv": recv,
                       "trade_id": d.get("trade_id"), "price": _f(d.get("price")), "size": _f(d.get("size")),
                       "side": d.get("side")})
        elif kind not in ("heartbeat", "last_match", "subscriptions"):
            events.append({"recv": recv, "src": "coinbase", "type": kind, "data": json.dumps(d)})
    return {"cb_ticker": pd.DataFrame(tk), "cb_match": pd.DataFrame(mt), "events": pd.DataFrame(events)}


def convert_simple(path):
    rows = list(iter_records(path))
    stream = Path(path).parent.parent.name
    if stream in ("markets", "settlements"):
        data = [dict(r["data"], recv=r["recv_ns"] // 1_000_000, type=r["type"]) for r in rows]
        return {stream: pd.DataFrame(data)}
    return {"events": pd.DataFrame([{"recv": r["recv_ns"] // 1_000_000, "src": r["src"], "type": r["type"],
                                     "data": json.dumps(r["data"])} for r in rows])}


def _job(path):
    stream = Path(path).parent.parent.name
    fn = {"webull": convert_webull, "coinbase": convert_coinbase}.get(stream, convert_simple)
    return fn(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO_ROOT / "backend/data/recorder"))
    p.add_argument("--out", default=None, help="default: <data>/parquet")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    data = Path(a.data)
    out = Path(a.out) if a.out else data / "parquet"
    out.mkdir(parents=True, exist_ok=True)
    files = [str(f) for s in ("webull", "coinbase", "markets", "settlements", "health")
             for f in hour_files(data, s)]
    parts = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(_job, files):
            for k, df in res.items():
                if len(df):
                    parts.setdefault(k, []).append(df)
    for k, dfs in parts.items():
        df = pd.concat(dfs, ignore_index=True)
        sort = [c for c in ("symbol", "product", "ts", "recv") if c in df.columns]
        df = df.sort_values(sort or ["recv"]).reset_index(drop=True)
        df.to_parquet(out / f"{k}.parquet", index=False)
        print(f"{k:12s} {len(df):>10,d} rows")


if __name__ == "__main__":
    main()
