"""Coinbase Exchange public WebSocket feed (no authentication) for underlying spot ticks.

Channels: `ticker` (best bid/ask + last trade), `matches` (every trade) and `heartbeat`.
Per-product trade_id continuity on `matches` detects dropped trades; a gap is written
as a `gap` record. Reconnects use capped exponential backoff, and each connect and
disconnect is recorded so downstream code can mask outage windows.
Docs: https://docs.cdp.coinbase.com/exchange/websocket-feed/overview
"""
import asyncio
import json
import time

import websockets

URL = "wss://ws-feed.exchange.coinbase.com"
STREAM = "coinbase"


class TradeGapDetector:
    def __init__(self):
        self.last = {}

    def check(self, product, trade_id):
        """Return number of missing trades before this one (0 if contiguous or first)."""
        prev = self.last.get(product)
        if prev is not None and trade_id <= prev:
            return 0  # duplicate or out of order; keep the high-water mark
        self.last[product] = trade_id
        return 0 if prev is None else trade_id - prev - 1


async def run(products, writer, stop: asyncio.Event):
    sub = {"type": "subscribe", "product_ids": list(products),
           "channels": ["ticker", "matches", "heartbeat"]}
    gaps = TradeGapDetector()
    backoff = 1
    while not stop.is_set():
        try:
            async with websockets.connect(URL, open_timeout=15, ping_interval=20, max_size=2**22) as ws:
                await ws.send(json.dumps(sub))
                writer.write(STREAM, "coinbase", "connected", {"url": URL, "subscribe": sub})
                backoff = 1
                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except TimeoutError:
                        continue
                    recv_ns = time.time_ns()
                    msg = json.loads(raw)
                    kind = msg.get("type", "unknown")
                    writer.write(STREAM, "coinbase", kind, msg, recv_ns)
                    if kind in ("match", "last_match") and "trade_id" in msg:
                        missing = gaps.check(msg["product_id"], int(msg["trade_id"]))
                        if missing and kind == "match":
                            writer.write(STREAM, "coinbase", "gap",
                                         {"product_id": msg["product_id"], "missing_trades": missing,
                                          "trade_id": msg["trade_id"]}, recv_ns)
                    if kind == "error":
                        raise RuntimeError(f"coinbase error: {msg}")
        except Exception as e:  # noqa: BLE001 - any failure means reconnect
            if stop.is_set():
                break
            writer.write(STREAM, "coinbase", "disconnected", {"error": repr(e)[:500], "retry_in": backoff})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
