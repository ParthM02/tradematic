"""Fetch official results for every recorded contract after it closes.

Uses Kalshi's public events endpoint (one call returns every market in an event, with
`result` and `expiration_value`). Polls from 2 minutes after close until each recorded
ticker is determined, giving up after 6 hours (recorded as `settlement_timeout`); the margin covers
contracts that close during Kalshi's Thursday maintenance and settle late.
"""
import time

from .discovery import kalshi_get, parse_ts

STREAM = "settlements"
FIRST_CHECK_S, RETRY_S, GIVE_UP_S = 120, 120, 6 * 3600


class SettlementTracker:
    def __init__(self, writer):
        self.writer = writer
        self.pending = {}  # event_ticker -> {"close": ts, "tickers": set, "next": ts}

    def track(self, market):
        ev = self.pending.setdefault(market["event_ticker"], {
            "close": parse_ts(market["close_time"]), "tickers": set(), "next": 0})
        ev["tickers"].add(market["ticker"])
        ev["next"] = max(ev["next"], ev["close"] + FIRST_CHECK_S)

    def poll(self, now=None):
        """Check due events. Blocking HTTP; call from a worker thread."""
        now = time.time() if now is None else now
        for event, ev in list(self.pending.items()):
            if now < ev["next"]:
                continue
            if now > ev["close"] + GIVE_UP_S:
                self.writer.write(STREAM, "kalshi", "settlement_timeout",
                                  {"event_ticker": event, "tickers": sorted(ev["tickers"])})
                del self.pending[event]
                continue
            try:
                payload = kalshi_get(f"/events/{event}")
            except Exception as e:  # noqa: BLE001
                self.writer.write(STREAM, "kalshi", "settlement_error", {"event_ticker": event, "error": repr(e)[:500]})
                ev["next"] = now + RETRY_S
                continue
            for m in payload.get("markets", []):
                if m.get("ticker") in ev["tickers"] and m.get("result") in ("yes", "no"):
                    self.writer.write(STREAM, "kalshi", "settlement", {
                        k: m.get(k) for k in ("ticker", "event_ticker", "result", "expiration_value",
                                              "floor_strike", "cap_strike", "strike_type",
                                              "open_time", "close_time", "settlement_ts", "status")})
                    ev["tickers"].discard(m["ticker"])
            if ev["tickers"]:
                ev["next"] = now + RETRY_S
            else:
                del self.pending[event]
