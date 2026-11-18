"""Find the contracts to record, using Kalshi's public (unauthenticated) REST API.

Webull lists the same Kalshi market tickers (for example KXRATECUTCOUNT-26DEC31-T3 in
Webull's order docs), so these tickers are used directly as Webull subscription symbols.
Selection per refresh:
  * 15-minute series (`...15M`): every active market (normally the current window).
  * Other series (hourly ladders): for the earliest-closing active event, the
    `hourly_strikes` markets whose quote midpoint is nearest 50 cents.
Symbols stay subscribed until their market closes, so a strike that drifts away from
the money is not dropped mid-window. The total is capped at `max_symbols`.
"""
import json
import time
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KALSHI = "https://external-api.kalshi.com/trade-api/v2"


def kalshi_get(path, params=None, timeout=20):
    """Public GET against Kalshi only, with bounded retries on rate limits and 5xx."""
    if not path.startswith("/"):
        raise ValueError("path must start with /")
    url = KALSHI + path + ("?" + urlencode(params) if params else "")
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "tradematic-recorder/0.1", "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()


def active_markets(series):
    """All open markets in a series (follows the pagination cursor)."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = kalshi_get("/markets", params)
        out += page.get("markets", [])
        cursor = page.get("cursor")
        if not cursor:
            return out


def _mid(m):
    """YES midpoint, or None without a real two-sided quote (empty books read 0/1)."""
    try:
        bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if bid <= 0 or ask >= 1 or ask <= bid:
        return None
    return (bid + ask) / 2


def select(markets_by_series, hourly_strikes, now=None):
    """Return {ticker: market} to record now. Pure function, for testing."""
    now = time.time() if now is None else now
    chosen = {}
    for series, markets in markets_by_series.items():
        live = [m for m in markets if parse_ts(m["close_time"]) > now and parse_ts(m["open_time"]) <= now]
        if not live:
            continue
        if series.endswith("15M"):
            chosen.update({m["ticker"]: m for m in live})
            continue
        first_close = min(parse_ts(m["close_time"]) for m in live)
        event = [m for m in live if parse_ts(m["close_time"]) == first_close]
        priced = [(abs(mid - 0.5), m["ticker"], m) for m in event if (mid := _mid(m)) is not None]
        priced.sort(key=lambda x: (x[0], x[1]))
        chosen.update({t: m for _, t, m in priced[:hourly_strikes]})
    return chosen


class Universe:
    """Desired subscription set, with sticky membership until each market closes."""

    def __init__(self, series, hourly_strikes, max_symbols):
        self.series, self.hourly_strikes, self.max_symbols = series, hourly_strikes, max_symbols
        self.markets = {}  # ticker -> market metadata, for everything currently desired
        self.seen = set()  # every ticker ever selected, for settlement tracking

    def refresh(self, markets_by_series, now=None):
        """Update the desired set. Returns (added, removed, new_market_records)."""
        now = time.time() if now is None else now
        keep = {t: m for t, m in self.markets.items() if parse_ts(m["close_time"]) > now}
        fresh = select(markets_by_series, self.hourly_strikes, now)
        for t in sorted(fresh, key=lambda t: parse_ts(fresh[t]["close_time"])):
            if t not in keep and len(keep) < self.max_symbols:
                keep[t] = fresh[t]
        added = sorted(set(keep) - set(self.markets))
        removed = sorted(set(self.markets) - set(keep))
        new_records = [keep[t] for t in added if t not in self.seen]
        self.seen |= set(added)
        self.markets = keep
        return added, removed, new_records

    def fetch(self):
        return {s: active_markets(s) for s in self.series}


def utc_iso(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, UTC).isoformat()
