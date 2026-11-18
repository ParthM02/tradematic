"""Read-only public backfill and bounded live book capture; no API credentials."""

import argparse
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"
START = pd.Timestamp("2026-08-15", tz="UTC")
END = pd.Timestamp("2026-09-12", tz="UTC")


def utc_now():
    return datetime.now(UTC).isoformat()


def get_json(base, path, parameters=None):
    """Public GET only. Bounded retry, including rate limits; no auth material."""
    if base not in (KALSHI, COINBASE) or not path.startswith("/"):
        raise ValueError("Unexpected public API destination")
    url = base + path + ("?" + urlencode(parameters) if parameters else "")
    for attempt in range(4):
        sent = utc_now()
        before = time.monotonic()
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "tradematic-public-research/0.1",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            return payload, {
                "url": url,
                "sent_at": sent,
                "received_at": utc_now(),
                "roundtrip_seconds": time.monotonic() - before,
            }
        except HTTPError as error:
            if error.code == 400:
                detail = error.read().decode("utf-8", errors="replace")[:800]
                raise ValueError(f"Public API rejected {path}: {detail}") from error
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After", "0")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0
            time.sleep(min(20, max(delay, 2**attempt)))
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Public request failed")


def cached_json(root, filename, base, path, parameters):
    dest = root / filename
    if dest.exists():
        with gzip.open(dest, "rt") as f:
            cached = json.load(f)
        expected = base + path + ("?" + urlencode(parameters) if parameters else "")
        if cached["request"]["url"] != expected:
            raise ValueError("Cached public request differs; use a new output directory")
        return cached["payload"]
    payload, timing = get_json(base, path, parameters)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".partial")
    with gzip.open(temp, "wt") as f:
        json.dump({"request": timing, "payload": payload}, f)
    temp.replace(dest)
    time.sleep(0.15)
    return payload


def price_field(block, key="close"):
    if not isinstance(block, dict):
        return np.nan
    value = block.get(key + "_dollars")
    if value is not None:
        return float(value)
    value = block.get(key)
    if value is None:
        return np.nan
    # Archived API fixed-point strings are dollars; legacy integers are cents.
    return float(value) if isinstance(value, str) else float(value) / 100


def flatten_quotes(ticker, candles, opened):
    rows, rejected = [], 0
    for candle in candles:
        bid, ask = price_field(candle.get("yes_bid")), price_field(candle.get("yes_ask"))
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask <= 1):
            rejected += 1
            continue
        ts = int(candle["end_period_ts"])
        elapsed = (ts - int(opened.timestamp())) / 60
        if elapsed != int(elapsed) or not 1 <= elapsed <= 16:
            continue
        rows.append(
            {
                "ticker": ticker,
                "end_ts": ts,
                "elapsed_minute": int(elapsed),
                "yes_bid": bid,
                "yes_ask": ask,
                "no_ask": 1 - bid,
                "last_price": price_field(candle.get("price")),
                "volume": float(candle.get("volume_fp", candle.get("volume", 0))),
            }
        )
    return rows, rejected


def market_pages(output):
    result = {}
    params = {
        "series_ticker": "KXSOL15M",
        "limit": 1000,
        "min_close_ts": int(START.timestamp()),
        "max_close_ts": int(END.timestamp()),
    }
    cursor, seen = "", set()
    for page in range(100):
        p = params | ({"cursor": cursor} if cursor else {})
        payload = cached_json(output / "raw", f"markets_{page:03d}.json.gz", KALSHI, "/markets", p)
        for item in payload.get("markets", []):
            opened, closed = pd.Timestamp(item["open_time"]), pd.Timestamp(item["close_time"])
            if START <= opened and closed <= END:
                result[item["ticker"]] = item
        cursor = payload.get("cursor", "")
        print(f"Public markets page {page + 1}: {len(result)} in fresh interval", flush=True)
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Repeated market pagination cursor")
        seen.add(cursor)
    else:
        raise ValueError("Market pagination limit exceeded")
    for item in result.values():
        item["archived"] = False
    cursor, seen = "", set()
    for page in range(100):
        params = {"series_ticker": "KXSOL15M", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = cached_json(
            output / "raw",
            f"archive_markets_{page:03d}.json.gz",
            KALSHI,
            "/historical/markets",
            params,
        )
        for item in payload.get("markets", []):
            opened, closed = pd.Timestamp(item["open_time"]), pd.Timestamp(item["close_time"])
            if START <= opened and closed <= END and item["ticker"] not in result:
                result[item["ticker"]] = item | {"archived": True}
        print(f"Archive markets page {page + 1}: {len(result)} total in fresh interval", flush=True)
        cursor = payload.get("cursor", "")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Repeated archived-market cursor")
        seen.add(cursor)
    else:
        raise ValueError("Historical pagination limit exceeded")
    if not result:
        raise ValueError("No fresh listed markets; cannot evaluate the prescribed interval")
    rows = []
    for ticker, m in result.items():
        rows.append(
            {
                "symbol": "SOLUSD",
                "series": "KXSOL15M",
                "ticker": ticker,
                "event_ticker": m.get("event_ticker"),
                "open_time": m["open_time"],
                "close_time": m["close_time"],
                "floor_strike": m.get("floor_strike"),
                "expiration_value": m.get("expiration_value"),
                "result": m.get("result"),
                "y_up": {"yes": 1, "no": 0}.get(m.get("result")),
                "volume": m.get("volume_fp", m.get("volume")),
                "status": m.get("status"),
                "archived": m["archived"],
            }
        )
    return pd.DataFrame(rows).sort_values("open_time")


def quote_batches(markets, budget=9000):
    """Budget rectangular ticker-count x time-span, not actual contract duration."""
    batch = []
    for row in markets.sort_values("open_time").itertuples():
        proposed = batch + [row]
        span = (
            pd.Timestamp(proposed[-1].close_time) - pd.Timestamp(proposed[0].open_time)
        ).total_seconds() / 60 + 2
        if batch and (len(proposed) > 100 or len(proposed) * span > budget):
            yield pd.DataFrame([r._asdict() for r in batch])
            batch = [row]
        else:
            batch = proposed
    if batch:
        yield pd.DataFrame([r._asdict() for r in batch])


def backfill(output, old_data):
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "manifest.json"
    if marker.exists() and json.loads(marker.read_text()).get("status") == "complete":
        print("Fresh public dataset already complete; retaining it unchanged", flush=True)
        return
    config = {
        "start": str(START),
        "end": str(END),
        "status": "downloading",
        "authentication": "none",
    }
    marker.write_text(json.dumps(config, indent=2) + "\n")
    m = market_pages(output)
    (output / "kalshi").mkdir(exist_ok=True)
    m.to_csv(output / "kalshi" / "markets_SOLUSD.csv", index=False)
    quotes, rejected = [], 0
    ordered = m.loc[~m.archived].sort_values("open_time").reset_index(drop=True)
    processed = 0
    for number, batch in enumerate(quote_batches(ordered)):
        opened, closed = (
            pd.to_datetime(batch.open_time, utc=True),
            pd.to_datetime(batch.close_time, utc=True),
        )
        payload = cached_json(
            output / "raw",
            f"quotes_small_{number:03d}.json.gz",
            KALSHI,
            "/markets/candlesticks",
            {
                "market_tickers": ",".join(batch.ticker),
                "start_ts": int(opened.min().timestamp()),
                "end_ts": int(closed.max().timestamp()) + 60,
                "period_interval": 1,
            },
        )
        lookup = dict(zip(batch.ticker, opened, strict=True))
        for block in payload.get("markets", []):
            ticker = block.get("market_ticker", block.get("ticker"))
            if ticker not in lookup:
                raise ValueError("Unexpected market in candlestick response")
            rows, bad = flatten_quotes(ticker, block.get("candlesticks", []), lookup[ticker])
            quotes.extend(rows)
            rejected += bad
        processed += len(batch)
        if number % 10 == 0:
            print(f"Public live-dataset quotes: {processed}/{len(ordered)} contracts", flush=True)
    archived = list(m.loc[m.archived].itertuples())

    def archived_quotes(row):
        opened, closed = pd.Timestamp(row.open_time), pd.Timestamp(row.close_time)
        payload = cached_json(
            output / "raw" / "archive_quotes",
            f"{row.ticker}.json.gz",
            KALSHI,
            f"/historical/markets/{row.ticker}/candlesticks",
            {
                "start_ts": int(opened.timestamp()),
                "end_ts": int(closed.timestamp()) + 60,
                "period_interval": 1,
            },
        )
        return flatten_quotes(row.ticker, payload.get("candlesticks", []), opened)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for number, (rows, bad) in enumerate(pool.map(archived_quotes, archived)):
            quotes.extend(rows)
            rejected += bad
            if number % 50 == 0:
                print(f"Public archived quotes: {number + 1}/{len(archived)} contracts", flush=True)
    q = pd.DataFrame(quotes)
    if q.empty:
        raise ValueError("No usable fresh quotes; refusing to score an empty backtest")
    q = q.drop_duplicates(["ticker", "end_ts"]).sort_values(["end_ts", "ticker"])
    q.to_csv(output / "kalshi" / "candles_SOLUSD.csv", index=False)
    (output / "coinbase").mkdir(exist_ok=True)
    asset_stats = {}
    for s in ("BTC", "ETH", "SOL"):
        old = pd.read_csv(old_data / f"{s.lower()}usd_m1_bars_coinbase.csv")
        old["dt"] = pd.to_datetime(old.dt, utc=True)
        current = old.dt.max() + pd.Timedelta(minutes=1)
        rows, page = [], 0
        while current < END:
            until = min(current + pd.Timedelta(minutes=300), END)
            payload = cached_json(
                output / "raw" / "coinbase",
                f"{s}_{page:04d}.json.gz",
                COINBASE,
                f"/products/{s}-USD/candles",
                {"start": current.isoformat(), "end": until.isoformat(), "granularity": 60},
            )
            if not isinstance(payload, list):
                raise TypeError("Coinbase candle response is not a list")
            for item in payload:
                if len(item) != 6:
                    raise ValueError("Unexpected Coinbase candle schema")
                ts, low, high, opened, close, volume = item
                dt = pd.Timestamp(ts, unit="s", tz="UTC")
                if current <= dt < until:
                    rows.append(
                        {
                            "symbol": s + "USD",
                            "dt": dt,
                            "date": dt.date(),
                            "hour": dt.floor("h"),
                            "minute": dt.minute,
                            "open": opened,
                            "high": high,
                            "low": low,
                            "close": close,
                            "volume": volume,
                        }
                    )
            current, page = until, page + 1
            if page % 25 == 0:
                print(f"Public {s} candles: {page} requests, through {current}", flush=True)
        combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        combined = combined[combined.dt.lt(END)].sort_values("dt")
        if combined.dt.duplicated().any():
            raise ValueError("Duplicate original/new Coinbase timestamp")
        combined.to_csv(output / "coinbase" / f"{s.lower()}usd_m1_bars_coinbase.csv", index=False)
        asset_stats[s] = {
            "new_minutes": len(rows),
            "total_minutes": len(combined),
            "last": str(combined.dt.max()),
        }
    config.update(
        status="complete",
        markets=len(m),
        usable_quote_rows=len(q),
        quoted_markets=q.ticker.nunique(),
        rejected_quote_rows=rejected,
        assets=asset_stats,
        csv_sha256={
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*.csv")
        },
    )
    marker.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Fresh public data ready: {output}", flush=True)


def record(output, seconds=60, interval=1.0):
    """Bounded REST snapshots, not a complete event stream or execution record."""
    output.mkdir(parents=True, exist_ok=True)
    dest = output / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".jsonl.gz")
    markets, _ = get_json(
        KALSHI, "/markets", {"series_ticker": "KXSOL15M", "status": "open", "limit": 100}
    )
    now = pd.Timestamp.now(tz="UTC")
    active = [
        m
        for m in markets.get("markets", [])
        if pd.Timestamp(m["open_time"]) <= now < pd.Timestamp(m["close_time"])
    ]
    if len(active) != 1:
        raise ValueError("Cannot identify exactly one active SOL contract")
    ticker, closed = active[0]["ticker"], pd.Timestamp(active[0]["close_time"])
    targets = [
        (s, COINBASE, f"/products/{s}-USD/book", {"level": 1}) for s in ("BTC", "ETH", "SOL")
    ]
    targets.append((ticker, KALSHI, f"/markets/{ticker}/orderbook", {"depth": 10}))
    began, rounds = time.monotonic(), 0

    def capture(target):
        symbol, base, path, parameters = target
        try:
            payload, timestamps = get_json(base, path, parameters)
            return {"symbol": symbol, **timestamps, "payload": payload}
        except (HTTPError, URLError, TimeoutError) as error:
            return {"symbol": symbol, "received_at": utc_now(), "error_type": type(error).__name__}

    with gzip.open(dest, "wt") as out, ThreadPoolExecutor(max_workers=4) as pool:
        while time.monotonic() - began < seconds and pd.Timestamp.now(tz="UTC") < closed:
            for row in pool.map(capture, targets):
                out.write(json.dumps(row) + "\n")
            out.flush()
            rounds += 1
            time.sleep(max(0, began + rounds * interval - time.monotonic()))
    print(
        json.dumps(
            {
                "file": str(dest),
                "rounds": rounds,
                "elapsed": time.monotonic() - began,
                "contract": ticker,
                "note": "REST snapshot smoke capture; not a full order-event stream",
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    back = sub.add_parser("backfill")
    back.add_argument("--output", type=Path, default=Path("backend/data/fresh_20260815_20260912"))
    back.add_argument("--old-data", type=Path, default=Path("backend/data/coinbase"))
    rec = sub.add_parser("record")
    rec.add_argument("--output", type=Path, default=Path("backend/data/live_public"))
    rec.add_argument("--seconds", type=int, default=60)
    args = p.parse_args()
    if args.command == "backfill":
        backfill(args.output, args.old_data)
    else:
        if not 1 <= args.seconds <= 3600:
            p.error("Capture duration must be 1..3600 seconds")
        record(args.output, args.seconds)


if __name__ == "__main__":
    main()
