"""Validate start-stamped UTC minute bars; keep labels separate from information."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ("BTC", "ETH", "SOL")
SCHEMA = {"symbol", "dt", "date", "hour", "minute", "open", "high", "low", "close", "volume"}
PRICES = ["open", "high", "low", "close"]


@dataclass
class MarketData:
    bars: dict[str, pd.DataFrame]
    windows: pd.DataFrame
    audit: dict


def read_bars(path: str | Path, asset: str, invalid_rows: str = "raise") -> pd.DataFrame:
    raw = pd.read_csv(path)
    if not SCHEMA.issubset(raw.columns):
        raise ValueError(f"{path}: missing columns {SCHEMA - set(raw.columns)}")
    names = raw.symbol.astype(str).str.upper().str.replace(r"[-_/]", "", regex=True)
    if (~names.isin([asset, asset + "USD", asset + "USDT"]) & raw.symbol.notna()).any():
        raise ValueError(f"{path}: symbols do not match {asset}")
    dt = pd.to_datetime(raw.dt, utc=True, errors="raise")
    if dt.isna().any() or dt.duplicated().any() or not dt.eq(dt.dt.floor("min")).all():
        raise ValueError(f"{path}: null, duplicate, or off-grid timestamps")
    if not pd.to_numeric(raw.minute, errors="raise").eq(dt.dt.minute).all():
        raise ValueError(f"{path}: minute disagrees with UTC dt")
    # The supplied 'hour' can be an hour timestamp, not an integer. dt is authoritative.
    numeric = raw[PRICES + ["volume"]].apply(pd.to_numeric, errors="raise")
    bad = (
        raw.symbol.isna()
        | ~np.isfinite(numeric.to_numpy()).all(axis=1)
        | numeric[PRICES].le(0).any(axis=1)
        | numeric.volume.lt(0)
        | numeric.high.lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric.low.gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    if invalid_rows not in ("raise", "quarantine"):
        raise ValueError("invalid_rows must be raise or quarantine")
    if bad.any() and invalid_rows == "raise":
        raise ValueError(f"{path}: {bad.sum()} invalid OHLCV rows")
    # Explicit quarantine: discard the entire bar, including any non-null fragments.
    numeric.loc[bad, :] = np.nan
    numeric.index = pd.DatetimeIndex(dt, name="dt")
    numeric = numeric.sort_index()
    numeric.attrs["quarantined_rows"] = int(bad.sum())
    numeric.attrs["quarantined_timestamps"] = dt[bad].astype(str).tolist()
    return numeric


def load_market(paths: dict[str, str | Path], invalid_rows: str = "raise") -> MarketData:
    """Use the common date span, preserving missing minutes as NaN, never forward filling.

    elapsed is the start-stamped offset 0..14. A row is available one minute later,
    so completed-minute k = elapsed + 1. Incomplete targets and exact dojis are
    retained in the audit and excluded from the binary evaluation universe.
    """
    if set(paths) != set(ASSETS):
        raise ValueError(f"Supply exactly {ASSETS}")
    frames = {s: read_bars(paths[s], s, invalid_rows) for s in ASSETS}
    start = max(f.index.min() for f in frames.values())
    stop = min(f.index.max() for f in frames.values())
    if start > stop:
        raise ValueError("Assets have no overlapping date span")
    grid = pd.date_range(start, stop, freq="min", name="dt")
    windows, audit = [], {"start": str(start), "stop": str(stop), "assets": {}}
    for s, f in frames.items():
        f = f.reindex(grid)
        w = grid.floor("15min")
        f["window_id"] = w
        f["elapsed"] = (grid.minute % 15).astype("int8")
        f["k"] = f.elapsed + 1
        f["available_at"] = grid + pd.Timedelta(minutes=1)
        f["anchor_open"] = f.open.where(f.elapsed.eq(0)).groupby(w).transform("first")
        # Labels are built here, but are NEVER passed into feature construction.
        counts = f.open.groupby(w).count()
        anchors = f.open.where(f.elapsed.eq(0)).groupby(w).first()
        closes = f.close.where(f.elapsed.eq(14)).groupby(w).first()
        labels = pd.DataFrame({"anchor_open": anchors, "final_close": closes})
        labels.index.name = "window_id"
        labels["complete"] = counts.eq(15) & anchors.notna() & closes.notna()
        labels["doji"] = closes.eq(anchors) & labels.complete
        labels["eligible"] = labels.complete & ~labels.doji
        labels["y"] = closes.gt(anchors).astype("int8")
        labels["symbol"] = s
        labels["label_end"] = labels.index + pd.Timedelta(minutes=15)
        audit["assets"][s] = {
            "quarantined_rows": int(f.attrs["quarantined_rows"]),
            "quarantined_timestamps": f.attrs["quarantined_timestamps"],
            "missing_minutes": int(f.open.isna().sum()),
            "windows": len(labels),
            "incomplete": int((~labels.complete).sum()),
            "doji": int(labels.doji.sum()),
            "binary_windows": int(labels.eligible.sum()),
        }
        frames[s] = f
        windows.append(labels.reset_index())
    return MarketData(frames, pd.concat(windows, ignore_index=True), audit)
