"""Lead-lag between Coinbase spot and Kalshi 15m contract prices (Webull feed).

Bins of 250 ms on exchange timestamps. For each 15m window with mid in (0.15, 0.85):
  dS = Coinbase log-mid change, dP = Kalshi YES mid change.
Cross-correlation corr(dP_t, dS_{t-lag}) for lag in -4..+20 bins; positive lag = spot leads.
Also the cumulative response: regress dP_t on dS_{t-k} for k=0..20 jointly (impulse response).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = str(ROOT / "data/recorder/parquet") + "/"
OUT = ROOT / "research/artifacts/live"
BIN = 250
q = pd.read_parquet(D + "wb_quote.parquet", columns=["symbol", "ts", "yes_bid", "yes_ask"])
cb = pd.read_parquet(D + "cb_ticker.parquet", columns=["product", "ts", "bid", "ask"])
mk = pd.read_parquet(D + "markets.parquet")
mk["close_ms"] = pd.to_datetime(mk.close_time).astype("int64") // 10**6
close = mk.set_index("ticker").close_ms

rows = []
for asset in ["BTC", "ETH", "SOL"]:
    c = cb[cb["product"] == f"{asset}-USD"].copy()
    c["b"] = c.ts // BIN
    s = np.log((c.bid + c.ask) / 2).groupby(c.b).last()
    qq = q[q.symbol.str.startswith(f"KX{asset}15M")].copy()
    qq["mid"] = (qq.yes_bid + qq.yes_ask) / 2
    qq["b"] = qq.ts // BIN
    X, Y = [], []
    for sym, g in qq.groupby("symbol"):
        p = g.groupby("b").mid.last()
        idx = np.arange(p.index.min(), min(p.index.max(), close[sym] // BIN - 4) + 1)
        p = p.reindex(idx).ffill()
        sp = s.reindex(idx).ffill()
        dP, dS = p.diff(), sp.diff()
        ok = (p.shift(1) > 0.15) & (p.shift(1) < 0.85)
        lagged = pd.concat({k: dS.shift(k) for k in range(-4, 21)}, axis=1)
        m = ok & lagged.notna().all(1) & dP.notna()
        X.append(lagged[m].values)
        Y.append(dP[m].values)
    X, Y = np.vstack(X), np.concatenate(Y)
    lags = np.arange(-4, 21)
    xc = [np.corrcoef(X[:, i], Y)[0, 1] for i in range(len(lags))]
    beta = np.linalg.lstsq(np.c_[np.ones(len(Y)), X], Y, rcond=None)[0][1:]
    pos = beta[lags >= 0]
    cum = np.cumsum(pos) / pos.sum()
    half = lags[lags >= 0][np.argmax(cum >= 0.5)] * BIN
    p90 = lags[lags >= 0][np.argmax(cum >= 0.9)] * BIN
    print(f"{asset}: n={len(Y):,} bins; corr by lag(ms): " +
          " ".join(f"{l*BIN}:{v:+.3f}" for l, v in zip(lags, xc) if l in (-2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 20)))
    print(f"   impulse response: 50% absorbed by {half} ms, 90% by {p90} ms; "
          f"share after 1s = {pos[lags[lags>=0]*BIN>1000].sum()/pos.sum():.2f}")
