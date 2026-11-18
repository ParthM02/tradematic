"""Does Kalshi taker flow (true aggressor side from Webull event-tick) predict future prices?

1 s bins per contract (15m series, plus hourly strikes while their mid is 0.15-0.85).
Signed flow: +contracts when the taker bought YES, - when the taker bought NO.
Features at second t (all causal): flow over last 5 s / 30 s / 120 s (in units of the
contract's trailing median 30 s |flow|), Coinbase log return over last 5 s / 30 s.
Targets measured from the quote at t+1 s (1 s execution latency):
  dmid over 10 s, 60 s, and settlement - mid  (the hold-to-expiry edge).
Reports per-decile mean moves of the flow features and OLS slopes with window-clustered SEs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = str(ROOT / "data/recorder/parquet") + "/"
OUT = ROOT / "research/artifacts/live"


def contract_panel():
    st = pd.read_parquet(D + "settlements.parquet")
    st["close_s"] = pd.to_datetime(st.close_time).astype("int64") // 10**9
    st["y"] = (st.result == "yes").astype(int)
    st["asset"] = st.ticker.str.extract(r"KX(BTC|ETH|SOL)")[0]
    q = pd.read_parquet(D + "wb_quote.parquet", columns=["symbol", "ts", "yes_bid", "no_bid"])
    tk = pd.read_parquet(D + "wb_tick.parquet", columns=["symbol", "time", "volume", "side", "trade_id"])
    tk = tk.drop_duplicates("trade_id")
    cb = pd.read_parquet(D + "cb_ticker.parquet", columns=["product", "ts", "bid", "ask"])
    spot = {}
    for a in ["BTC", "ETH", "SOL"]:
        c = cb[cb["product"] == f"{a}-USD"]
        s = np.log((c.bid + c.ask) / 2).groupby(c.ts // 1000).last()
        spot[a] = s.reindex(np.arange(s.index.min(), s.index.max() + 1)).ffill()
    q["s"] = q.ts // 1000
    q["mid"] = (q.yes_bid + 1 - q.no_bid) / 2
    q["spr"] = 1 - q.no_bid - q.yes_bid
    tk["s"] = tk.time // 1000
    tk["sf"] = np.where(tk.side == "yes", tk.volume, -tk.volume)
    qg, tg = dict(tuple(q.groupby("symbol"))), dict(tuple(tk.groupby("symbol")))
    out = []
    for _, m in st.iterrows():
        if m.ticker not in qg:
            continue
        g = qg[m.ticker].groupby("s")[["mid", "spr"]].last()
        idx = np.arange(g.index.min(), m.close_s)
        g = g.reindex(idx).ffill()
        f = tg[m.ticker].groupby("s").sf.sum().reindex(idx, fill_value=0) if m.ticker in tg else pd.Series(0.0, idx)
        sp = spot[m.asset].reindex(idx)
        df = pd.DataFrame({"mid": g.mid.values, "spr": g.spr.values}, index=idx)
        for w in (5, 30, 120):
            df[f"f{w}"] = f.rolling(w, min_periods=1).sum().values
        scale = df.f30.abs().rolling(900, min_periods=60).median().shift(1).clip(lower=1)
        for w in (5, 30, 120):
            df[f"f{w}"] = df[f"f{w}"] / scale
        df["s5"], df["s30"] = (sp - sp.shift(5)).values * 1e4, (sp - sp.shift(30)).values * 1e4
        m1 = df.mid.shift(-1)
        df["entry_mid"] = m1
        df["entry_spr"] = df.spr.shift(-1)
        df["d10"], df["d60"] = df.mid.shift(-11) - m1, df.mid.shift(-61) - m1
        df["dset"] = m.y - m1
        df["ticker"], df["event"], df["series"] = m.ticker, m.event_ticker, m.ticker.split("-")[0]
        df["rem"] = m.close_s - idx
        out.append(df.reset_index(names="t"))
    P = pd.concat(out, ignore_index=True)
    return P[(P.mid > 0.15) & (P.mid < 0.85) & (P.rem > 62) & P.f120.notna() & P.s30.notna()]


def ols_cluster(X, y, groups):
    X = np.c_[np.ones(len(X)), X]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = pd.DataFrame(X * e[:, None]).groupby(groups.values).sum().values
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return b, np.sqrt(np.diag(V))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    P = contract_panel()
    P.to_parquet(OUT / "flow_panel.parquet")
    feats = ["f5", "f30", "f120", "s5", "s30"]
    for fam, d in [("15m", P[P.series.str.endswith("15M")]), ("hourly", P[~P.series.str.endswith("15M")])]:
        print(f"===== {fam}: {len(d):,} seconds, {d.ticker.nunique()} contracts, {d.event.nunique()} events")
        for tgt in ["d10", "d60", "dset"]:
            x = d.dropna(subset=[tgt])
            b, se = ols_cluster(x[feats].values, x[tgt].values, x.event)
            print(f"  {tgt:5s} cents per 1 sd: " + "  ".join(
                f"{f}={100*b[i+1]*x[f].std():+.2f}(t={b[i+1]/se[i+1]:+.1f})" for i, f in enumerate(feats)))
        d = d.dropna(subset=["d60"]).copy()
        d["fq"] = pd.qcut(d.f30.rank(method="first"), 10, labels=False)
        t = d.groupby("fq").agg(f30=("f30", "mean"), d10=("d10", "mean"), d60=("d60", "mean"),
                                dset=("dset", "mean"), spr=("entry_spr", "mean"), n=("t", "size"))
        t[["d10", "d60", "dset", "spr"]] *= 100
        print("  by 30s-flow decile (moves in cents, from the t+1s mid):")
        print(t.round(2).to_string())
