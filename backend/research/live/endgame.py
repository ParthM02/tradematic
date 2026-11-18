"""Endgame pricing test: in the last N seconds, does a Coinbase-based model of the 60s
settlement average beat Webull/Kalshi quotes by more than the 2c fee?

Settlement = mean of CF BRTI over the 60 s before close. Proxy: Coinbase 1 s mid,
plus a per-asset basis estimated ONLY from events that closed earlier (expanding mean;
the first 5 events per asset are skipped). At decision time t (seconds before close = r):
  known  = seconds already inside the averaging window, with their realized mean
  future = remaining points follow a driftless random walk from the current mid
  var(final average) = sigma^2 * [d + sum_{i=1..m} i^2 / 60^2]   (m = future points,
                       d = seconds until the window opens if r > 60), plus basis noise.
sigma = trailing 300 s RMS of 1 s log mid changes (causal).
Execution: buy YES at yes_ask (= 1 - best no bid) or NO at no_ask, 1 contract, $0.02 fee,
held to official settlement. Quotes are taken at t + latency (default 0, 0.5 s, 1 s).
One trade per contract: first second the edge clears the margin.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
D = str(ROOT / "data/recorder/parquet") + "/"
OUT = ROOT / "research/artifacts/live"
FEE, LOOKBACK, BASIS_SD = 0.02, 300, 0.7e-4


def mids_1s(cb, asset):
    c = cb[cb["product"] == f"{asset}-USD"]
    s = ((c.bid + c.ask) / 2).groupby(c.ts // 1000).last()
    s = s.reindex(np.arange(s.index.min(), s.index.max() + 1)).ffill()
    lr = np.log(s).diff()
    sig = np.sqrt((lr**2).rolling(300, min_periods=120).mean())
    return s, sig


def build():
    st = pd.read_parquet(D + "settlements.parquet")
    st["close_s"] = pd.to_datetime(st.close_time).astype("int64") // 10**9
    st["asset"] = st.ticker.str.extract(r"KX(BTC|ETH|SOL)")[0]
    st["y"] = (st.result == "yes").astype(int)
    cb = pd.read_parquet(D + "cb_ticker.parquet", columns=["product", "ts", "bid", "ask"])
    q = pd.read_parquet(D + "wb_quote.parquet", columns=["symbol", "ts", "yes_bid", "no_bid"])
    q = q[q.symbol.isin(st.ticker)].sort_values("ts")
    rows = []
    for a, sa in st.groupby("asset"):
        s, sig = mids_1s(cb, a)
        # causal basis: expanding mean of (cb60/settle - 1) over previously closed events
        ev = sa.drop_duplicates("event_ticker").sort_values("close_s").copy()
        ev["cb60"] = [s.loc[c - 60:c - 1].mean() for c in ev.close_s]
        ev["b"] = np.log(ev.cb60 / ev.expiration_value.astype(float))
        ev["b_prior"] = ev.b.expanding().mean().shift(1)
        ev["n_prior"] = np.arange(len(ev))
        sa = sa.merge(ev[["event_ticker", "b_prior", "n_prior"]], on="event_ticker")
        sa = sa[sa.n_prior >= 5]
        for _, m in sa.iterrows():
            t = np.arange(m.close_s - LOOKBACK, m.close_s - 1)
            if t[0] - 300 < s.index[0] or t[-1] > s.index[-1]:
                continue
            S = s.reindex(t).values
            r = m.close_s - t                      # seconds to close
            k = np.clip(60 - r, 0, 60)            # points already inside the window
            win = s.loc[m.close_s - 60:m.close_s - 1].values
            csum = np.concatenate([[0], np.cumsum(win)])
            known_sum = csum[k.astype(int)]
            mpts = 60 - k                          # future points (incl. pre-window drift)
            d = np.clip(r - 60, 0, None)
            exp_avg = (known_sum + mpts * S) / 60
            sd = S * sig.reindex(t).values * np.sqrt(d * (mpts / 60) ** 2 + (mpts * (mpts + 1) * (2 * mpts + 1) / 6) / 3600)
            adj = np.exp(-m.b_prior)               # Coinbase -> BRTI level
            mu = exp_avg * adj
            sd_tot = np.sqrt(sd**2 * adj**2 + (BASIS_SD * mu) ** 2)
            p = norm.cdf((mu - m.floor_strike) / sd_tot)
            rows.append(pd.DataFrame({"ticker": m.ticker, "event": m.event_ticker, "asset": a,
                                      "series": m.ticker.split("-")[0], "t": t, "r": r, "p": p, "y": m.y}))
    P = pd.concat(rows, ignore_index=True)
    # attach the latest quote at t + latency
    for lat in (0, 500, 1000):
        key = P[["ticker", "t"]].assign(ts=P.t * 1000 + lat).sort_values("ts")
        j = pd.merge_asof(key, q.rename(columns={"symbol": "ticker"}), on="ts", by="ticker",
                          direction="backward", tolerance=5000)
        j = j.set_index(["ticker", "t"])
        P[f"ya{lat}"] = (1 - j.no_bid).reindex(pd.MultiIndex.from_frame(P[["ticker", "t"]])).values
        P[f"na{lat}"] = (1 - j.yes_bid).reindex(pd.MultiIndex.from_frame(P[["ticker", "t"]])).values
    return P


def evaluate(P, margins=(0.0, 0.01, 0.02, 0.04), windows=((0, 300), (0, 60), (60, 300))):
    out = []
    P = P.dropna(subset=["ya0", "na0"])
    P = P[(P.ya0 > 0) & (P.ya0 < 1) & (P.na0 > 0) & (P.na0 < 1)]
    for lo, hi in windows:
        w = P[(P.r > lo) & (P.r <= hi)]
        for mg in margins:
            eY, eN = w.p - w.ya0 - FEE, (1 - w.p) - w.na0 - FEE
            sel = w[(np.maximum(eY, eN) > mg)].copy()
            sel["yes"] = (eY >= eN)[sel.index]
            sel = sel.sort_values("t").groupby("ticker").head(1)
            for lat in (0, 500, 1000):
                px = np.where(sel.yes, sel[f"ya{lat}"], sel[f"na{lat}"])
                win = np.where(sel.yes, sel.y, 1 - sel.y)
                pnl = win - px - FEE
                ev = pd.Series(pnl, index=sel.event).groupby(level=0).mean()
                out.append({"window": f"{lo}-{hi}s", "margin": mg, "latency_ms": lat, "trades": len(sel),
                                "events": len(ev), "win": win.mean(), "px": np.nanmean(px),
                                "pred_edge": np.maximum(eY, eN)[sel.index].mean(),
                                "pnl": np.nanmean(pnl), "pnl_event_se": ev.std() / np.sqrt(len(ev))})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    P = build()
    P.to_parquet(OUT / "endgame_panel.parquet")
    print("panel", len(P), "contracts", P.ticker.nunique(), "events", P.event.nunique())
    # calibration of model vs market at decision time
    Q = P.dropna(subset=["ya0"])
    Q = Q[(Q.ya0 > 0) & (Q.na0 > 0)]
    Q["mid"] = (Q.ya0 + 1 - Q.na0) / 2
    for nm, col in [("model", "p"), ("market", "mid")]:
        x = Q[col].clip(1e-6, 1 - 1e-6)
        print(f"{nm:6s} Brier {((x - Q.y) ** 2).mean():.5f}  logloss {-(Q.y*np.log(x)+(1-Q.y)*np.log(1-x)).mean():.4f}")
    print(evaluate(P).round(4).to_string())
