"""How well does a Coinbase 60s average reproduce Kalshi's settlement value (CF BRTI avg)?
Compare expiration_value with Coinbase mid/trade averaged over [close-60s, close)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = str(ROOT / "data/recorder/parquet") + "/"
OUT = ROOT / "research/artifacts/live"
st = pd.read_parquet(D + "settlements.parquet")
st["close_ms"] = pd.to_datetime(st.close_time).astype("int64") // 10**6
st["ev"] = st.expiration_value.astype(float)
st["asset"] = st.ticker.str.extract(r"KX(BTC|ETH|SOL)")[0]
ev = st.drop_duplicates("event_ticker")[["event_ticker", "asset", "close_ms", "ev"]]
cb = pd.read_parquet(D + "cb_ticker.parquet", columns=["product", "ts", "bid", "ask", "price"])
out = []
for a, g in ev.groupby("asset"):
    c = cb[cb["product"] == f"{a}-USD"].sort_values("ts")
    c["mid"] = (c.bid + c.ask) / 2
    sec = c.set_index(pd.to_datetime(c.ts, unit="ms")).mid.resample("1s").last().ffill()
    for _, r in g.iterrows():
        e = pd.to_datetime(r.close_ms, unit="ms")
        w = sec.loc[e - pd.Timedelta("60s"): e - pd.Timedelta("1s")]
        if len(w) < 55:
            continue
        out.append({"asset": a, "event": r.event_ticker, "ev": r.ev, "cb60": w.mean(), "cblast": w.iloc[-1]})
o = pd.DataFrame(out)
o["err_bp"] = (o.cb60 / o.ev - 1) * 1e4
o["err_last_bp"] = (o.cblast / o.ev - 1) * 1e4
print(o.groupby("asset").agg(events=("ev", "size"), mean_bp=("err_bp", "mean"), sd_bp=("err_bp", "std"),
                             mae_bp=("err_bp", lambda s: s.abs().mean()), sd_last_bp=("err_last_bp", "std")).round(2))
# outcome agreement: does sign(cb60 - strike) match result for every recorded contract?
m = st.merge(o[["event", "cb60", "err_bp"]], left_on="event_ticker", right_on="event")
m["pred"] = np.where(m.cb60 > m.floor_strike, "yes", "no")
m["dist_bp"] = (m.ev / m.floor_strike - 1) * 1e4
print("contracts", len(m), "Coinbase-60s sign agrees with result:", (m.pred == m.result).mean().round(4))
bad = m[m.pred != m.result]
print("disagreements: |settle-strike| bp", bad.dist_bp.abs().round(2).tolist())
o.to_parquet(OUT / "basis.parquet")
