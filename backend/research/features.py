"""Vectorized causal features. No target, final close, or future window eligibility input."""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .data_loader import ASSETS

EPS = 1e-12
METADATA = {"symbol", "dt", "window_id", "available_at", "k", "ready"}


def _divide(a, b):
    return a / b.replace(0, np.nan)


def _freeze(series, window, elapsed):
    return series.where(elapsed.eq(0)).groupby(window).transform("first")


def _own(b: pd.DataFrame) -> pd.DataFrame:
    w, k, elapsed = b.window_id, b.k.astype(float), b.elapsed
    o, h, lo, c, v = (b[name] for name in ("open", "high", "low", "close", "volume"))
    anchor = b.anchor_open
    r = np.log(c).diff()  # NaN across gaps, no implicit padding.
    prior_sigma = r.rolling(60, min_periods=40).std(ddof=1).shift(1)
    sigma = _freeze(prior_sigma, w, elapsed).clip(lower=1e-7)
    d = c / anchor - 1
    high = h.groupby(w).cummax() / anchor - 1
    low = 1 - lo.groupby(w).cummin() / anchor
    total_v = v.groupby(w).cumsum()
    spread = h - lo
    clv = ((2 * c - h - lo) / spread.replace(0, np.nan)).fillna(0)
    signed = clv * v
    body = c - o
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - lo
    sign = np.sign(d)
    # Carry zero closes within a window to count transitions between nonzero signs.
    nonzero = sign.replace(0, np.nan).groupby(w).ffill()
    prev_sign = nonzero.groupby(w).shift(1)
    cross = ((nonzero * prev_sign) < 0).astype(float)
    n = c.notna().groupby(w).cumsum()
    f = {
        "anchor_return": d,
        "pre_sigma": sigma,
        "z_elapsed": d / (sigma * np.sqrt(k)),
        "z_remaining": d / (sigma * np.sqrt(15 - k).clip(lower=1)),
        "remaining_minutes": 15 - k,
        "excursion_up": high / sigma,
        "excursion_down": low / sigma,
        "excursion_log_ratio": np.log((high.clip(lower=0) + EPS) / (low.clip(lower=0) + EPS)),
        "excursion_balance": (high - low) / (high + low + EPS),
        "retracement_from_high": (high - d) / sigma,
        "bounce_from_low": (low + d) / sigma,
        "time_above_open": d.gt(0).astype(float).groupby(w).cumsum() / k,
        "midpoint_above_open": ((h + lo) / 2 > anchor).astype(float).groupby(w).cumsum() / k,
        "zero_crossings": cross.groupby(w).cumsum(),
        "window_signed_volume": signed.groupby(w).cumsum(),
        "window_delta_fraction": _divide(signed.groupby(w).cumsum(), total_v),
        "bar_clv": clv,
        "bar_body_fraction": body / (spread + EPS),
        "window_range_sigma": (high + low) / sigma,
        "distance_x_orderflow": d / sigma * _divide(signed.groupby(w).cumsum(), total_v),
    }
    # Weighted least squares through the known anchor y(0)=0. For k=1, use
    # a straight line and zero curvature; for k>=2 fit y=a*x + b*x^2.
    weight = v.clip(lower=EPS)
    moments = {j: (weight * k**j).groupby(w).cumsum() for j in (2, 3, 4)}
    t1 = (weight * k * d).groupby(w).cumsum()
    t2 = (weight * k**2 * d).groupby(w).cumsum()
    det = moments[2] * moments[4] - moments[3] ** 2
    a = (t1 * moments[4] - t2 * moments[3]) / det.where(det.abs() > EPS)
    b2 = (t2 * moments[2] - t1 * moments[3]) / det.where(det.abs() > EPS)
    f["path_velocity"] = a.where(k.gt(1), d) / sigma
    f["path_acceleration"] = 2 * b2.where(k.gt(1), 0) / sigma
    f["end_velocity"] = f["path_velocity"] + k * f["path_acceleration"]
    steps = (c - c.groupby(w).shift(1)).where(elapsed.ne(0), c - anchor).abs()
    f["window_efficiency"] = (c - anchor) / (steps.groupby(w).cumsum() + EPS)
    for length in (5, 10, 15, 30):
        net = np.log(c / c.shift(length))
        path = r.abs().rolling(length).sum()
        f[f"efficiency_{length}"] = net / (path + EPS)
        f[f"return_{length}"] = net
    five = np.log(c / c.shift(5))
    f["variance_ratio_5_60"] = five.rolling(60).var() / (5 * r.rolling(60).var() + EPS)
    # Overlapping-return variance ratio is a regime proxy, not an iid test statistic.
    f["realized_skew_15"] = r.rolling(15).skew()
    f["realized_excess_kurtosis_15"] = r.rolling(15).kurt()
    f["wick_pressure_15"] = np.log(
        ((v * upper / (o + EPS)).rolling(15).sum() + EPS)
        / ((v * lower / (o + EPS)).rolling(15).sum() + EPS)
    )
    f["delta_fraction_15"] = signed.rolling(15).sum() / (v.rolling(15).sum() + EPS)
    f["window_delta_vs_prior"] = f["window_delta_fraction"] - _freeze(
        f["delta_fraction_15"].shift(1), w, elapsed
    )
    impact = r.abs() / (v * c).replace(0, np.nan)  # Dollar-volume normalization.
    f["amihud_10"] = impact.rolling(10, min_periods=5).mean()
    up = impact.where(r.gt(0)).rolling(10, min_periods=1).mean()
    down = impact.where(r.lt(0)).rolling(10, min_periods=1).mean()
    f["impact_asymmetry_10"] = (up - down) / (up + down + EPS)
    logvol = np.log1p(v * c)
    vm = logvol.shift(1).rolling(60, min_periods=40).mean()
    vs = logvol.shift(1).rolling(60, min_periods=40).std()
    f["volume_z"] = (logvol - vm) / vs.clip(lower=1e-6)
    f["window_volume_surprise"] = total_v / (
        k * _freeze(v.shift(1).rolling(60, min_periods=40).mean(), w, elapsed) + EPS
    )
    for name in ("efficiency_15", "variance_ratio_5_60", "realized_skew_15", "wick_pressure_15"):
        f["pre_" + name] = _freeze(f[name].shift(1), w, elapsed)
    # Previous extremum timing is available only after that whole candle closed.
    complete = o.groupby(w).count().eq(15)
    for name, series, extreme in (("high", h, "max"), ("low", lo, "min")):
        ex = series.groupby(w).transform(extreme)
        timing = elapsed.where(series.eq(ex)).groupby(w).min().where(complete)
        timing.index = timing.index + pd.Timedelta(minutes=15)
        f[f"previous_{name}_minute"] = w.map(timing)
    index = b.index
    minute = index.hour * 60 + index.minute + 1  # Information availability clock.
    f["time_sin"] = np.sin(2 * np.pi * minute / 1440)
    f["time_cos"] = np.cos(2 * np.pi * minute / 1440)
    f["week_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    f["week_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    f["quarter_hour"] = w.dt.minute / 15
    f["top_of_hour_window"] = w.dt.minute.eq(0).astype(float)
    # DST-aware clock proxies; exchange holidays are deliberately not inferred.
    available = index + pd.Timedelta(minutes=1)
    ny, chicago = available.tz_convert("America/New_York"), available.tz_convert("America/Chicago")
    ny_clock, ct_clock = ny.hour * 60 + ny.minute, chicago.hour * 60 + chicago.minute
    f["us_equity_open_clock"] = ((ny.dayofweek < 5) & (abs(ny_clock - 570) <= 15)).astype(float)
    legacy = available < pd.Timestamp("2026-05-29 16:00", tz="America/Chicago")
    f["cme_legacy_close_clock"] = (
        legacy & (chicago.dayofweek < 5) & (abs(ct_clock - 960) <= 15)
    ).astype(float)
    f["funding_clock"] = ((minute % 480 <= 10) | (minute % 480 >= 470)).astype(float)
    f["funding_minutes"] = minute % 480
    out = pd.DataFrame(f, index=b.index).replace([np.inf, -np.inf], np.nan)
    out["ready"] = n.eq(k) & anchor.notna() & sigma.notna() & c.notna()
    return out


def build_features(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return k=1..10 snapshots, with only prices/volumes known at available_at.

    NaNs are retained for LightGBM's missing-value handling. No fitted scaler,
    whole-sample volatility estimate, or cross-sectional future fill is used.
    """
    own = {s: _own(bars[s]) for s in ASSETS}
    returns = {s: np.log(bars[s].close).diff() for s in ASSETS}
    z = pd.DataFrame({s: own[s].z_remaining for s in ASSETS})
    vz = pd.DataFrame({s: own[s].volume_z for s in ASSETS})
    available = z.notna().all(axis=1)
    aligned = ((z > 0).all(axis=1) | (z < 0).all(axis=1)).astype(float).where(available)
    mass = vz.clip(lower=0) + EPS
    shares = mass.div(mass.sum(axis=1), axis=0)
    entropy = -(shares * np.log(shares)).sum(axis=1).where(vz.notna().all(axis=1))
    snapshots = []
    for s in ASSETS:
        f = own[s]
        peers = [p for p in ASSETS if p != s]
        extra = {
            "cross_all_aligned": aligned,
            "cross_z_sum": z.sum(axis=1, min_count=3),
            "cross_z_dispersion": z.std(axis=1).where(available),
            "cross_volume_entropy": entropy,
            "cross_volume_ignitions": vz.gt(2).sum(axis=1).where(vz.notna().all(axis=1)),
            "peer_anchor_gap": z[peers].mean(axis=1).where(available) - z[s],
        }
        for p in peers:
            extra[f"{p}_z_remaining"] = z[p]
            extra[f"{p}_volume_z"] = vz[p]
            # Betas fit on strictly prior one-minute pairs, never on future rows.
            cov = returns[s].rolling(120, min_periods=60).cov(returns[p]).shift(1)
            var = returns[p].rolling(120, min_periods=60).var().shift(1)
            beta = (cov / var.clip(lower=EPS)).clip(-5, 5)
            extra[f"beta_{p}"] = beta
            for lag in (1, 3, 5):
                rs = np.log(bars[s].close / bars[s].close.shift(lag))
                rp = np.log(bars[p].close / bars[p].close.shift(lag))
                extra[f"residual_{p}_{lag}"] = (rs - beta * rp) / (f.pre_sigma * np.sqrt(lag))
                extra[f"lagged_{p}_{lag}"] = rp.shift(1) / (f.pre_sigma * np.sqrt(lag))
        f = pd.concat([f, pd.DataFrame(extra, index=f.index)], axis=1)
        mask = bars[s].k.between(1, 10)
        f = f.loc[mask].copy()
        ready = f.pop("ready")
        f = f.replace([np.inf, -np.inf], np.nan).astype("float32")
        f["ready"] = ready
        f["symbol"] = s
        f["dt"] = f.index
        for name in ("window_id", "available_at", "k"):
            f[name] = bars[s].loc[mask, name]
        snapshots.append(f.reset_index(drop=True))
    return pd.concat(snapshots, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in METADATA]


def bridge_probability(z_remaining):
    """Zero-drift Gaussian remaining-increment proxy; not a fitted probability."""
    return ndtr(z_remaining)
