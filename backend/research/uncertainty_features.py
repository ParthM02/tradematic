"""Causal contract-price and technical features for uncertain SOL markets."""

import numpy as np
import pandas as pd
from scipy.special import logit, ndtr

from .features import feature_columns


def causal_kalman(log_price, prior_variance):
    """Local linear trend FILTER (never a smoother), with fixed variance ratios.

    Observation variance is the lagged rolling return variance. State resets
    after missing prices. Parameters are fixed before any fold is examined.
    """
    output = np.full((len(log_price), 3), np.nan)
    initialized = False
    for i, (price, variance) in enumerate(zip(log_price, prior_variance, strict=True)):
        if not np.isfinite(price):
            initialized = False
            continue
        r = max(float(variance), 1e-10) if np.isfinite(variance) else 1e-6
        if not initialized:
            level, velocity, p00, p01, p11 = price, 0.0, r, 0.0, r
            initialized = True
            output[i] = 0
            continue
        predicted = level + velocity
        a, b, c = p00 + 2 * p01 + p11 + 0.01 * r, p01 + p11, p11 + 0.001 * r
        innovation, total = price - predicted, a + r
        gain0, gain1 = a / total, b / total
        level, velocity = predicted + gain0 * innovation, velocity + gain1 * innovation
        p00, p01, p11 = max(a - a * gain0, 1e-15), b - b * gain0, max(c - b * gain1, 1e-15)
        output[i] = [
            (price - level) / np.sqrt(r),
            velocity / np.sqrt(r),
            innovation / np.sqrt(total),
        ]
    return output


def technical_features(bars):
    price = np.log(bars.close)
    variance = price.diff().rolling(60, min_periods=40).var().shift(1)
    sigma = np.sqrt(variance).clip(lower=1e-7)
    features = {}
    for length in (5, 15, 30, 60):
        sma = price.rolling(length).mean()
        ema = price.ewm(span=length, adjust=False, min_periods=length).mean()
        features[f"sma_gap_{length}"] = (price - sma) / sigma
        features[f"ema_gap_{length}"] = (price - ema) / sigma
        features[f"sma_slope_{length}"] = (sma - sma.shift(3)) / (sigma * 3)
    fast = price.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = price.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd"] = macd / sigma
    features["macd_signal"] = signal / sigma
    features["macd_histogram"] = (macd - signal) / sigma
    features["bollinger_position_20"] = (price - price.rolling(20).mean()) / price.rolling(
        20
    ).std().clip(lower=1e-8)
    features["bollinger_width_20"] = 4 * price.rolling(20).std() / sigma
    kalman = causal_kalman(price.to_numpy(), variance.to_numpy())
    for i, name in enumerate(("kalman_price_gap", "kalman_trend", "kalman_innovation")):
        features[name] = kalman[:, i]
    return (
        pd.DataFrame(features, index=bars.index)
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )


def build_uncertainty_snapshots(bars, path_snapshots, quotes, market_info):
    """No settlement result, expiration_value, or future eligibility is accepted.

    Only the whitelist of contract metadata below enters feature construction.
    Quotes are assumed observable at end_ts, matching completed-minute k.
    """
    info = market_info[["ticker", "open_time", "close_time", "floor_strike"]].copy()
    if info.open_time.duplicated().any() or info.floor_strike.le(0).any():
        raise ValueError("Ambiguous contract opens or invalid strikes")
    data = path_snapshots[path_snapshots.symbol.eq("SOL")].copy()
    path_columns = feature_columns(data)
    data = data.merge(info, left_on="window_id", right_on="open_time", validate="many_to_one")
    qfields = ["ticker", "quote_time", "yes_bid", "yes_ask", "no_ask", "volume"]
    data = data.merge(
        quotes[qfields].rename(columns={"volume": "quote_volume"}),
        left_on=["ticker", "available_at"],
        right_on=["ticker", "quote_time"],
        how="left",
        validate="one_to_one",
    ).sort_values(["ticker", "k"])
    g = data.groupby("ticker", sort=False)
    data["market_mid"] = (data.yes_bid + data.yes_ask) / 2
    data["market_logit"] = logit(data.market_mid.clip(1e-4, 1 - 1e-4))
    data["market_spread"] = data.yes_ask - data.yes_bid
    data["market_logit_x_k"] = data.market_logit * data.k / 10
    data["market_volume_log"] = np.log1p(data.quote_volume)
    data["market_volume_cumulative"] = np.log1p(g.quote_volume.cumsum())
    for lag in (1, 3, 5):
        data[f"market_move_{lag}"] = data.market_mid - g.market_mid.shift(lag)
    data["market_high_so_far"] = g.market_mid.cummax()
    data["market_low_so_far"] = g.market_mid.cummin()
    data["market_observed_fraction"] = (
        data.quote_time.notna().groupby(data.ticker).cumsum() / data.k
    )
    market_columns = [c for c in data if c.startswith("market_")] + ["k"]
    tech = technical_features(bars)
    data = data.merge(tech, left_on="dt", right_index=True, how="left", validate="many_to_one")
    own = bars.reindex(pd.DatetimeIndex(data.dt))
    distance = own.close.to_numpy() / data.floor_strike.to_numpy() - 1
    data["strike_distance_z"] = distance / data.pre_sigma
    data["strike_remaining_z"] = data.strike_distance_z / np.sqrt(15 - data.k)
    data["strike_anchor_basis_z"] = (
        own.anchor_open.to_numpy() / data.floor_strike.to_numpy() - 1
    ) / data.pre_sigma
    data["strike_gaussian_probability"] = ndtr(data.strike_remaining_z)
    data["strike_probability_market_gap"] = data.strike_gaussian_probability - data.market_mid
    data["strike_time_above"] = (own.close.to_numpy() > data.floor_strike).astype(float).groupby(
        data.ticker
    ).cumsum() / data.k
    path_columns += [c for c in data if c.startswith("strike_")]
    groups = {
        "market_platt": ["market_logit", "market_spread", "k", "market_logit_x_k"],
        "market": market_columns,
        "path": market_columns + path_columns,
        "classic": market_columns + path_columns + [c for c in tech if not c.startswith("kalman_")],
        "technical": market_columns + path_columns + list(tech.columns),
    }
    numeric = list(dict.fromkeys(groups["technical"]))
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    data = data.copy()  # Consolidate numeric blocks before attaching flags and labels.
    data["observable"] = data.ready & data.quote_time.notna() & data.market_mid.notna()
    return data.reset_index(drop=True), groups
