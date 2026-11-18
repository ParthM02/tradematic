import numpy as np
import pandas as pd
from scipy.special import logit

from backend.research.data_loader import load_market
from backend.research.features import build_features
from backend.research.tests.test_pipeline import make_csvs
from backend.research.uncertainty_features import build_uncertainty_snapshots, technical_features
from backend.research.uncertainty_trainer import (
    first_value_passage,
    fit_forecast,
    learn_value_policy,
    replay_values,
    window_weights,
)


def contracts_and_quotes(bars):
    ids = pd.DatetimeIndex(bars.window_id.unique())
    info = pd.DataFrame(
        {
            "ticker": [f"test_{i}" for i in range(len(ids))],
            "open_time": ids,
            "close_time": ids + pd.Timedelta(minutes=15),
            "floor_strike": bars.loc[ids, "open"].to_numpy(),
            "result": "yes",
            "expiration_value": 999.0,
        }
    )
    rows = []
    for row in info.itertuples():
        for k in range(1, 11):
            rows.append(
                {
                    "ticker": row.ticker,
                    "quote_time": row.open_time + pd.Timedelta(minutes=k),
                    "yes_bid": 0.48,
                    "yes_ask": 0.52,
                    "no_ask": 0.52,
                    "volume": k,
                }
            )
    return info, pd.DataFrame(rows)


def test_future_and_settlement_changes_cannot_affect_observable_features(tmp_path):
    paths = make_csvs(tmp_path)
    market = load_market(paths)
    info, quotes = contracts_and_quotes(market.bars["SOL"])
    path = build_features(market.bars)
    before, groups = build_uncertainty_snapshots(market.bars["SOL"], path, quotes, info)
    assert not {"result", "expiration_value", "y"}.intersection(before.columns)
    assert "kalman_trend" in groups["technical"] and "kalman_trend" not in groups["classic"]
    cutoff = pd.Timestamp("2025-01-01 04:10", tz="UTC")
    for p in paths.values():
        frame = pd.read_csv(p)
        future = pd.to_datetime(frame.dt, utc=True).ge(cutoff)
        frame.loc[future, ["open", "high", "low", "close"]] *= 2
        frame.loc[future, "volume"] *= 7
        frame.to_csv(p, index=False)
    later = quotes.quote_time.gt(cutoff)
    quotes.loc[later, ["yes_bid", "yes_ask", "no_ask"]] = [0.10, 0.14, 0.90]
    info["result"], info["expiration_value"] = "no", 0.0
    changed = load_market(paths)
    after, _ = build_uncertainty_snapshots(
        changed.bars["SOL"], build_features(changed.bars), quotes, info
    )
    keys = ["window_id", "k"]
    a = before[before.available_at.le(cutoff)].sort_values(keys).reset_index(drop=True)
    b = after[after.available_at.le(cutoff)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    prefix = market.bars["SOL"].loc[: cutoff - pd.Timedelta(minutes=1)]
    pd.testing.assert_frame_equal(
        technical_features(market.bars["SOL"]).loc[prefix.index], technical_features(prefix)
    )


def test_market_quote_lags_do_not_skip_missing_minutes_or_cross_contracts(tmp_path):
    market = load_market(make_csvs(tmp_path))
    info, quotes = contracts_and_quotes(market.bars["SOL"])
    missing = (quotes.ticker == "test_8") & quotes.quote_time.eq(
        info.open_time.iloc[8] + pd.Timedelta(minutes=4)
    )
    quotes = quotes[~missing]
    frame, _ = build_uncertainty_snapshots(
        market.bars["SOL"], build_features(market.bars), quotes, info
    )
    contract = frame[frame.ticker.eq("test_8")].set_index("k")
    assert pd.isna(contract.loc[5, "market_move_1"])
    assert not contract.loc[4, "observable"]
    assert pd.isna(contract.loc[1, "market_move_1"])


def test_value_policy_can_buy_less_likely_side_and_locks_first_action():
    p = np.full((2, 10), 0.45)
    matrices = {
        "p": p,
        "yes_ask": np.full((2, 10), 0.35),
        "no_ask": np.full((2, 10), 0.67),
        "eligible": np.ones((2, 10), bool),
    }
    matrices["eligible"][1] = False
    result = first_value_passage(matrices, 0.03)
    np.testing.assert_array_equal(result["trigger_k"], [1, 0])
    np.testing.assert_array_equal(result["side"], [1, -1])
    p[0, 1:] = 0.01
    assert first_value_passage(matrices, 0.03)["side"][0] == 1
    universe = pd.DataFrame(
        {"window_id": pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"), "y": [1, 0]}
    )
    scored = replay_values(universe, matrices, 0.03)
    np.testing.assert_allclose(scored.net_pnl, [0.63, 0])


def test_tuning_cannot_manufacture_25_percent_coverage_or_force_losing_trades():
    matrices = {
        "p": np.full((100, 10), 0.8),
        "yes_ask": np.full((100, 10), 0.5),
        "no_ask": np.full((100, 10), 0.52),
        "eligible": np.zeros((100, 10), bool),
    }
    matrices["eligible"][:20] = True
    universe = pd.DataFrame(
        {"window_id": pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC"), "y": 0}
    )
    policy = learn_value_policy(universe, matrices, constrained=True)
    assert not policy["validation_feasible_25"]
    assert policy["tuning_coverage"] == 0.2
    selective = learn_value_policy(universe, matrices)
    assert np.isinf(selective["margin"]) and selective["tuning_coverage"] == 0


def test_residual_model_adds_market_offset_back_and_windows_have_equal_weight():
    rng = np.random.default_rng(378)

    def frame(n, start):
        midpoint = rng.choice([0.2, 0.8], size=n)
        return pd.DataFrame(
            {
                "window_id": pd.date_range(start, periods=n, freq="15min", tz="UTC"),
                "market_mid": midpoint,
                "constant": 0.0,
                "y": rng.binomial(1, midpoint),
            }
        )

    train, cal = frame(300, "2026-01-01"), frame(150, "2026-02-01")
    model = fit_forecast(train, cal, ["constant"], "residual", trees=2, threads=1)
    # With no splitting features, the correction is zero; the market offset must survive.
    np.testing.assert_allclose(model.raw(cal), logit(cal.market_mid), atol=1e-10)
    duplicate = pd.concat([train.iloc[:1]] * 5 + [train.iloc[1:2]], ignore_index=True)
    weights = window_weights(duplicate)
    assert np.isclose(weights[:5].sum(), weights[5])


def test_zero_validation_actions_choose_explicit_abstention_not_an_unobserved_tail():
    matrices = {
        "p": np.full((100, 10), 0.6),
        "yes_ask": np.full((100, 10), 0.55),
        "no_ask": np.full((100, 10), 0.47),
        "eligible": np.ones((100, 10), bool),
    }
    universe = pd.DataFrame(
        {"window_id": pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC"), "y": 0}
    )
    policy = learn_value_policy(universe, matrices)
    assert policy["tuning_coverage"] == 0 and np.isinf(policy["margin"])
    matrices["p"][:] = 0.99
    assert not replay_values(universe, matrices, policy["margin"]).triggered.any()
