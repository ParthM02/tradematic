"""External invariants: future perturbations, split bounds, and window counting."""

import numpy as np
import pandas as pd
import pytest

from backend.research.data_loader import ASSETS, load_market
from backend.research.features import build_features
from backend.research.selective_trainer import (
    Config,
    first_passage,
    learn_policies,
    optimize_fixed,
    replay,
    walk_forward,
)


def make_csvs(tmp_path, minutes=600):
    rng = np.random.default_rng(378)
    index = pd.date_range("2025-01-01", periods=minutes, freq="min", tz="UTC")
    paths = {}
    for i, s in enumerate(ASSETS):
        close = (100 + i * 30) * np.exp(np.cumsum(rng.normal(0, 0.001, minutes)))
        opened = np.r_[close[0] * 0.9999, close[:-1]]
        frame = pd.DataFrame(
            {
                "symbol": s + "USD",
                "dt": index.astype(str),
                "date": index.date,
                "hour": index.floor("h").astype(str),
                "minute": index.minute,
                "open": opened,
                "high": np.maximum(close, opened) * 1.0003,
                "low": np.minimum(close, opened) * 0.9997,
                "close": close,
                "volume": rng.uniform(1, 100, minutes),
            }
        )
        path = tmp_path / f"{s}.csv"
        frame.to_csv(path, index=False)
        paths[s] = path
    return paths


def test_alignment_and_missing_minutes_are_not_filled(tmp_path):
    paths = make_csvs(tmp_path)
    raw = pd.read_csv(paths["ETH"])
    raw.drop(index=[170]).to_csv(paths["ETH"], index=False)
    market = load_market(paths)
    t0 = pd.Timestamp("2025-01-01 02:45", tz="UTC")
    assert pd.isna(market.bars["ETH"].loc[t0 + pd.Timedelta(minutes=5), "open"])
    target = market.windows.query("symbol == 'ETH'").set_index("window_id").loc[t0]
    assert not target.complete
    btc = market.windows.query("symbol == 'BTC'").set_index("window_id").loc[t0]
    bars = market.bars["BTC"]
    assert btc.anchor_open == bars.loc[t0, "open"]
    assert btc.final_close == bars.loc[t0 + pd.Timedelta(minutes=14), "close"]
    assert btc.label_end == t0 + pd.Timedelta(minutes=15)
    assert bars.loc[t0, "elapsed"] == 0
    assert bars.loc[t0 + pd.Timedelta(minutes=9), "k"] == 10


def test_future_mutation_cannot_change_any_available_feature(tmp_path):
    paths = make_csvs(tmp_path)
    market = load_market(paths)
    before = build_features(market.bars)
    # At 04:10 we can use bars through 04:09, including all contemporaneous peers.
    cutoff = pd.Timestamp("2025-01-01 04:10", tz="UTC")
    for path in paths.values():
        raw = pd.read_csv(path)
        future = pd.to_datetime(raw.dt, utc=True).ge(cutoff)
        raw.loc[future, ["open", "high", "low", "close"]] *= 3
        raw.loc[future, "volume"] *= 11
        raw.to_csv(path, index=False)
    after = build_features(load_market(paths).bars)
    pd.testing.assert_frame_equal(
        before[before.available_at.le(cutoff)].reset_index(drop=True),
        after[after.available_at.le(cutoff)].reset_index(drop=True),
    )
    assert not before[before.available_at.gt(cutoff)].equals(after[after.available_at.gt(cutoff)])


def test_prefix_replay_matches_full_data(tmp_path):
    paths = make_csvs(tmp_path)
    market = load_market(paths)
    full = build_features(market.bars)
    cutoff = pd.Timestamp("2025-01-01 05:09", tz="UTC")
    prefix = build_features({s: f.loc[:cutoff] for s, f in market.bars.items()})
    # Tests feature availability at a partially completed window, without a final label.
    for s in ASSETS:
        expected = full[full.symbol.eq(s) & full.dt.le(cutoff)].reset_index(drop=True)
        observed = prefix[prefix.symbol.eq(s)].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, observed)


def test_doji_excluded_and_duplicates_rejected(tmp_path):
    paths = make_csvs(tmp_path)
    raw = pd.read_csv(paths["BTC"])
    raw.loc[14, "close"] = raw.loc[0, "open"]
    raw.loc[14, "high"] = max(raw.loc[14, "high"], raw.loc[14, "close"])
    raw.loc[14, "low"] = min(raw.loc[14, "low"], raw.loc[14, "close"])
    raw.to_csv(paths["BTC"], index=False)
    target = load_market(paths).windows.query("symbol == 'BTC'").iloc[0]
    assert target.doji and target.complete and not target.eligible
    pd.concat([raw, raw.iloc[[0]]]).to_csv(paths["BTC"], index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_market(paths)


def test_explicit_quarantine_removes_entire_malformed_bar(tmp_path):
    paths = make_csvs(tmp_path)
    raw = pd.read_csv(paths["BTC"])
    raw.loc[100, "close"] = np.nan
    raw.loc[130, "symbol"] = np.nan
    raw.to_csv(paths["BTC"], index=False)
    with pytest.raises(ValueError, match="invalid OHLCV"):
        load_market(paths)
    market = load_market(paths, invalid_rows="quarantine")
    bar = market.bars["BTC"].iloc[100]
    assert bar[["open", "high", "low", "close", "volume"]].isna().all()
    assert market.audit["assets"]["BTC"]["quarantined_rows"] == 2
    assert not market.windows.query("symbol == 'BTC'").iloc[6].complete


def test_first_passage_locks_and_future_scores_do_not_change_trigger():
    p = np.full((4, 10), 0.6)
    p[0, 2], p[0, 8] = 0.95, 0.01
    p[1, 5] = 0.03
    p[2, :] = np.nan
    minute, probability = first_passage(p, np.full(10, 0.9))
    np.testing.assert_array_equal(minute, [3, 6, 0, 0])
    np.testing.assert_allclose(probability[:2], [0.95, 0.03])
    p[0, 3:] = np.nan
    assert first_passage(p, np.full(10, 0.9))[0][0] == 3


def test_coverage_uses_windows_and_validation_thresholds_respect_ties():
    y = np.r_[np.ones(30), np.zeros(70)]
    p = np.full((100, 10), 0.55)
    p[:30] = 0.9  # 300 high-confidence snapshots represent just 30 windows.
    policies = learn_policies(p, y, 0.25)
    for policy in policies.values():
        minute, _ = first_passage(p, policy.thresholds)
        assert (minute > 0).sum() >= 25
    assert policies["fixed_1"].tuning_coverage == 0.30
    assert optimize_fixed(p[:, 0], y, 0.25) == 0.9
    universe = pd.DataFrame(
        {"window_id": pd.date_range("2025-01-01", periods=100, freq="15min", tz="UTC"), "y": y}
    )
    result = replay(universe, p, policies["dynamic"], fold=0, symbol="BTC", name="dynamic")
    assert len(result) == 100 and result.window_id.is_unique
    assert result.triggered.sum() == 30
    # Frozen cutoffs must abstain on a shifted test distribution, not refit to fix coverage.
    shifted = np.full((100, 10), 0.7)
    assert first_passage(shifted, policies["dynamic"].thresholds)[0].sum() == 0


def test_purged_partitions_do_not_overlap_and_test_folds_are_contiguous():
    ids = pd.date_range("2025-01-01", periods=200 * 96, freq="15min", tz="UTC")
    windows = pd.DataFrame({"window_id": ids, "label_end": ids + pd.Timedelta(minutes=15)})
    config = Config(folds=3, test_days=14, min_train_days=60)
    folds = list(walk_forward(windows, config))
    for fold in folds:
        masks = [fold.mask(windows, p) for p in ("train", "calibration", "tuning", "test")]
        assert np.stack(masks).sum(axis=0).max() <= 1
        gap = pd.Timedelta(minutes=config.embargo_minutes)
        assert fold.calibration_start - fold.train_end == gap
        assert fold.tuning_start - fold.calibration_end == gap
        assert fold.test_start - fold.tuning_end == gap
        for p in ("train", "calibration", "tuning", "test"):
            f = windows[fold.mask(windows, p)]
            assert f.label_end.le(getattr(fold, p + "_end")).all()
    assert folds[0].test_end == folds[1].test_start
