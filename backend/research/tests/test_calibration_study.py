import copy
import gzip
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from backend.research import calibration_study
from backend.research.calibration_models import fit_calibrators
from backend.research.public_data import cached_json, flatten_quotes, price_field, quote_batches


class FrozenModel:
    def __init__(self):
        self.coefficient = 1.0

    def raw(self, frame):
        return frame.raw_score.to_numpy() * self.coefficient

    def predict(self, frame):
        return expit(self.raw(frame) * 0.8)

    def fit(self, *args, **kwargs):
        raise AssertionError("Classifier must remain frozen")


def sample_calibration():
    rng = np.random.default_rng(42)
    n = 1500
    raw = rng.normal(size=n)
    return pd.DataFrame(
        {
            "raw_score": raw,
            "market_mid": expit(raw * 0.7),
            "market_spread": 0.02,
            "k": np.tile(np.arange(1, 11), n // 10),
            "y": rng.binomial(1, expit(raw * 0.6)),
            "window_id": np.repeat(np.arange(n // 10), 10),
        }
    )


def test_calibration_does_not_mutate_classifier_and_unweighted_matches_reference():
    model, frame = FrozenModel(), sample_calibration()
    saved = copy.deepcopy(model.__dict__)
    mappings = fit_calibrators(model, frame)
    assert model.__dict__ == saved
    reference = LogisticRegression(C=1.0, max_iter=1000).fit(
        frame[["raw_score"]].to_numpy(), frame.y
    )
    np.testing.assert_allclose(
        mappings["pooled_unweighted"].predict(model, frame),
        reference.predict_proba(frame[["raw_score"]].to_numpy())[:, 1],
    )
    np.testing.assert_allclose(
        mappings["legacy_weighted"].predict(model, frame), model.predict(frame)
    )


def test_predictions_ignore_future_labels_counts_and_input_order():
    model, frame = FrozenModel(), sample_calibration()
    mappings = fit_calibrators(model, frame)
    changed = frame.copy()
    changed["y"] = 1 - changed.y
    changed["window_id"] = 0
    for mapping in mappings.values():
        expected = mapping.predict(model, frame)
        np.testing.assert_allclose(expected, mapping.predict(model, changed))
        shuffled = frame.sample(frac=1, random_state=1)
        observed = pd.Series(mapping.predict(model, shuffled), index=shuffled.index).sort_index()
        np.testing.assert_allclose(expected, observed)


def test_sparse_horizons_use_declared_pooled_fallback():
    frame = sample_calibration().iloc[:500]
    mappings = fit_calibrators(FrozenModel(), frame)
    assert mappings["per_minute"].fits[1] is mappings["pooled_unweighted"].fits["pooled"]


def test_public_quote_parser_rejects_missing_quotes_without_filling():
    opened = pd.Timestamp("2026-08-15", tz="UTC")
    ts = int(opened.timestamp())
    candles = [
        {
            "end_period_ts": ts + 60,
            "yes_bid": {"close_dollars": "0.40"},
            "yes_ask": {"close_dollars": "0.43"},
            "volume_fp": "2",
        },
        {
            "end_period_ts": ts + 120,
            "yes_bid": {"close_dollars": None},
            "yes_ask": {"close_dollars": "0.43"},
        },
        {
            "end_period_ts": ts + 180,
            "yes_bid": {"close": 42},
            "yes_ask": {"close": 44},
            "volume": 3,
        },
    ]
    rows, bad = flatten_quotes("test", candles, opened)
    assert bad == 1 and [r["elapsed_minute"] for r in rows] == [1, 3]
    assert rows[0]["no_ask"] == 0.6 and rows[1]["yes_ask"] == 0.44


def test_public_cache_cannot_replay_a_different_date_query(tmp_path):
    path = tmp_path / "request.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(
            {
                "request": {"url": "https://api.exchange.coinbase.com/test?start=1"},
                "payload": [1, 2],
            },
            f,
        )
    assert cached_json(
        tmp_path, path.name, "https://api.exchange.coinbase.com", "/test", {"start": 1}
    ) == [1, 2]
    with pytest.raises(ValueError, match="differs"):
        cached_json(tmp_path, path.name, "https://api.exchange.coinbase.com", "/test", {"start": 2})


def test_archived_price_strings_and_batch_rectangular_budget():
    assert price_field({"close": "0.5600"}) == 0.56
    assert price_field({"close": 56}) == 0.56
    assert price_field({"close_dollars": "0.5600"}) == 0.56
    start = pd.Timestamp("2026-08-15", tz="UTC")
    markets = pd.DataFrame(
        {
            "ticker": [str(i) for i in range(100)],
            "open_time": pd.date_range(start, periods=100, freq="15min"),
        }
    )
    markets["close_time"] = markets.open_time + pd.Timedelta(minutes=15)
    batches = list(quote_batches(markets))
    assert sum(len(b) for b in batches) == 100 and len(batches) > 1
    for b in batches:
        span = (b.close_time.max() - b.open_time.min()).total_seconds() / 60 + 2
        assert len(b) * span <= 9000


def test_blind_fresh_replay_never_fits_or_reselects(tmp_path, monkeypatch):
    from types import SimpleNamespace

    study = tmp_path / "study"
    fresh = tmp_path / "fresh_data"
    study.mkdir()
    fresh.mkdir()
    model = FrozenModel()
    mappings = fit_calibrators(model, sample_calibration())
    policy = {"profit_dynamic": {"margin": 0.0, "fixed_k": 0, "validation_feasible_25": True}}
    bundle = {
        "families": {
            "champion": {
                "model": model,
                "mappings": mappings,
                "selected": "context",
                "policies": {v: policy for v in [*mappings, "selected"]},
            }
        }
    }
    joblib.dump(bundle, study / "frozen_final.joblib")
    (study / "selection.json").write_text("[]")
    (study / "protocol.json").write_text(
        json.dumps(
            {
                "source": str(tmp_path),
                "frozen_final_sha256": calibration_study.sha(study / "frozen_final.joblib"),
            }
        )
    )
    ids = pd.date_range(calibration_study.FRESH_START, periods=12, freq="15min")
    targets = pd.DataFrame(
        {
            "ticker": [f"t{i}" for i in range(12)],
            "open_time": ids,
            "close_time": ids + pd.Timedelta(minutes=15),
            "kalshi_y": np.arange(12) % 2,
        }
    )
    frame = pd.DataFrame(
        {
            "window_id": np.repeat(ids, 10),
            "k": np.tile(np.arange(1, 11), 12),
            "raw_score": np.tile(np.linspace(-1, 1, 10), 12),
            "observable": True,
            "market_mid": 0.5,
            "market_spread": 0.02,
            "yes_ask": 0.51,
            "no_ask": 0.51,
        }
    )
    monkeypatch.setattr(
        calibration_study, "load_market", lambda *a, **k: SimpleNamespace(bars={"SOL": None})
    )
    monkeypatch.setattr(calibration_study, "load_kalshi", lambda *a: (None, targets))
    monkeypatch.setattr(calibration_study, "build_features", lambda *a: None)
    monkeypatch.setattr(
        calibration_study, "build_uncertainty_snapshots", lambda *a: (frame.copy(), {})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Fresh data must never fit calibration or select a policy")

    monkeypatch.setattr(calibration_study, "fit_calibrators", forbidden)
    monkeypatch.setattr(calibration_study, "fit_policies", forbidden)
    calibration_study.evaluate_fresh(study, fresh)
    assert (study / "fresh" / "evaluation_manifest.json").exists()
    assert (
        calibration_study.sha(study / "frozen_final.joblib")
        == json.loads((study / "protocol.json").read_text())["frozen_final_sha256"]
    )
