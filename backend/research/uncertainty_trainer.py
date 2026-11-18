"""Market-conditioned forecast models and label-free, fee-aware stopping."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODEL_SPECS = {
    "market_platt": ("linear", "market_platt", False),
    "market_tree": ("tree", "market", False),
    "path_tree": ("tree", "path", False),
    "classic_tree": ("tree", "classic", False),
    "technical_tree": ("tree", "technical", False),
    "technical_linear": ("linear", "technical", False),
    "technical_extra_trees": ("extra", "technical", False),
    "technical_residual": ("residual", "technical", False),
    "technical_all_states": ("tree", "technical", True),
}
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, np.inf)


def uncertain(frame, lower=0.2, upper=0.8, max_spread=0.10):
    return (
        frame.observable
        & frame.market_mid.between(lower, upper)
        & frame.market_spread.le(max_spread + 1e-8)
    )


def window_weights(frame):
    weights = 1 / frame.groupby("window_id").window_id.transform("size").to_numpy(float)
    return weights / weights.mean()


@dataclass
class ForecastModel:
    estimator: object
    calibrator: object
    columns: list[str]
    kind: str
    imputer: object = None
    scaler: object = None

    def raw(self, frame):
        x = frame[self.columns]
        if self.imputer is not None:
            x = self.imputer.transform(x)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.kind == "residual":
            return self.estimator.booster_.predict(
                x, raw_score=True, num_threads=self.estimator.n_jobs
            ) + logit(frame.market_mid.clip(1e-4, 1 - 1e-4))
        if self.kind == "linear":
            return self.estimator.decision_function(x)
        return logit(np.clip(self.estimator.predict_proba(x)[:, 1], 1e-5, 1 - 1e-5))

    def predict(self, frame):
        raw = np.asarray(self.raw(frame))
        return self.calibrator.predict_proba(raw[:, None])[:, 1]


def fit_forecast(train, calibration, columns, kind, *, trees=160, threads=4, seed=378):
    if train.window_id.nunique() < 100 or calibration.window_id.nunique() < 100:
        raise ValueError("Not enough independent windows for model/calibration")
    if train.y.nunique() != 2 or calibration.y.nunique() != 2:
        raise ValueError("Both outcomes are required in each fit partition")
    model = ForecastModel(None, None, columns, kind)
    x, weights = train[columns], window_weights(train)
    if kind in ("linear", "extra"):
        model.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        x = model.imputer.fit_transform(x)
    if kind == "linear":
        model.scaler = StandardScaler()
        x = model.scaler.fit_transform(x)
        estimator = LogisticRegression(C=0.05, max_iter=2000)
    elif kind == "extra":
        estimator = ExtraTreesClassifier(
            n_estimators=trees,
            max_depth=8,
            min_samples_leaf=100,
            max_features=0.7,
            n_jobs=threads,
            random_state=seed,
        )
    else:
        estimator = LGBMClassifier(
            n_estimators=trees,
            learning_rate=0.03,
            num_leaves=8,
            max_depth=4,
            min_child_samples=300,
            reg_lambda=10,
            n_jobs=threads,
            random_state=seed,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    with threadpool_limits(limits=threads):
        if kind == "residual":
            estimator.fit(
                x,
                train.y,
                sample_weight=weights,
                init_score=logit(train.market_mid.clip(1e-4, 1 - 1e-4)),
            )
        else:
            estimator.fit(x, train.y, sample_weight=weights)
        model.estimator = estimator
        raw = np.asarray(model.raw(calibration))
        model.calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(
            raw[:, None], calibration.y, sample_weight=window_weights(calibration)
        )
    return model


def forecast_metrics(frame, probability):
    valid = frame[probability].notna()
    f = frame.loc[valid]
    if f.empty:
        return {
            "snapshots": 0,
            "windows": 0,
            "brier": np.nan,
            "log_loss": np.nan,
            "accuracy": np.nan,
        }
    p, y = np.clip(f[probability].to_numpy(), 1e-7, 1 - 1e-7), f.y.to_numpy()
    weights = window_weights(f)
    return {
        "snapshots": len(f),
        "windows": f.window_id.nunique(),
        "brier": np.average((p - y) ** 2, weights=weights),
        "log_loss": np.average(-(y * np.log(p) + (1 - y) * np.log1p(-p)), weights=weights),
        "accuracy": np.average((p >= 0.5) == y, weights=weights),
    }


def action_matrix(frame, universe, probability):
    def matrix(column):
        return (
            frame.pivot(index="window_id", columns="k", values=column)
            .reindex(index=universe.window_id, columns=range(1, 11))
            .to_numpy()
        )

    return {
        "p": matrix(probability).astype(float),
        "yes_ask": matrix("yes_ask").astype(float),
        "no_ask": matrix("no_ask").astype(float),
        "eligible": matrix("eligible").astype(float) == 1,
    }


def first_value_passage(matrices, margin, *, fixed_k=0, fee=0.02, slippage=0.0):
    """No labels accepted. Choose side by value even if its probability is <50%."""
    p = matrices["p"]
    yes, no = (
        np.minimum(matrices["yes_ask"] + slippage, 1),
        np.minimum(matrices["no_ask"] + slippage, 1),
    )
    ey, en = p - yes - fee, 1 - p - no - fee
    side = (ey >= en).astype(int)
    edge = np.maximum(ey, en)
    original_ask = np.where(side == 1, matrices["yes_ask"], matrices["no_ask"])
    hit = (
        matrices["eligible"]
        & np.isfinite(p)
        & (original_ask > 0)
        & (original_ask < 1)
        & (edge > margin)
    )
    if fixed_k:
        hit[:, np.arange(10) != fixed_k - 1] = False
    active, j = hit.any(axis=1), hit.argmax(axis=1)
    idx = np.arange(len(p))
    return {
        "trigger_k": np.where(active, j + 1, 0),
        "side": np.where(active, side[idx, j], -1),
        "entry_price": np.where(active, np.where(side == 1, yes, no)[idx, j], np.nan),
        "p_yes": np.where(active, p[idx, j], np.nan),
        "expected_edge": np.where(active, edge[idx, j], np.nan),
    }


def replay_values(universe, matrices, margin, *, fixed_k=0, fee=0.02, slippage=0.0):
    actions = first_value_passage(matrices, margin, fixed_k=fixed_k, fee=fee, slippage=slippage)
    d = universe[["window_id", "y"]].reset_index(drop=True).copy()
    for name, values in actions.items():
        d[name] = values
    active = d.trigger_k.gt(0)
    d["triggered"] = active
    d["win"] = np.where(active, d.side.eq(d.y), np.nan)
    d["entry_fee"] = np.where(active, fee, 0)
    d["gross_pnl"] = np.where(active, d.win - d.entry_price, 0)
    d["net_pnl"] = d.gross_pnl - d.entry_fee
    return d


def learn_value_policy(universe, matrices, *, constrained=False, dynamic=True, fee=0.02):
    candidates = []
    for k in [0] if dynamic else range(1, 11):
        for margin in MARGINS:
            d = replay_values(universe, matrices, margin, fixed_k=k, fee=fee)
            candidates.append(
                {
                    # A cutoff with no validation actions has no evidence for
                    # future executions; canonicalize it to explicit abstention.
                    "margin": float(margin) if d.triggered.any() else np.inf,
                    "fixed_k": k,
                    "tuning_coverage": float(d.triggered.mean()),
                    "tuning_net_per_window": float(d.net_pnl.mean()),
                }
            )
    feasible = [c for c in candidates if c["tuning_coverage"] >= 0.25]
    choices = feasible if constrained and feasible else candidates
    chosen = max(
        choices, key=lambda c: (c["tuning_net_per_window"], -c["tuning_coverage"], -c["fixed_k"])
    )
    if constrained and not feasible:
        # Show maximum available positive-EV coverage rather than manufacture trades.
        chosen = max(candidates, key=lambda c: (c["tuning_coverage"], c["tuning_net_per_window"]))
    return chosen | {"validation_feasible_25": bool(feasible)}


def block_brier_difference(frame, left, right, seed=378):
    """Left-minus-right Brier, equally weighted windows then paired 7-day blocks."""
    f = frame.dropna(subset=[left, right]).copy()
    f["difference"] = (f[left] - f.y) ** 2 - (f[right] - f.y) ** 2
    w = f.groupby("window_id").difference.mean().reset_index()
    daily = w.assign(day=w.window_id.dt.floor("D")).groupby("day").difference.agg(["sum", "count"])
    daily = daily.reindex(
        pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0
    )
    values, n = daily.to_numpy(), len(daily)
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(2000, int(np.ceil(n / 7))))
    indices = ((starts[:, :, None] + np.arange(7)) % n).reshape(2000, -1)[:, :n]
    totals = values[indices].sum(axis=1)
    delta = totals[:, 0] / totals[:, 1]
    low, high = np.quantile(delta, [0.025, 0.975])
    return {
        "brier_difference": w.difference.mean(),
        "ci95_low": low,
        "ci95_high": high,
        "windows": len(w),
        "calendar_days": n,
    }
