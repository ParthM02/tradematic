"""Purged expanding folds, separate calibration, and causal window-level stopping."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Config:
    folds: int = 3
    test_days: int = 28
    calibration_days: int = 14
    tuning_days: int = 14
    min_train_days: int = 60
    embargo_minutes: int = 120
    min_coverage: float = 0.25
    coverage_buffer: float = 0.0
    calibration: str = "platt"
    trees: int = 160
    threads: int = 4
    seed: int = 378

    @property
    def tuning_coverage(self):
        return self.min_coverage + self.coverage_buffer

    def __post_init__(self):
        if not 0 < self.tuning_coverage <= 1 or not 0 < self.min_coverage <= 1:
            raise ValueError("Coverage and coverage + buffer must be in (0, 1]")
        if self.coverage_buffer < 0 or self.embargo_minutes < 0:
            raise ValueError("Buffer and embargo must be nonnegative")
        if (
            min(
                self.folds,
                self.test_days,
                self.calibration_days,
                self.tuning_days,
                self.min_train_days,
                self.trees,
                self.threads,
            )
            < 1
        ):
            raise ValueError("Fold lengths, model size and threads must be positive")
        if self.calibration not in ("platt", "isotonic"):
            raise ValueError("Calibration must be platt or isotonic")


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    tuning_start: pd.Timestamp
    tuning_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def mask(self, frame, partition):
        # Purge any label whose availability exceeds its partition boundary.
        start, end = getattr(self, partition + "_start"), getattr(self, partition + "_end")
        return frame.window_id.ge(start) & frame.label_end.le(end)

    def to_dict(self):
        return {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in asdict(self).items()}


def walk_forward(windows, config):
    start, end = windows.window_id.min(), windows.label_end.max()
    gap = pd.Timedelta(minutes=config.embargo_minutes)
    for number in range(config.folds):
        test_end = end - pd.Timedelta(days=(config.folds - 1 - number) * config.test_days)
        test_start = test_end - pd.Timedelta(days=config.test_days)
        tuning_end = test_start - gap
        tuning_start = tuning_end - pd.Timedelta(days=config.tuning_days)
        cal_end = tuning_start - gap
        cal_start = cal_end - pd.Timedelta(days=config.calibration_days)
        train_end = cal_start - gap
        if train_end - start < pd.Timedelta(days=config.min_train_days):
            raise ValueError("Insufficient history: shorten folds or supply more data")
        yield Fold(
            number,
            start,
            train_end,
            cal_start,
            cal_end,
            tuning_start,
            tuning_end,
            test_start,
            test_end,
        )


@dataclass
class CalibratedModel:
    estimator: LGBMClassifier
    calibrator: object
    method: str
    columns: list[str]

    def predict(self, features):
        raw = self.estimator.predict_proba(features[self.columns])[:, 1]
        if self.method == "isotonic":
            return self.calibrator.predict(raw)
        return self.calibrator.predict_proba(logit(np.clip(raw, 1e-6, 1 - 1e-6))[:, None])[:, 1]


def fit_model(train, calibration, columns, config):
    if min(len(train), len(calibration)) < 100:
        raise ValueError("Need at least 100 ready samples in training and calibration")
    if train.y.nunique() != 2 or calibration.y.nunique() != 2:
        raise ValueError("Training and calibration each require both directions")
    estimator = LGBMClassifier(
        n_estimators=config.trees,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        reg_lambda=5.0,
        colsample_bytree=0.9,
        random_state=config.seed,
        n_jobs=config.threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    estimator.fit(train[columns], train.y)
    raw = estimator.predict_proba(calibration[columns])[:, 1]
    if config.calibration == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, calibration.y)
    else:
        calibrator = LogisticRegression(C=1.0, solver="lbfgs").fit(
            logit(np.clip(raw, 1e-6, 1 - 1e-6))[:, None], calibration.y
        )
    return CalibratedModel(estimator, calibrator, config.calibration, columns)


def probability_matrix(frame, universe, probability="p"):
    """N windows x ten decisions. Universe includes windows without ready features."""
    matrix = frame.pivot(index="window_id", columns="k", values=probability)
    return matrix.reindex(index=universe.window_id, columns=range(1, 11)).to_numpy(float)


def first_passage(probabilities, thresholds):
    """Label-free inference. NaNs abstain. There is at most one action per row."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise ValueError("Expected window x 10 probability matrix")
    thresholds = np.asarray(thresholds)
    if thresholds.shape != (10,):
        raise ValueError("Expected ten thresholds")
    confidence = np.maximum(probabilities, 1 - probabilities)
    hits = np.isfinite(probabilities) & (confidence >= thresholds[None, :])
    active = hits.any(axis=1)
    minute = hits.argmax(axis=1)
    selected = probabilities[np.arange(len(probabilities)), minute]
    minute = np.where(active, minute + 1, 0)
    return minute, np.where(active, selected, np.nan)


def _objective(matrix, y, thresholds, min_coverage):
    minute, p = first_passage(matrix, thresholds)
    active = minute > 0
    n = active.sum()
    if n < int(np.ceil(min_coverage * len(y))):
        return (-np.inf, -np.inf, -np.inf)
    accuracy = np.mean((p[active] >= 0.5) == y[active])
    return (float(accuracy), -float(n / len(y)), -float(minute[active].mean()))


def optimize_fixed(probabilities, y, target):
    """Exact validation frontier over distinct confidence values, including ties."""
    conf = np.maximum(probabilities, 1 - probabilities)
    available = np.isfinite(conf)
    idx = np.flatnonzero(available)
    idx = idx[np.argsort(-conf[idx], kind="stable")]
    needed = int(np.ceil(target * len(y)))
    if len(idx) < needed:
        raise ValueError("Insufficient ready windows to meet tuning coverage")
    hits = ((probabilities[idx] >= 0.5) == y[idx]).cumsum()
    # A threshold admits ALL ties. Partial tie selection would not be causal.
    ends = np.r_[np.flatnonzero(np.diff(conf[idx]) != 0), len(idx) - 1]
    ends = ends[ends + 1 >= needed]
    precisions = hits[ends] / (ends + 1)
    best = ends[np.argmax(precisions)]
    return float(conf[idx[best]])


@dataclass
class Policy:
    thresholds: np.ndarray
    tuning_precision: float
    tuning_coverage: float
    tuning_mean_minute: float

    def to_dict(self):
        return {
            "thresholds": [float(t) if np.isfinite(t) else None for t in self.thresholds],
            "tuning_precision": self.tuning_precision,
            "tuning_coverage": self.tuning_coverage,
            "tuning_mean_minute": self.tuning_mean_minute,
        }


def learn_policies(matrix, y, target):
    """Fixed k is exact on tuning; dynamic coordinate search is approximate.

    Search uses first-passage coverage, never row-level coverage. Three initial
    policies and a bounded confidence grid limit search cost and tuning variance.
    No optimization or coverage repair is performed on the test set.
    """
    y = np.asarray(y)
    policies = {}
    for k in range(10):
        thresholds = np.full(10, np.inf)
        thresholds[k] = optimize_fixed(matrix[:, k], y, target)
        obj = _objective(matrix, y, thresholds, target)
        policies[f"fixed_{k + 1}"] = Policy(thresholds, obj[0], -obj[1], -obj[2])
    best = max(policies.values(), key=lambda p: (p.tuning_precision, -p.tuning_mean_minute))
    policies["fixed_selected"] = best
    conf = np.maximum(matrix, 1 - matrix)
    grids = []
    for k in range(10):
        values = conf[:, k][np.isfinite(conf[:, k])]
        grids.append(np.unique(np.r_[0.5, np.quantile(values, np.linspace(0, 1, 25)), np.inf]))
    seeds = [best.thresholds.copy(), policies["fixed_10"].thresholds.copy(), np.full(10, 0.5)]
    winner, winning_obj = best.thresholds.copy(), _objective(matrix, y, best.thresholds, target)
    for thresholds in seeds:
        current = _objective(matrix, y, thresholds, target)
        for _ in range(3):
            changed = False
            for k in range(10):
                chosen, chosen_obj = thresholds[k], current
                for threshold in grids[k]:
                    candidate = thresholds.copy()
                    candidate[k] = threshold
                    obj = _objective(matrix, y, candidate, target)
                    if obj > chosen_obj:
                        chosen, chosen_obj = threshold, obj
                if chosen != thresholds[k]:
                    thresholds[k], current, changed = chosen, chosen_obj, True
            if not changed:
                break
        if current > winning_obj:
            winner, winning_obj = thresholds.copy(), current
    policies["dynamic"] = Policy(winner, winning_obj[0], -winning_obj[1], -winning_obj[2])
    return policies


def replay(universe, matrix, policy, *, fold, symbol, name):
    minute, p = first_passage(matrix, policy.thresholds)
    out = universe[["window_id", "y"]].copy()
    out["fold"], out["symbol"], out["policy"] = fold, symbol, name
    out["trigger_k"], out["p"] = minute, p
    out["triggered"] = minute > 0
    out["correct"] = np.where(minute > 0, (p >= 0.5) == out.y.to_numpy(), np.nan)
    out["available_at"] = out.window_id + pd.to_timedelta(minute, unit="min")
    out.loc[~out.triggered, "available_at"] = pd.NaT
    return out
