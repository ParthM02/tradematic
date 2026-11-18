"""Serializable post-hoc mappings; no estimator/feature training."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = ("legacy_weighted", "pooled_unweighted", "per_minute", "context")


def context_design(raw, frame):
    r = np.asarray(raw)
    m = logit(frame.market_mid.clip(1e-4, 1 - 1e-4).to_numpy())
    k = frame.k.to_numpy() / 10
    gap = r - m
    return np.column_stack(
        [r, m, k, r * k, m * k, gap * np.abs(gap), frame.market_spread.to_numpy(), m * np.abs(m)]
    )


@dataclass
class Recalibrator:
    variant: str
    fits: dict

    def predict(self, model, frame):
        if self.variant == "legacy_weighted":
            return model.predict(frame)
        raw = np.asarray(model.raw(frame))
        if self.variant == "context":
            return self.fits["pooled"].predict_proba(context_design(raw, frame))[:, 1]
        if self.variant == "pooled_unweighted":
            return self.fits["pooled"].predict_proba(raw[:, None])[:, 1]
        out = np.empty(len(frame))
        for k in range(1, 11):
            mask = frame.k.to_numpy() == k
            if mask.any():
                out[mask] = self.fits[k].predict_proba(raw[mask, None])[:, 1]
        return out


def fit_calibrators(model, calibration):
    """No future eligibility counts, estimator fitting, or legacy-state mutation."""
    if calibration.y.nunique() != 2:
        raise ValueError("Calibration requires both outcomes")
    raw = np.asarray(model.raw(calibration))
    pooled = LogisticRegression(C=1.0, max_iter=1000).fit(raw[:, None], calibration.y)
    per_minute = {}
    for k in range(1, 11):
        mask = calibration.k.to_numpy() == k
        f = calibration.loc[mask]
        per_minute[k] = (
            pooled
            if len(f) < 100 or f.y.nunique() != 2
            else (LogisticRegression(C=1.0, max_iter=1000).fit(raw[mask, None], f.y))
        )
    context = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    context.fit(context_design(raw, calibration), calibration.y)
    return {
        "legacy_weighted": Recalibrator("legacy_weighted", {}),
        "pooled_unweighted": Recalibrator("pooled_unweighted", {"pooled": pooled}),
        "per_minute": Recalibrator("per_minute", per_minute),
        "context": Recalibrator("context", {"pooled": context}),
    }
