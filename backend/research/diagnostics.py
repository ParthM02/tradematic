"""Window-level metrics, matched-coverage benchmarks, block bootstrap, TreeSHAP."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def summarize(decisions, min_coverage=0.25):
    rows = []
    groupings = [("fold", "symbol", "policy"), ("symbol", "policy")]
    for keys in groupings:
        for key, frame in decisions.groupby(list(keys), sort=True):
            row = dict(zip(keys, key, strict=True))
            active = frame[frame.triggered]
            coverage = len(active) / len(frame)
            row.update(
                windows=len(frame),
                triggered=len(active),
                coverage=coverage,
                precision=active.correct.mean(),
                mean_k=active.trigger_k.mean(),
                eligible_at_25=coverage >= min_coverage,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def minute_breakdown(decisions):
    rows = []
    for (symbol, policy), group in decisions.groupby(["symbol", "policy"]):
        for k in range(11):
            # k=0 has no completed minute bar, hence no model and no predictions.
            part = group[group.triggered & group.trigger_k.eq(k)]
            rows.append(
                {
                    "symbol": symbol,
                    "policy": policy,
                    "k": k,
                    "triggered": len(part),
                    "precision": part.correct.mean(),
                    "share_of_windows": len(part) / len(group),
                }
            )
    return pd.DataFrame(rows)


def _ranked_decisions(matrix, universe, k, fraction, name, scores=None):
    """Retrospective top confidence only: NOT a causally deployable test policy."""
    p = matrix[:, k - 1]
    score = np.maximum(p, 1 - p) if scores is None else scores[:, k - 1]
    idx = np.flatnonzero(np.isfinite(p))
    idx = idx[np.argsort(-score[idx], kind="stable")]
    n = int(np.ceil(len(universe) * fraction))
    if len(idx) < n:
        raise ValueError("Not enough ready test rows for matched-coverage diagnostic")
    selected = idx[:n]
    out = universe[["window_id", "y"]].copy()
    out["triggered"] = False
    out["correct"] = np.nan
    out.loc[out.index[selected], "triggered"] = True
    out.loc[out.index[selected], "correct"] = (
        (p[selected] >= 0.5) == out.y.to_numpy()[selected]
    ).astype(float)
    out["policy"] = name
    return out


def matched_coverage(
    model_matrix, baseline_matrix, universe, *, fold, symbol, baseline_scores=None, fraction=0.25
):
    rows = []
    for k in range(1, 11):
        for name, matrix in (("ml", model_matrix), ("distance", baseline_matrix)):
            scores = baseline_scores if name == "distance" else None
            out = _ranked_decisions(matrix, universe, k, fraction, name, scores)
            out["fold"], out["symbol"], out["k"] = fold, symbol, k
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def calibration_metrics(predictions):
    rows = []
    for key, f in predictions.dropna(subset=["p"]).groupby(["fold", "symbol", "k"]):
        p = np.clip(f.p.to_numpy(), 1e-7, 1 - 1e-7)
        rows.append(
            dict(zip(("fold", "symbol", "k"), key, strict=True))
            | {
                "n": len(f),
                "brier": brier_score_loss(f.y, p),
                "log_loss": log_loss(f.y, p, labels=[0, 1]),
                "accuracy_all_ready": ((p >= 0.5) == f.y).mean(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_precision_difference(left, right, seed=378, repetitions=1000, block_days=7):
    """Paired circular moving-block bootstrap of calendar days, retaining abstentions.

    Resample the same seven-day blocks for both policies. Returns a descriptive
    95% interval, with no correction for many horizons/assets or policy searches.
    """

    def daily(f):
        g = f.assign(day=f.window_id.dt.floor("D"))
        return g.groupby("day").agg(hits=("correct", "sum"), n=("triggered", "sum"))

    a, b = daily(left), daily(right)
    days = pd.date_range(
        min(a.index.min(), b.index.min()), max(a.index.max(), b.index.max()), freq="D"
    )
    av, bv = a.reindex(days, fill_value=0).to_numpy(), b.reindex(days, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(days), size=(repetitions, int(np.ceil(len(days) / block_days))))
    sample = ((starts[:, :, None] + np.arange(block_days)) % len(days)).reshape(repetitions, -1)
    sample = sample[:, : len(days)]
    ac, bc = av[sample].sum(axis=1), bv[sample].sum(axis=1)
    valid = (ac[:, 1] > 0) & (bc[:, 1] > 0)
    delta = ac[valid, 0] / ac[valid, 1] - bc[valid, 0] / bc[valid, 1]
    point = left.correct.mean() - right.correct.mean()
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {
        "precision_difference": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "calendar_days": len(days),
        "block_days": block_days,
        "replicates": repetitions,
    }


def shap_triggered(model, snapshots, decisions, *, fold, symbol, policy, max_rows=1000, seed=378):
    """Native TreeSHAP explains the uncalibrated LightGBM log odds, not Platt output."""
    selected = decisions.loc[decisions.triggered, ["window_id", "trigger_k"]].rename(
        columns={"trigger_k": "k"}
    )
    sample = snapshots.merge(selected, on=["window_id", "k"], validate="one_to_one")
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=seed)
    rows = []
    for k, frame in sample.groupby("k"):
        fitted = model[int(k)]
        contributions = fitted.estimator.booster_.predict(
            frame[fitted.columns], pred_contrib=True, num_threads=fitted.estimator.n_jobs
        )
        for j, feature in enumerate(fitted.columns):
            rows.append(
                {
                    "fold": fold,
                    "symbol": symbol,
                    "policy": policy,
                    "k": k,
                    "feature": feature,
                    "samples": len(frame),
                    "sum_abs_shap": np.abs(contributions[:, j]).sum(),
                    "sum_shap": contributions[:, j].sum(),
                }
            )
    return rows


def _table(frame, columns, percent=()):
    f = frame[columns].copy()
    for c in percent:
        f[c] = f[c].map(lambda x: f"{100 * x:.2f}%" if pd.notna(x) else "—")
    for c in f.select_dtypes("float").columns:
        f[c] = f[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    # No tabulate dependency for a small Markdown table.
    return (
        "| "
        + " | ".join(f.columns)
        + " |\n| "
        + " | ".join(["---"] * len(f.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |" for row in f.itertuples(index=False, name=None)
        )
    )


def write_report(output: Path, decisions, matched, predictions, shap_rows, config, audit):
    metrics = summarize(decisions, config.min_coverage)
    metrics.to_csv(output / "policy_metrics.csv", index=False)
    minute_breakdown(decisions).to_csv(output / "trigger_minute_metrics.csv", index=False)
    calibration_metrics(predictions).to_csv(output / "calibration_metrics.csv", index=False)
    frontier = (
        matched.groupby(["symbol", "k", "policy"])
        .agg(windows=("y", "size"), triggered=("triggered", "sum"), precision=("correct", "mean"))
        .reset_index()
    )
    frontier["coverage"] = frontier.triggered / frontier.windows
    frontier.to_csv(output / "matched_25_frontier.csv", index=False)
    comparisons = []
    for (symbol, k), f in matched.groupby(["symbol", "k"]):
        result = bootstrap_precision_difference(
            f[f.policy.eq("ml")], f[f.policy.eq("distance")], seed=config.seed
        )
        comparisons.append(dict(symbol=symbol, k=k, comparison="retrospective_top25", **result))
    for symbol, f in decisions.groupby("symbol"):
        for name in ("fixed_selected", "dynamic"):
            result = bootstrap_precision_difference(
                f[f.policy.eq("ml_" + name)], f[f.policy.eq("distance_" + name)], seed=config.seed
            )
            comparisons.append(dict(symbol=symbol, k=0, comparison="frozen_" + name, **result))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "precision_differences.csv", index=False)
    shap = pd.DataFrame(shap_rows)
    if not shap.empty:
        shap.to_csv(output / "shap_by_fold_minute.csv", index=False)
        agg = shap.groupby(["symbol", "policy", "feature"])[
            ["sum_abs_shap", "sum_shap", "samples"]
        ].sum()
        agg["mean_abs_shap"] = agg.sum_abs_shap / agg.samples
        agg["mean_shap"] = agg.sum_shap / agg.samples
        agg.reset_index().sort_values("mean_abs_shap", ascending=False).to_csv(
            output / "shap_triggered.csv", index=False
        )
    selected = metrics[
        metrics.fold.isna() & metrics.policy.str.endswith(("fixed_selected", "dynamic"))
    ]
    failures = metrics[metrics.fold.notna() & ~metrics.eligible_at_25]
    main_failures = failures[failures.policy.str.endswith(("fixed_selected", "dynamic"))]
    content = [
        "# Out-of-sample directional research",
        "",
        (
            "k counts completed bars; actions occur at t0+k. Dojis and incomplete target candles are "
            "excluded and counted in data_audit.json. Coverage denominators include all remaining "
            "test windows, including those without ready features. Each asset is a separate experiment."
        ),
        "",
        (
            f"Chronological expanding folds: {config.folds}; test days/fold: {config.test_days}; "
            f"calibration days: {config.calibration_days}; policy-tuning days: {config.tuning_days}; "
            f"boundary embargo: {config.embargo_minutes} minutes. Tuning coverage target: "
            f"{config.tuning_coverage:.1%}; test eligibility floor: {config.min_coverage:.1%}."
        ),
        "",
        "## Frozen policies",
        "",
        _table(
            selected,
            ["symbol", "policy", "windows", "coverage", "precision", "mean_k", "eligible_at_25"],
            ("coverage", "precision"),
        ),
        "",
        (
            f"Selected/dynamic policies failing the coverage floor in individual folds: {len(main_failures)}. "
            "A pooled pass does not rescue a failed fold. See policy_metrics.csv for all ten fixed horizons."
        ),
        "",
        "## Identical retrospective 25% coverage",
        "",
        (
            "Each fold/asset/horizon selects ceil(0.25*N) test windows by confidence, with stable timestamp "
            "tie-breaking. This is an ex-post ranking diagnostic, not a tradable stopping rule. The baseline "
            "predicts the current sign and ranks |z_remaining|. It uses no test outcomes to rank, but its "
            "cutoff uses the full test score distribution. No test ranking changes frozen policies."
        ),
        "",
        _table(
            frontier, ["symbol", "k", "policy", "coverage", "precision"], ("coverage", "precision")
        ),
        "",
        "## Evidence beyond anchor distance",
        "",
        (
            "precision_differences.csv reports seven-calendar-day moving-block bootstrap intervals for "
            "ML minus distance precision. Frozen comparisons can have different realized coverages; "
            "retrospective_top25 comparisons have identical counts at each horizon. Intervals are "
            "descriptive and unadjusted for multiple comparisons. Fixed horizons are chosen on tuning "
            "only. Dynamic search is a local coordinate search, not a guaranteed global optimum. "
            "High absolute precision alone is not evidence of alpha; require coverage eligibility, "
            "a positive out-of-sample improvement, stability across folds, and an untouched replication "
            "before making that claim."
        ),
        "",
        "## Attributions and limitations",
        "",
        (
            "shap_triggered.csv contains native TreeSHAP mean absolute and signed contributions on "
            "up to 1,000 triggered windows per selected policy/asset/fold, sampled without reference to "
            "outcomes. Contributions explain the underlying model's raw log odds; the separate probability "
            "calibrator is not attributed. They describe associations, not causal mechanisms. "
            "trigger_minute_metrics.csv explicitly includes k=0 with zero actions (only the anchor is known). "
            "No trading-profit, fee, fill, exchange latency, funding schedule, or order-book claims follow "
            "from directional accuracy. This is minute-bar research, not an HFT execution simulator."
        ),
        "",
        f"Common data span: {audit['start']} to {audit['stop']}.",
    ]
    (output / "report.md").write_text("\n".join(content) + "\n")
    return selected
