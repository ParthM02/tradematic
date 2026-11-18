"""Post-hoc calibration ablation on frozen estimators, with blind fresh replay."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration_models import VARIANTS, fit_calibrators
from .data_loader import ASSETS, load_market
from .features import build_features
from .kalshi_overlay import load_kalshi
from .run_uncertainty import decision_metrics
from .selective_trainer import Fold
from .uncertainty_features import build_uncertainty_snapshots
from .uncertainty_trainer import action_matrix, learn_value_policy, replay_values, uncertain

FRESH_START = pd.Timestamp("2026-08-15", tz="UTC")
FRESH_END = pd.Timestamp("2026-09-12", tz="UTC")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def score(frame, probability):
    f = frame.loc[frame.eligible & frame[probability].notna()]
    p = np.clip(f[probability].to_numpy(), 1e-7, 1 - 1e-7)
    y = f.y.to_numpy()
    brier = (p - y) ** 2
    return {
        "snapshots": len(f),
        "windows": f.window_id.nunique(),
        "brier": brier.mean(),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean(),
        "window_weighted_brier": pd.Series(brier, index=f.index).groupby(f.window_id).mean().mean(),
    }


def fit_policies(frame, universe, probability):
    matrices = action_matrix(frame, universe, probability)
    choices = {}
    for constrained in (False, True):
        for dynamic in (False, True):
            name = ("coverage25" if constrained else "profit") + (
                "_dynamic" if dynamic else "_fixed"
            )
            choices[name] = learn_value_policy(
                universe, matrices, constrained=constrained, dynamic=dynamic
            )
    return choices


def replay(frame, universe, probability, policies, fold, family, variant, period):
    matrices = action_matrix(frame, universe, probability)
    records = []
    for name, policy in policies.items():
        d = replay_values(universe, matrices, policy["margin"], fixed_k=policy["fixed_k"])
        d["fold"], d["family"], d["variant"], d["policy"], d["period"] = (
            fold,
            family,
            variant,
            name,
            period,
        )
        d["validation_feasible_25"] = policy["validation_feasible_25"]
        d["market_mid"] = np.nan
        current = frame[["window_id", "k", "market_mid"]].rename(
            columns={"k": "trigger_k", "market_mid": "mid_at_entry"}
        )
        d = d.merge(current, on=["window_id", "trigger_k"], how="left", validate="one_to_one")
        d["market_mid"] = d.pop("mid_at_entry")
        records.append(d)
    return records


def edge_summary(frame):
    active = frame[frame.triggered]
    predicted = active.expected_edge.mean()
    realized = active.net_pnl.mean()
    return decision_metrics(frame) | {
        "predicted_net_per_trade": predicted,
        "edge_overstatement": predicted - realized,
    }


def reliability_tables(decisions):
    rows = []
    active = decisions[decisions.triggered].copy()
    active["predicted_win"] = np.where(active.side.eq(1), active.p_yes, 1 - active.p_yes)
    active["market_side_prob"] = np.where(
        active.side.eq(1), active.market_mid, 1 - active.market_mid
    )
    active["gap_band"] = pd.cut(
        active.predicted_win - active.market_side_prob, [-np.inf, 0, 0.02, 0.05, 0.10, np.inf]
    ).astype(str)
    active["market_band"] = pd.cut(active.market_mid, [0, 0.2, 0.4, 0.6, 0.8, 1]).astype(str)
    base = ["period", "family", "variant", "policy"]
    for dimension in ("trigger_k", "market_band", "gap_band"):
        for key, f in active.groupby(base + [dimension]):
            rows.append(
                dict(zip(base + ["bin"], key, strict=True))
                | {
                    "dimension": dimension,
                    "trades": len(f),
                    "predicted_win": f.predicted_win.mean(),
                    "observed_win": f.win.mean(),
                    "predicted_net": f.expected_edge.mean(),
                    "realized_net": f.net_pnl.mean(),
                }
            )
    return pd.DataFrame(rows)


def save_results(output, predictions, decisions, summaries, frozen_audit):
    predictions.to_parquet(output / "predictions.parquet", index=False)
    decisions.to_parquet(output / "decisions.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output / "forecast_metrics.csv", index=False)
    rows = []
    keys = ["period", "family", "variant", "policy"]
    for grouping in (keys, keys + ["fold"]):
        for key, f in decisions.groupby(grouping):
            rows.append(dict(zip(grouping, key, strict=True)) | edge_summary(f))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "policy_metrics.csv", index=False)
    reliability_tables(decisions).to_csv(output / "triggered_reliability.csv", index=False)
    pd.DataFrame(frozen_audit).to_csv(output / "fixed_actions_calibration.csv", index=False)
    stress = []
    for key, f in decisions.groupby(keys):
        for slip in (0, 0.01, 0.02):
            cost = np.where(f.triggered, np.minimum(f.entry_price + slip, 1) - f.entry_price, 0)
            stress.append(
                dict(zip(keys, key, strict=True))
                | {
                    "slippage": slip,
                    "trades": int(f.triggered.sum()),
                    "net_pnl": f.net_pnl.sum() - cost.sum(),
                }
            )
    pd.DataFrame(stress).to_csv(output / "execution_stress.csv", index=False)
    pooled = metrics[metrics.fold.isna() & metrics.family.eq("champion")]
    cols = [
        "period",
        "variant",
        "policy",
        "trades",
        "coverage",
        "predicted_net_per_trade",
        "net_per_trade",
        "net_pnl",
    ]
    report = [
        "# Frozen-model SOL calibration study",
        "",
        (
            "Classifier estimators, feature sets, fitted imputers/scalers and original architecture choices "
            "are frozen from uncertainty_sol_v2. Only post-hoc calibration and the unchanged validation-only "
            "stopping-policy search vary. Model files are hash-checked before/after preparation."
        ),
        "",
        "## Selected-trade accounting",
        "",
        "```text",
        pooled[cols].to_string(index=False),
        "```",
        "",
        (
            "The purchase fee is $0.02 per contract, resolution fee zero. All outputs assume one contract "
            "at the quoted ask. A predicted-vs-realized edge gap equals average win-probability overstatement "
            "on selected trades. Forecast scores alone do not establish reliable trade probabilities."
        ),
        "",
        "## Controlled comparisons",
        "",
        (
            "legacy_weighted uses the original saved pooled Platt mapping. pooled_unweighted changes only "
            "its calibration weights to one per observed state. per_minute fits separate unweighted Platt "
            "mappings at each completed minute, with pooled fallback below 100 calibration samples or for "
            "one-class bins. context is a regularized unweighted logistic calibration of raw model log odds, "
            "market log odds, k, spread and fixed interactions for time/market disagreement. These are "
            "already-observed model/market variables, not new price indicators. All fits use the original "
            "14-day calibration partition; architecture/feature training is never rerun."
        ),
        "",
        (
            "Each family chooses its selected calibration variant on unweighted tuning Brier score before "
            "test evaluation. Trading margins/horizons use the same separate tuning period and original "
            "finite search grid. Both unconstrained-profit and 25%-coverage policies are reported. Coverage "
            "denominators include all listed contracts, not only uncertain states. Check each fold: pooling "
            "cannot make a failed coverage fold eligible. Zero-action validation choices explicitly abstain."
        ),
        "",
        (
            "fixed_actions_calibration.csv evaluates every mapping on identical original windows, sides "
            "and entry times, without changing the actions or entry prices. This separates probability "
            "correction from changes in trade selection. triggered_reliability.csv describes realized "
            "selected trades by minute, market-price band and predicted market disagreement. Its bins "
            "are diagnostic, not additional test-tuned trading rules."
        ),
        "",
        "## Time separation",
        "",
        (
            "Old three-fold test dates were already examined and remain exploratory. The protocol "
            "predeclares a new test interval [2026-08-15, 2026-09-12) UTC. The final frozen bundle is "
            "selected/calibrated/tuned using only the original final fold's pre-test dates, and its "
            "choices are saved before scoring the fresh interval. No new-model training, recalibration, "
            "threshold selection, or winner selection occurs on that new interval. Confirmatory results "
            "require sufficient data coverage, correct causal quote timing and conservative execution."
        ),
        "",
        (
            "The original inverse within-window weights depend on the completed historical window's "
            "number of eligible states. They change the statistical sampling target and can distort "
            "prospective state probabilities. The new calibrators remove that weighting; the frozen "
            "classifiers still retain their original training weights. Per-minute calibration additionally "
            "avoids pooling different prediction times. The experiment cannot guarantee conditional "
            "calibration for every selected subgroup."
        ),
        "",
        "## Limits",
        "",
        (
            "Quote timestamps are treated as observable at bar end. Minute snapshots do not prove "
            "publication latency, resting depth or actual fills. Execution stress charges extra costs "
            "to the same frozen trades. No benchmark tick data is inferred from Coinbase candles."
        ),
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return pooled


def prepare(source, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(source / "snapshots.parquet")
    universe = pd.read_parquet(source / "universe.parquet")
    original = pd.read_parquet(source / "decisions.parquet")
    source_selection = json.loads((source / "selection.json").read_text())
    fold_dicts = json.loads((source / "folds.json").read_text())
    files = sorted((source / "models").glob("*.joblib"))
    model_hashes = {str(p): sha(p) for p in files}
    protocol = {
        "fresh_start": str(FRESH_START),
        "fresh_end": str(FRESH_END),
        "source": str(source.resolve()),
        "frozen_model_sha256": model_hashes,
        "source_manifest_sha256": sha(source / "manifest.json"),
        "calibration_variants": VARIANTS,
        "fee": 0.02,
        "resolution_fee": 0,
        "selection": "Per original fold/family, minimize unweighted tuning Brier; no test selection",
        "source_code_sha256": sha(__file__),
        "status": "preparing",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    predictions, decisions, summaries, audits, frozen = [], [], [], [], []
    for raw_fold, selection in zip(fold_dicts, source_selection, strict=True):
        fold = Fold(**{k: pd.Timestamp(v) if k != "number" else v for k, v in raw_fold.items()})
        models = joblib.load(source / "models" / f"fold_{fold.number}.joblib")
        cal = data[fold.mask(data, "calibration") & data.eligible]
        tune = data[fold.mask(data, "tuning")].copy()
        test = data[fold.mask(data, "test")].copy()
        families = {
            "champion": selection["choices"]["selected_champion"],
            "market": selection["choices"]["selected_market"],
        }
        bundle = {"fold": raw_fold, "families": {}}
        for family, model_name in families.items():
            print(
                f"Fold {fold.number + 1}: frozen {family}={model_name}; fitting calibration only",
                flush=True,
            )
            model = models[model_name]
            mappings = fit_calibrators(model, cal)
            scores, policies = {}, {}
            for variant, mapping in mappings.items():
                for frame in (tune, test):
                    frame[variant] = np.nan
                    mask = frame.observable
                    frame.loc[mask, variant] = mapping.predict(model, frame.loc[mask])
                scores[variant] = score(tune, variant)["brier"]
                policies[variant] = fit_policies(
                    tune, universe[fold.mask(universe, "tuning")], variant
                )
            selected = min(VARIANTS, key=scores.get)
            tune["selected"], test["selected"] = tune[selected], test[selected]
            policies["selected"] = policies[selected]
            for variant in (*VARIANTS, "selected"):
                summaries.append(
                    {
                        "period": "exploratory",
                        "fold": fold.number,
                        "family": family,
                        "variant": variant,
                        **score(test, variant),
                    }
                )
                decisions.extend(
                    replay(
                        test,
                        universe[fold.mask(universe, "test")],
                        variant,
                        policies[variant],
                        fold.number,
                        family,
                        variant,
                        "exploratory",
                    )
                )
            # Replicate the original action cohort, then evaluate all mappings on
            # those same rows and sides; do not refit on its test outcomes.
            for old_name in ("coverage25_fixed_selected", "coverage25_dynamic"):
                old = original[
                    original.fold.eq(fold.number)
                    & original.model.eq("selected_" + family)
                    & original.policy.eq(old_name)
                    & original.triggered
                ]
                paired = old.merge(
                    test[["window_id", "k", *VARIANTS]],
                    left_on=["window_id", "trigger_k"],
                    right_on=["window_id", "k"],
                    validate="one_to_one",
                )
                for variant in VARIANTS:
                    pwin = np.where(paired.side.eq(1), paired[variant], 1 - paired[variant])
                    audits.append(
                        {
                            "fold": fold.number,
                            "family": family,
                            "original_policy": old_name,
                            "variant": variant,
                            "trades": len(paired),
                            "predicted_win": pwin.mean(),
                            "observed_win": paired.win.mean(),
                            "predicted_net": (pwin - paired.entry_price - 0.02).mean(),
                            "realized_net": paired.net_pnl.mean(),
                            "brier_on_fixed_actions": ((pwin - paired.win) ** 2).mean(),
                        }
                    )
            test_out = test[
                [
                    "window_id",
                    "k",
                    "y",
                    "eligible",
                    "market_mid",
                    "market_spread",
                    *VARIANTS,
                    "selected",
                ]
            ].copy()
            test_out["family"], test_out["fold"], test_out["period"] = (
                family,
                fold.number,
                "exploratory",
            )
            predictions.append(test_out)
            bundle["families"][family] = {
                "model": model,
                "model_name": model_name,
                "mappings": mappings,
                "selected": selected,
                "tuning_brier": scores,
                "policies": policies,
            }
            print(f"  Calibration selected on tuning: {selected}", flush=True)
        frozen.append(bundle)
    if model_hashes != {str(p): sha(p) for p in files}:
        raise RuntimeError("Frozen source model files changed")
    joblib.dump(frozen[-1], output / "frozen_final.joblib", compress=3)
    joblib.dump(frozen, output / "frozen_folds.joblib", compress=3)
    selected_choices = [
        {
            "fold": b["fold"]["number"],
            "families": {
                f: {
                    "model": v["model_name"],
                    "selected": v["selected"],
                    "tuning_brier": v["tuning_brier"],
                    "policies": v["policies"],
                }
                for f, v in b["families"].items()
            },
        }
        for b in frozen
    ]
    (output / "selection.json").write_text(
        json.dumps(selected_choices, indent=2).replace("Infinity", "null") + "\n"
    )
    result = save_results(
        output,
        pd.concat(predictions, ignore_index=True),
        pd.concat(decisions, ignore_index=True),
        summaries,
        audits,
    )
    protocol["status"] = "frozen_before_fresh_evaluation"
    protocol["frozen_final_sha256"] = sha(output / "frozen_final.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)


def evaluate_fresh(study, fresh_dir):
    protocol = json.loads((study / "protocol.json").read_text())
    if sha(study / "frozen_final.joblib") != protocol["frozen_final_sha256"]:
        raise ValueError("Frozen final bundle hash mismatch")
    output = study / "fresh"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh interval already scored; never overwrite its evaluation")
    output.mkdir(exist_ok=True)
    source = Path(protocol["source"])
    bundle = joblib.load(study / "frozen_final.joblib")
    market = load_market(
        {s: fresh_dir / "coinbase" / f"{s.lower()}usd_m1_bars_coinbase.csv" for s in ASSETS},
        invalid_rows="quarantine",
    )
    quotes, markets = load_kalshi(
        fresh_dir / "kalshi" / "candles_SOLUSD.csv", fresh_dir / "kalshi" / "markets_SOLUSD.csv"
    )
    targets = markets[
        markets.open_time.ge(FRESH_START)
        & markets.close_time.le(FRESH_END)
        & markets.kalshi_y.notna()
    ]
    if targets.empty:
        raise ValueError("No settled markets in the predeclared fresh interval")
    # Full old price history is retained for exactly the same EMA/Kalman state.
    path = build_features(market.bars)
    frame, _ = build_uncertainty_snapshots(market.bars["SOL"], path, quotes, targets)
    universe = targets.rename(
        columns={"open_time": "window_id", "close_time": "label_end", "kalshi_y": "y"}
    )
    frame = frame.merge(
        universe[["window_id", "label_end", "y"]], on="window_id", validate="many_to_one"
    )
    frame["eligible"] = uncertain(frame)
    predictions, decisions, scores = [], [], []
    for family, frozen in bundle["families"].items():
        current = frame.copy()
        for variant, mapping in frozen["mappings"].items():
            current[variant] = np.nan
            mask = current.observable
            current.loc[mask, variant] = mapping.predict(frozen["model"], current.loc[mask])
        current["selected"] = current[frozen["selected"]]
        for variant in (*VARIANTS, "selected"):
            scores.append(
                {
                    "period": "fresh",
                    "fold": 3,
                    "family": family,
                    "variant": variant,
                    **score(current, variant),
                }
            )
            decisions.extend(
                replay(
                    current,
                    universe,
                    variant,
                    frozen["policies"][variant],
                    3,
                    family,
                    variant,
                    "fresh",
                )
            )
        current["family"], current["fold"], current["period"] = family, 3, "fresh"
        predictions.append(
            current[
                [
                    "window_id",
                    "k",
                    "y",
                    "eligible",
                    "market_mid",
                    "family",
                    "fold",
                    "period",
                    *VARIANTS,
                    "selected",
                ]
            ]
        )
    result = save_results(
        output,
        pd.concat(predictions, ignore_index=True),
        pd.concat(decisions, ignore_index=True),
        scores,
        [],
    )
    report = {
        "fresh_start": str(FRESH_START),
        "fresh_end": str(FRESH_END),
        "listed_settled_markets": len(targets),
        "expected_quarter_hours": 28 * 96,
        "observable_windows": int(frame.loc[frame.observable, "window_id"].nunique()),
        "frozen_final_sha256": protocol["frozen_final_sha256"],
        "inputs_sha256": {str(p): sha(p) for p in fresh_dir.rglob("*.csv")},
        "selection_sha256": sha(study / "selection.json"),
        "source_study": str(source),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("backend/research/artifacts/uncertainty_sol_v2")
    )
    p.add_argument(
        "--output", type=Path, default=Path("backend/research/artifacts/calibration_sol_v1")
    )
    p.add_argument("--fresh-dir", type=Path)
    args = p.parse_args()
    if args.fresh_dir:
        evaluate_fresh(args.output, args.fresh_dir)
    else:
        prepare(args.source, args.output)


if __name__ == "__main__":
    main()
