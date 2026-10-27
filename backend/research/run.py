"""Run from the checkout root: uv run python -m backend.research.run --help."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .data_loader import ASSETS, load_market
from .diagnostics import matched_coverage, shap_triggered, write_report
from .features import bridge_probability, build_features, feature_columns
from .selective_trainer import (
    Config,
    fit_model,
    learn_policies,
    probability_matrix,
    replay,
    walk_forward,
)


def log(message):
    print(message, flush=True)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data/coinbase"))
    parser.add_argument("--output", type=Path, default=Path("backend/research/artifacts/latest"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--calibration-days", type=int, default=14)
    parser.add_argument("--tuning-days", type=int, default=14)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--embargo-minutes", type=int, default=120)
    parser.add_argument(
        "--coverage-buffer",
        type=float,
        default=0.0,
        help="Optional validation coverage reserve above 0.25; never repairs test coverage",
    )
    parser.add_argument("--calibration", choices=("platt", "isotonic"), default="platt")
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=378)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument(
        "--invalid-rows",
        choices=("raise", "quarantine"),
        default="raise",
        help="Quarantine discards malformed OHLCV bars as audited gaps",
    )
    args = parser.parse_args(argv)
    config = Config(**{k: getattr(args, k) for k in asdict(Config()) if hasattr(args, k)})
    output = args.output
    # Preserve previous experiments. Reproducible artifacts must never be silently mixed.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is nonempty: {output}; choose a new experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    paths = {s: args.data_dir / f"{s.lower()}usd_m1_bars_coinbase.csv" for s in ASSETS}
    manifest = {
        "config": asdict(config),
        "invalid_rows": args.invalid_rows,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "lightgbm", "scikit-learn", "scipy")
        },
        "inputs": {s: {"path": str(p.resolve()), "sha256": checksum(p)} for s, p in paths.items()},
    }
    try:
        manifest["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        manifest["git_head"] = None
    manifest["source_sha256"] = {p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = perf_counter()
    log("Loading and auditing the three assets...")
    market = load_market(paths, invalid_rows=args.invalid_rows)
    (output / "data_audit.json").write_text(json.dumps(market.audit, indent=2) + "\n")
    market.windows.to_parquet(output / "window_audit.parquet", index=False)
    log(f"Common span {market.audit['start']} through {market.audit['stop']}")
    log("Building causal feature snapshots...")
    snapshots = build_features(market.bars)
    columns = feature_columns(snapshots)
    (output / "feature_columns.json").write_text(json.dumps(columns, indent=2) + "\n")
    snapshots.to_parquet(output / "features.parquet", index=False)
    windows = market.windows[market.windows.eligible].copy()
    del market.bars
    gc.collect()
    log(
        f"Built {len(snapshots):,} snapshots / {len(columns)} features in {perf_counter() - start:.1f}s"
    )
    if args.features_only:
        return
    folds = list(walk_forward(windows, config))
    (output / "folds.json").write_text(json.dumps([f.to_dict() for f in folds], indent=2) + "\n")
    decisions, matched, predictions, shap_rows, policy_json = [], [], [], [], []
    for fold in folds:
        for s in ASSETS:
            universe = windows[windows.symbol.eq(s)].sort_values("window_id").reset_index(drop=True)
            data = snapshots[snapshots.symbol.eq(s)].merge(
                universe[["window_id", "label_end", "y"]], on="window_id", validate="many_to_one"
            )
            tune_u = universe[fold.mask(universe, "tuning")].reset_index(drop=True)
            test_u = universe[fold.mask(universe, "test")].reset_index(drop=True)
            if min(len(tune_u), len(test_u)) < 100:
                raise ValueError(f"{s} fold {fold.number}: insufficient tuning/test windows")
            partition = {
                p: data[fold.mask(data, p)] for p in ("train", "calibration", "tuning", "test")
            }
            models, tune_parts, test_parts = {}, [], []
            for k in range(1, 11):
                train = partition["train"].loc[
                    partition["train"].k.eq(k) & partition["train"].ready
                ]
                cal = partition["calibration"].loc[
                    partition["calibration"].k.eq(k) & partition["calibration"].ready
                ]
                model = fit_model(train, cal, columns, config)
                models[k] = model
                for p, parts in (("tuning", tune_parts), ("test", test_parts)):
                    f = partition[p].loc[partition[p].k.eq(k)].copy()
                    f["p"] = np.nan
                    if f.ready.any():
                        f.loc[f.ready, "p"] = model.predict(f.loc[f.ready])
                    f["baseline_p"] = bridge_probability(f.z_remaining).where(f.ready)
                    f["baseline_score"] = f.z_remaining.abs().where(f.ready)
                    parts.append(f[["window_id", "k", "y", "p", "baseline_p", "baseline_score"]])
                if k in (1, 5, 10):
                    log(
                        f"Fold {fold.number + 1}/{config.folds} {s} k={k}: "
                        f"fit {len(train):,}, calibrate {len(cal):,}"
                    )
            tune, test = pd.concat(tune_parts), pd.concat(test_parts)
            test["fold"], test["symbol"] = fold.number, s
            predictions.append(test)
            model_dir = output / "models" / f"fold_{fold.number}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(models, model_dir / f"{s}.joblib", compress=3)
            matrices = {}
            for name, probability in (("ml", "p"), ("distance", "baseline_p")):
                tm = probability_matrix(tune, tune_u, probability)
                xm = probability_matrix(test, test_u, probability)
                matrices[name] = xm
                policies = learn_policies(tm, tune_u.y.to_numpy(), config.tuning_coverage)
                policy_json.append(
                    {
                        "fold": fold.number,
                        "symbol": s,
                        "method": name,
                        "policies": {n: p.to_dict() for n, p in policies.items()},
                    }
                )
                for n, policy in policies.items():
                    d = replay(test_u, xm, policy, fold=fold.number, symbol=s, name=f"{name}_{n}")
                    decisions.append(d)
                    if name == "ml" and n in ("fixed_selected", "dynamic"):
                        shap_rows.extend(
                            shap_triggered(
                                models,
                                partition["test"],
                                d,
                                fold=fold.number,
                                symbol=s,
                                policy=n,
                                seed=config.seed,
                            )
                        )
                best_k = int(
                    np.flatnonzero(np.isfinite(policies["fixed_selected"].thresholds))[0] + 1
                )
                log(
                    f"  {name}: tuning selected k*={best_k}; "
                    f"dynamic tuning precision={policies['dynamic'].tuning_precision:.3%}, "
                    f"coverage={policies['dynamic'].tuning_coverage:.3%}"
                )
            matched.append(
                matched_coverage(
                    matrices["ml"],
                    matrices["distance"],
                    test_u,
                    fold=fold.number,
                    symbol=s,
                    baseline_scores=probability_matrix(test, test_u, "baseline_score"),
                )
            )
            # Durable per-asset/fold progress if a later model fails.
            (output / "policies.json").write_text(json.dumps(policy_json, indent=2) + "\n")
            del data, partition, models, tune, test, tune_parts, test_parts, train, cal, f
            gc.collect()
    decisions = pd.concat(decisions, ignore_index=True)
    matched = pd.concat(matched, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    decisions.to_parquet(output / "decisions.parquet", index=False)
    matched.to_parquet(output / "matched_25_decisions.parquet", index=False)
    predictions.to_parquet(output / "oos_probabilities.parquet", index=False)
    selected = write_report(
        output, decisions, matched, predictions, shap_rows, config, market.audit
    )
    log(selected.to_string(index=False))
    log(f"Finished in {(perf_counter() - start) / 60:.1f} minutes. Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
