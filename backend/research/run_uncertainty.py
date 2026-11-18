"""Compare SOL uncertainty specialists against calibrated Kalshi-only models."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .data_loader import ASSETS, load_market
from .features import build_features
from .kalshi_overlay import load_kalshi, pnl_interval
from .selective_trainer import Config, walk_forward
from .uncertainty_features import build_uncertainty_snapshots
from .uncertainty_trainer import (
    MODEL_SPECS,
    action_matrix,
    block_brier_difference,
    fit_forecast,
    forecast_metrics,
    learn_value_policy,
    replay_values,
    uncertain,
)


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decision_metrics(d):
    active = d[d.triggered]
    low, high = pnl_interval(d.assign(scored_trade=d.triggered))
    return {
        "windows": len(d),
        "trades": len(active),
        "coverage": d.triggered.mean(),
        "precision": active.win.mean(),
        "mean_k": active.trigger_k.mean(),
        "mean_entry": active.entry_price.mean(),
        "gross_pnl": d.gross_pnl.sum(),
        "entry_fees": d.entry_fee.sum(),
        "net_pnl": d.net_pnl.sum(),
        "net_per_trade": active.net_pnl.mean(),
        "net_per_window": d.net_pnl.mean(),
        "net_per_trade_ci_low": low,
        "net_per_trade_ci_high": high,
        "test_coverage_25": bool(d.triggered.mean() >= 0.25),
    }


def forecast_tables(predictions, names):
    rows, calibration = [], []
    partitions = [("all", predictions)] + list(predictions.groupby("fold"))
    for fold, frame in partitions:
        for lower, upper in ((0.2, 0.8), (0.3, 0.7), (0.4, 0.6)):
            f = frame[frame.eligible & frame.market_mid.between(lower, upper)]
            for name in names:
                rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "market_low": lower,
                        "market_high": upper,
                        "k": 0,
                        **forecast_metrics(f, name),
                    }
                )
                if lower == 0.2:
                    for k in range(1, 11):
                        rows.append(
                            {
                                "fold": fold,
                                "model": name,
                                "market_low": lower,
                                "market_high": upper,
                                "k": k,
                                **forecast_metrics(f[f.k.eq(k)], name),
                            }
                        )
        primary = frame[frame.eligible]
        for name in names:
            binned = primary[["window_id", "y", name]].dropna().copy()
            binned["bin"] = np.minimum((binned[name] * 10).astype(int), 9)
            for bucket, part in binned.groupby("bin"):
                # Each bin is descriptive; repeated states are not independent trials.
                calibration.append(
                    {
                        "fold": fold,
                        "model": name,
                        "bin": bucket,
                        "snapshots": len(part),
                        "windows": part.window_id.nunique(),
                        "mean_probability": part[name].mean(),
                        "outcome_rate": part.y.mean(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data/coinbase"))
    parser.add_argument("--kalshi-dir", type=Path, default=Path("backend/data/kalshi"))
    parser.add_argument(
        "--output", type=Path, default=Path("backend/research/artifacts/uncertainty_sol_v2")
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    config = Config(
        folds=args.folds, test_days=args.test_days, trees=args.trees, threads=args.threads
    )
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; choose a new experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    paths = {s: args.data_dir / f"{s.lower()}usd_m1_bars_coinbase.csv" for s in ASSETS}
    qp, mp = args.kalshi_dir / "candles_SOLUSD.csv", args.kalshi_dir / "markets_SOLUSD.csv"
    manifest = {
        "config": asdict(config),
        "inputs_sha256": {str(p): checksum(p) for p in [*paths.values(), qp, mp]},
        "source_sha256": {p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")},
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "lightgbm", "scikit-learn")
        },
        "models": MODEL_SPECS,
        "fee_per_contract": 0.02,
        "resolution_fee": 0,
        "market_mid_band": [0.2, 0.8],
        "max_spread": 0.10,
        "seed": 378,
        "selection": "Calibration then tuning only; no test model/threshold selection",
        "status": "running",
        "interpretation": "Exploratory reuse of previously examined calendar periods",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Loading bars and Kalshi contracts; building causal features...", flush=True)
    market = load_market(paths, invalid_rows="quarantine")
    quotes, markets = load_kalshi(qp, mp)
    end = min(b.available_at.max() for b in market.bars.values()).floor("15min")
    universe = markets[
        markets.open_time.ge(next(iter(market.bars.values())).index.min())
        & markets.close_time.le(end)
        & markets.kalshi_y.notna()
    ].copy()
    universe = universe.rename(
        columns={"open_time": "window_id", "close_time": "label_end", "kalshi_y": "y"}
    )
    universe = (
        universe[["ticker", "window_id", "label_end", "y"]]
        .sort_values("window_id")
        .reset_index(drop=True)
    )
    # Unlike the original Coinbase-target experiment, Coinbase dojis and later
    # missing Coinbase bars do NOT remove valid Kalshi settlement targets.
    info = markets[markets.ticker.isin(universe.ticker)]
    path = build_features(market.bars)
    snapshots, groups = build_uncertainty_snapshots(market.bars["SOL"], path, quotes, info)
    snapshots = snapshots.merge(
        universe[["window_id", "label_end", "y"]], on="window_id", validate="many_to_one"
    )
    snapshots["eligible"] = uncertain(snapshots)
    snapshots.to_parquet(output / "snapshots.parquet", index=False)
    universe.to_parquet(output / "universe.parquet", index=False)
    (output / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    (output / "data_audit.json").write_text(json.dumps(market.audit, indent=2) + "\n")
    del path, market
    gc.collect()
    folds = list(walk_forward(universe, config))
    (output / "folds.json").write_text(json.dumps([f.to_dict() for f in folds], indent=2) + "\n")
    print(
        f"{len(universe):,} Kalshi windows; {len(snapshots):,} snapshots; "
        f"{len(groups['technical'])} full features; preparation {perf_counter() - start:.1f}s",
        flush=True,
    )
    predictions, decisions, selection, frontier = [], [], [], []
    fields = ["window_id", "k", "y", "eligible", "market_mid", "market_spread", "yes_ask", "no_ask"]
    model_names = list(MODEL_SPECS)
    report_names = [
        "market_raw",
        *model_names,
        "selected_market",
        "selected_augmented",
        "selected_champion",
    ]
    for fold in folds:
        partitions = {
            p: snapshots[fold.mask(snapshots, p)].copy()
            for p in ("train", "calibration", "tuning", "test")
        }
        tune, test = partitions["tuning"][fields].copy(), partitions["test"][fields].copy()
        tune["market_raw"], test["market_raw"] = tune.market_mid, test.market_mid
        models = {}
        for name, (kind, feature_group, all_states) in MODEL_SPECS.items():
            train = partitions["train"]
            train = train[
                (train.observable & train.market_spread.le(0.10 + 1e-8))
                if all_states
                else train.eligible
            ]
            cal = partitions["calibration"].loc[partitions["calibration"].eligible]
            print(
                f"Fold {fold.number + 1}/{config.folds}: {name}, "
                f"{train.window_id.nunique():,} training windows / {len(train):,} states",
                flush=True,
            )
            model = fit_forecast(
                train,
                cal,
                groups[feature_group],
                kind,
                trees=config.trees,
                threads=config.threads,
                seed=config.seed,
            )
            models[name] = model
            tune[name] = np.nan
            visible = partitions["tuning"].observable
            tune.loc[visible, name] = model.predict(partitions["tuning"].loc[visible])
        tuning_scores = {
            name: forecast_metrics(tune[tune.eligible], name)["brier"] for name in model_names
        }
        baseline = min(("market_platt", "market_tree"), key=tuning_scores.get)
        augmented = min(
            (n for n in model_names if not n.startswith("market_")), key=tuning_scores.get
        )
        champion = min(model_names, key=tuning_scores.get)
        choices = {
            "selected_market": baseline,
            "selected_augmented": augmented,
            "selected_champion": champion,
        }
        for alias, name in choices.items():
            tune[alias] = tune[name]
        tune_u = universe[fold.mask(universe, "tuning")].reset_index(drop=True)
        test_u = universe[fold.mask(universe, "test")].reset_index(drop=True)
        policies = {}
        for name in report_names:
            tm = action_matrix(tune, tune_u, name)
            for constrained in (False, True):
                for dynamic in (False, True):
                    policy_name = ("coverage25" if constrained else "profit") + (
                        "_dynamic" if dynamic else "_fixed_selected"
                    )
                    policies[(name, policy_name)] = learn_value_policy(
                        tune_u, tm, constrained=constrained, dynamic=dynamic
                    )
            # Fixed-k frontier is learned on tuning, not sorted by test outcomes.
            for k in range(1, 11):
                best = None
                for margin in (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, np.inf):
                    d = replay_values(tune_u, tm, margin, fixed_k=k)
                    item = {
                        "margin": margin if d.triggered.any() else np.inf,
                        "coverage": d.triggered.mean(),
                        "net": d.net_pnl.mean(),
                    }
                    if best is None or item["net"] > best["net"]:
                        best = item
                frontier.append(
                    {
                        "fold": fold.number,
                        "model": name,
                        "k": k,
                        "margin": best["margin"],
                        "tuning_coverage": best["coverage"],
                        "tuning_net_per_window": best["net"],
                    }
                )
        # All model and policy choices are now frozen; only then score test rows.
        for name, model in models.items():
            test[name] = np.nan
            visible = partitions["test"].observable
            test.loc[visible, name] = model.predict(partitions["test"].loc[visible])
        for alias, name in choices.items():
            test[alias] = test[name]
        test["fold"] = fold.number
        predictions.append(test)
        selection.append(
            {
                "fold": fold.number,
                "choices": choices,
                "tuning_brier": tuning_scores,
                "policies": {f"{n}/{p}": v for (n, p), v in policies.items()},
            }
        )
        for name in report_names:
            xm = action_matrix(test, test_u, name)
            for (n, policy_name), policy in policies.items():
                if n != name:
                    continue
                d = replay_values(test_u, xm, policy["margin"], fixed_k=policy["fixed_k"])
                d["fold"], d["model"], d["policy"] = fold.number, name, policy_name
                d["validation_feasible_25"] = policy["validation_feasible_25"]
                decisions.append(d)
            for row in frontier:
                if row["fold"] == fold.number and row["model"] == name:
                    d = replay_values(test_u, xm, row["margin"], fixed_k=row["k"])
                    row.update(
                        test_coverage=d.triggered.mean(),
                        test_precision=d.win.mean(),
                        test_net_pnl=d.net_pnl.sum(),
                        test_trades=int(d.triggered.sum()),
                    )
        model_dir = output / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(models, model_dir / f"fold_{fold.number}.joblib", compress=3)
        print(f"  Tuning choices: {choices}", flush=True)
        del models, partitions
        gc.collect()
    predictions, decisions = (
        pd.concat(predictions, ignore_index=True),
        pd.concat(decisions, ignore_index=True),
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    decisions.to_parquet(output / "decisions.parquet", index=False)
    # JSON has no Infinity; null explicitly means disabled/abstain.
    selection_text = json.dumps(selection, indent=2).replace("Infinity", "null")
    (output / "selection.json").write_text(selection_text + "\n")
    pd.DataFrame(frontier).to_csv(output / "fixed_frontier.csv", index=False)
    scores, reliability = forecast_tables(predictions, report_names)
    scores.to_csv(output / "forecast_metrics.csv", index=False)
    reliability.to_csv(output / "reliability.csv", index=False)
    summaries, stress = [], []
    for keys in (("model", "policy"), ("fold", "model", "policy")):
        for key, frame in decisions.groupby(list(keys)):
            summaries.append(dict(zip(keys, key, strict=True)) | decision_metrics(frame))
            if len(keys) == 2:
                for slip in (0.0, 0.01, 0.02):
                    # Same frozen actions; do not reselect after adverse execution.
                    cost = np.where(
                        frame.triggered,
                        np.minimum(frame.entry_price + slip, 1) - frame.entry_price,
                        0,
                    )
                    stress.append(
                        {
                            "model": key[0],
                            "policy": key[1],
                            "adverse_slippage": slip,
                            "trades": int(frame.triggered.sum()),
                            "net_pnl": frame.net_pnl.sum() - cost.sum(),
                        }
                    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "policy_metrics.csv", index=False)
    pd.DataFrame(stress).to_csv(output / "execution_stress.csv", index=False)
    eligible = predictions[predictions.eligible]
    comparisons = []
    for left, right in [
        ("selected_augmented", "selected_market"),
        ("selected_champion", "selected_market"),
        ("technical_tree", "path_tree"),
        ("technical_tree", "classic_tree"),
        ("technical_tree", "technical_all_states"),
    ]:
        comparisons.append(
            {"left": left, "right": right, **block_brier_difference(eligible, left, right)}
        )
    pd.DataFrame(comparisons).to_csv(output / "brier_differences.csv", index=False)
    primary = scores[(scores.fold.eq("all")) & scores.k.eq(0) & scores.market_low.eq(0.2)]
    selected = summary[summary.fold.isna() & summary.model.str.startswith("selected_")]
    report = [
        "# SOL forecasts in uncertain Kalshi markets",
        "",
        (
            "This is an exploratory follow-up using calendar periods already examined in the earlier research. "
            "Models, calibration and thresholds are causal within each fold, but these dates are not a fresh "
            "untouched confirmation set. No CF Benchmarks tick history was available or synthesized."
        ),
        "",
        "## Probability quality: YES midpoint 20–80 cents, spread at most 10 cents",
        "",
        "```text",
        primary[["model", "windows", "snapshots", "brier", "log_loss", "accuracy"]].to_string(
            index=False
        ),
        "```",
        "",
        (
            "Lower Brier/log loss is better. Each market window has equal total weight across its eligible "
            "minute states; minute states do not count as independent trials. market_raw is the quote midpoint. "
            "market_platt and market_tree use only Kalshi quotes/history and clock position. Architecture "
            "and baseline selection use tuning Brier only, separately in each fold."
        ),
        "",
        "## Frozen fee-aware actions for the selected models",
        "",
        "```text",
        selected[
            [
                "model",
                "policy",
                "trades",
                "coverage",
                "precision",
                "mean_entry",
                "net_pnl",
                "test_coverage_25",
            ]
        ].to_string(index=False),
        "```",
        "",
        (
            "One contract at the selected YES/NO ask, a flat $0.02 fee only when buying, no resolution fee. "
            "Buy the side with larger predicted payout minus ask minus fee, even if its probability is "
            "below 50%. Trigger at the first eligible k=1..10 whose expected edge exceeds the frozen margin. "
            "Margins are a predeclared small grid. profit policies may abstain entirely. coverage25 policies "
            "require at least 25% tuning coverage; if no positive-EV margin can do that, record the maximum "
            "available coverage and mark validation infeasible. No negative-EV trades are forced to fill a quota. "
            "Test coverage uses ALL listed test markets, including unobservable windows. See selection.json "
            "and per-fold metrics; a pooled coverage pass cannot rescue a failed fold."
        ),
        "",
        "## Feature and architecture experiments",
        "",
        (
            "All specialist fits and calibrations use the uncertain band. path_tree adds causal anchor/path, "
            "order-flow proxies, cross-asset features and distance to the actual known Kalshi strike. "
            "classic_tree adds SMA/EMA gaps and slopes, MACD and Bollinger geometry. technical_tree also "
            "adds a forward-only local-linear Kalman filter, with fixed process/observation variance ratios. "
            "technical_linear and technical_extra_trees test regularized logistic and randomized forests. "
            "technical_residual fits LightGBM corrections to raw market log odds using init_score, and "
            "explicitly adds the market offset back at inference. technical_all_states is the controlled "
            "all-state training comparison, with the same feature set and uncertain-state calibration. "
            "Imputation, scaling and calibration fit their designated historical partitions only. Repeated "
            "states get inverse within-window weights for train/calibration fits."
        ),
        "",
        (
            "brier_differences.csv reports left-minus-right Brier with paired seven-day block-bootstrap "
            "descriptive intervals (2,000 resamples); negative favors the left model. They are not adjusted "
            "for multiple experiments. forecast_metrics.csv includes 30–70 and 40–60 secondary bands and "
            "every minute. fixed_frontier.csv contains validation-chosen margins for each fixed horizon. "
            "reliability.csv contains descriptive probability bins. execution_stress.csv applies one/two "
            "cents adverse execution to the SAME frozen trades, without changing selection."
        ),
        "",
        "## Data and practical limits",
        "",
        (
            "Actual Kalshi results supply labels, with availability at contract close. Contract strike is "
            "assumed available by the first minute. Neither expiration_value nor result enters features. "
            "Coinbase dojis and future Coinbase data gaps do not remove Kalshi targets. Missing features "
            "can prevent observation/trading; missing quotes are never filled across minutes or contracts. "
            "Bar and quote end timestamps are assumed observable at that timestamp. The files do not "
            "establish historical publication latency, resting depth, fills, or revision-free strike history. "
            "These are quote-level settlement markouts. Successful specialization, extra indicators, "
            "or a positive point estimate alone does not prove executable market alpha."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    manifest["status"], manifest["elapsed_seconds"] = "complete", perf_counter() - start
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(primary[["model", "brier", "log_loss", "accuracy"]].to_string(index=False), flush=True)
    print(
        f"Complete in {(perf_counter() - start) / 60:.1f} minutes: {output / 'report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
