"""Replay frozen SOL signals against minute Kalshi asks and actual settlements.

This is a quote-level markout, with assumed one-contract taker fills. It does
not infer liquidity from candle trade volume or use retrospective top-25% picks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = ("ml_fixed_selected", "ml_dynamic", "distance_fixed_selected", "distance_dynamic")
SOURCES = {
    "settlement": "https://help.kalshi.com/en/articles/13823838-crypto-markets",
    "quotes": "https://docs.kalshi.com/api-reference/market/get-market-candlesticks",
    "complement": "https://docs.kalshi.com/getting_started/orderbook_responses",
    "fees": "https://kalshi.com/docs/kalshi-fee-schedule.pdf",
}


def load_kalshi(candles_path, markets_path):
    q, m = pd.read_csv(candles_path), pd.read_csv(markets_path)
    q_required = {"ticker", "end_ts", "elapsed_minute", "yes_bid", "yes_ask", "no_ask", "volume"}
    m_required = {"ticker", "series", "open_time", "close_time", "result", "y_up"}
    if not q_required.issubset(q) or not m_required.issubset(m):
        raise ValueError("Missing required Kalshi quote or market columns")
    if m.ticker.duplicated().any() or q.duplicated(["ticker", "end_ts"]).any():
        raise ValueError("Duplicate markets or market/minute quotes")
    if not m.series.eq("KXSOL15M").all():
        raise ValueError("Expected SOL 15-minute markets only")
    for name in ("open_time", "close_time"):
        m[name] = pd.to_datetime(m[name], utc=True, errors="raise")
    if (
        m.open_time.duplicated().any()
        or not (m.close_time - m.open_time).eq(pd.Timedelta(minutes=15)).all()
    ):
        raise ValueError("Ambiguous windows or market duration is not 15 minutes")
    if not m.open_time.eq(m.open_time.dt.floor("15min")).all():
        raise ValueError("Market start does not lie on a standard quarter hour")
    result = m.result.map({"yes": 1.0, "no": 0.0})
    if (result.notna() & result.ne(m.y_up)).any():
        raise ValueError("Market result and y_up disagree")
    m["kalshi_y"] = result
    q["quote_time"] = pd.to_datetime(q.end_ts, unit="s", utc=True, errors="raise")
    prices = q[["yes_bid", "yes_ask", "no_ask"]]
    if (
        not np.isfinite(prices.to_numpy()).all()
        or not prices.ge(0).all().all()
        or not prices.le(1).all().all()
    ):
        raise ValueError("Non-finite or out-of-range quotes")
    if q.yes_bid.gt(q.yes_ask + 1e-9).any() or not np.allclose(q.no_ask, 1 - q.yes_bid, atol=1e-9):
        raise ValueError("Crossed quotes or NO ask is not complementary to YES bid")
    joined = q.merge(m[["ticker", "open_time"]], on="ticker", how="left", validate="many_to_one")
    if (
        joined.open_time.isna().any()
        or not ((joined.quote_time - joined.open_time).dt.total_seconds() / 60)
        .eq(joined.elapsed_minute)
        .all()
    ):
        raise ValueError("Quote elapsed minute does not agree with market metadata")
    # Input files are not guaranteed to be in timestamp order.
    return q.sort_values(["quote_time", "ticker"]), m.sort_values("open_time")


def taker_fee(price, quantity=1, rate=0.07, rounding=0.01):
    """Explicit scenario, not a claim to reconstruct historical/account fee schedules.

    Fee = ceil(rate * quantity * price * (1-price) / rounding) * rounding.
    Default is conservative whole-cent rounding of each one-contract order.
    """
    if quantity < 1 or rate < 0 or rounding <= 0:
        raise ValueError("Invalid fee parameters")
    fee = rate * quantity * np.asarray(price) * (1 - np.asarray(price))
    return np.ceil(np.maximum(fee / rounding - 1e-10, 0)) * rounding


def align_signals(decisions, quotes, markets, delay_minutes=0):
    """Exact clock joins only; never carry a quote across markets or forward fill.

    delay=0 uses the quote observed at signal time. delay=1 is explicitly a
    future execution-price scenario, not information available to the model.
    """
    d = decisions[decisions.symbol.eq("SOL") & decisions.policy.isin(POLICIES)].copy()
    if d.duplicated(["policy", "window_id"]).any():
        raise ValueError("More than one action per policy/window")
    d["signal_time"] = pd.to_datetime(d.available_at, utc=True)
    expected = d.window_id + pd.to_timedelta(d.trigger_k, unit="min")
    if not d.loc[d.triggered, "signal_time"].eq(expected[d.triggered]).all():
        raise ValueError("Signal time does not match completed-minute k")
    if not d.loc[d.triggered, "trigger_k"].between(1, 10).all():
        raise ValueError("Prediction was emitted outside k=1..10")
    d["entry_time"] = d.signal_time + pd.Timedelta(minutes=delay_minutes)
    # Actual settlement labels are joined for scoring, never used to choose an action.
    fields = ["ticker", "open_time", "close_time", "kalshi_y"]
    optional = [n for n in ("floor_strike", "expiration_value") if n in markets]
    d = d.merge(
        markets[fields + optional],
        left_on="window_id",
        right_on="open_time",
        how="left",
        validate="many_to_one",
    )
    qfields = ["ticker", "quote_time", "yes_bid", "yes_ask", "no_ask", "volume"]
    d = d.merge(
        quotes[qfields].rename(columns={"volume": "quote_minute_volume"}),
        left_on=["ticker", "entry_time"],
        right_on=["ticker", "quote_time"],
        how="left",
        validate="many_to_one",
    )
    if (
        not d.loc[d.quote_time.notna(), "quote_time"]
        .eq(d.loc[d.quote_time.notna(), "entry_time"])
        .all()
    ):
        raise AssertionError("Quote alignment failed")
    d["side"] = np.where(d.p.ge(0.5), "yes", "no")
    d.loc[~d.triggered, "side"] = "abstain"
    d["ask"] = np.where(d.side.eq("yes"), d.yes_ask, d.no_ask)
    d["side_probability_proxy"] = np.maximum(d.p, 1 - d.p)
    d["label_disagrees"] = d.kalshi_y.ne(d.y) & d.kalshi_y.notna()
    return d


def price_overlay(
    aligned,
    *,
    quantity=1,
    flat_fee_per_contract=0.02,
    rate=0.07,
    rounding=0.01,
    slippage=0.0,
    positive_edge_only=False,
):
    if quantity < 1 or (
        flat_fee_per_contract is not None
        and (not np.isfinite(flat_fee_per_contract) or flat_fee_per_contract < 0)
    ):
        raise ValueError("Invalid quantity or flat entry fee")
    d = aligned.copy()
    # Ask=1 is a non-executable/no-offer sentinel; no depth is assumed there.
    available = (
        d.triggered
        & d.ticker.notna()
        & d.quote_time.notna()
        & d.ask.gt(0)
        & d.ask.lt(1)
        & d.entry_time.lt(d.close_time)
    )
    d["entry_price"] = (d.ask + slippage).clip(upper=1)
    if flat_fee_per_contract is None:
        d["entry_fee"] = taker_fee(d.entry_price, quantity, rate, rounding)
    else:
        d["entry_fee"] = np.where(d.entry_price.notna(), quantity * flat_fee_per_contract, np.nan)
    d["proxy_expected_net"] = quantity * (d.side_probability_proxy - d.entry_price) - d.entry_fee
    d["assumed_trade"] = available & (d.proxy_expected_net.gt(0) if positive_edge_only else True)
    d["scored_trade"] = d.assumed_trade & d.kalshi_y.notna()
    win = np.where(d.side.eq("yes"), d.kalshi_y, 1 - d.kalshi_y)
    d["kalshi_win"] = np.where(d.scored_trade, win, np.nan)
    d["coinbase_win"] = np.where(d.scored_trade, d.correct, np.nan)
    d["gross_pnl"] = np.where(d.scored_trade, quantity * (win - d.entry_price), 0.0)
    d["net_pnl"] = d.gross_pnl - np.where(d.scored_trade, d.entry_fee, 0.0)
    d["coinbase_counterfactual_net"] = np.where(
        d.scored_trade, quantity * (d.correct - d.entry_price) - d.entry_fee, 0.0
    )
    d["cash_outlay"] = np.where(d.scored_trade, quantity * d.entry_price + d.entry_fee, 0.0)
    d["skip_reason"] = np.select(
        [
            ~d.triggered,
            d.ticker.isna(),
            d.quote_time.isna(),
            ~available,
            available & ~d.assumed_trade,
            d.assumed_trade & d.kalshi_y.isna(),
        ],
        [
            "no_signal",
            "no_market",
            "no_exact_quote",
            "unavailable_ask_or_closed",
            "nonpositive_proxy_edge",
            "unsettled",
        ],
        default="assumed_fill",
    )
    return d


def pnl_interval(frame, seed=378, repetitions=2000):
    daily = (
        frame.assign(day=frame.window_id.dt.floor("D"))
        .groupby("day")
        .agg(pnl=("net_pnl", "sum"), n=("scored_trade", "sum"))
    )
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    values = daily.reindex(days, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(len(days), size=(repetitions, int(np.ceil(len(days) / 7))))
    idx = ((starts[:, :, None] + np.arange(7)) % len(days)).reshape(repetitions, -1)[:, : len(days)]
    totals = values[idx].sum(axis=1)
    nonempty = totals[:, 1] > 0
    if not nonempty.any():
        return np.nan, np.nan
    return tuple(np.quantile(totals[nonempty, 0] / totals[nonempty, 1], [0.025, 0.975]))


def summarize_overlay(frame):
    rows = []
    for keys in (("policy",), ("fold", "policy")):
        for key, group in frame.groupby(list(keys)):
            row = dict(zip(keys, key, strict=True))
            trades = group[group.scored_trade]
            equity = np.r_[0, group.sort_values("window_id").net_pnl.cumsum().to_numpy()]
            low, high = pnl_interval(group)
            row.update(
                windows=len(group),
                matched_markets=int(group.ticker.notna().sum()),
                signals=int(group.triggered.sum()),
                assumed_trades=int(group.assumed_trade.sum()),
                trades=len(trades),
                coverage=len(trades) / len(group),
                quote_coverage_of_signals=len(trades) / max(int(group.triggered.sum()), 1),
                kalshi_win_rate=trades.kalshi_win.mean(),
                coinbase_win_rate=trades.coinbase_win.mean(),
                label_disagreements=int(trades.label_disagrees.sum()),
                mean_entry=trades.entry_price.mean(),
                fees=trades.entry_fee.sum(),
                gross_pnl=trades.gross_pnl.sum(),
                net_pnl=trades.net_pnl.sum(),
                net_per_trade=trades.net_pnl.mean(),
                return_on_outlay=trades.net_pnl.sum() / trades.cash_outlay.sum()
                if len(trades)
                else np.nan,
                max_drawdown=float(np.max(np.maximum.accumulate(equity) - equity)),
                net_per_trade_ci_low=low,
                net_per_trade_ci_high=high,
                zero_volume_quote_minutes=int(trades.quote_minute_volume.eq(0).sum()),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--research", type=Path, default=Path("backend/research/artifacts/walk_forward_v1_audited")
    )
    p.add_argument("--candles", type=Path, default=Path("backend/data/kalshi/candles_SOLUSD.csv"))
    p.add_argument("--markets", type=Path, default=Path("backend/data/kalshi/markets_SOLUSD.csv"))
    p.add_argument(
        "--output", type=Path, default=Path("backend/research/artifacts/kalshi_sol_overlay_flat_2c")
    )
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument("--fee-mode", choices=("flat", "probability"), default="flat")
    p.add_argument(
        "--flat-fee-per-contract",
        type=float,
        default=0.02,
        help="Purchase fee in dollars per contract; resolution fee is zero",
    )
    p.add_argument("--fee-rate", type=float, default=0.07)
    p.add_argument("--fee-rounding", type=float, default=0.01)
    args = p.parse_args(argv)
    if (
        args.quantity < 1
        or args.fee_rate < 0
        or args.fee_rounding <= 0
        or not np.isfinite(args.flat_fee_per_contract)
        or args.flat_fee_per_contract < 0
    ):
        p.error("Invalid quantity or fee assumptions")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is nonempty; select a new experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    quotes, markets = load_kalshi(args.candles, args.markets)
    decisions_path = args.research / "decisions.parquet"
    decisions = pd.read_parquet(decisions_path)
    all_rows, summaries = [], []
    for delay in (0, 1):
        aligned = align_signals(decisions, quotes, markets, delay_minutes=delay)
        for slip in (0.0, 0.01, 0.02):
            for edge_filter in (False, True):
                result = price_overlay(
                    aligned,
                    quantity=args.quantity,
                    flat_fee_per_contract=args.flat_fee_per_contract
                    if args.fee_mode == "flat"
                    else None,
                    rate=args.fee_rate,
                    rounding=args.fee_rounding,
                    slippage=slip,
                    positive_edge_only=edge_filter,
                )
                result["delay_minutes"] = delay
                result["slippage"] = slip
                result["entry_rule"] = "positive_proxy_ev" if edge_filter else "all_signals"
                summary = summarize_overlay(result)
                for col in ("delay_minutes", "slippage", "entry_rule"):
                    summary[col] = result[col].iloc[0]
                summaries.append(summary)
                all_rows.append(result)
    rows, summary = pd.concat(all_rows, ignore_index=True), pd.concat(summaries, ignore_index=True)
    rows.to_parquet(args.output / "aligned_decisions.parquet", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    keys = ["delay_minutes", "slippage", "entry_rule", "policy"]
    rows.assign(day=rows.window_id.dt.floor("D")).groupby(keys + ["day"]).agg(
        pnl=("net_pnl", "sum"), trades=("scored_trade", "sum")
    ).reset_index().to_csv(args.output / "daily_pnl.csv", index=False)
    rows.groupby(keys + ["skip_reason"]).size().rename("windows").reset_index().to_csv(
        args.output / "skip_reasons.csv", index=False
    )
    base = summary[
        summary.fold.isna()
        & summary.delay_minutes.eq(0)
        & summary.slippage.eq(0)
        & summary.entry_rule.eq("all_signals")
    ]
    filtered = summary[
        summary.fold.isna()
        & summary.delay_minutes.eq(0)
        & summary.slippage.eq(0)
        & summary.entry_rule.eq("positive_proxy_ev")
    ]
    universe = rows[
        (rows.delay_minutes.eq(0))
        & rows.slippage.eq(0)
        & rows.entry_rule.eq("all_signals")
        & rows.policy.eq("ml_dynamic")
    ]
    audit = {
        "oos_start": str(universe.window_id.min()),
        "oos_end": str(universe.window_id.max()),
        "quote_start": str(quotes.quote_time.min()),
        "quote_end": str(quotes.quote_time.max()),
        "oos_windows": len(universe),
        "matched_markets": int(universe.ticker.notna().sum()),
        "comparable_settlements": int(universe.kalshi_y.notna().sum()),
        "all_window_label_disagreements": int(universe.label_disagrees.sum()),
    }
    manifest = {
        "audit": audit,
        "quantity": args.quantity,
        "fee_mode": args.fee_mode,
        "flat_fee_per_contract": args.flat_fee_per_contract if args.fee_mode == "flat" else None,
        "resolution_fee": 0.0,
        "fee_rate": args.fee_rate,
        "fee_rounding": args.fee_rounding,
        "sources": SOURCES,
        "sha256": {
            str(path): _sha(path)
            for path in (args.candles, args.markets, decisions_path, Path(__file__))
        },
        "signal_manifest_sha256": _sha(args.research / "manifest.json"),
        "note": "No model or threshold retraining; fee/slippage/latency are scenario assumptions.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = [
        "policy",
        "trades",
        "coverage",
        "kalshi_win_rate",
        "mean_entry",
        "gross_pnl",
        "fees",
        "net_pnl",
        "net_per_trade",
    ]
    if args.fee_mode == "flat":
        fee_description = (
            f"User-specified flat purchase fee: ${args.flat_fee_per_contract:.4f} per contract, "
            "charged once on entry, independently of price and outcome. No resolution fee. "
            "This assumption supersedes the earlier probability-based fee scenarios."
        )
    else:
        fee_description = (
            f"Fee scenario: ceil({args.fee_rate} × contracts × ask × (1−ask) / {args.fee_rounding}) "
            f"× {args.fee_rounding} dollars per order. No resolution fee. "
            "This legacy scenario does not establish historical/account fees."
        )
    report = [
        "# SOL signals over Kalshi quotes",
        "",
        (
            f"Frozen out-of-sample windows: {audit['oos_start']} to {audit['oos_end']}. "
            f"Matched {audit['matched_markets']} of {audit['oos_windows']} Coinbase-evaluable windows. "
            f"Kalshi and Coinbase labels disagree on {audit['all_window_label_disagreements']} "
            f"of {audit['comparable_settlements']} comparable windows."
        ),
        "",
        "## Same-minute quote markout, all signals",
        "",
        "```text",
        base[columns].to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        "```",
        "",
        "## Predeclared positive proxy-EV filter",
        "",
        "```text",
        filtered[columns].to_string(index=False, float_format=lambda x: f"{x:.5f}"),
        "```",
        "",
        (
            "The filter buys only when the frozen Coinbase directional probability minus the selected "
            "Kalshi ask and assumed fee is positive. This is a proxy: the model is calibrated for "
            "Coinbase candle color, not Kalshi settlement. It adds no retraining, threshold sweep, "
            "or second trigger after a rejection. It may reduce coverage below 25%."
        ),
        "",
        "## Accounting and interpretation",
        "",
        (
            f"Assume {args.quantity} contract(s) at the YES ask for a green prediction, or NO ask for red, "
            "held to the recorded Kalshi result. Gross P&L = contracts × (settlement payout − entry ask). "
            + fee_description
        ),
        "",
        (
            "Exact UTC joins use market open_time and close_time, never ticker-date inference. Quotes "
            "are used only when end_ts equals the chosen entry time and that time precedes market close. "
            "No missing quote is forward filled. An ask of 1 is treated as unavailable. CSV input order "
            "is ignored. Baseline and ML are distinct strategies, never combined into duplicate positions."
        ),
        "",
        (
            "summary.csv includes both 0/1-minute execution-delay assumptions, 0/1/2-cent adverse price "
            "slippage, all-signals/positive-proxy-EV entry, and each fold. A delayed price is an execution "
            "scenario after the signal, never a model input. Slippage prices are capped at $1. "
            "Coverage uses all original OOS Coinbase-evaluable windows, so missing markets/quotes count "
            "as skipped windows. Actual settlement labels are used only for scoring; unresolved assumed "
            "trades are counted explicitly but not scored. None were silently relabeled from Coinbase."
        ),
        "",
        (
            "Net-per-trade intervals in summary.csv use 2,000 paired-time seven-day block resamples of "
            "calendar days for each strategy; these are descriptive individual-policy intervals, not a "
            "multiple-comparison-adjusted significance test. daily_pnl.csv permits equity reconstruction. "
            "aligned_decisions.parquet records asks, model confidence, entry timing, settlement, costs, "
            "label disagreements, and skip reasons for every scenario/window."
        ),
        "",
        (
            "This file contains minute quote observations without sizes or queue history. Same-minute "
            "price availability, bar publication latency, and fills cannot be established from these "
            "CSVs. Trade volume in the preceding minute is not resting depth. These results are "
            "hypothetical quote-level settlement markouts, not demonstrated executable returns. "
            "The source model's earlier SOL top-25% finding is retrospective; only the original frozen "
            "fixed-selected/dynamic actions are overlaid here."
        ),
        "",
        (
            "Kalshi's crypto target uses CF Benchmarks averages, unlike Coinbase OHLC candle color. "
            "Settlement and contract outcomes come from markets_SOLUSD.csv."
        ),
        "",
        *[f"- [{name}]({url})" for name, url in SOURCES.items()],
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(base[columns].to_string(index=False), flush=True)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
