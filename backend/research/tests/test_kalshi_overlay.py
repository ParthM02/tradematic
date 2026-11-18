import numpy as np
import pandas as pd

from backend.research.kalshi_overlay import align_signals, price_overlay, taker_fee


def fixture():
    t0 = pd.Timestamp("2026-06-01", tz="UTC")
    d = pd.DataFrame(
        {
            "symbol": ["SOL"] * 3,
            "policy": ["ml_dynamic"] * 3,
            "fold": [0] * 3,
            "window_id": [t0 + pd.Timedelta(minutes=i * 15) for i in range(3)],
            "available_at": [t0 + pd.Timedelta(minutes=i * 15 + 5) for i in range(3)],
            "triggered": [True] * 3,
            "trigger_k": [5] * 3,
            "p": [0.95, 0.05, 0.90],
            "y": [1, 0, 1],
            "correct": [1.0, 1.0, 1.0],
        }
    )
    m = pd.DataFrame(
        {
            "ticker": ["a", "b", "c"],
            "open_time": d.window_id,
            "close_time": d.window_id + pd.Timedelta(minutes=15),
            "kalshi_y": [0.0, 0.0, 1.0],
        }
    )
    q = pd.DataFrame(
        {
            "ticker": ["a", "a", "b", "c"],
            "quote_time": [t0 + pd.Timedelta(minutes=i) for i in (5, 6, 20, 36)],
            "yes_bid": [0.80, 0.85, 0.80, 0.85],
            "yes_ask": [0.85, 0.90, 0.85, 0.90],
            "no_ask": [0.20, 0.15, 0.20, 0.15],
            "volume": [0.0, 100.0, 10.0, 20.0],
        }
    )
    return d, q, m


def test_exact_quotes_no_future_fill_and_actual_kalshi_settlement():
    d, q, m = fixture()
    aligned = align_signals(d, q, m)
    result = price_overlay(aligned)
    assert result.scored_trade.tolist() == [True, True, False]
    assert result.side.tolist() == ["yes", "no", "yes"]
    assert result.kalshi_win.iloc[0] == 0  # Coinbase called this a correct green.
    np.testing.assert_allclose(result.net_pnl, [-0.87, 0.78, 0.0])
    assert result.skip_reason.iloc[2] == "no_exact_quote"
    # Future quotes and settlement may change scores, but cannot change entry actions.
    q.loc[q.quote_time.dt.minute.eq(6), "yes_ask"] = 0.99
    m["kalshi_y"] = 1 - m.kalshi_y
    changed = price_overlay(align_signals(d, q, m))
    np.testing.assert_array_equal(result.assumed_trade, changed.assumed_trade)
    np.testing.assert_allclose(result.entry_price, changed.entry_price, equal_nan=True)


def test_ev_filter_is_frozen_signal_price_comparison_not_a_second_trigger():
    d, q, m = fixture()
    d.loc[0, "p"] = 0.82  # Quoted YES is 0.85 plus fees.
    result = price_overlay(align_signals(d, q, m), positive_edge_only=True)
    assert not result.assumed_trade.iloc[0]
    assert result.skip_reason.iloc[0] == "nonpositive_proxy_edge"
    assert result.assumed_trade.iloc[1]
    delayed = align_signals(d, q, m, delay_minutes=1)
    assert delayed.ask.iloc[0] == 0.90
    assert delayed.quote_time.iloc[0] == d.available_at.iloc[0] + pd.Timedelta(minutes=1)


def test_whole_cent_fees_and_unavailable_ask():
    np.testing.assert_allclose(taker_fee([0.01, 0.5, 0.99]), [0.01, 0.02, 0.01])
    d, q, m = fixture()
    q.loc[0, "yes_ask"] = 1.0
    result = price_overlay(align_signals(d, q, m))
    assert not result.assumed_trade.iloc[0]
    assert result.skip_reason.iloc[0] == "unavailable_ask_or_closed"


def test_flat_fee_is_per_contract_once_independent_of_price_and_outcome():
    d, q, m = fixture()
    result = price_overlay(align_signals(d, q, m), quantity=3)
    np.testing.assert_allclose(result.entry_fee.iloc[:2], [0.06, 0.06])
    np.testing.assert_allclose(result.net_pnl, [-2.61, 2.34, 0.0])
    assert result.cash_outlay.iloc[0] == 3 * 0.85 + 0.06
    d.loc[0, "p"] = 0.865
    aligned = align_signals(d, q, m)
    assert not price_overlay(aligned, positive_edge_only=True).assumed_trade.iloc[0]
    assert price_overlay(
        aligned, flat_fee_per_contract=None, positive_edge_only=True
    ).assumed_trade.iloc[0]
