"""
utils/metrics.py
-----------------
Two things live here:

1. PortfolioState - the execution/state manager. Tracks cash, inventory,
   and processes fills. This is intentionally simple (no partial fills,
   no queue position modeling) but captures the essentials: every fill
   moves cash and inventory in opposite directions.

2. Performance metrics - PnL time series analysis: running mark-to-market
   PnL, Sharpe ratio, and max drawdown, computed the same way you'd
   analyze any strategy's equity curve.
"""

import numpy as np


class PortfolioState:
    """Tracks the bot's cash and inventory as fills happen over time."""

    def __init__(self, starting_cash=0.0, starting_inventory=0):
        self.cash = starting_cash
        self.inventory = starting_inventory

        # History for analytics / plotting
        self.cash_history = [starting_cash]
        self.inventory_history = [starting_inventory]
        self.mid_price_history = []
        self.pnl_history = []
        self.trade_log = []  # list of dicts: {time, side, price, size}

    def process_fill(self, side, price, size, t):
        """
        side: 'buy' (we bought, hitting our bid) or 'sell' (we sold, our ask was lifted)
        """
        if side == "buy":
            self.cash -= price * size
            self.inventory += size
        elif side == "sell":
            self.cash += price * size
            self.inventory -= size
        else:
            raise ValueError(f"Unknown side: {side}")

        self.trade_log.append({"time": t, "side": side, "price": price, "size": size})

    def mark_to_market(self, mid_price):
        """Total portfolio value = cash + inventory valued at current mid-price."""
        return self.cash + self.inventory * mid_price

    def record_step(self, mid_price):
        """Call once per simulation step to snapshot state for later analysis."""
        self.cash_history.append(self.cash)
        self.inventory_history.append(self.inventory)
        self.mid_price_history.append(mid_price)
        self.pnl_history.append(self.mark_to_market(mid_price))


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_returns(pnl_series):
    """Simple period-over-period changes in mark-to-market PnL."""
    pnl_series = np.asarray(pnl_series, dtype=float)
    return np.diff(pnl_series)


def sharpe_ratio_from_returns(period_returns, periods_per_year=252):
    """
    The statistically correct way to compute Sharpe: pass in a series of
    INDEPENDENT PERIOD returns (e.g. one number per trading day) and this
    annualizes by sqrt(periods_per_year), which defaults to 252 (trading
    days/year).

    Do NOT pass tick-level / intraday mark-to-market changes here. Ticks
    within the same simulated day are highly autocorrelated (same vol
    regime, same inventory path), so treating them as independent draws
    and scaling by sqrt(huge_number_of_ticks_per_year) massively overstates
    the true annualized Sharpe. See run_multi_day_backtest() in main.py for
    how to build a proper period-return series.
    """
    period_returns = np.asarray(period_returns, dtype=float)
    if len(period_returns) < 2 or period_returns.std() == 0:
        return 0.0
    return (period_returns.mean() / period_returns.std()) * np.sqrt(periods_per_year)


def sharpe_ratio_ticklevel_naive(pnl_series, periods_per_year=252 * 390):
    """
    NAIVE, MISLEADING tick-level Sharpe -- kept only so you can see the
    difference vs sharpe_ratio_from_returns(). This treats every
    simulation step's mark-to-market change as an independent return and
    scales by sqrt(periods_per_year). For a smooth, high-frequency PnL
    path like a market maker's, that assumption is false: it will report
    Sharpe ratios in the hundreds that are impossible to realize live.
    Use sharpe_ratio_from_returns() on independent daily returns instead.
    """
    rets = compute_returns(pnl_series)
    if rets.std() == 0 or len(rets) == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(periods_per_year)


def max_drawdown(pnl_series):
    """
    Largest peak-to-trough decline in the equity curve, expressed both as
    an absolute dollar figure and as a fraction of the peak value.
    """
    pnl_series = np.asarray(pnl_series, dtype=float)
    running_max = np.maximum.accumulate(pnl_series)
    drawdowns = pnl_series - running_max
    max_dd_abs = drawdowns.min()  # most negative value

    # Avoid divide-by-zero if running_max hits 0
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns_pct = np.where(running_max != 0, drawdowns / running_max, 0.0)
    max_dd_pct = drawdowns_pct.min()

    return max_dd_abs, max_dd_pct


def summary_stats(state: PortfolioState, final_mid_price):
    """Bundle the key end-of-run metrics into a dict for easy printing/logging."""
    final_pnl = state.mark_to_market(final_mid_price)
    total_trades = len(state.trade_log)
    buy_trades = sum(1 for t in state.trade_log if t["side"] == "buy")
    sell_trades = total_trades - buy_trades

    dd_abs, dd_pct = max_drawdown(state.pnl_history) if state.pnl_history else (0.0, 0.0)

    return {
        "final_pnl": final_pnl,
        "final_inventory": state.inventory,
        "total_trades": total_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        # NOTE: this is the naive tick-level Sharpe -- it will look
        # unrealistically high (see sharpe_ratio_ticklevel_naive's
        # docstring). For a trustworthy annualized Sharpe, use
        # run_multi_day_backtest() + sharpe_ratio_from_returns() in
        # main.py, which is what the printed script output does.
        "tick_level_sharpe_not_reliable": sharpe_ratio_ticklevel_naive(state.pnl_history) if state.pnl_history else 0.0,
        "max_drawdown_abs": dd_abs,
        "max_drawdown_pct": dd_pct * 100,
    }
