"""
main.py
-------
Orchestrates the full simulation:

    for each time step:
        1. advance the market (mid-price + stochastic vol)             [market.simulator]
        2. ask the strategy for bid/ask quotes given current inventory  [strategy.avellaneda]
        3. simulate whether market orders arrive and fill our quotes    [market.simulator]
        4. update cash/inventory on any fills                           [utils.metrics]
        5. record everything for later analysis                        [utils.metrics]

At the end, prints summary statistics (Sharpe ratio, PnL, drawdown, win
rate) and renders an analytics dashboard (price path, volatility,
inventory, PnL).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config
from market.simulator import MidPriceProcess, OrderFlowSimulator
from strategy.avellaneda import AvellanedaStoikov
from utils.metrics import PortfolioState, summary_stats, sharpe_ratio_from_returns

# Output directory sits next to this script, so it works the same way on
# any machine regardless of where the project is checked out.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def realized_sigma(mid_prices, i, window_steps, dt):
    """
    Causal realized-volatility estimate using only past price data
    (mid_prices[0..i]) -- exactly what a real market maker could compute
    in real time, with no lookahead.

    This is used alongside the model's "true" sigma_t because sigma_t is
    a hidden parameter of the simulated process that a real strategy
    could never observe directly, only estimate from realized price
    moves. It also catches a case the model sigma alone would miss: a
    sustained directional drift (a stretch of a random walk that happens
    to trend, without any change in the instantaneous diffusion
    coefficient) still produces large realized price moves and is just
    as dangerous for a stale resting quote as a spike in sigma_t.
    """
    start = max(0, i - window_steps)
    window = mid_prices[start:i + 1]
    if len(window) < 3:
        return 0.0
    rets = np.diff(window)
    return rets.std() / np.sqrt(dt)


def risk_horizon_remaining(t):
    """
    "Time remaining" fed into the AS risk/skew formulas, as a sawtooth
    that resets every RISK_HORIZON units instead of only tapering once
    at the very end of the whole simulation. See config.RISK_HORIZON.
    """
    within_day = config.RISK_HORIZON - (t % config.RISK_HORIZON)
    time_to_true_end = config.T - t
    tr = min(within_day, time_to_true_end)  # also taper to zero at the actual final tick
    return max(tr, config.DT)


def run_simulation(seed=None):
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)

    # --- Instantiate the three engines ---------------------------------
    market = MidPriceProcess(
        s0=config.S0,
        mu=config.MU,
        sigma0=config.SIGMA0,
        kappa=config.VOL_KAPPA,
        theta=config.VOL_THETA,
        vol_of_vol=config.VOL_OF_VOL,
        rng=rng,
    )
    order_flow = OrderFlowSimulator(
        A=config.ORDER_FLOW_A,
        k=config.ORDER_FLOW_K,
        rng=rng,
        adverse_selection_strength=config.ADVERSE_SELECTION_STRENGTH,
    )
    strategy = AvellanedaStoikov(
        gamma=config.GAMMA,
        k=config.K_AS,
        max_inventory=config.MAX_INVENTORY,
        inventory_skew_multiplier=config.INVENTORY_SKEW_MULTIPLIER,
        quote_latency=config.QUOTE_LATENCY,
        latency_buffer_multiplier=config.LATENCY_BUFFER_MULTIPLIER,
        vol_baseline=config.LATENCY_BUFFER_VOL_BASELINE,
    )
    state = PortfolioState(starting_cash=config.STARTING_CASH, starting_inventory=0)

    # --- Precompute the full mid-price / vol path up front --------------
    # The (optional) adverse-selection fill model needs one step of
    # lookahead to bias fills toward informed flow. The strategy itself
    # never receives this -- it only ever sees mid_prices[i], never
    # mid_prices[i+1] -- so there's no lookahead bias in the trading
    # logic, only in the exogenous order-flow model.
    times = np.linspace(0, config.T, config.N_STEPS + 1)
    mid_prices = np.empty(config.N_STEPS + 1)
    sigmas = np.empty(config.N_STEPS + 1)
    mid_prices[0], sigmas[0] = config.S0, config.SIGMA0
    for i in range(1, config.N_STEPS + 1):
        mid_prices[i], sigmas[i] = market.step(config.DT)

    bids, asks = [np.nan], [np.nan]
    reservation_prices = [config.S0]
    current_bid, current_ask = None, None
    last_refresh_time = None
    current_day = 0
    daily_pnls = []  # end-of-day mark-to-market PnL, one entry per completed day
    day_start_value = config.STARTING_CASH
    halted_today = False

    # --- Main loop --------------------------------------------------------
    for i in range(config.N_STEPS):
        t = times[i]
        mid_price, sigma = mid_prices[i], sigmas[i]

        # --- End-of-day flattening --------------------------------------
        # Every trading desk closes out (or at least resets risk limits
        # on) its book at the end of each day. Without this, inventory
        # would silently carry across RISK_HORIZON boundaries while the
        # risk/skew formulas reset to "start of day" -- a leftover
        # position would get treated as if a full fresh day of risk lay
        # ahead of it. Flattening here makes a long run behave like many
        # independent single-day sessions stitched together.
        day_idx = int(t // config.RISK_HORIZON)
        if day_idx > current_day:
            if state.inventory != 0:
                side = "sell" if state.inventory > 0 else "buy"
                size = abs(state.inventory)
                state.process_fill(side, mid_price, size, t)
            daily_pnls.append(state.mark_to_market(mid_price))
            current_day = day_idx
            current_bid, current_ask = None, None  # force a fresh quote for the new day
            last_refresh_time = None
            day_start_value = state.mark_to_market(mid_price)
            halted_today = False

        # --- Daily stop-loss circuit breaker -----------------------------
        # A defense against a sustained adverse trend that develops
        # gradually: individually small unfavorable fills can compound
        # into a large loss over a single bad day. This mirrors a real
        # trading desk's daily loss limit, which halts trading (not just
        # widens spreads) once breached -- capping the size of the
        # worst-case day directly rather than trying to price around it.
        if not halted_today:
            running_pnl = state.mark_to_market(mid_price) - day_start_value
            if running_pnl <= -config.MAX_DAILY_LOSS:
                halted_today = True
                if state.inventory != 0:
                    side = "sell" if state.inventory > 0 else "buy"
                    size = abs(state.inventory)
                    state.process_fill(side, mid_price, size, t)
                current_bid, current_ask = None, None

        # 2. Get quotes from the strategy given current inventory/vol.
        # Only actually recompute once QUOTE_LATENCY time units have
        # passed since the last refresh -- in between, the previous quote
        # stays resting in the book exactly like a real limit order
        # would, even as the true mid moves on. This latency is the main
        # source of realistic adverse-selection risk in the simulation:
        # see config.QUOTE_LATENCY and fill_intensity() in
        # market/simulator.py for how stale, crossed quotes get picked
        # off.
        if not halted_today and (last_refresh_time is None or (t - last_refresh_time) >= config.QUOTE_LATENCY):
            tr = risk_horizon_remaining(t)
            window_steps = max(1, int(config.REALIZED_VOL_WINDOW / config.DT))
            r_sigma = realized_sigma(mid_prices, i, window_steps, config.DT)
            # Use whichever is larger: the model's own (in reality
            # unobservable) sigma_t, or what's actually been realized in
            # the price recently. Either signals danger for a resting
            # quote -- using only the model sigma misses sustained
            # trends, using only realized vol misses a sigma_t spike that
            # hasn't yet shown up in a short price window.
            buffer_sigma = max(sigma, r_sigma)
            current_bid, current_ask = strategy.quote(
                mid_price=mid_price,
                inventory=state.inventory,
                sigma=sigma,
                time_remaining=tr,
                tick_size=config.TICK_SIZE,
                buffer_sigma=buffer_sigma,
            )
            last_refresh_time = t
        bid, ask = current_bid, current_ask
        # Hard inventory cap always applies, even between quote refreshes.
        if state.inventory >= config.MAX_INVENTORY:
            bid = None
        if state.inventory <= -config.MAX_INVENTORY:
            ask = None
        tr = risk_horizon_remaining(t)
        r = strategy.reservation_price(mid_price, state.inventory, sigma, tr)
        bids.append(bid if bid is not None else np.nan)
        asks.append(ask if ask is not None else np.nan)
        reservation_prices.append(r)

        # 3. Simulate order flow hitting our quotes. Two components:
        #    (a) uninformed/noise flow -- distance-based Poisson process
        #    (b) informed flow (optional, off by default) -- biased
        #        toward whichever side a future price move proves wrong
        delta_b, delta_a = strategy.quote_distances(mid_price, bid, ask)
        lookahead_idx = min(i + config.ADVERSE_SELECTION_LOOKAHEAD_STEPS, config.N_STEPS)
        price_move = mid_prices[lookahead_idx] - mid_price

        if bid is not None:
            intensity = order_flow.informed_intensity(price_move, "bid")
            if order_flow.sample_fill(delta_b, config.DT, informed_intensity=intensity):
                size = order_flow.sample_fill_size()
                state.process_fill("buy", bid, size, t)

        if ask is not None:
            intensity = order_flow.informed_intensity(price_move, "ask")
            if order_flow.sample_fill(delta_a, config.DT, informed_intensity=intensity):
                size = order_flow.sample_fill_size()
                state.process_fill("sell", ask, size, t)

        # 4 & 5. Record state for this step
        state.record_step(mid_price)

    # Capture the final (possibly partial) day too.
    daily_pnls.append(state.mark_to_market(mid_prices[-1]))

    return {
        "times": times,
        "mid_prices": list(mid_prices),
        "sigmas": list(sigmas),
        "bids": bids,
        "asks": asks,
        "reservation_prices": reservation_prices,
        "state": state,
        "daily_pnls": np.array(daily_pnls),
    }


def daily_returns_from_values(daily_values, starting_cash):
    """
    Converts a sequence of end-of-day mark-to-market VALUES (as captured
    by run_simulation's end-of-day flattening) into day-over-day RETURNS
    suitable for sharpe_ratio_from_returns(). The first day's return is
    measured against the starting cash balance.
    """
    daily_values = np.asarray(daily_values, dtype=float)
    prev = np.concatenate([[starting_cash], daily_values[:-1]])
    return daily_values - prev


def run_multi_day_backtest(n_days=100, base_seed=None):
    """
    Runs many independent trading days (each a fresh call to
    run_simulation with a different seed) and collects the end-of-day
    mark-to-market PnL for each one.

    This exists so the Sharpe ratio can be computed correctly. Computing
    Sharpe from a single day's intraday tick-by-tick PnL changes and
    scaling by sqrt(periods_per_year) is a well-known statistical
    fallacy for high-frequency strategies: consecutive ticks are highly
    autocorrelated (same vol regime, same inventory path), so naive
    sqrt(N) scaling wildly overstates the true annualized Sharpe. Sharpe
    should come from a distribution of independent period (daily)
    returns, which is what this function produces.
    """
    rng_master = np.random.default_rng(base_seed if base_seed is not None else config.RANDOM_SEED)
    daily_pnls = []
    for _ in range(n_days):
        seed = int(rng_master.integers(0, 2**31 - 1))
        results = run_simulation(seed=seed)
        state = results["state"]
        end_value = state.mark_to_market(results["mid_prices"][-1])
        daily_pnls.append(end_value - config.STARTING_CASH)
    return np.array(daily_pnls)


def collect_stable_daily_returns(target_days=500):
    """
    Pools day-over-day returns across as many independent runs as needed
    to reach roughly `target_days` total samples, regardless of how long
    a single run (config.T) is.

    A single run is still just one random realization -- it can be an
    unusually good or bad stretch (e.g. a long trending path) purely by
    chance, and reporting statistics from only that one draw can make a
    genuinely profitable-on-average strategy look like a loser, or vice
    versa. Pooling across multiple independent seeds gives a report that
    reflects the strategy's actual expected behavior.
    """
    n_seeds = max(1, int(np.ceil(target_days / max(config.T, 1))))
    rng_master = np.random.default_rng(config.RANDOM_SEED)
    all_returns = []
    for _ in range(n_seeds):
        seed = int(rng_master.integers(0, 2**31 - 1))
        results = run_simulation(seed=seed)
        all_returns.append(daily_returns_from_values(results["daily_pnls"], config.STARTING_CASH))
    return np.concatenate(all_returns)


def plot_dashboard(results, save_path=None):
    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "mm_bot_dashboard.png")
    times = results["times"]
    mid_prices = results["mid_prices"]
    sigmas = results["sigmas"]
    bids = results["bids"]
    asks = results["asks"]
    r = results["reservation_prices"]
    state = results["state"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # --- Price / quotes panel ---
    ax = axes[0]
    ax.plot(times, mid_prices, label="Mid Price", color="black", linewidth=1)
    ax.plot(times, bids, label="Bid Quote", color="green", linewidth=0.8, alpha=0.7)
    ax.plot(times, asks, label="Ask Quote", color="red", linewidth=0.8, alpha=0.7)
    ax.plot(times, r, label="Reservation Price", color="blue", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylabel("Price")
    ax.set_title("Mid-Price, Quotes, and Reservation Price")
    ax.legend(loc="upper left", fontsize=8)

    # --- Volatility panel ---
    ax = axes[1]
    ax.plot(times, sigmas, color="purple", linewidth=1)
    ax.set_ylabel("Sigma (instantaneous vol)")
    ax.set_title("Stochastic Volatility")

    # --- Inventory panel ---
    ax = axes[2]
    ax.plot(times, state.inventory_history, color="darkorange", linewidth=1)
    ax.axhline(config.MAX_INVENTORY, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(-config.MAX_INVENTORY, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Inventory (q)")
    ax.set_title("Inventory Over Time")

    # --- PnL panel ---
    ax = axes[3]
    pnl = [state.cash_history[0]] + state.pnl_history  # prepend starting cash as t=0 mark-to-market
    ax.plot(times, pnl, color="teal", linewidth=1)
    ax.set_ylabel("Mark-to-Market PnL")
    ax.set_xlabel("Time")
    ax.set_title("Portfolio Value Over Time")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Dashboard saved to {save_path}")
    plt.close(fig)


def main():
    # Single detailed run, used only for the dashboard plots -- a
    # visualization of what one representative run looks like. All
    # reported statistics (Sharpe, mean/std daily PnL) come from
    # collect_stable_daily_returns() instead, which pools across multiple
    # independent runs so the numbers reflect the strategy's actual
    # expected behavior rather than one run's luck.
    results = run_simulation()
    stats = summary_stats(results["state"], results["mid_prices"][-1])

    print("=" * 50)
    print("MARKET-MAKING BOT — SINGLE-RUN DASHBOARD SUMMARY")
    print("(one representative run -- see pooled stats below for the")
    print(" statistically reliable picture)")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:,.4f}")
        else:
            print(f"{k:20s}: {v}")
    print("=" * 50)

    # Sharpe ratio and PnL stats, pooled across multiple independent runs.
    # See collect_stable_daily_returns's docstring for why this matters,
    # and run_multi_day_backtest's docstring for why Sharpe has to come
    # from independent daily returns rather than tick-level PnL changes.
    daily_returns = collect_stable_daily_returns(target_days=500)
    n_days = len(daily_returns)
    daily_sharpe = sharpe_ratio_from_returns(daily_returns, periods_per_year=252)
    print(f"\nSharpe ratio (pooled from {n_days} simulated days across independent runs,")
    print("annualized from daily returns): "
          f"{daily_sharpe:,.2f}")
    print(f"Mean daily PnL: {daily_returns.mean():,.2f}   Std daily PnL: {daily_returns.std():,.2f}")
    print(f"(Win rate: {(daily_returns > 0).mean():.1%} of days profitable)")
    print("=" * 50)

    plot_dashboard(results)


if __name__ == "__main__":
    main()
