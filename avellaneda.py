"""
strategy/avellaneda.py
-----------------------
Implementation of the Avellaneda-Stoikov (2008) "High-frequency trading in
a limit order book" optimal market-making model.

Core idea: a risk-neutral market maker would just quote symmetrically
around the mid-price. But holding inventory is risky (the market can move
against you), so the AS model derives, from a utility-maximization
argument (CARA utility, risk aversion gamma), a *reservation price* that
skews your quotes away from the mid based on how much inventory you're
carrying, plus an *optimal spread* that widens as risk (volatility, time
horizon, risk-aversion) increases.

Reservation price:
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    - s: current mid-price
    - q: current inventory (positive = long, negative = short)
    - gamma: risk aversion coefficient
    - sigma: instantaneous volatility
    - (T - t): remaining time horizon

    Interpretation: if you're long inventory (q > 0), your reservation
    price shifts *below* the mid — you're effectively willing to sell
    cheaper because you want to reduce risk. This asymmetry is what makes
    your ask more aggressive and your bid less aggressive when you're
    already long (and vice-versa when short).

Optimal spread:
    delta_total = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    - The first term grows with risk aversion, volatility, and remaining
      time: more risk in the world -> wider spread to compensate.
    - The second term comes from the order-flow arrival-rate decay
      parameter k (how fast fill probability drops as you move away from
      mid) - it sets a baseline spread even under zero risk, related to
      how "competitive" the order flow environment is.

Quotes are then placed symmetrically around the reservation price:
    bid = r - delta_total / 2
    ask = r + delta_total / 2

Latency risk buffer (an addition on top of the textbook AS formula):
    A resting quote isn't updated instantly -- it sits for roughly
    QUOTE_LATENCY time units before it's refreshed (see main.py). During
    that window the price can move by roughly sigma * sqrt(QUOTE_LATENCY),
    and once the true mid crosses a stale quote, fill probability spikes
    fast (see fill_intensity() in market/simulator.py). The classic AS
    risk_term above widens the spread using sigma^2 * (T - t) -- the
    *daily* risk horizon -- which is a completely different timescale
    from the latency window, so it does nothing to specifically protect
    against this. Real market makers explicitly size their quote buffer
    to the time they expect an order to rest, which is what
    latency_buffer() below adds: an extra sigma * sqrt(QUOTE_LATENCY)
    term on each side. Without it, high-volatility periods get
    disproportionately (super-linearly) more expensive to quote through,
    since the classic spread widens only in proportion to sigma^2 * T,
    not to the much larger effective danger from wider price swings
    during the (fixed) latency window.
"""

import numpy as np


class AvellanedaStoikov:
    def __init__(self, gamma, k, max_inventory, inventory_skew_multiplier=1.0,
                 quote_latency=0.0, latency_buffer_multiplier=0.0, vol_baseline=None):
        self.gamma = gamma
        self.k = k
        self.max_inventory = max_inventory
        self.inventory_skew_multiplier = inventory_skew_multiplier
        # Both default to 0.0, which exactly recovers the textbook AS
        # quote with no latency protection -- opt-in via config.py.
        self.quote_latency = quote_latency
        self.latency_buffer_multiplier = latency_buffer_multiplier
        # Baseline ("typical") vol level -- the buffer only kicks in for
        # vol ABOVE this, so ordinary quoting near the typical vol regime
        # is untouched and only unusually volatile excursions get the
        # extra protection. See latency_buffer() docstring.
        self.vol_baseline = vol_baseline if vol_baseline is not None else 0.0

    def reservation_price(self, mid_price, inventory, sigma, time_remaining):
        """
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        """
        skew = (
            inventory
            * self.gamma
            * (sigma ** 2)
            * time_remaining
            * self.inventory_skew_multiplier
        )
        return mid_price - skew

    def optimal_spread(self, sigma, time_remaining):
        """
        delta_total = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        """
        risk_term = self.gamma * (sigma ** 2) * time_remaining
        flow_term = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        return risk_term + flow_term

    def latency_buffer(self, sigma):
        """
        Extra HALF-spread buffer (applied to each side) sized to the
        EXCESS volatility (above vol_baseline) expected to be realized
        while a quote rests unrefreshed for quote_latency time units.
        Scales with (sigma - vol_baseline), not sigma directly -- so
        ordinary quoting near the typical/baseline vol regime is left
        alone (matching how the strategy was originally calibrated), and
        the extra protection only activates for genuinely elevated
        volatility. This matters because over many simulated days, high-
        vol excursions are sampled far more often in aggregate than in
        any single "typical" day, and without this the strategy gets
        disproportionately hurt by exactly those excursions as the total
        simulated time grows (see config.LATENCY_BUFFER_MULTIPLIER).
        """
        if self.latency_buffer_multiplier <= 0.0 or self.quote_latency <= 0.0:
            return 0.0
        excess_sigma = max(sigma - self.vol_baseline, 0.0)
        return self.latency_buffer_multiplier * excess_sigma * np.sqrt(self.quote_latency)

    def quote(self, mid_price, inventory, sigma, time_remaining, tick_size=None, buffer_sigma=None):
        """
        Returns (bid_price, ask_price) given current market/inventory state.

        `sigma` drives the textbook AS risk/skew formulas as before.
        `buffer_sigma` (defaults to `sigma` if not given) drives the
        latency buffer specifically -- pass a causal realized-vol estimate
        here to protect against sustained directional moves that a hidden
        "true" sigma_t wouldn't reflect. See realized_sigma() in main.py.

        Inventory limits: once |inventory| >= max_inventory, we stop quoting
        the side that would increase inventory further (a hard risk cap on
        top of the soft skew already provided by the reservation price).
        """
        r = self.reservation_price(mid_price, inventory, sigma, time_remaining)
        spread = self.optimal_spread(sigma, time_remaining)
        buffer = self.latency_buffer(buffer_sigma if buffer_sigma is not None else sigma)

        bid = r - spread / 2.0 - buffer
        ask = r + spread / 2.0 + buffer

        # Hard inventory cap: don't add to a position that's already maxed out.
        if inventory >= self.max_inventory:
            bid = None  # withdraw bid, we won't buy more
        if inventory <= -self.max_inventory:
            ask = None  # withdraw ask, we won't sell more

        if tick_size:
            if bid is not None:
                bid = round(bid / tick_size) * tick_size
            if ask is not None:
                ask = round(ask / tick_size) * tick_size

        return bid, ask

    def quote_distances(self, mid_price, bid, ask):
        """
        Convenience helper: how far are our bid/ask from the mid? Used by
        the OrderFlowSimulator to compute fill probabilities.
        """
        delta_b = (mid_price - bid) if bid is not None else None
        delta_a = (ask - mid_price) if ask is not None else None
        return delta_b, delta_a
