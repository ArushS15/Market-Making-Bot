"""
market/simulator.py
--------------------
Simulates the "true" market our bot trades against:

1. MidPriceProcess  - the mid-price follows a Brownian motion whose
   instantaneous volatility is itself a mean-reverting stochastic process
   (an Ornstein-Uhlenbeck process on sigma). This gives us realistic vol
   clustering instead of a flat constant-vol Black-Scholes world.

2. OrderFlowSimulator - models the arrival of market orders that could
   "hit" our bid or "lift" our ask as an inhomogeneous Poisson process
   whose intensity decays exponentially with distance from the mid-price,
   plus an optional adverse-selection component: a fraction of order flow
   can be "informed" and preferentially trade against us right before the
   price moves in its favor (see `informed_intensity` below). Without
   this, fills are pure noise-trader flow with zero correlation to future
   price moves, which lets a market maker collect the spread almost
   risk-free.

Both classes are stepped forward one `dt` at a time by the main loop so
that the strategy can react at every tick, just like a live system would.
"""

import numpy as np


class MidPriceProcess:
    """
    Stochastic-volatility mid-price process.

    Price:      dS_t = mu * dt + sigma_t * dW_t^S
    Volatility: dsigma_t = kappa * (theta - sigma_t) * dt + xi * dW_t^sigma

    The volatility SDE is a simple OU process (Vasicek-style). It's not as
    realistic as a full Heston square-root process, but it's numerically
    trivial, can't blow up as easily in a teaching simulation, and still
    gives us the mean-reverting, clustering volatility we want for testing
    how the strategy widens/narrows spreads under changing risk.
    """

    def __init__(self, s0, mu, sigma0, kappa, theta, vol_of_vol, rng=None):
        self.s = s0
        self.mu = mu
        self.sigma = sigma0
        self.kappa = kappa
        self.theta = theta
        self.xi = vol_of_vol
        self.rng = rng if rng is not None else np.random.default_rng()

    def step(self, dt):
        """Advance both processes by one time increment `dt`. Returns new mid-price."""
        # Two independent Brownian increments (could add correlation via rho if desired)
        dW_s = self.rng.normal(0.0, np.sqrt(dt))
        dW_sigma = self.rng.normal(0.0, np.sqrt(dt))

        # Volatility update (OU / mean-reverting), floored at a small positive
        # number so it can never go negative or zero (which would break the
        # strategy's spread formula).
        self.sigma = self.sigma + self.kappa * (self.theta - self.sigma) * dt + self.xi * dW_sigma
        self.sigma = max(self.sigma, 1e-4)

        # Price update using the *current* (just-updated) volatility level
        self.s = self.s + self.mu * dt + self.sigma * dW_s
        self.s = max(self.s, 1e-4)  # prices can't go negative

        return self.s, self.sigma


class OrderFlowSimulator:
    """
    Models market-order arrivals that can execute against our resting
    limit quotes as two independent inhomogeneous Poisson processes (bid
    side, ask side), made of two components:

    1. Uninformed ("noise") flow -- fill intensity decays exponentially
       with distance from mid:
            lambda_uninformed(delta) = A * exp(-k * delta)
       Quotes placed right at the mid fill very frequently but are more
       exposed; quotes placed far away rarely fill but are safer. This
       part alone carries no directional information -- it's liquidity
       demand that shows up regardless of where the price is headed.

    2. Informed ("adverse-selection") flow -- an extra fill probability
       applied to whichever side is about to be proven wrong by the next
       price move. If the price is about to jump up, informed buyers
       preferentially lift our ask (we sell right before it gets more
       expensive); if the price is about to drop, informed sellers hit
       our bid. This is what actually creates adverse selection risk:
       real fills carry a systematic tax, not just a risk-free spread.
    """

    def __init__(self, A, k, rng=None, adverse_selection_strength=0.0):
        self.A = A
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng()
        # Scales how strongly an imminent price move predicts which side
        # gets hit. 0.0 recovers the old (unrealistic) noise-only model.
        self.adverse_selection_strength = adverse_selection_strength

    def fill_intensity(self, delta):
        """
        lambda(delta): expected uninformed-flow fills per unit time at a
        given quote distance from mid.

        Note: delta is allowed to go NEGATIVE (a quote that's stale and
        has been crossed by the current mid -- e.g. your ask sitting
        below the current fair value). The exponential naturally spikes
        fill intensity in that case (exp(-k*delta) with delta<0 grows
        quickly), representing near-certain, near-arbitrage execution.
        This is the main channel through which quote staleness (see
        config.QUOTE_LATENCY) creates real adverse-selection cost: if a
        quote lags the market even a little, it can get badly mispriced
        and picked off almost for free the moment the true price crosses
        it.
        """
        return self.A * np.exp(-self.k * delta)

    def sample_fill(self, delta, dt, informed_intensity=0.0):
        """
        Bernoulli/Poisson-thinning approximation: over a small dt, the
        probability of at least one fill is ~ (lambda_uninformed(delta) +
        informed_intensity) * dt. Both components are RATES (fills per
        unit time), combined before multiplying by dt, so they're on
        consistent units regardless of how fine dt is.
        Returns True/False for "did a market order arrive and hit this quote".
        """
        prob = (self.fill_intensity(delta) + max(informed_intensity, 0.0)) * dt
        prob = min(max(prob, 0.0), 1.0)  # guard against dt too large / delta too small
        return self.rng.random() < prob

    def informed_intensity(self, price_move, side):
        """
        Extra fill RATE (not probability -- gets multiplied by dt in
        sample_fill, same as the uninformed intensity) for `side` ('bid'
        or 'ask'), driven by informed flow that knows the price will move
        by `price_move` over the strategy's approximate holding horizon.
        Only ever helps the informed trader: a price about to rise boosts
        ask fills (we sell too cheap), a price about to fall boosts bid
        fills (we buy too high). This is the channel through which the
        simulation charges the strategy a genuine adverse-selection cost
        instead of a risk-free spread.
        """
        if self.adverse_selection_strength <= 0.0:
            return 0.0
        if side == "ask":
            return self.adverse_selection_strength * max(price_move, 0.0)
        elif side == "bid":
            return self.adverse_selection_strength * max(-price_move, 0.0)
        else:
            raise ValueError(f"Unknown side: {side}")

    def sample_fill_size(self):
        """
        Order sizes aren't all identical in real markets. We draw a random
        integer lot size to make fills more realistic than a fixed unit size.
        """
        return int(self.rng.integers(1, 4))  # fills of size 1-3
