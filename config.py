"""
config.py
---------
Central place for every hyperparameter used across the simulation,
strategy, and execution layers. Keeping these in one file makes it easy
to run parameter sweeps (e.g. risk-aversion sensitivity, volatility
regimes) without touching the core logic.
"""

# ---------------------------------------------------------------------------
# Simulation horizon
# ---------------------------------------------------------------------------
T = 1.0                    # total simulation time (e.g. "1 trading day" in normalized units)

# STEPS_PER_UNIT_TIME keeps the discretization step DT constant regardless
# of how long T is. N_STEPS is derived automatically -- change
# STEPS_PER_UNIT_TIME rather than N_STEPS directly. Every latency- and
# fill-related parameter below (QUOTE_LATENCY, the adverse-selection
# model, etc.) is calibrated against a specific DT, so keeping it fixed
# as T grows is what lets the same parameters work whether you simulate
# a single day or several hundred.
STEPS_PER_UNIT_TIME = 4000
N_STEPS = int(T * STEPS_PER_UNIT_TIME)   # number of discrete time steps
DT = T / N_STEPS                          # time increment per step

# RISK_HORIZON is the length of one repeating "trading day" for the
# risk/skew formulas, in the same time units as T. Rather than a single
# (T - t) that only tapers once at the very end of the whole simulation,
# the strategy uses (RISK_HORIZON - (t mod RISK_HORIZON)): a sawtooth
# that resets every RISK_HORIZON units. This gives every simulated day
# the same "wind down risk near the close" behavior a real market maker
# has, however many days T spans.
RISK_HORIZON = 1.0

# ---------------------------------------------------------------------------
# Mid-price process (stochastic volatility)
# ---------------------------------------------------------------------------
S0 = 100.0                 # initial mid-price
MU = 0.0                   # drift of mid-price (zero -- a pure random walk)

# Volatility itself is modeled as a mean-reverting Ornstein-Uhlenbeck
# process rather than a constant, so the strategy has to react to
# changing risk instead of quoting against a single fixed vol number.
SIGMA0 = 2.0                # initial instantaneous volatility
VOL_KAPPA = 5.0             # speed of mean reversion of volatility
VOL_THETA = 2.0             # long-run mean volatility level
VOL_OF_VOL = 1.5            # volatility of volatility ("vol-of-vol")

# ---------------------------------------------------------------------------
# Order flow (Poisson arrival process for market orders hitting our quotes)
# ---------------------------------------------------------------------------
# Market order arrival intensity as a function of distance from mid-price:
#   lambda(delta) = A * exp(-k * delta)
# The standard Avellaneda-Stoikov / Cartea-style exponential decay in
# fill probability the further a quote sits from the mid.
ORDER_FLOW_A = 140.0        # base arrival intensity (orders per unit time at delta=0)
ORDER_FLOW_K = 1.5          # decay rate of fill probability with distance

# Optional secondary adverse-selection channel: an extra fill-rate boost
# for whichever side a longer-horizon price move is about to prove
# wrong (see ADVERSE_SELECTION_LOOKAHEAD_STEPS below). Disabled by
# default -- QUOTE_LATENCY (below) is the primary, well-behaved driver
# of realistic risk-adjusted returns in this project. This lookahead
# channel is left in as an optional extra risk factor to experiment
# with, not because it's required.
ADVERSE_SELECTION_STRENGTH = 0.0

# Adverse selection has to be measured over a horizon comparable to the
# spread being captured, not the next infinitesimal tick -- a single dt
# step moves the price by roughly sigma*sqrt(dt), which is tiny next to
# a typical half-spread. This many steps ahead is used instead of just
# i+1 when computing the (optional) adverse-selection signal above.
ADVERSE_SELECTION_LOOKAHEAD_STEPS = 200

# How much simulated time passes between the strategy actually
# recomputing its quotes (in time units, not step counts, so the knob
# stays meaningful regardless of how fine DT is). A value of 0 (or <=
# DT) means zero latency: quotes are always perfectly priced against the
# current mid and can never be picked off, which isn't realistic. Real
# market makers have nonzero latency between observing the market and
# updating resting orders; during that gap the market can move and order
# flow can trade against a stale, mispriced quote almost risk-free. This
# is the primary mechanism controlling the strategy's realized risk.
QUOTE_LATENCY = 0.0021

# ---------------------------------------------------------------------------
# Avellaneda-Stoikov strategy parameters
# ---------------------------------------------------------------------------
GAMMA = 0.1                 # risk aversion coefficient (higher = more inventory-averse)
K_AS = ORDER_FLOW_K         # the "k" in the AS optimal spread formula reuses the order-flow decay

# Latency-risk buffer: an extra half-spread added to each side of the
# quote, sized to volatility above LATENCY_BUFFER_VOL_BASELINE (using
# whichever is higher of the model's instantaneous sigma or a causal
# realized-vol estimate -- see AvellanedaStoikov.latency_buffer() and
# main.py's realized_sigma()). This protects the strategy from getting
# disproportionately hurt during elevated-volatility or sustained-trend
# stretches, when quotes are resting stale for QUOTE_LATENCY time units.
# 0.0 disables it and recovers the textbook AS quote.
LATENCY_BUFFER_MULTIPLIER = 16.0

# The vol level above which the latency buffer activates. Set above
# VOL_THETA (not equal to it) deliberately: sigma spends roughly half
# its time above VOL_THETA from ordinary OU fluctuation alone (the
# process's stationary std here is ~0.47), so using VOL_THETA itself as
# the threshold would make the buffer active most of the time and
# distort normal-regime quoting. ~1.5 stationary-std above VOL_THETA
# means the buffer only engages for genuinely elevated volatility.
LATENCY_BUFFER_VOL_BASELINE = VOL_THETA + 1.5 * (VOL_OF_VOL / (2 * VOL_KAPPA) ** 0.5)

# Lookback window (in time units) for the causal realized-volatility
# estimate used alongside the model's sigma when sizing the latency
# buffer. Sized as a modest multiple of QUOTE_LATENCY -- long enough for
# a stable estimate, short enough to react quickly to an emerging trend.
REALIZED_VOL_WINDOW = 25 * QUOTE_LATENCY

# ---------------------------------------------------------------------------
# Inventory / risk management
# ---------------------------------------------------------------------------
MAX_INVENTORY = 50                 # hard cap on |inventory| -- bot stops quoting one side beyond this
INVENTORY_SKEW_MULTIPLIER = 1.0    # extra scaling knob on the inventory skew term, for experimentation

# ---------------------------------------------------------------------------
# Execution / capital
# ---------------------------------------------------------------------------
STARTING_CASH = 100_000.0
TICK_SIZE = 0.01            # minimum price increment; quotes are rounded to this

# Daily stop-loss: once a day's running mark-to-market loss reaches this,
# the strategy flattens immediately and stops quoting for the rest of
# that day. This caps the size of the worst-case day directly, the same
# way a real trading desk's risk limits work, rather than relying purely
# on spread-widening to price the tail risk away.
MAX_DAILY_LOSS = 800.0

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42             # set to None for a fresh random run each time
