# Market-Making Bot with Inventory Control

A simulation of an optimal market-making strategy based on the
**Avellaneda-Stoikov (2008)** model, built from scratch in Python.
Instead of trying to predict price direction, the bot continuously
quotes bid and ask prices around a simulated limit order book, managing
inventory risk and adverse selection the way a real market maker would.

<img width="1000" height="1200" alt="dashboard_example" src="https://github.com/user-attachments/assets/909b44ee-802a-423a-9e00-7527bebfeda7" />

## What this project demonstrates

- A stochastic-volatility market simulator (mean-reverting vol, Poisson
  order arrivals)
- The Avellaneda-Stoikov reservation-price and optimal-spread formulas,
  derived from a CARA utility-maximization argument
- Several realism layers that a textbook implementation of AS leaves out
  entirely: quote latency, causal realized-volatility estimation,
  adverse selection from stale quotes, and a daily stop-loss
- A statistically correct way to measure a high-frequency strategy's
  Sharpe ratio (the naive tick-level approach massively overstates it)

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

This runs a one-day simulation, prints performance statistics, and saves
a four-panel dashboard to `output/mm_bot_dashboard.png`.

To simulate a longer period, just change `T` in `config.py` (see
[Configuration](#configuration) below) -- everything else scales
automatically.

## Project structure

```
mm_bot/
├── config.py             # All hyperparameters in one place
├── main.py                # Simulation loop, orchestration, reporting, plotting
├── market/
│   └── simulator.py       # Mid-price SDE + Poisson order-flow simulation
├── strategy/
│   └── avellaneda.py      # Reservation price / optimal spread math
└── utils/
    └── metrics.py          # Portfolio state tracking, Sharpe ratio, drawdown
```

## The model

### Reservation price

A risk-neutral market maker would just quote symmetrically around the
mid-price. But holding inventory is risky, so Avellaneda-Stoikov derive
a *reservation price* that skews quotes away from the mid based on
current inventory:

```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

If you're long inventory (`q > 0`), the reservation price shifts below
the mid -- you become more willing to sell, which makes your ask more
aggressive and your bid less aggressive, nudging your position back
toward flat.

### Optimal spread

```
delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
```

The first term widens the spread with volatility, risk aversion, and
time remaining. The second term sets a baseline spread from the
competitiveness of order flow (`k`), even under zero risk.

Quotes are placed symmetrically around the reservation price:
`bid = r - delta/2`, `ask = r + delta/2`.

### Market simulation

- **Mid-price**: Brownian motion whose volatility is itself a
  mean-reverting Ornstein-Uhlenbeck process, giving realistic
  volatility clustering instead of a flat constant-vol world.
- **Order flow**: an inhomogeneous Poisson process whose fill intensity
  decays exponentially with distance from the mid (`lambda(delta) = A *
  exp(-k*delta)`) -- quotes near the mid fill often but are riskier,
  quotes further away fill rarely but are safer.

## Beyond the textbook model

A literal implementation of Avellaneda-Stoikov, re-quoting with zero
latency at every simulated tick, is unrealistically profitable: it can
capture the bid-ask spread almost risk-free, since its quotes are
always perfectly priced against the current mid. This project adds
several mechanisms to make the simulation behave more like live trading:

- **Quote latency** (`QUOTE_LATENCY`) -- quotes rest for a short period
  before refreshing, exactly like real resting limit orders. During
  that window the market can move, and a stale, crossed quote can get
  picked off almost for free. This is the primary source of realistic
  risk in the simulation.
- **Causal realized volatility** -- the latency-risk buffer uses
  whichever is higher of the model's (in reality unobservable)
  instantaneous volatility, or a rolling realized-volatility estimate
  computed only from past prices. This protects against both
  volatility spikes and sustained directional drift, which a model-only
  volatility signal would miss entirely.
- **Rolling risk horizon** -- the "time remaining" fed into the AS
  formulas resets every `RISK_HORIZON` units rather than tapering once
  across the whole simulation, so a multi-day run behaves like many
  independent trading days rather than one that never reaches "end of
  day" risk-reduction pressure.
- **End-of-day flattening** -- inventory is closed out at each day
  boundary, matching how the strategy's parameters were calibrated.
- **Daily stop-loss** (`MAX_DAILY_LOSS`) -- trading halts for the rest
  of the day once a loss threshold is breached. Spread-widening alone
  prices around tail risk but doesn't cap it; a hard daily loss limit,
  the same control every real trading desk uses, directly bounds the
  worst-case day.

## Measuring performance correctly

Computing a Sharpe ratio from tick-level PnL changes and annualizing
with `sqrt(periods_per_year)` is a well-known statistical trap for
high-frequency strategies: consecutive ticks are highly autocorrelated,
so naive scaling can report Sharpe ratios in the hundreds -- something
no real strategy achieves. This project instead:

1. Aggregates PnL into **daily returns** (independent-ish periods)
2. Annualizes with `sqrt(252)`, the standard convention
3. **Pools returns across multiple independent simulation runs**
   (`collect_stable_daily_returns`) rather than relying on a single
   random realization, which can look like a winner or a loser purely
   by chance

`utils/metrics.py` also keeps the naive tick-level calculation
(`sharpe_ratio_ticklevel_naive`) around for comparison, clearly labeled
as unreliable, so the difference is visible rather than hidden.

## Configuration

All hyperparameters live in `config.py`. The most relevant ones to
experiment with:

| Parameter | What it controls |
|---|---|
| `T` | Total simulation length, in trading days |
| `GAMMA` | Risk aversion -- higher means more inventory-averse, wider spreads |
| `SIGMA0`, `VOL_KAPPA`, `VOL_THETA`, `VOL_OF_VOL` | Volatility process parameters |
| `QUOTE_LATENCY` | How long a quote rests before refreshing |
| `LATENCY_BUFFER_MULTIPLIER` | Strength of the extra defensive spread during elevated volatility |
| `MAX_INVENTORY` | Hard position limit |
| `MAX_DAILY_LOSS` | Daily stop-loss threshold |

Changing `T` alone is safe -- `N_STEPS` and `DT` derive automatically so
the simulation's time resolution (and therefore every latency-dependent
calibration) stays consistent regardless of how long the run is.

## Example output

```
Sharpe ratio (pooled from 500 simulated days across independent runs,
annualized from daily returns): 1.41
Mean daily PnL: 64.12   Std daily PnL: 158.78
(Win rate: 92.0% of days profitable)
```

The dashboard (`output/mm_bot_dashboard.png`) shows four panels: the
mid-price with bid/ask quotes and reservation price overlaid, the
stochastic volatility path, inventory over time, and the portfolio's
mark-to-market value.

## References

- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a
  limit order book.* Quantitative Finance, 8(3), 217-224.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and
  High-Frequency Trading.* Cambridge University Press.

