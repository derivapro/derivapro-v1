## Barrier Options

Barrier options are path-dependent options whose payoff depends on whether the underlying price breaches a specified barrier during the monitoring period.

DerivaPro currently supports four single-underlying barrier styles:

- **Up-and-out:** payoff is extinguished if the path breaches an upper barrier.
- **Down-and-out:** payoff is extinguished if the path breaches a lower barrier.
- **Up-and-in:** payoff is activated only if the path breaches an upper barrier.
- **Down-and-in:** payoff is activated only if the path breaches a lower barrier.

The current product-standard workflow uses explicit user assumptions, Monte Carlo path simulation, finite-difference Greeks, a European vanilla benchmark, barrier breach diagnostics, first-pass driver sensitivity, and a structured run summary.

Full methodology documentation is available in [`docs/methodology/barrier_option.md`](../../docs/methodology/barrier_option.md).
