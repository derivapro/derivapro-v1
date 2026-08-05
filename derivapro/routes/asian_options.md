## Asian Options

Asian options are path-dependent options whose payoff depends on an average underlying level observed over a specified period.

DerivaPro currently supports:

- **Average price / fixed strike:** payoff compares the observed average to a fixed strike.
- **Average strike / floating strike:** payoff compares terminal spot to the observed average.
- **Arithmetic averaging:** simple average of observed prices.
- **Geometric averaging:** exponential of the average observed log price.

The current product-standard workflow uses explicit user assumptions, configurable averaging windows, Monte Carlo simulation, finite-difference Greeks, benchmark diagnostics, first-pass driver sensitivity, and a structured run summary.

Full methodology documentation is available in [`docs/methodology/asian_option.md`](../../docs/methodology/asian_option.md).
