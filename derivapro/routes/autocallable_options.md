# Autocallable / Phoenix Notes

DerivaPro supports a Phoenix-style autocallable structured-note workflow for transparent product valuation and review.

The current implementation covers:

- Single-underlying notes.
- Worst-of basket notes.
- Scheduled autocall observation times.
- Coupon barrier.
- Autocall barrier.
- Knock-in protection barrier.
- Memory coupon toggle.
- Notional and maturity inputs.
- Static/user-supplied volatility assumptions.
- Flat basket correlation.
- Monte Carlo path simulation using the newer DerivaPro simulation engine.

For basket notes, the payoff is evaluated on the worst-performing underlying relative to its initial level. The workflow reports present value, standard error, autocall probability, protection-breach probability, average coupon count, and worst-of final-level diagnostics.

The product-standard methodology page is maintained under `docs/methodology/autocallable_note.md`.
