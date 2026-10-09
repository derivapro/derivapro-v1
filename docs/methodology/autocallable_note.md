# Autocallable / Phoenix Note Methodology

## Product Scope

This methodology note covers the DerivaPro autocallable structured-note workflow currently implemented for Phoenix-style notes. The model supports:

- Single-underlying notes.
- Worst-of basket notes.
- Scheduled autocall observation dates expressed as year fractions.
- Coupon barrier, autocall barrier, and knock-in protection barrier.
- Memory coupon and non-memory coupon logic.
- Notional redemption with downside exposure when final worst-of performance breaches the protection barrier.
- Flat volatility assumptions per underlying.
- Flat equicorrelation for basket underlyings.

The current page is intended for transparent valuation and product-level analytics. It is not yet a full termsheet payoff builder for every structured-note variation traded in the market.

## Payoff Overview

The simulation tracks each underlying relative to its initial level. For basket notes, DerivaPro evaluates the product against the worst-performing underlying at each observation date.

At each scheduled observation:

1. The coupon condition is tested against the coupon barrier.
2. If the coupon condition is met, the current coupon is paid.
3. If memory coupon is enabled, previously missed coupons are accrued and paid once the coupon condition is met.
4. The autocall condition is tested against the autocall barrier.
5. If the autocall condition is met, the note redeems early at notional plus eligible coupons.

If the note does not autocall, final redemption is evaluated at maturity. If the final worst-of level is above the protection barrier, notional is repaid. If the final worst-of level is below the protection barrier, redemption is reduced proportionally by the final worst-of performance.

## Monte Carlo Methodology

The autocallable workflow uses the newer DerivaPro Monte Carlo path engine. The engine simulates correlated geometric-Brownian-motion style equity paths under user-supplied market assumptions:

- Initial spot or indexed levels.
- Risk-free rate.
- Dividend yield.
- Flat volatility per underlying.
- Maturity.
- Number of paths.
- Number of time steps.
- Random sequence type.
- Equicorrelation matrix for basket notes.

For a basket with multiple underlyings, the model constructs a flat equicorrelation matrix with the user-specified pairwise correlation.

## Key Outputs

The page reports:

- Present value.
- Present value as a percentage of notional.
- Monte Carlo standard error.
- Standard error as a percentage of notional.
- Autocall probability.
- Protection-breach probability.
- Average number of coupons paid.
- Worst-of final-level mean and selected percentiles.
- Simulation configuration used for the run.

These outputs are designed to support product review, model diagnostics, and future report generation.

## Current Assumptions and Limitations

The current implementation has important simplifications:

- Volatility is flat by underlying; the workflow does not yet consume a calibrated volatility surface.
- Basket dependence uses flat equicorrelation, not a full user-supplied correlation matrix.
- Observation schedules are entered as year fractions, not business-day adjusted calendar schedules.
- Interest rates and dividends are deterministic flat inputs.
- Greeks and scenario grids for the Phoenix payoff are planned but not yet implemented in the product-standard page.
- Issuer credit spread, funding spread, secondary-market liquidity, and margin adjustments are outside the current valuation.
- Regulatory, accounting, tax, and suitability analysis are outside the model scope.

## Planned Extensions

Recommended next enhancements include:

- Step-down autocall barriers.
- Snowball and accumulator-style coupon variants.
- Callable yield notes.
- Reverse convertible and capital-at-risk certificate variants.
- Principal-protected autocallable notes.
- Full correlation-matrix input for baskets.
- Calendar-based observation schedule generation.
- Scenario analysis across spot, volatility, correlation, and rates.
- Convergence diagnostics for path count, time steps, and random-sequence selection.
- Product-specific reporting templates linked to saved pricing runs.

## Validation Considerations

Before using this workflow for production or commercial valuation, users should independently validate:

- Payoff interpretation against the actual legal termsheet.
- Observation schedule and barrier conventions.
- Market data inputs and calibration choices.
- Monte Carlo convergence under the selected path count and time-step grid.
- Sensitivity to volatility, correlation, and dividend assumptions.
- Any issuer-specific credit, funding, or liquidity adjustments required by the valuation policy.
