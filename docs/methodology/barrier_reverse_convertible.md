# Barrier Reverse Convertible Methodology

## Product Scope

This note covers the DerivaPro first-pass workflow for barrier reverse convertible notes. The product pays a fixed coupon and exposes principal to downside performance if the reference underlying finishes below the protection barrier.

## Pricing Framework

The current implementation uses Monte Carlo simulation of a single equity-style reference asset under a flat geometric-Brownian-motion assumption. At maturity:

- Coupon is paid according to the stated annual coupon rate.
- Full notional is redeemed if the final reference level is at or above the protection barrier.
- If the final reference level is below the protection barrier, redemption is reduced in proportion to final underlying performance.

The present value is the risk-neutral expected discounted payoff.

## Key Outputs

- Present value and PV as a percentage of notional.
- Monte Carlo standard error.
- Barrier breach probability.
- Expected redemption.
- Expected coupon.
- Final-level distribution diagnostics.

## Analysis Framework

The product page includes an explicit, user-triggered risk review. Pricing can be run on its own; analysis modules are selected and run separately so users can control runtime and compare multiple review packages.

- Scenario analysis reprices the note under product-relevant combined or single-factor shocks, including volatility shock, protection-barrier step-up, carry compression, defensive terms, and rate shock.
- Driver sensitivity reprices the note under isolated shocks to volatility, protection barrier, coupon rate, and risk-free rate.
- P&L attribution applies a sequential downside-risk repricing case and reports the incremental PV contribution from each driver step.
- Payoff profile shows undiscounted redemption, coupon, and total payoff across selected terminal underlying levels.
- Risk diagnostics summarize initial cushion to barrier, estimated breach probability, expected redemption shortfall, and left-tail terminal level.

The application keeps a standard BRC risk-review package as the default, while allowing controlled customization of scenario package, shock sizes, payoff-profile range, payoff grid density, and optional visual summaries. This is intended to balance consistency across users with the flexibility needed for desk, review, and model-risk use cases.

The current implementation intentionally avoids spot-only shocks because this first-pass model treats the note at inception with terms normalized to initial reference level. Secondary-market spot sensitivity requires a separate current-reference-level input and will be added in a later model upgrade.

## Current Limitations

- Single-underlying workflow only.
- Final barrier observation only; intraday or continuous knock-in monitoring is not yet modeled.
- Flat volatility, rate, and dividend assumptions.
- Current spot and initial reference level are not yet separated for secondary-market valuation.
- Issuer credit, funding, bid/ask, tax, and suitability adjustments are outside scope.

## Planned Extensions

- Calendar-based coupon and maturity schedules.
- Alternative knock-in monitoring conventions.
- Basket and worst-of reverse convertible variants.
- Issuer credit/funding spread overlays.
- Product-specific report template.
