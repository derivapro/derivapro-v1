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

## Current Limitations

- Single-underlying workflow only.
- Final barrier observation only; intraday or continuous knock-in monitoring is not yet modeled.
- Flat volatility, rate, and dividend assumptions.
- Issuer credit, funding, bid/ask, tax, and suitability adjustments are outside scope.

## Planned Extensions

- Calendar-based coupon and maturity schedules.
- Alternative knock-in monitoring conventions.
- Basket and worst-of reverse convertible variants.
- Issuer credit/funding spread overlays.
- Product-specific report template.
