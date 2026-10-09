# Digital Coupon / Contingent Income Note Methodology

## Product Scope

This note covers the DerivaPro first-pass workflow for contingent income notes. Coupons are paid on scheduled observation dates when the reference level is above the coupon barrier. Principal is protected unless the final level breaches the protection barrier.

## Pricing Framework

The implementation simulates reference paths at scheduled coupon observation dates. For each observation:

- A coupon is paid if the observed level is at or above the coupon barrier.
- If memory coupon is enabled, missed coupons accrue and are paid once a later coupon condition is satisfied.
- At maturity, full notional is redeemed if the final level is above the protection barrier.
- If the protection barrier is breached, redemption is reduced in proportion to final underlying performance.

The present value is the sum of discounted expected coupon and redemption cash flows.

## Key Outputs

- Present value and PV as a percentage of notional.
- Coupon payment probability.
- Average coupon count.
- Protection breach probability.
- Expected redemption.
- Final-level distribution diagnostics.

## Current Limitations

- Single-underlying path simulation.
- Observation dates are generated from coupon frequency rather than a business-day schedule.
- Flat volatility, rate, and dividend assumptions.
- No autocall feature; autocallable Phoenix notes are handled separately.

## Planned Extensions

- Calendar-based observation schedules.
- Basket and worst-of contingent income notes.
- Step-down coupon barriers.
- Scenario grids across spot, volatility, and coupon barrier.
