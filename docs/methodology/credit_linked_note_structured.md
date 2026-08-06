# Credit-Linked Note Methodology

## Product Scope

This note covers the DerivaPro first-pass workflow for credit-linked notes in the structured-products workspace. The product pays coupons while the reference credit survives and exposes principal to loss if a default event occurs before maturity.

## Pricing Framework

The current implementation uses a flat reduced-form default-intensity model:

- Default time is simulated from an exponential distribution using the user-supplied hazard rate.
- Coupons are paid on scheduled dates only while the note has survived.
- If default occurs before maturity, recovery is paid at default time.
- If no default occurs, full notional is redeemed at maturity.
- Cash flows are discounted using the flat risk-free rate.

## Key Outputs

- Present value and PV as a percentage of notional.
- Default probability.
- Survival probability.
- Expected loss.
- Average coupon count.
- Monte Carlo standard error.

## Current Limitations

- Flat hazard rate and flat recovery assumption.
- No calibrated credit curve or CDS bootstrap in this first-pass page.
- No issuer/reference-entity basis, counterparty risk, funding spread, or liquidity adjustment.
- No accrued coupon convention or legal settlement convention.

## Planned Extensions

- CDS-curve bootstrapping and survival-curve input.
- Coupon accrual and default settlement conventions.
- Multi-name and basket credit-linked notes.
- Spread and recovery sensitivity grids.
