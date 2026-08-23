# Amortizing / Step-Up / Sinking Bond Methodology

## Scope

This page covers bonds whose coupon rate and outstanding principal may vary over time. Supported first-wave structures include:

- Bullet bonds with scheduled step-up coupons.
- Straight-line amortizing bonds.
- Bonds with explicit sinking-fund principal schedules.
- Combinations of coupon schedules and principal paydown schedules.

## Pricing Framework

The workflow uses the shared DerivaPro generic bond cash-flow engine:

1. Build contractual payment dates from valuation date, maturity date, and payment frequency.
2. Select the effective coupon rate for each payment date from the optional coupon schedule.
3. Calculate coupon on opening outstanding notional.
4. Apply scheduled principal reduction from the selected amortization type.
5. Discount coupon and principal cash flows on the supplied zero-rate curve.

The model PV is the sum of all discounted cash flows.

## Schedule Inputs

Coupon schedules use `YYYY-MM-DD:rate` entries. The latest effective entry on or before a payment date is applied.

Sinking schedules use `YYYY-MM-DD:pct_original` entries. Each percentage represents principal reduction as a percentage of original notional.

## Risk Measures

The page reports model PV, model price, yield to maturity, duration, convexity, DV01, and parallel rate-shock scenario PV.

## Current Assumptions

- Sinking payments are aligned to coupon periods.
- Accrued interest, settlement lag, ex-coupon periods, business-day adjustment, and odd stubs are simplified.
- Principal reduction is floored at remaining outstanding balance.
- No embedded call/put optionality is included on this page; callable/putable structures are handled separately.

## Validation Priorities

- Add explicit cash-flow import/export.
- Add richer amortization schedule editing.
- Add key-rate duration and non-parallel curve scenarios.
- Compare generated cash flows and risk statistics against controlled benchmark cases.
