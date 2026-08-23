# Custom Structured Bond Methodology

## Scope

Custom structured bonds are configurable fixed-income instruments whose contractual cash flows do not fit a standard bullet, amortizing, or floating-rate template. The current page supports user-entered coupon schedules, principal schedules, and additional fixed-payment schedules.

## Pricing Framework

The product uses the generic bond cash-flow engine:

1. Build a payment schedule.
2. Apply a fallback coupon rate unless an effective coupon schedule is supplied.
3. Calculate coupons on outstanding notional.
4. Apply scheduled principal reductions where supplied.
5. Add fixed payment amounts whose event dates fall in each payment period.
6. Discount all generated cash flows with the supplied zero-rate curve.

## Supported Schedule Tables

- Coupon schedule: `YYYY-MM-DD:rate`.
- Principal / sinking schedule: `YYYY-MM-DD:pct_original`.
- Fixed payment schedule: `YYYY-MM-DD:amount`.

These lightweight table formats are intended for first-wave review. A richer grid editor and CSV import should be added before production use.

## Analytics

The page reports model PV, model price, yield to maturity, duration, convexity, DV01, cash-flow diagnostics, and parallel rate-shock scenario PV.

## Current Assumptions

- Cash flows are deterministic.
- Coupon and principal schedules are interpreted on effective/payment-date boundaries.
- Embedded optionality, stochastic rates, credit migration, and default risk are not included.
- Settlement, accrued interest, holiday calendars, and business-day rules are simplified.

## Validation Priorities

- Add table-based editing with validation warnings.
- Add fixed cash-flow import/export.
- Add PIK/accreting notional support.
- Add z-spread and credit-spread discounting.
