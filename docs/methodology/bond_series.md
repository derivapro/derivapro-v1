# Bond Series Methodology

## Scope

Bond series analytics cover multiple related bonds or maturities valued together as one issuer-level program. This is useful for serial bonds, municipal-style programs, and fixed-income books where aggregate cash-flow and maturity exposure matter.

## Pricing Framework

Each row in the bond series table represents one bond maturity with principal, coupon rate, and redemption price. DerivaPro values each row as a level coupon bond using the shared generic bond engine, then aggregates:

- Series present value.
- Aggregate yield.
- Weighted-average maturity.
- Per-maturity model PV and model price.

## Input Format

The current first-wave table format is:

```text
maturity|principal|coupon_rate|redemption_pct; maturity|principal|coupon_rate|redemption_pct
```

Example:

```text
2028-08-13|1000000|0.040|100; 2030-08-13|1500000|0.045|100
```

## Analytics

The page reports series-level PV, aggregate yield, weighted-average maturity, number of bonds, per-maturity diagnostics, and parallel rate-shock scenario PV.

## Current Assumptions

- Each series component is currently treated as a non-callable level coupon bond.
- Coupon frequency is semiannual in the first-wave implementation.
- Callable bond series, sinking series, premium/discount series pricing, and tax-specific conventions are planned extensions.
- Accrued interest and business-day adjustment are simplified.

## Validation Priorities

- Add an editable grid for series rows.
- Add callable bond series support.
- Add aggregate cash-flow export.
- Add series-level duration, convexity, and key-rate risk.
