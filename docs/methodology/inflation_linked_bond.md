# Inflation-Linked Bond Methodology

## Scope

This note covers the DerivaPro first-pass Inflation-Linked Bond workflow under Fixed Income Extensions. The workflow is aligned conceptually with the local methodology references `Math/InflationBonds.html`, `Math/InflationDerivs.html`, and `Math/InflCurve.html`, while keeping public repo documentation original and implementation-specific.

## Product Description

An inflation-linked bond adjusts coupon and redemption cash flows using an inflation index ratio. The instrument is commonly used to obtain real-rate exposure and inflation compensation through indexed principal and coupon payments.

Supported conventions in the current workflow:

- Fixed real coupon rate.
- User-supplied base CPI and current CPI.
- User-supplied projected annual inflation rate.
- User-supplied indexation lag in months.
- Optional principal floor at par.
- Annual, semiannual, or quarterly coupon frequency.
- ACT/360, ACT/365, or 30/360 coupon day count.
- Real discount curve and nominal curve inputs.
- Real-rate and inflation scenario shocks.

## Pricing Framework

The current index ratio is:

```text
Current index ratio = Current CPI / Base CPI
```

For each future payment date, the workflow projects an indexed ratio after applying the indexation lag:

```text
Projected index ratio_i =
    Current index ratio * exp(Projected inflation rate * max(T_i - lag, 0))
```

If the principal floor is enabled, the projected index ratio is floored at `1.0` for principal protection purposes.

Coupons and principal are projected as:

```text
Coupon_i = Notional * Projected index ratio_i * Real coupon rate * accrual_i
Principal_T = Notional * Projected index ratio_T
```

The present value is the sum of indexed cash flows discounted on the supplied real discount curve.

The page also reports a simple breakeven proxy:

```text
Breakeven proxy = Nominal zero rate at maturity - Real zero rate at maturity
```

## Outputs

The page reports:

- Real discounted PV.
- Current index ratio.
- Projected maturity index ratio.
- Breakeven inflation proxy.
- Real-rate and inflation shock scenario PV.
- Period-level projected index ratios, coupons, principal, and discounted PVs.

## Current Assumptions

- CPI is projected from a flat user-supplied inflation rate.
- Interpolated daily/monthly CPI mechanics are not yet modeled.
- Clean/dirty price, accrued inflation, settlement conventions, and regional index rules are not yet modeled.
- Seasonality, deflation floors by jurisdiction, and market quote conventions are simplified.
- The nominal curve is used only for the breakeven proxy.

## Recommended Validation

- Add country-specific CPI lag and interpolation conventions.
- Add accrued inflation and clean/dirty indexed price.
- Add inflation curve bootstrapping and zero-coupon inflation swap inputs.
- Compare against TIPS, linker, and inflation-swap benchmark cases.
- Add real DV01, inflation DV01, breakeven sensitivity, and carry/roll-down analytics.
