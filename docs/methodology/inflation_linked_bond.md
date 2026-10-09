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

## Detailed Methodology Notes

### Index Ratio

Inflation-linked bonds adjust real cash flows using an index ratio:

```text
Index ratio(date) = Reference CPI(date) / Base CPI
```

For many sovereign linkers, the reference CPI is lagged and interpolated. The current DerivaPro implementation uses a simplified projected ratio:

```text
Projected ratio_i =
    Current CPI / Base CPI
    * exp(projected_inflation_rate * max(T_i - lag, 0))
```

### Indexed Coupon and Principal

The indexed coupon is:

```text
Coupon_i = Notional * IndexRatio_i * real_coupon_rate * alpha_i
```

At maturity:

```text
Principal_T = Notional * IndexRatio_T
```

If the principal floor is enabled:

```text
Principal_T = Notional * max(IndexRatio_T, 1.0)
```

Some jurisdictions floor only principal, while coupons continue to use the actual index ratio. This distinction should be made configurable.

### Real Yield Versus Nominal Yield

Inflation-linked bonds may be quoted on real yield or real clean price. In a real-yield framework:

```text
Real price = sum_i RealCF_i / (1 + y_real)^(tau_i)
```

Nominal cash flows are obtained by multiplying real cash flows by projected index ratios.

### Breakeven Inflation

A simple breakeven proxy is:

```text
Breakeven = nominal zero rate - real zero rate
```

This is only an approximation. Production breakeven analysis should include seasonality, inflation risk premium, carry, roll-down, liquidity effects, and index lag.

### Alternative Methodologies

Production linker analytics should support country-specific CPI rules, monthly CPI interpolation, indexation lag by jurisdiction, real clean/dirty price, accrued inflation, deflation floor conventions, inflation curve bootstrapping, zero-coupon inflation swap inputs, and seasonality adjustments.

### Additional Risk Measures

Recommended analytics:

- Real DV01.
- Nominal DV01.
- Inflation DV01.
- Breakeven sensitivity.
- Carry and roll-down.
- CPI fixing sensitivity.
