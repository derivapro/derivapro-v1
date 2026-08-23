# Level Coupon Bond Methodology

## Scope

This page covers standard fixed-rate bullet bonds with level periodic coupons and principal redemption at maturity. It is the product-standard replacement path for the legacy fixed-rate bond page.

## Pricing Framework

DerivaPro generates contractual coupon dates from valuation date, maturity date, and coupon frequency. For each payment period:

1. The accrual factor is calculated from the selected day-count convention.
2. Coupon cash flow equals outstanding notional multiplied by coupon rate and accrual factor.
3. Principal is paid at maturity.
4. Each cash flow is discounted using the supplied zero-rate discount curve.

The model PV is the sum of discounted coupon and principal cash flows.

## Yield and Risk Measures

The page solves yield to maturity by finding the single annual yield that equates generated cash flows to the supplied market clean-price reference.

Duration, convexity, and DV01 are first-pass cash-flow measures calculated from discounted cash-flow weights. Rate scenarios are produced by applying parallel shifts to the supplied zero-rate curve.

## Current Assumptions

- The current implementation uses deterministic fixed cash flows.
- Accrued interest and settlement-date conventions are simplified.
- Holiday calendars, ex-coupon dates, odd first/last coupons, and issuer-specific settlement rules are not yet modeled.
- Credit spread, z-spread, G-spread, I-spread, and OAS are planned extensions.

## Validation Priorities

- Compare PV, yield, duration, and convexity against QuantLib or desk benchmark cases.
- Add clean/dirty price and accrued-interest decomposition.
- Add key-rate duration and curve twist scenarios.
- Add import/export support for cash-flow schedules.
