# Floating-Rate Bond Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for non-callable floating-rate bond valuation and parallel rate-shock analysis.

## 2. Product Definition

A floating-rate bond pays coupons linked to a reference index plus or minus a contractual spread and repays principal at maturity. Coupon amounts reset over the bond's life, so valuation depends on both discount-curve and index-curve assumptions.

For accrual period `i`, the coupon is approximated by:

$$C_i = N \times (I_i + s) \times \Delta_i$$

where `N` is notional, `I_i` is the projected index rate, `s` is the contractual spread, and `Δ_i` is the accrual fraction.

## 3. Valuation Framework

The model builds separate zero curves from the dated spot-rate and index-rate inputs. The index curve projects floating coupons, while the spot curve discounts projected coupons and principal redemption.

The present value is:

$$PV = \sum_{i=1}^{n} C_i D(t_i) + N D(T)$$

where `D(t_i)` is the discount factor for payment date `t_i` and `T` is maturity.

## 4. User Inputs

| Input group | Inputs |
|---|---|
| Bond terms | Issue date, maturity date, notional, contractual spread, payment tenor |
| Discount curve | Evaluation date, spot dates, spot rates |
| Index curve | Index dates, index rates |
| Conventions | Currency, day count, calendar, interpolation, compounding, compounding frequency |
| Scenarios | Parallel rate shocks entered as a comma-separated list |

## 5. Outputs

For each requested shock, the workflow reports NPV, clean price, yield to maturity, duration, dollar duration, convexity, and dollar convexity.

## 6. Assumptions and Limitations

- Discount and index curves are constructed directly from user-supplied zero-rate pillars.
- Rate shocks are parallel shifts applied to the supplied rates.
- Historical coupon fixings, payment lags, reset lags, caps, floors, and fallback provisions are simplified or not explicitly modeled.
- The reference index is represented by the supplied index curve rather than a full market instrument bootstrap.
- Credit spread, liquidity, funding, tax, and transaction-cost effects are not separately modeled beyond the entered contractual spread.
- Yield, duration, and convexity measures for floating-rate instruments depend on the model conventions and should be interpreted with the projected reset behavior in mind.

## 7. Validation Expectations

Production use should include benchmark comparison against an independent bond engine, curve-input validation, reset-date and accrual testing, shock-sign tests, and regression coverage for price and risk measures.

## 8. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/ncfloatingbonds.html` | Floating-rate bond interface |
| `derivapro/routes/bonds.py` | Request handling and valuation orchestration |
| `derivapro/models/mdls_bonds.py` | QuantLib-backed floating-rate bond valuation |
