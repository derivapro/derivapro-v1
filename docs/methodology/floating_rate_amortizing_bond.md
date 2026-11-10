# Floating-Rate Amortizing Bond Methodology

## 1. Purpose

This document describes the DerivaPro methodology for non-callable floating-rate amortizing bond valuation and parallel curve-shock analysis.

## 2. Product Definition

A floating-rate amortizing bond combines scheduled principal reduction with coupons linked to a reference index plus a contractual spread. Both coupon size and rate sensitivity evolve as principal declines.

For payment period `i`:

$$CF_i = A_i + N_{i-1}(I_i+s)\Delta_i$$

where `A_i` is principal amortization, `N_{i-1}` is opening principal, `I_i` is the projected reference rate, `s` is contractual spread, and `Δ_i` is the accrual fraction.

## 3. Valuation

The index curve projects coupon rates and the discount curve discounts projected coupons and principal cash flows:

$$PV = \sum_{i=1}^{n} CF_i D(t_i)$$

Notional amounts and notional dates define when each outstanding balance becomes effective.

## 4. Inputs

| Group | Inputs |
|---|---|
| Bond and amortization | Issue date, maturity date, spread, payment tenor, notionals, notional dates |
| Discount curve | Evaluation date, spot dates, spot rates |
| Index curve | Index dates, index rates |
| Conventions | Currency, day count, calendar, interpolation, compounding, frequency |
| Scenarios | Parallel shocks entered as a comma-separated list |

## 5. Outputs

For each shock, the workflow reports NPV, clean price, yield to maturity, duration, dollar duration, convexity, and dollar convexity. An optional generated assessment can provide a narrative review.

## 6. Assumptions and Limitations

- Notional dates and amounts must define a valid declining principal schedule within the bond life.
- Discount and index curves are built from user-entered zero-rate pillars.
- Historical fixings, reset/payment lags, caps, floors, and fallback provisions are simplified or not modeled.
- Shocks are parallel shifts rather than key-rate or curve-shape scenarios.
- Prepayment, default, liquidity, taxes, funding, and transaction costs are not explicitly modeled.

## 7. Validation Expectations

Production use should include independent coupon and principal reconciliation, reset-date testing, notional-schedule validation, curve benchmarks, shock-sign tests, and regression coverage for all reported risk measures.

## 8. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/ncfloatingamortbonds.html` | Product interface |
| `derivapro/routes/bonds.py` | Request handling and orchestration |
| `derivapro/models/mdls_bonds.py` | QuantLib-backed bond valuation |
