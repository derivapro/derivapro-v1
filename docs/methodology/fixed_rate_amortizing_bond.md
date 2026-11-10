# Fixed-Rate Amortizing Bond Methodology

## 1. Purpose

This document describes the DerivaPro methodology for non-callable fixed-rate amortizing bond valuation and parallel curve-shock analysis.

## 2. Product Definition

A fixed-rate amortizing bond pays a fixed coupon on outstanding principal and repays principal gradually over its life. The declining balance changes both coupon cash flows and interest-rate sensitivity relative to a bullet bond.

For payment period `i`:

$$CF_i = N_{i-1} c \Delta_i + A_i$$

where `N_{i-1}` is opening principal, `c` is the fixed coupon rate, `Δ_i` is the accrual fraction, and `A_i` is scheduled principal amortization.

## 3. Valuation

The model discounts all scheduled coupon and principal cash flows using a zero curve constructed from user-supplied dated spot rates:

$$PV = \sum_{i=1}^{n} CF_i D(t_i)$$

The notional list represents the declining outstanding balance used by the amortizing bond engine.

## 4. Inputs

| Group | Inputs |
|---|---|
| Bond and amortization | Issue date, maturity date, fixed coupon, payment tenor, outstanding notionals |
| Curve | Evaluation date, spot dates, spot rates |
| Conventions | Day count, calendar, interpolation, compounding, frequency |
| Scenarios | Parallel shocks entered as a comma-separated list |

## 5. Outputs

For each shock, the workflow reports NPV, clean price, yield to maturity, duration, dollar duration, convexity, and dollar convexity. An optional generated assessment can provide a narrative review of the latest results.

## 6. Assumptions and Limitations

- Principal balances must align with the generated payment schedule expected by the model.
- Curves are built directly from user-entered zero-rate pillars.
- Shocks are parallel shifts rather than key-rate or curve-shape scenarios.
- Prepayment, default, liquidity, taxes, funding, and transaction costs are not explicitly modeled.
- Calendar and schedule behavior follows the conventions supported by the underlying pricing engine.

## 7. Validation Expectations

Production use should include independent cash-flow reconciliation, principal-runoff checks, bullet-bond boundary tests, curve benchmarks, shock-sign tests, and regression coverage for all reported risk measures.

## 8. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/ncfixedamortbonds.html` | Product interface |
| `derivapro/routes/bonds.py` | Request handling and orchestration |
| `derivapro/models/mdls_bonds.py` | QuantLib-backed bond valuation |
