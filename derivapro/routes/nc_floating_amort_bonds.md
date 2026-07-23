**Description:**
A non-callable floating-rate amortizing bond combines periodic principal amortization with rate-resetting coupons. As the outstanding balance declines, both coupon size and interest-rate sensitivity evolve over time.

Each period cash flow typically follows:

* $CF_i = Amortization_i + Outstanding_{i-1} \times (Index_i + Spread) \times \Delta_i$

The instrument value is the discounted sum of all scheduled amortization and floating coupon cash flows.

This structure is frequently used in loan-style products where principal paydown and floating exposure must be modeled together.

**Instructions:**
Use this workflow to value amortizing floating-rate bonds and evaluate scenario impacts.

1. Enter amortization and instrument setup inputs (dates, notional, schedule, frequency, spread).
2. Provide term structure and index/reset assumptions for floating coupon projection.
3. Choose interpolation and compounding methods for discounting.
4. Add optional shocks for risk and sensitivity analysis.
5. Run pricing and compare base versus stressed valuation outcomes.