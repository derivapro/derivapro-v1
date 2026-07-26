**Description:**
A non-callable fixed-rate amortizing bond repays principal gradually over the life of the instrument while paying coupons on the outstanding balance. Because principal declines over time, coupon cash flows and duration behavior differ from bullet bonds.

The price is computed by discounting each scheduled coupon-plus-principal cash flow:

* $P = \sum_{i=1}^{N} \frac{CF_i}{(1+r_i)^{t_i}}$

where $CF_i$ includes both coupon and amortization at period $i$.

This structure is commonly used in mortgage-style and project-finance cash flow profiles where repayment is not concentrated at maturity.

**Instructions:**
Use this workflow to price fixed-rate amortizing structures and test curve sensitivity.

1. Enter bond setup fields, including amortization schedule assumptions and payment frequency.
2. Provide spot tenors/rates and choose interpolation and compounding settings.
3. Add optional shock scenarios to test valuation and cash flow sensitivity.
4. Run pricing to view present value outputs based on declining principal balances.
5. Compare results across scenarios for risk review and model validation.