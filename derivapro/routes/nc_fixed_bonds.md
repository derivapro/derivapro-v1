**Description:**
A non-callable fixed-rate bond pays a predetermined coupon over its life and repays principal at maturity. The bond value is the present value of all future cash flows discounted using the selected term structure assumptions.

For a fixed-rate bond, the clean price can be represented as:

* $P = \sum_{i=1}^{N} \frac{C_i}{(1+r_i)^{t_i}} + \frac{F}{(1+r_N)^{t_N}}$

where $C_i$ is the coupon cash flow at period $i$, $F$ is face value, and $r_i$ is the discount rate for tenor $t_i$.

This workflow supports scenario shocks and curve assumptions to evaluate pricing sensitivity, accrued interest impact, and valuation consistency.

**Instructions:**
Use this page to value a fixed-rate non-callable bond under your chosen market assumptions.

1. Enter bond setup inputs (issue/evaluation date, maturity, coupon, frequency, face value, and day count convention).
2. Provide curve inputs (spot tenors and spot rates) and select interpolation/compounding assumptions.
3. Optionally add shock values (parallel or custom list) to compare shocked versus base valuation.
4. Run the calculation to review price, yield-related outputs, and shocked valuation comparisons.
5. Use generated outputs for downstream analysis, reporting, or model validation checks.