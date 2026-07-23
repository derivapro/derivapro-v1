**Description:**
A non-callable floating-rate bond pays coupons linked to a reference index (for example, SOFR/LIBOR-style rate inputs) plus or minus a spread. Coupon amounts reset on scheduled dates, so valuation depends on both discount factors and forward/index assumptions.

At each reset period, coupon cash flow can be approximated as:

* $C_i = Notional \times (Index_i + Spread) \times \Delta_i$

The bond value is the discounted sum of floating coupons and principal redemption at maturity.

Floating-rate instruments are useful for managing duration exposure and assessing rate-reset dynamics under different curve environments.

**Instructions:**
Use this page to value floating-rate non-callable bonds under configurable market data.

1. Enter bond setup inputs (dates, notional/face value, spread, payment frequency, and conventions).
2. Provide spot curve and index/reset inputs required for projected floating coupons.
3. Select interpolation and compounding methods for curve construction.
4. Optionally apply shocks to test price sensitivity to rate moves.
5. Run the calculator and review valuation outputs across base and stressed assumptions.