# Level Coupon Bond Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro level coupon bond workflow under the Fixed Income workspace. A level coupon bond is a deterministic fixed-income instrument that pays a fixed coupon on a fixed schedule and returns principal at maturity. It is the baseline methodology for many other bond workflows: amortizing bonds, callable bonds, asset swaps, bond forwards, and bond portfolios all depend on the same core cash-flow, discounting, price/yield, and risk-statistic framework.

This methodology is aligned conceptually with the local reference material in:

- `Math/lcb.html` for general bond price/yield and risk-statistic formulas.
- `Math/GenericBonds.html` for generic date generation, level coupon versus day-count cash-flow generation, and term-structure discounting.
- `Math/YTM.html` for yield-to-maturity solving.
- `Math/BondTable.html` for cash-flow table and bond-statistic design patterns.

The public DerivaPro documentation is implementation-specific and intentionally written in original wording rather than copied from those references.

## 2. Instrument Economics

The current DerivaPro level coupon bond supports:

- Valuation date.
- Maturity date.
- Face value / notional.
- Fixed annual coupon rate.
- Market clean-price reference as a percentage of par.
- Coupon frequency: annual, semiannual, or quarterly.
- Day-count convention: `30/360`, `ACT/360`, or `ACT/365`.
- User-supplied zero-rate discount curve.
- Parallel rate-shock scenario size.

The product is currently a non-callable bullet bond. There is no embedded optionality, no floating-rate index, no inflation indexation, and no stochastic credit default process in this page.

## 3. Cash-Flow Generation

Let:

\[
\begin{aligned}
N       &= \text{face value / notional} \\
c       &= \text{annual coupon rate} \\
m       &= \text{coupon payments per year} \\
d_i     &= \text{the } i\text{-th coupon payment date} \\
\alpha_i &= \text{accrual factor for period } i \\
DF_i    &= \text{discount factor to } d_i
\end{aligned}
\]

DerivaPro first creates a payment schedule from valuation date to maturity using the selected frequency. For each coupon period, the accrual factor is calculated from the previous coupon date to the current coupon date using the selected day-count convention.

For day-count-based coupons:

\[
\text{Coupon}_i = N \cdot c \cdot \alpha_i
\]

For a regular semiannual level-coupon convention, this is often equivalent to:

\[
\text{Coupon}_i = \frac{N \cdot c}{m}
\]

DerivaPro currently uses the day-count accrual formulation because it generalizes more naturally to irregular periods, custom schedules, amortizing principal, and fixed-leg swap style products.

At maturity, principal is added:

\[
\begin{aligned}
\text{Principal}_i &= 0, && i < T \\
\text{Principal}_T &= N, && \text{at maturity} \\
CF_i &= \text{Coupon}_i + \text{Principal}_i
\end{aligned}
\]

## 4. Discount-Curve Method

The page accepts a continuously compounded zero-rate curve through comma-separated tenors and zero rates. The internal curve object linearly interpolates zero rates by maturity and returns discount factors:

\[
DF(t) = e^{-z(t)t}
\]

where `z(t)` is the interpolated continuously compounded zero rate for maturity `t`.

The model present value is:

\[
PV = \sum_i CF_i \cdot DF(t_i)
\]

where `t_i` is the time in years from valuation date to payment date.

The model price as a percentage of par is:

\[
\text{Model Price (\% of Par)} = \frac{PV}{N} \times 100
\]

## 5. Price/Yield Relationship

The yield-to-maturity calculation solves a single annual yield `y` such that the present value of generated cash flows equals the supplied market clean-price reference:

\[
\text{Market Price} = \sum_i \frac{CF_i}{(1+y)^{\tau_i}}
\]

where:

\[
\begin{aligned}
\text{Market Price} &= N \cdot \frac{\text{market clean price pct}}{100} \\
\tau_i &= \text{year fraction from valuation date to cash-flow date } i
\end{aligned}
\]

DerivaPro currently solves this equation using a bounded root search. If the cash-flow stream and market price do not produce a valid root in the supported search interval, the page returns `n/a` rather than forcing an unstable yield.

### Yield Convention Note

The current workflow uses a single annual effective yield formulation. Market conventions can differ materially across regions and instruments, including street yield, true yield, money-market yield, semiannual bond-equivalent yield, Japanese simple yield, and treasury-specific conventions. These are planned extensions.

## 6. Risk Statistics

The page calculates first-pass deterministic cash-flow risk measures.

### Present-Value Weighted Duration

The current duration calculation is a discounted cash-flow weighted average maturity:

\[
\begin{aligned}
\text{Duration} &= \frac{\sum_i t_i PV_i}{PV} \\
PV_i &= CF_i \cdot DF(t_i)
\end{aligned}
\]

This is a practical first-pass duration proxy. In a production bond analytics engine, the exact definition should be selected by quote convention: Macaulay duration, modified duration, effective duration, spread duration, or key-rate duration.

### Convexity

The current convexity proxy is:

\[
\text{Convexity} = \frac{\sum_i t_i(t_i+1)PV_i}{PV}
\]

This mirrors the idea that convexity is the second-order sensitivity of price to yield, but it should eventually be refined to support convention-specific formulas and finite-difference checks.

### DV01

DerivaPro reports a first-pass DV01:

\[
DV01 = \frac{\text{Duration} \cdot PV}{10{,}000}
\]

This approximates the dollar value of a one-basis-point rate move. For production use, finite-difference DV01 should be added:

\[
DV01_{\text{fd}} = \frac{PV_{\text{down 1bp}} - PV_{\text{up 1bp}}}{2}
\]

## 7. Scenario Analysis

The page applies parallel shifts to the zero-rate curve:

\[
\begin{aligned}
z_{\text{up}}(t) &= z(t) + \frac{\text{shock}_{bp}}{10{,}000} \\
z_{\text{down}}(t) &= z(t) - \frac{\text{shock}_{bp}}{10{,}000}
\end{aligned}
\]

It then reprices the same cash flows under the shifted curves. This produces base, up-shock, and down-shock PVs.

## 8. Assumptions in the Current Implementation

- Cash flows are deterministic.
- The bond is non-callable and non-putable.
- Principal is repaid fully at maturity.
- Discounting uses a user-supplied continuously compounded zero curve.
- Accrued interest is not yet separated from clean price.
- The clean-price input is used as a price reference for yield solving.
- Business-day calendars and holiday adjustment are simplified.
- Odd first and odd last coupon periods are not yet explicitly configurable.
- Ex-coupon / ex-dividend treatment is not included.
- Credit spread and default risk are not modeled on this page.

## 9. Alternative Methodologies

### Yield-Based Valuation

Instead of curve discounting, a bond can be valued directly from a quoted yield:

\[
P(y) = \sum_i \frac{CF_i}{(1+y)^{\tau_i}}
\]

This is useful when market quotes are expressed primarily as yield rather than curve PV.

### Curve Plus Spread Valuation

For credit-risky bonds, discounting may use a risk-free curve plus a constant spread:

\[
DF_{\text{spread}}(t) = e^{-(z(t)+s)t}
\]

The spread can be interpreted as z-spread under deterministic cash flows.

### Key-Rate Risk

Rather than shifting the entire curve in parallel, key-rate duration shocks one tenor bucket at a time and interpolates the impact across the curve. This is more useful for hedging and book-level risk.

### Full Street-Convention Bond Analytics

A production bond analytics engine should support settlement lag, accrued interest, clean/dirty conversion, calendars, business-day conventions, end-of-month rules, odd coupons, ex-coupon behavior, and regional yield conventions.

## 10. Validation Plan

Recommended validation cases:

1. Par bond at flat curve: coupon approximately equal to yield should price near par.
2. Zero-coupon bond special case: PV should equal principal times discount factor.
3. Premium/discount bonds: yield should move inversely with price.
4. Duration monotonicity: longer maturities should generally produce higher duration, all else equal.
5. Coupon sensitivity: higher coupons should reduce duration relative to lower coupons with same maturity.
6. Curve-shock symmetry: small up/down shocks should produce reasonable first-order and second-order behavior.
7. Benchmark comparison against QuantLib or desk/vendor examples.

## 11. Current DerivaPro Status

Implemented now:

- Cash-flow generation.
- Curve discounting.
- Market clean-price yield solve.
- Duration, convexity, DV01.
- Parallel rate-shock scenarios.
- Cash-flow table display.

Planned:

- Clean/dirty price decomposition.
- Accrued interest.
- Convention-specific yield calculations.
- Key-rate duration.
- Spread measures.
- Calendar and business-day controls.
