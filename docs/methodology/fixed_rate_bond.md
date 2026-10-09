# Fixed-Rate Bond Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro **Fixed-Rate Bond** workflow under the Fixed Income workspace (`/noncallable-bonds/fixed_bonds`, backed by `NCFixedBonds` in `derivapro/models/mdls_bonds.py`). The instrument is a non-callable, bullet-repayment fixed-rate bond: it pays a level coupon on a regular schedule and returns the full face value at maturity. Pricing is engine-driven through QuantLib rather than a closed-form formula, using a discount curve bootstrapped from user-supplied spot rates.

This page is distinct from the "Level Coupon Bond" methodology under Fixed Income Extensions, which prices from a market clean-price quote and solves for yield. The Fixed-Rate Bond workflow documented here instead builds its own zero curve from scratch and reads NPV, clean price, and yield directly off the priced QuantLib bond object.

## 2. Instrument Economics

The workflow accepts:

- Valuation date (`value_date`).
- A user-supplied zero-rate curve: comma-separated spot dates and spot zero rates.
- One or more parallel rate shocks, expressed in basis points, applied simultaneously to every point on the curve.
- Day-count convention: `ActualActual`, `Thirty360`, `Actual360`, or `Actual365Fixed`.
- Calendar: United States, TARGET, United Kingdom, or China.
- Interpolation method for the zero curve: Linear, LogLinear, or Cubic.
- Compounding convention: Compounded, Simple, or Continuous, with a compounding frequency (Annual, Semiannual, Quarterly, Monthly, or Daily).
- Issue date, maturity date, coupon frequency (tenor), fixed annual coupon rate, and notional (face value).

The product is a plain-vanilla bullet bond: no call/put optionality, no floating-rate leg, no amortization schedule (that case is handled by the separate Fixed Amortizing Bonds workflow), and no credit-default modeling.

## 3. Zero-Curve Construction

DerivaPro builds one discount curve per requested shock scenario. Given spot dates \(d_1, \dots, d_n\) and spot zero rates \(z_1, \dots, z_n\), and a shock size \(s\) (in decimal, e.g. `0.01` for +100bp):

\[
z_i^{\text{shocked}} = z_i + s
\]

The shocked rates are passed to `ql.ZeroCurve`, which builds an interpolated zero curve under the selected interpolation, calendar, day-count, compounding, and compounding-frequency conventions. Each shock scenario therefore reprices the bond against its own independently-constructed curve rather than reusing a single base curve and bumping it afterward.

## 4. Bond Construction and Pricing

For each shock scenario, DerivaPro:

1. Builds a coupon schedule from issue date to maturity date using the selected tenor, with the `Following` business-day convention on both accrual start/end and a backward date-generation rule.
2. Constructs a `ql.FixedRateBond` with zero settlement days, the supplied face value, the schedule, the fixed coupon rate, and the day-count convention.
3. Attaches a `ql.DiscountingBondEngine` built from the shocked zero curve.

The bond's net present value and clean price are then read directly from the priced QuantLib object:

\[
NPV = \sum_i CF_i \cdot DF(t_i), \qquad \text{Clean Price} = \frac{NPV_{\text{ex-coupon}}}{N} \times 100
\]

where \(CF_i\) are the coupon and principal cash flows, \(DF(t_i)\) is the discount factor to payment date \(i\), and \(N\) is the face value. QuantLib handles the accrued-interest adjustment internally when computing clean price from dirty NPV.

## 5. Yield to Maturity

Unlike the Level Coupon Bond workflow (which solves yield from a market price quote), this engine asks QuantLib to imply the yield to maturity directly from the priced bond:

\[
NPV = \sum_i \frac{CF_i}{(1+y)^{\tau_i}}
\]

solved for \(y\) under the selected day-count, compounding, and compounding-frequency convention (`bond.bondYield(...)`). Because the bond was priced off the curve rather than a market quote, this YTM is the curve-implied yield for the given shock scenario, not a market-observed yield.

## 6. Risk Statistics

For each shock scenario, DerivaPro constructs an `ql.InterestRate` from the scenario's own YTM and asks QuantLib's `BondFunctions` for duration and convexity at that yield:

\[
\text{Duration} = \text{ql.BondFunctions.duration}(\text{bond}, y), \qquad
\text{Convexity} = \frac{\text{ql.BondFunctions.convexity}(\text{bond}, y)}{100}
\]

Dollar-scaled versions are also reported:

\[
\text{Dollar Duration} = \frac{NPV}{100} \times \text{Duration}, \qquad
\text{Dollar Convexity} = \frac{NPV}{100} \times \text{Convexity}
\]

These dollar measures scale the per-100-face risk statistic by the bond's actual NPV, giving a face-value-adjusted sensitivity rather than a percentage-of-par sensitivity.

## 7. Scenario Analysis

Every shock the user enters (a comma-separated list of basis-point offsets, e.g. `-100, 0, 100`) is priced independently and returned as its own row: NPV, Price, YTM, Duration, Dollar Duration, Convexity, and Dollar Convexity, keyed by the shock size in basis points. This lets a user see how each risk statistic moves under parallel curve shifts without repeated form submissions.

## 8. Assumptions in the Current Implementation

- Cash flows are deterministic; the bond is non-callable and non-putable.
- The discount curve is a flat-extrapolated interpolation of user-supplied spot rates — no bootstrapping from market instruments (swaps, futures, deposits) is performed.
- Shocks are applied in parallel across the entire curve; no key-rate or non-parallel shock support on this page (that is handled separately by the Interest Rate Curves workflow).
- Settlement lag is zero business days.
- Business-day adjustment uses the `Following` convention uniformly; no separate termination-date convention is exposed.
- Yield, duration, and convexity are curve-implied (from the priced bond), not solved against an independent market quote.
- Notional/face value must be strictly positive; the page validates this server-side and shows a corrective message rather than a raw pricing-library error.

## 9. Alternative Methodologies

### Market-Quote Yield Solve

Rather than reading YTM off a curve-priced bond, a market clean price could be supplied directly and yield solved against it (as the Level Coupon Bond workflow does). This is preferable when the objective is benchmarking a curve model against an observed market quote.

### Bootstrapped Discount Curve

Production term-structure construction typically bootstraps the zero curve from a ladder of money-market deposits, futures, and swap rates rather than accepting spot rates directly, which removes interpolation risk between sparse user-supplied points.

### Key-Rate Duration

Parallel-shock duration is a simple, widely-used risk measure but does not capture curve-shape risk. Key-rate duration, which shocks one tenor bucket at a time, is better suited to hedging a non-parallel curve exposure.

## 10. Validation Plan

Recommended validation cases:

1. Par bond at flat curve: with coupon rate approximately equal to the flat zero rate, clean price should be close to 100.
2. Zero-shock baseline should reproduce a plain discounted-cash-flow NPV calculable by hand for a short-maturity bond.
3. Duration monotonicity: longer-maturity bonds should show higher duration at a comparable coupon and yield.
4. Shock symmetry: small symmetric up/down shocks should produce NPV changes consistent with the reported duration and convexity to second order.
5. Negative notional/face value should be rejected with a validation message rather than an unhandled exception.
6. Cross-check NPV, clean price, and YTM against an independently constructed QuantLib script for at least one reference bond.

## 11. Current DerivaPro Status

Implemented now:

- User-supplied zero-curve construction with selectable interpolation, calendar, day-count, and compounding conventions.
- QuantLib-engine bond pricing (NPV, clean price).
- Curve-implied yield to maturity.
- Duration, convexity, and their dollar-scaled equivalents.
- Multiple simultaneous parallel rate-shock scenarios.
- Server-side validation rejecting non-positive notional.

Planned:

- Bootstrapped discount curves from market instruments.
- Non-parallel / key-rate shock scenarios on this page.
- Accrued interest and clean/dirty price breakdown in the displayed results.
- Market-quote-based yield solve as an alternative to curve-implied yield.
