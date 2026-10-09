# Custom Structured Bond Methodology

## 1. Scope and Product Definition

The Custom Structured Bond page is a generic fixed-income cash-flow workbench. It is intended for deterministic bond-like instruments whose cash flows cannot be represented cleanly by a single bullet coupon, a simple amortization rule, or a standard floating-rate reset.

The current workflow is conceptually aligned with the generic bond framework in `Math/GenericBonds.html`, especially the use of flexible coupon, notional, sinking, and fixed-payment tables. It also uses the price/yield and risk-statistic concepts from `Math/lcb.html` and `Math/YTM.html`.

This page should be viewed as infrastructure for future product types, not only as a standalone calculator.

## 2. Supported Structure Components

The current page supports:

- Valuation date.
- Maturity date.
- Original face value.
- Fallback coupon rate.
- Market clean-price reference.
- Coupon schedule.
- Principal / sinking schedule.
- Fixed payment schedule.
- Payment frequency.
- Day-count convention.
- User-supplied discount curve.
- Parallel rate-shock scenario.

The schedule inputs are intentionally simple in this first-wave implementation.

## 3. Schedule Input Formats

### Coupon Schedule

```text
YYYY-MM-DD:coupon_rate, YYYY-MM-DD:coupon_rate, ...
```

The coupon rate effective on a payment date is the latest schedule entry whose date is on or before that payment date. If no entry applies, the fallback coupon rate is used.

### Principal / Sinking Schedule

```text
YYYY-MM-DD:pct_original_notional, YYYY-MM-DD:pct_original_notional, ...
```

Each amount is interpreted as a scheduled principal payment equal to the percentage multiplied by original notional.

### Fixed Payment Schedule

```text
YYYY-MM-DD:amount, YYYY-MM-DD:amount, ...
```

Fixed payments are added to the coupon and principal cash flow in the period in which the event date falls.

## 4. Cash-Flow Model

Let:

```text
N_0       = original notional
N_i       = opening notional for period i
c_i       = effective coupon rate
alpha_i   = accrual factor
P_i       = scheduled principal amount
F_i       = scheduled fixed payment amount
DF_i      = discount factor
```

Coupon:

```text
Coupon_i = N_i * c_i * alpha_i
```

Principal:

```text
Principal_i = scheduled principal amount during period i
```

Fixed payment:

```text
Fixed_i = sum fixed payments with event dates in period i
```

Total cash flow:

```text
CF_i = Coupon_i + Principal_i + Fixed_i
```

Outstanding notional updates after principal payment:

```text
N_{i+1} = N_i - Principal_i
```

The final period repays remaining outstanding notional under the current generic-bond implementation when a sinking-style structure is selected.

## 5. Present Value

The present value is the discounted sum of generated cash flows:

```text
PV = sum_i CF_i * DF(t_i)
```

Discount factors come from the supplied continuously compounded zero curve:

```text
DF(t) = exp(-z(t) * t)
```

## 6. Yield and Risk

Yield is solved against the market clean-price reference:

```text
N_0 * clean_price_pct / 100 = sum_i CF_i / (1 + y)^(tau_i)
```

The page reports:

- Model PV.
- Model price as percentage of original notional.
- Yield to maturity when a stable root exists.
- Duration.
- Convexity.
- DV01.
- Parallel rate-shock scenarios.
- Detailed cash-flow diagnostics.

## 7. Modeling Interpretation

The custom structured bond engine is suitable for deterministic promised cash flows. Examples include:

- Step coupon notes.
- Scheduled redemption notes.
- Bonds with special fixed payments.
- Bonds with custom principal profiles.
- Simplified project-bond cash-flow schedules.

It is not yet suitable for payoff structures whose cash flows depend on stochastic market paths, credit events, exercise decisions, index performance, or borrower behavior.

## 8. Assumptions in the Current Implementation

- Cash flows are deterministic.
- Schedule strings must be valid and date ordered conceptually, though the parser sorts them.
- Principal schedule percentages are based on original notional.
- Fixed payments are absolute currency amounts.
- Coupon calculations use opening notional for the period.
- Accrued interest is not separately calculated.
- Business-day adjustment, ex-coupon treatment, holidays, settlement lag, and odd stubs are simplified.
- No default, recovery, callable exercise, conversion feature, or inflation indexation is included.

## 9. Alternative Methodologies

### Full Freestyle Cash-Flow Table

A production implementation should allow explicit cash-flow rows:

```text
date, coupon, principal, fixed_payment, currency, discount_curve, spread
```

and discount each row directly.

### Component Table Method

A more structured alternative is to maintain separate tables for:

- Coupon rule.
- Notional rule.
- Principal repayment rule.
- PIK/accretion rule.
- Fixed payment rule.
- Optionality rule.

The engine then generates a reconciled cash-flow table.

### Scenario Cash-Flow Engine

For products whose cash flows depend on scenarios, the deterministic cash-flow table should be replaced by:

```text
Expected PV = average over scenarios of discounted scenario cash flows
```

This is required for callable, prepayable, credit-contingent, inflation-linked, and equity-linked instruments.

## 10. Validation Plan

Recommended validation:

1. Empty schedules should reduce to a level coupon bullet bond.
2. Coupon schedule should apply the latest effective rate.
3. Principal schedule should reduce outstanding notional correctly.
4. Fixed payments should appear only in intended periods.
5. Total undiscounted principal should reconcile to original notional.
6. Discounted cash-flow totals should reconcile to model PV.
7. Yield solve should reproduce the market price when plugged back into the cash-flow equation.

## 11. Current DerivaPro Status

Implemented now:

- Generic schedule parsing.
- Coupon, principal, and fixed-payment schedules.
- Curve PV.
- Yield solve.
- Duration, convexity, DV01.
- Parallel rate scenarios.

Planned:

- Editable table UI.
- Cash-flow import/export.
- Validation warnings for inconsistent schedules.
- PIK/accreting notional.
- Clean/dirty price.
- Spread curves.
- Linkage to portfolio holdings.
