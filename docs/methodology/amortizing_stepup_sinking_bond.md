# Amortizing / Step-Up / Sinking Bond Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro workflow for bonds with time-varying coupon rates and scheduled principal reduction. The page is intended to cover practical structures such as fixed-rate amortizing bonds, step-up or step-down coupon bonds, bonds with sinking-fund principal schedules, and hybrid structures with both changing coupons and declining notional.

The methodology is aligned conceptually with:

- `Math/GenericBonds.html` for generic bonds, coupon tables, sinking funds, payment-in-kind, and flexible cash-flow generation.
- `Math/lcb.html` for price/yield and risk-statistic formulas.
- `Math/BondTable.html` for cash-flow table structures and diagnostics.

## 2. Product Economics

The workflow supports:

- Valuation date.
- Maturity date.
- Original face value.
- Base coupon rate.
- Market clean-price reference.
- Payment frequency.
- Day-count convention.
- Principal schedule type: bullet, straight-line amortization, or explicit sinking schedule.
- Optional coupon schedule.
- Optional sinking/principal schedule.
- Discount curve and parallel rate-shock scenario.

The page is deterministic. It does not include embedded optionality, default risk, stochastic prepayment, floating coupon resets, or inflation indexation.

## 3. Schedule Table Conventions

DerivaPro uses compact schedule strings for first-wave review.

Coupon schedule:

```text
YYYY-MM-DD:coupon_rate, YYYY-MM-DD:coupon_rate, ...
```

Example:

```text
2026-08-13:0.045, 2029-08-13:0.055
```

The latest effective coupon rate on or before a payment date is applied. If no schedule entry applies, the base coupon rate is used.

Sinking schedule:

```text
YYYY-MM-DD:pct_original_notional, YYYY-MM-DD:pct_original_notional, ...
```

Example:

```text
2029-08-13:0.20, 2030-08-13:0.20, 2031-08-13:0.20
```

Each percentage is interpreted as a reduction of original notional, not current outstanding balance.

## 4. Cash-Flow Generation

Let:

```text
N_0       = original notional
N_i       = opening outstanding notional for period i
c_i       = effective coupon rate for period i
alpha_i   = day-count accrual factor
S_i       = scheduled principal reduction during period i
DF_i      = discount factor to payment date i
```

Coupon cash flow:

```text
Coupon_i = N_i * c_i * alpha_i
```

Principal cash flow depends on the selected amortization style.

### Bullet

```text
Principal_i = 0       for i < maturity
Principal_T = N_0     at maturity
```

### Straight-Line Amortization

If there are `M` payment dates:

```text
Principal_i = N_0 / M
```

The final period is adjusted to avoid over- or under-amortization due to rounding or schedule truncation.

### Sinking Schedule

For explicit sinking events:

```text
Principal_i = sum_j scheduled_pct_j * N_0
```

where event `j` falls inside payment period `i`. Outstanding balance evolves as:

```text
N_{i+1} = N_i - Principal_i
```

The total period cash flow is:

```text
CF_i = Coupon_i + Principal_i
```

## 5. Present Value and Yield

The present value is:

```text
PV = sum_i CF_i * DF_i
```

The model price is:

```text
Model price pct = PV / N_0 * 100
```

Yield to maturity is solved against the supplied market clean-price reference:

```text
Market price = sum_i CF_i / (1 + y)^(tau_i)
```

Because amortizing and sinking structures return principal over time, their yield, weighted-average life, duration, and convexity can differ substantially from a bullet bond with the same final maturity.

## 6. Risk Measures

The page currently reports:

- Model PV.
- Model price.
- Yield to maturity.
- Present-value weighted duration.
- Convexity proxy.
- DV01.
- Parallel rate-shock scenario PV.
- Period-level coupon, principal, outstanding balance, discount factor, and PV diagnostics.

For amortizing bonds, risk is distributed across the principal repayment schedule. Principal that returns earlier reduces weighted-average life and duration.

## 7. Assumptions in the Current Implementation

- Coupon schedule entries are effective-date based.
- Sinking schedule entries are aligned into payment periods by event date.
- Sinking percentages are percentages of original notional.
- Principal payments are capped at remaining outstanding balance.
- The final payment is adjusted so remaining outstanding principal is fully repaid for amortizing and sinking structures.
- Accrued interest, clean/dirty decomposition, holiday calendars, and business-day adjustment are simplified.
- Payment-in-kind and accreting-notional treatment are not yet implemented.
- Call/put optionality is handled on the callable/putable bond page, not here.

## 8. Alternative Methodologies

### Full Generic Bond Table Engine

A more complete implementation would allow separate tables for coupon rates, notional amounts, sinking fund amounts, PIK/accretion amounts, fixed payments, and custom cash-flow overrides. The engine would then generate a reconciled cash-flow table from these components.

### Amortization Schedule Import

For real-world loans, project bonds, and private credit instruments, amortization is often loaded as a schedule rather than generated from simple formulas. A production page should allow CSV import with validation.

### Spread Discounting

Credit-sensitive amortizing bonds should support discounting on:

```text
DF_spread(t) = exp(-(z(t) + s(t)) * t)
```

where `s(t)` may be a flat z-spread or a term structure of credit spreads.

### Prepayment or Extension Risk

Some amortizing assets include borrower optionality or prepayment behavior. Those require stochastic or scenario-based prepayment models and should not be priced as deterministic bonds.

## 9. Validation Plan

Recommended validation:

1. Bullet mode should match the level coupon bond page for equivalent inputs.
2. Straight-line amortization should fully reduce outstanding balance to zero.
3. Sinking schedule should reduce outstanding principal only on intended dates.
4. Increasing coupon schedule should increase PV relative to a flat lower coupon.
5. Earlier principal reduction should reduce duration.
6. Scenario PV should decrease under positive rate shocks for ordinary positive cash-flow bonds.
7. Cash-flow table totals should reconcile to model PV.

## 10. Current DerivaPro Status

Implemented now:

- Coupon schedule parsing.
- Sinking schedule parsing.
- Bullet, straight-line, and explicit sinking principal logic.
- Curve discounting.
- Yield solve.
- Duration, convexity, DV01.
- Parallel rate scenarios.

Planned:

- Rich editable schedule table.
- CSV import/export.
- PIK/accreting notional.
- Odd coupon periods.
- Clean/dirty price and accrued interest.
- Key-rate duration and spread measures.
