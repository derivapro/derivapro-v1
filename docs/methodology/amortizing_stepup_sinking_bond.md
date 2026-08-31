# Structured Amortizing Bonds Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro workflow for bonds with time-varying coupon rates and scheduled principal reduction. The page is intended to cover practical structures such as fixed-rate amortizing bonds, step-up or step-down coupon bonds, bonds with sinking-fund principal schedules, and hybrid structures with both changing coupons and declining notional.

The methodology is aligned conceptually with:

- `Math/GenericBonds.html` for generic bonds, coupon tables, sinking funds, payment-in-kind, and flexible cash-flow generation.
- `Math/lcb.html` for price/yield and risk-statistic formulas.
- `Math/BondTable.html` for cash-flow table structures and diagnostics.

## 2. Product Economics

The workflow supports:

- Settlement / valuation date.
- Dated date.
- First coupon date after the dated date.
- Last coupon date before maturity.
- Maturity date.
- Original face value.
- Base coupon rate.
- Market clean-price reference.
- Yield-to-maturity input for street-style price-from-yield benchmarks.
- Payment frequency.
- Day-count convention.
- Principal schedule type: bullet, straight-line amortization, or explicit sinking schedule.
- Optional coupon schedule.
- Optional sinking/principal schedule.
- Discount curve and parallel rate-shock scenario.

The page is deterministic. It does not include embedded optionality, default risk, stochastic prepayment, floating coupon resets, or inflation indexation.

## 3. External Calculator Benchmark Inputs

For benchmarking against spreadsheet calculators and commercial fixed-income libraries, the page supports the main fields needed to reproduce a clean/dirty price workflow:

| Field | Role in valuation |
| --- | --- |
| Settlement / value date | Date from which future cash flows are discounted and accrued interest is calculated. |
| Dated date | Start of the first accrual period for a newly issued or recently dated bond. |
| First coupon date | First scheduled coupon date after the dated date. |
| Last coupon before maturity | Final regular coupon date before principal redemption. |
| Maturity date | Final contractual redemption date. |
| Coupon frequency | Number of coupon periods per year. |
| Accrual method | Day-count rule used for coupon accrual and accrued interest. |
| Opening notional amount | Calculated from original face value less prior principal payments. This is not an independent valuation input. |
| Principal payment amount | User-editable contractual principal reduction for each payment date; the final row redeems remaining principal. |
| Pricing basis | Either curve discounting or price from yield. |
| Yield to maturity | Required for price-from-yield benchmarking. |

The external calculator benchmark cases reviewed for DerivaPro use:

- Settlement date: 30 Aug 2026.
- Dated date: 20 Jun 2026.
- First coupon date: 20 Dec 2026.
- Last coupon before maturity: 20 Dec 2040.
- Maturity date: 20 Jun 2041.
- Coupon frequency: semiannual.
- Coupon rate: 5.00%.
- Yield to maturity: 6.00%.
- Accrual method: Actual/Actual ISMA-style coupon-period accrual.

The benchmark is sensitive to the fractional first coupon period. The first discount exponent is based on the fraction of the current coupon period remaining from settlement to the next coupon date.

## 4. Schedule Table Conventions

DerivaPro uses compact schedule strings for first-wave review.

Coupon schedule:

```text
YYYY-MM-DD:coupon_rate, YYYY-MM-DD:coupon_rate, ...
```

Example:

```text
2026-06-20:0.050, 2031-06-20:0.060
```

The latest effective coupon rate on or before a payment date is applied. If no schedule entry applies, the base coupon rate is used.

Sinking schedule:

```text
YYYY-MM-DD:pct_original_notional, YYYY-MM-DD:pct_original_notional, ...
```

Example:

```text
2029-06-20:0.20, 2030-06-20:0.20, 2031-06-20:0.20
```

Each percentage is interpreted as a reduction of original notional, not current outstanding balance.

## 5. Cash-Flow Generation

Let:

\[
\begin{aligned}
N_0 &= \text{original notional},\\
N_i &= \text{opening outstanding notional for period } i,\\
c_i &= \text{effective coupon rate for period } i,\\
\alpha_i &= \text{coupon accrual factor},\\
S_i &= \text{scheduled principal reduction during period } i,\\
DF_i &= \text{discount factor to payment date } i.
\end{aligned}
\]

Coupon cash flow:

\[
\text{Coupon}_i = N_i c_i \alpha_i
\]

Principal cash flow depends on the selected amortization style.

### Bullet

\[
\text{Principal}_i =
\begin{cases}
0, & i<T,\\
N_0, & i=T.
\end{cases}
\]

### Straight-Line Amortization

If there are \(M\) payment dates:

\[
\text{Principal}_i = \frac{N_0}{M}
\]

The final period is adjusted to avoid over- or under-amortization due to rounding or schedule truncation.

### Sinking Schedule

For explicit sinking events:

\[
\text{Principal}_i = \sum_{j \in i} p_j N_0
\]

where event \(j\) falls inside payment period \(i\). Outstanding balance evolves as:

\[
N_{i+1}=N_i-\text{Principal}_i
\]

The total period cash flow is:

\[
CF_i=\text{Coupon}_i+\text{Principal}_i+\text{FixedPayment}_i
\]

## 6. Present Value, Yield, and Clean Price

The dirty present value is:

\[
PV_{\text{dirty}}=\sum_i CF_i DF_i
\]

For curve-based pricing:

\[
DF_i = P(0,t_i)
\]

For yield-based benchmarking with coupon frequency \(m\), yield \(y\), and first coupon fraction \(w\):

\[
DF_i=\left(1+\frac{y}{m}\right)^{-(i-1+w)}
\]

where:

\[
w=\frac{\text{days from settlement to next coupon date}}{\text{days in current coupon period}}
\]

Accrued interest for an Actual/Actual ISMA-style period is:

\[
AI=N_{\text{current}}\frac{c}{m}
\frac{\text{days from previous coupon date to settlement}}{\text{days from previous coupon date to next coupon date}}
\]

Clean value is:

\[
PV_{\text{clean}}=PV_{\text{dirty}}-AI
\]

The model price is:

\[
\text{Model price percent}=\frac{PV_{\text{clean}}}{N_0}\times 100
\]

For curve-mode diagnostics, yield to maturity is solved against the supplied market clean-price reference:

\[
\text{Market price}=\sum_i\frac{CF_i}{(1+y)^{\tau_i}}
\]

Because amortizing and sinking structures return principal over time, their yield, weighted-average life, duration, and convexity can differ substantially from a bullet bond with the same final maturity.

## 7. Risk Measures

The page currently reports:

- Fair value clean price.
- Accrued interest.
- Fair value plus accrued interest.
- Yield to maturity.
- Macaulay duration.
- Modified duration.
- Modified convexity.
- BPV / price change for a +1bp yield move.
- Parallel rate-shock scenario PV.
- Period-level coupon, principal, outstanding balance, discount factor, and PV diagnostics.

For yield-based benchmarking:

\[
D_{\text{Mac}}=
\frac{\sum_i \left(\frac{i-1+w}{m}\right)PV_i}{\sum_i PV_i}
\]

\[
D_{\text{Mod}}=\frac{D_{\text{Mac}}}{1+y/m}
\]

\[
C_{\text{Mod}}=
\frac{1}{PV_{\text{dirty}}}
\sum_i
\frac{CF_i(i-1+w)(i+w)}
{m^2(1+y/m)^{i+1+w}}
\]

For amortizing bonds, risk is distributed across the principal repayment schedule. Principal that returns earlier reduces weighted-average life and duration.

## 8. Benchmark Results

With the external calculator benchmark terms above, DerivaPro reproduces the following reference outputs:

| Case | Principal behavior | Clean value | Accrued interest | Dirty value | Duration | Modified duration | Modified convexity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Bullet principal paid at maturity | 902,702.15 | 9,699.45 | 912,401.60 | 10.2737 | 9.9745 | 129.8359 |
| B | Straight-line principal amortization over 30 periods | 943,393.01 | 9,699.45 | 953,092.46 | 5.8385 | 5.6684 | 50.7291 |

The straight-line amortizing case repays \(1{,}000{,}000/30=33{,}333.33\) principal per semiannual period, with the final period adjusted for rounding.

## 9. Assumptions in the Current Implementation

- Coupon schedule entries are effective-date based.
- In the editable payment table, opening notional is derived from original face value and previous principal payments.
- User-entered principal payments are the source of truth for the outstanding-balance roll-forward.
- The final payment date redeems remaining outstanding principal after previous amortization or sinking payments.
- Sinking schedule entries are aligned into payment periods by event date.
- Sinking percentages are percentages of original notional.
- Principal payments are capped at remaining outstanding balance.
- The final payment is adjusted so remaining outstanding principal is fully repaid for amortizing and sinking structures.
- Accrued interest and clean/dirty price are supported for the benchmark workflow.
- Business-day adjustment, ex-dividend handling, and holiday calendars remain simplified.
- Payment-in-kind and accreting-notional treatment are not yet implemented.
- Call/put optionality is handled on the callable/putable bond page, not here.

## 10. Alternative Methodologies

### Full Generic Bond Table Engine

A more complete implementation would allow separate tables for coupon rates, notional amounts, sinking fund amounts, payment-in-kind/accretion amounts, fixed payments, and custom cash-flow overrides. The engine would then generate a reconciled cash-flow table from these components.

### Amortization Schedule Import

For real-world loans, project bonds, and private credit instruments, amortization is often loaded as a schedule rather than generated from simple formulas. A production page should allow CSV import with validation.

### Spread Discounting

Credit-sensitive amortizing bonds should support discounting on:

\[
DF_{\text{spread}}(t)=\exp\left(-(z(t)+s(t))t\right)
\]

where \(s(t)\) may be a flat z-spread or a term structure of credit spreads.

### Prepayment or Extension Risk

Some amortizing assets include borrower optionality or prepayment behavior. Those require stochastic or scenario-based prepayment models and should not be priced as deterministic bonds.

## 11. Validation Plan

Recommended validation:

1. Bullet mode should match the level coupon bond page for equivalent inputs.
2. Straight-line amortization should fully reduce outstanding balance to zero.
3. Sinking schedule should reduce outstanding principal only on intended dates.
4. Increasing coupon schedule should increase PV relative to a flat lower coupon.
5. Earlier principal reduction should reduce duration.
6. Scenario PV should decrease under positive rate shocks for ordinary positive cash-flow bonds.
7. Cash-flow table totals should reconcile to dirty value before accrued interest.
8. Price-from-yield mode should reproduce the external calculator bullet and straight-line amortizing benchmark cases.

## 12. Current DerivaPro Status

Implemented now:

- Coupon schedule parsing.
- Sinking schedule parsing.
- Bullet, straight-line, and explicit sinking principal logic.
- Settlement / dated / first coupon / last coupon inputs.
- Price-from-yield benchmarking.
- Clean and dirty value decomposition.
- Actual/Actual ISMA-style accrued interest for regular coupon periods.
- Curve discounting.
- Yield solve.
- Duration, convexity, BPV.
- Parallel rate scenarios.

Planned:

- Rich editable schedule table.
- CSV import/export.
- Payment-in-kind/accreting notional.
- Odd coupon periods.
- Business-day adjustment and ex-dividend convention controls.
- Key-rate duration and spread measures.
