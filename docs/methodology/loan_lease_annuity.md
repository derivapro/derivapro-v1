# Loan / Lease / Annuity Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro Loans / Leases / Annuities workflow. These instruments are modeled as contractual payment streams with principal and interest components. The current implementation is deterministic and uses present-value cash-flow methods.

The methodology is aligned conceptually with:

- `Math/PVCF.html` for present-value cash-flow functions.
- `Math/YTM.html` for yield solving.
- `Math/GenericBonds.html` for scheduled cash-flow generation concepts.
- Mortgage and prepayment references in the Math folder for future extensions, although the current page does not yet include prepayment modeling.

## 2. Supported Structures

The page currently supports:

- Level-payment loan or annuity.
- Equal-principal amortization.
- Interest-only / bullet structure.

Supported inputs:

- Valuation date.
- Start date.
- Maturity date.
- Principal / financed amount.
- Contract rate.
- Payment frequency: annual, semiannual, quarterly, or monthly.
- Day-count convention.
- Discount curve.
- Parallel rate-shock scenario.

## 3. Payment Schedule

A payment schedule is generated from start date to maturity date using the selected frequency.

Let:

```text
B_i       = opening balance for period i
r         = annual contract rate
alpha_i   = day-count accrual factor
M         = number of payments
DF_i      = discount factor to payment date i
```

Interest for the period is:

```text
Interest_i = B_i * r * alpha_i
```

The principal component depends on the selected structure.

## 4. Level-Payment Structure

For a level-payment loan, DerivaPro estimates a constant scheduled payment. With payment frequency `m`, the period rate is:

```text
q = r / m
```

For `M` payments, the level payment is:

```text
Payment = Principal * q / (1 - (1 + q)^(-M))
```

Then each period is decomposed:

```text
Principal_i = Payment - Interest_i
B_{i+1}     = B_i - Principal_i
```

In the final period, DerivaPro adjusts principal repayment to clear remaining balance.

## 5. Equal-Principal Structure

For equal-principal amortization:

```text
Principal_i = Principal / M
Payment_i   = Principal_i + Interest_i
```

Payments decline over time as the outstanding balance declines.

## 6. Interest-Only / Bullet Structure

For interest-only periods before maturity:

```text
Payment_i   = Interest_i
Principal_i = 0
```

At maturity:

```text
Payment_T = Interest_T + remaining principal
```

## 7. Present Value

The present value is:

```text
PV = sum_i Payment_i * DF(t_i)
```

where the discount factor is obtained from the supplied zero-rate curve:

```text
DF(t) = exp(-z(t) * t)
```

Price as a percentage of principal is:

```text
Price pct = PV / Principal * 100
```

## 8. Implied Yield

The page solves for a yield that equates scheduled payments to principal:

```text
Principal = sum_i Payment_i / (1 + y)^(tau_i)
```

This is a simplified internal-rate-of-return style yield. For loans and leases, production systems may use APR, effective annual rate, money-market yield, lease implicit rate, or accounting-specific yield definitions.

## 9. Scenario Analysis

The rate-shock scenarios reprice the same contractual payment stream under shifted discount curves:

```text
z_up(t)   = z(t) + shock_bp / 10,000
z_down(t) = z(t) - shock_bp / 10,000
```

This measures discount-rate sensitivity only. It does not currently change contractual rate, prepayment behavior, credit behavior, or residual value.

## 10. Assumptions in the Current Implementation

- Payments are deterministic.
- Contract rate is fixed.
- No prepayment is modeled.
- No default, delinquency, recovery, servicing fee, tax, or accounting treatment is modeled.
- No residual value is modeled for leases.
- No floating-rate reset is modeled.
- Business-day adjustment and holidays are simplified.
- Fees, upfront points, origination costs, and balloon structures beyond interest-only bullet repayment are not yet included.

## 11. Alternative Methodologies

### Present-Value Cash-Flow Table

The most general deterministic method is to import all projected payments:

```text
payment_date, interest, principal, fee, residual, total_payment
```

and discount each row.

### IRR / Effective Yield

Loans are often assessed through internal rate of return:

```text
0 = initial_advance - sum_i cashflow_i / (1 + IRR)^(tau_i)
```

This can incorporate fees and irregular payments.

### Prepayment Modeling

For mortgage-like or callable loan assets, expected cash flows depend on borrower behavior. A prepayment model can make principal payment a function of current loan rate versus market rate, seasoning, borrower credit attributes, loan-to-value, burnout, and macroeconomic variables.

### Lease Residual Modeling

Lease valuation may require a residual value assumption:

```text
PV = PV(lease payments) + PV(expected residual value)
```

Residual value may be deterministic, scenario-based, or option-like.

## 12. Validation Plan

Recommended validation:

1. Level-payment schedule should amortize to zero.
2. Equal-principal schedule should have constant principal payments.
3. Interest-only structure should repay principal at maturity.
4. Undiscounted principal payments should sum to original principal.
5. At discount rate equal to contract yield, PV should be near principal for a plain level-payment loan.
6. Positive discount-rate shocks should reduce PV.
7. Monthly, quarterly, semiannual, and annual schedules should produce reasonable payment counts.

## 13. Current DerivaPro Status

Implemented now:

- Level-payment, equal-principal, and interest-only structures.
- Monthly, quarterly, semiannual, and annual schedules.
- Payment decomposition.
- Curve discounting.
- Implied yield.
- Parallel rate scenarios.

Planned:

- Cash-flow import/export.
- Fees and upfront costs.
- Balloon payments.
- Floating-rate reset loans.
- Lease residual values.
- Prepayment and credit behavior.
- Portfolio integration.
