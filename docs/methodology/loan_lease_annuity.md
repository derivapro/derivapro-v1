# Loan / Lease / Annuity Methodology

## Scope

This page covers contractual payment streams such as loans, leases, and annuities. The first-wave implementation supports level-payment, equal-principal, and interest-only / bullet structures.

## Pricing Framework

The workflow builds scheduled payment dates from start date, maturity date, and payment frequency. For each payment period:

1. Interest is calculated from opening balance, contract rate, and day-count accrual.
2. Principal payment is determined by the selected structure.
3. Scheduled payment equals interest plus principal.
4. Scheduled payments are discounted on the supplied zero-rate curve.

The present value is the sum of discounted scheduled payments.

## Supported Structures

- Level payment: payment amount is solved so the principal amortizes over the term.
- Equal principal: principal is repaid evenly across periods.
- Interest only / bullet: interest is paid during the term and principal is repaid at maturity.

## Analytics

The page reports PV, price as a percentage of principal, implied yield, payment count, cash-flow diagnostics, and parallel rate-shock scenario PV.

## Current Assumptions

- No prepayment, delinquency, default, recovery, servicing cost, residual value, or tax treatment is included.
- Floating-rate loan resets are not yet modeled.
- Lease-specific residual-value and purchase-option features are planned extensions.
- Business-day calendars and settlement conventions are simplified.

## Validation Priorities

- Add amortization schedule import/export.
- Add prepayment and residual-value assumptions.
- Add floating-rate loan and lease variants.
- Add portfolio aggregation hooks for book-level exposure.
