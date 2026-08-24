# Forward Rate Agreement Methodology

## Scope

This note covers the DerivaPro first-wave Forward Rate Agreement (FRA) workflow under Fixed Income Extensions. The workflow values a single future accrual-period rate contract using user-supplied discount and forward curves.

## Product Description

An FRA locks a fixed contract rate for a future interest accrual period. The contract payoff depends on the difference between the projected forward rate and the agreed fixed rate over the accrual period.

Supported conventions in the current workflow:

- Pay fixed / receive floating.
- Receive fixed / pay floating.
- User-supplied notional.
- User-supplied contract rate.
- ACT/360, ACT/365, or 30/360 accrual.
- Separate discount and forward zero-rate curves.
- Parallel curve shock scenarios.

## Pricing Framework

The model computes the accrual factor between the FRA start and end dates:

```text
tau = year_fraction(start_date, end_date)
```

The projected forward rate is implied from the forward curve discount factors:

```text
F = (DF_fwd(T_start) / DF_fwd(T_end) - 1) / tau
```

For a pay-fixed / receive-floating position, the projected payoff is:

```text
Payoff = Notional * tau * (F - K)
```

For a receive-fixed / pay-floating position, the sign is reversed. The payoff is discounted using the supplied discount curve.

## Outputs

The page reports:

- Present value.
- Implied forward rate.
- Contract strike rate.
- Discount factor.
- Base/up/down rate shock scenario PV.
- Projected payoff and discounted PV for the FRA accrual period.

## Current Assumptions

- Curves are entered as continuously compounded zero-rate tenors.
- Curve interpolation is inherited from the internal piecewise-linear curve helper.
- The current first-pass implementation discounts the projected payoff to the payment date.
- Convexity adjustment, collateral discounting details, settlement-date conventions, calendars, and holiday adjustments are not yet modeled.

## Recommended Validation

- Compare against QuantLib or desk-system FRA examples.
- Validate sign conventions for pay-fixed and receive-fixed cases.
- Add settlement-at-start convention and compare against settlement-at-end approximation.
- Add curve interpolation and compounding controls.
- Add calibration from money-market and futures instruments.

## Detailed Methodology Notes

### Forward Rate Derivation

For an accrual period beginning at `T1` and ending at `T2`, the no-arbitrage forward rate implied by a discount curve is:

```text
F(T1,T2) = (DF(T1) / DF(T2) - 1) / alpha
```

where `alpha` is the accrual year fraction for the reference period. If a separate projection curve is supplied, DerivaPro uses the projection curve for the forward rate and the discount curve for present value. This is consistent with multi-curve methodology, where projection and discounting curves may differ.

### Payoff Timing

A classical FRA is often settled at the start of the loan period, with the end-of-period interest differential discounted back to the settlement date:

```text
Settlement_start = Notional * alpha * (F - K) / (1 + alpha * F)
```

The current DerivaPro page uses a transparent end-period discounted payoff:

```text
Payoff_end = Notional * alpha * (F - K)
PV         = Payoff_end * DF(T2)
```

This is acceptable for first-pass analytics but should be enhanced with explicit settlement timing.

### Position Convention

For a pay-fixed / receive-floating FRA:

```text
PV = Notional * alpha * (F - K) * DF(T2)
```

For a receive-fixed / pay-floating FRA:

```text
PV = -Notional * alpha * (F - K) * DF(T2)
```

### Alternative Methodologies

Production FRA valuation should support start-date settlement, collateral discounting with OIS curves, separate index projection curves, IMM date conventions, forward-starting stub periods, and convexity adjustments where the underlying market quote requires them.

### Additional Risk Measures

Recommended future analytics:

- Forward-rate DV01.
- Discount-curve DV01.
- Key-rate DV01 by projection and discount curve.
- Curve carry and roll-down.
- Scenario grids across start/end forward rates and discount rates.
