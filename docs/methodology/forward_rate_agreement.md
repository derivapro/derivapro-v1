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
