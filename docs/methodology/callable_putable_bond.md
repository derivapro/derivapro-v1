# Callable / Putable Bond Methodology

## Scope

This note covers the DerivaPro first-wave callable/putable fixed-rate bond workflow under Fixed Income Extensions. The workflow extends the existing fixed-income bond coverage by adding embedded exercise optionality.

## Product Description

A callable bond gives the issuer the right to redeem the bond before maturity, usually when rates fall. A putable bond gives the investor the right to sell the bond back to the issuer before maturity, usually when rates rise or credit/liquidity preferences change.

Supported conventions in the current workflow:

- Callable or putable fixed-rate bond.
- User-supplied face value.
- User-supplied coupon rate.
- User-supplied market clean price.
- Annual, semiannual, or quarterly coupon frequency.
- User-supplied exercise price as a percentage of par.
- User-supplied first exercise year.
- User-supplied short-rate volatility.
- User-supplied discount curve.
- Parallel rate shock scenarios.
- Yield-to-best and yield-to-worst diagnostics across maturity and eligible exercise cases.

## Pricing Framework

The workflow reports both:

- Straight-bond present value using deterministic discounting.
- Option-adjusted present value using a recombining short-rate lattice approximation.

The straight-bond benchmark is the discounted value of coupon and redemption cash flows:

```text
PV_straight = sum(Coupon_i * DF(T_i)) + Principal * DF(T_maturity)
```

The option-adjusted value is computed by backward induction over a short-rate lattice. At each node, continuation value is discounted from the next time step. When exercise is allowed:

```text
Callable value = min(Continuation value, Exercise price + due coupon)
Putable value  = max(Continuation value, Exercise price + due coupon)
```

The embedded option value is reported as:

```text
Callable option value = PV_straight - PV_callable
Putable option value  = PV_putable - PV_straight
```

Yield-to-best and yield-to-worst are calculated by solving the yield for each eligible exercise-date cash-flow case plus the final maturity case. The current implementation uses the supplied market clean-price reference for these diagnostics.

## Outputs

The page reports:

- Option-adjusted PV.
- Straight-bond PV.
- Embedded option value.
- Effective duration from up/down rate shocks.
- Yield to worst.
- Yield to best.
- Base/up/down rate shock scenario PV.
- Lattice cash-flow step diagnostics.
- Exercise-date yield diagnostics.

## Current Assumptions

- The lattice is a transparent first-pass approximation, not a calibrated Hull-White, Black-Karasinski, or market-standard term-structure model.
- Short-rate volatility is user supplied and constant.
- Exercise schedule is simplified to allow exercise after the first exercise year.
- Call/put price is flat as a percentage of par.
- Credit spread, OAS calibration, settlement conventions, accrued interest, holiday calendars, and issuer-specific call schedules are not yet modeled.
- Yield-to-best/worst uses the supplied clean-price percentage and simplified accrued-interest handling until full clean/dirty price decomposition is introduced.

## Recommended Validation

- Add calibrated Hull-White and Black-Karasinski engines.
- Support full call/put schedules with date-specific prices.
- Add clean/dirty price, accrued interest, OAS, and key-rate duration.
- Compare against QuantLib callable bond examples and desk benchmark cases.
- Add credit spread curve and liquidity spread controls.
