# Bond Forward / Treasury Lock Methodology

## Scope

This note covers the DerivaPro first-pass Bond Forward / Treasury Lock workflow under Fixed Income Extensions. The workflow is aligned conceptually with the local methodology references `Math/BFWD.html`, `Math/TreasuryLocks.html`, `Math/IRFuture.html`, `Math/ConvFactor.html`, and `Math/CTDAnalysis.html`, while keeping public repo documentation original and implementation-specific.

## Product Description

Bond forwards define future delivery economics for a bond. Treasury locks are rate-lock instruments used to hedge the future level of treasury yields or bond-linked financing rates before issuance, purchase, or portfolio rebalancing.

Supported conventions in the current workflow:

- User-supplied spot dirty price as a percentage of par.
- User-supplied financing or repo rate.
- Annual, semiannual, or quarterly coupon frequency.
- ACT/360, ACT/365, or 30/360 coupon day count.
- User-supplied locked forward yield.
- User-supplied modified duration for treasury-lock PV approximation.
- Receive-fixed / long-duration or pay-fixed / short-duration lock direction.
- Parallel rate shock scenarios.

## Pricing Framework

The workflow computes coupon income before the delivery date:

```text
Income PV = sum coupons paid before delivery * DF(T_i)
```

The first-pass forward dirty price uses cost-of-carry logic:

```text
Forward dirty price =
    (Spot dirty price - Income PV) * exp(Financing rate * T_delivery)
```

The implied forward yield is approximated from coupon, forward price, and remaining maturity:

```text
Forward yield =
    (Coupon + (Par - Forward price) / remaining years)
  / average(Par, Forward price)
```

Treasury-lock PV is estimated with a duration-based approximation:

```text
Treasury lock PV =
    position_sign * -Modified duration * Notional
    * (Forward yield - Locked forward yield)
    * DF(T_delivery)
```

## Outputs

The page reports:

- Treasury lock PV.
- Forward dirty price.
- Forward price as a percentage of par.
- Implied forward yield.
- Up/down rate shock scenario PV.
- Pre-delivery coupon income diagnostics.

## Current Assumptions

- The forward price is based on user-supplied spot dirty price and financing rate.
- Coupon reinvestment and delivery option value are not yet modeled.
- Conversion factor, cheapest-to-deliver, invoice price, and futures margining are not yet modeled.
- The treasury-lock valuation is a duration approximation, not a full term-structure hedge valuation.
- Settlement, repo specialness, accrued interest, calendars, and holiday adjustment are not yet modeled.

## Recommended Validation

- Add clean/dirty conversion and accrued interest.
- Add deliverable basket, conversion factor, CTD, net basis, and implied repo analytics.
- Compare against benchmark treasury futures and bond forward examples.
- Add curve carry, roll-down, key-rate DV01, and hedge-ratio outputs.
- Add settlement calendar and coupon-date business-day adjustments.
