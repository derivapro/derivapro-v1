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

## Detailed Methodology Notes

### Cost-of-Carry Forward Pricing

For a bond forward, the forward price reflects spot dirty price, financing cost, and coupon income before delivery.

Let:

```text
P_dirty = spot dirty price
I       = present value of coupons paid before delivery
R_repo  = financing / repo rate
T       = time to delivery
```

The current DerivaPro approximation is:

```text
Forward dirty price = (P_dirty - I) * exp(R_repo * T)
```

This is a clean first-pass carry model. A production model should handle coupon reinvestment, settlement calendars, accrued interest at delivery, and repo curve term structure.

### Implied Forward Yield

The page estimates a forward yield from forward price, coupon, and remaining maturity using a bond-yield approximation:

```text
Forward yield =
    [Coupon + (Par - Forward price) / remaining_years]
    / average(Par, Forward price)
```

This is not a full yield solve. A production workflow should generate post-delivery bond cash flows and solve yield exactly.

### Treasury Lock PV

A treasury lock is approximated with duration:

```text
PV_lock =
    position_sign
    * -Modified duration
    * Notional
    * (Forward yield - Locked yield)
    * DF(T_delivery)
```

This treats the lock as a linear exposure to the forward yield. Convexity and curve-shape effects are not included.

### Futures and CTD Extensions

For treasury futures, bond forward methodology must be extended to include deliverable basket, conversion factor, cheapest-to-deliver selection, invoice price, net basis, implied repo rate, daily variation margin, and delivery option value.

### Alternative Methodologies

Production alternatives include full cash-flow forward valuation from spot and repo curve, futures valuation with CTD optimization, hedge-ratio calculation using DV01, key-rate hedge decomposition, and scenario analysis across repo, yield, and curve shocks.

### Additional Risk Measures

Recommended analytics:

- Forward DV01.
- Carry and roll-down.
- Repo sensitivity.
- Conversion-factor adjusted futures basis.
- Hedge ratio versus cash bonds or futures contracts.
