# Asset Swap Methodology

## Scope

This note covers the DerivaPro first-pass Asset Swap workflow under Fixed Income Extensions. The workflow is aligned conceptually with the local methodology reference `Math/AssetSwaps.html`, while keeping the public repo documentation original and implementation-specific.

## Product Description

An asset swap combines a cash bond position with an interest-rate swap overlay. The structure is commonly used to convert fixed-rate bond exposure into floating-rate exposure and to compare bond credit/liquidity value against the swap curve.

Supported conventions in the current workflow:

- Fixed-rate bullet bond economics.
- User-supplied market clean price as a percentage of par.
- Annual, semiannual, or quarterly coupon frequency.
- ACT/360, ACT/365, or 30/360 coupon day count.
- User-supplied quoted asset-swap spread.
- User-supplied zero-rate discount curve.
- Parallel rate-shock and spread-shock scenarios.

## Pricing Framework

The workflow first computes the fixed bond coupon annuity and the par swap rate from the supplied discount curve:

```text
Annuity = sum_i accrual_i * DF(T_i)
Par swap rate = (1 - DF(T_maturity)) / Annuity
```

The first-pass par asset-swap spread is estimated as:

```text
Par ASW spread =
    Bond coupon rate
  - Par swap rate
  + (Par amount - Market dirty price) / (Par amount * Annuity)
```

For an entered quoted spread, package PV is reported as:

```text
Asset swap PV = Notional * Annuity * (Quoted spread - Par ASW spread)
```

The page also reports the model PV of the fixed-rate bond cash flows using deterministic discounting.

## Outputs

The page reports:

- Asset-swap package PV.
- Par asset-swap spread.
- Par swap rate.
- Bond model PV.
- Rate and spread shock scenario PV.
- Period-level coupon, accrual, discount-factor, and PV diagnostics.

## Current Assumptions

- Clean price is treated as dirty price for first-pass analytics; accrued interest is not yet modeled.
- The bond is assumed to be a fixed-rate bullet bond.
- Funding, settlement, ex-dividend treatment, business-day adjustment, and bond-specific street conventions are not yet modeled.
- The spread formula is a transparent approximation, not a full dealer asset-swap engine.
- Credit curve and collateral discounting are represented only through user-entered curve/spread assumptions.

## Recommended Validation

- Add accrued interest and clean/dirty price conversion.
- Add settlement date, holiday calendar, ex-dividend, and coupon stub controls.
- Compare par asset-swap spread against benchmark vendor or QuantLib-style examples.
- Add z-spread, G-spread, I-spread, and OAS comparisons.
- Add key-rate DV01 and credit-spread DV01.

## Detailed Methodology Notes

### Asset Swap Interpretation

An asset swap combines a cash bond purchase with a swap overlay that transforms fixed bond coupon exposure into floating-rate exposure. The par asset-swap spread is the floating spread that makes the package value zero at inception, given the bond price, bond coupon, and swap curve.

### Bond Premium / Discount Adjustment

When the bond is purchased away from par, the premium or discount must be amortized economically through the swap package. DerivaPro approximates this as:

```text
Price pull-to-par spread = (Par amount - Market dirty price) / (Par amount * Annuity)
```

This term increases the par spread for discount bonds and decreases it for premium bonds.

### Coupon Versus Swap Adjustment

The fixed coupon on the bond is compared against the par swap rate:

```text
Coupon spread component = Bond coupon - Par swap rate
```

Combining both effects:

```text
Par ASW spread =
    Bond coupon - Par swap rate
  + (Par amount - Market dirty price) / (Par amount * Annuity)
```

### Package PV

If the market quotes an asset-swap spread `S_quote`, the first-pass package PV is:

```text
PV_package = Notional * Annuity * (S_quote - S_par)
```

### Alternative Spread Measures

Asset-swap spread is only one relative-value measure. Production fixed-income analytics should also include:

- Z-spread: flat spread over the zero curve that discounts bond cash flows to market price.
- G-spread: spread over the government curve at comparable maturity.
- I-spread: spread over the interpolated swap curve.
- OAS: spread after adjusting for embedded optionality.
- CDS basis: comparison between bond-implied spread and CDS spread.

### Production Enhancements

The current approximation should be extended with clean/dirty price conversion, accrued interest, settlement date, floating leg reset schedule, projection/discount curve separation, bond-specific calendars, coupon stubs, funding assumptions, and collateral assumptions.
