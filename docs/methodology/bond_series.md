# Bond Series Methodology

## 1. Scope and Product Definition

Bond Series analytics cover multiple related bond maturities valued together as one issuer-level or program-level structure. This is common in municipal finance, serial bond issuance, and fixed-income books where different maturities are part of one financing program.

The methodology is aligned conceptually with `Math/BondSeries.html`, which describes serial bond functions, aggregate cash-flow tables, price/yield/risk statistics for a whole series, and callable bond series extensions.

The current DerivaPro implementation is a first-pass non-callable bond series engine.

## 2. Product Economics

Each series row currently contains:

```text
maturity date
principal
coupon rate
redemption percentage
```

The page accepts rows in this compact format:

```text
maturity|principal|coupon_rate|redemption_pct; maturity|principal|coupon_rate|redemption_pct
```

Example:

```text
2028-08-13|1000000|0.040|100; 2030-08-13|1500000|0.045|100
```

The current implementation assumes semiannual coupons for each component bond and values each component as a level coupon bond.

## 3. Component Bond Valuation

For each component bond `j`, DerivaPro constructs a level coupon cash-flow stream:

```text
Coupon_{j,i}    = Principal_j * coupon_j * alpha_{j,i}
Principal_{j,T} = Principal_j * redemption_pct_j / 100
CF_{j,i}        = Coupon_{j,i} + Principal_{j,i}
```

Component PV:

```text
PV_j = sum_i CF_{j,i} * DF(t_{j,i})
```

Component model price:

```text
Price_j = PV_j / Principal_j * 100
```

## 4. Series Aggregation

The series present value is:

```text
PV_series = sum_j PV_j
```

Total principal:

```text
Principal_series = sum_j Principal_j
```

Weighted-average maturity:

```text
WAM = sum_j Principal_j * maturity_years_j / Principal_series
```

DerivaPro also builds an aggregate cash-flow map by payment date:

```text
Aggregate CF(date) = sum_j CF_j(date)
```

This aggregate cash-flow stream is used for the first-pass aggregate yield solve.

## 5. Aggregate Yield

The aggregate yield is the single annual yield that equates aggregate series cash flows to total principal:

```text
Principal_series = sum_k AggregateCF_k / (1 + y)^(tau_k)
```

This is a simplified par-reference yield. A production serial bond engine should also support market prices by maturity, total issue price, and premium/discount allocations.

## 6. Scenario Analysis

Parallel rate scenarios shift the supplied zero-rate curve:

```text
z_shifted(t) = z(t) +/- shock_bp / 10,000
```

Each component bond is repriced under the shifted curve, and the series PV is re-aggregated.

## 7. Assumptions in the Current Implementation

- All component bonds are treated as non-callable.
- Coupon frequency is semiannual.
- Component bond clean prices are not separately entered.
- Aggregate yield is solved against total principal.
- Accrued interest, issue date, first coupon date, last coupon date, odd coupons, and ex-coupon conventions are not modeled.
- Business-day adjustment and holiday calendars are simplified.
- Callable serial bond analysis is not yet included.

## 8. Alternative Methodologies

### Full Serial Bond Table

A richer bond series model should include:

```text
dated date
first coupon date
last coupon date
maturity date
principal
redemption value
coupon rate
market price
```

This supports odd coupons, issue-level analysis, and maturity-specific pricing.

### Callable Bond Series

Callable bond series analysis should incorporate a call schedule and determine price/yield to worst or best for each maturity. For a callable series, expected redemption dates may differ from legal maturity dates.

### Portfolio-of-Bonds Method

An alternative view is to treat the series as a portfolio of individual term bonds. This supports component-level price/yield, component-level duration and convexity, series-level aggregation, and key-rate risk by maturity bucket.

### Curve-Based Strip Pricing

Instead of solving a single aggregate yield, the entire series can be discounted from a curve:

```text
PV_series = sum_j sum_i CF_{j,i} * DF(t_{j,i})
```

This is the current DerivaPro valuation basis.

## 9. Recommended Outputs for Production

A production bond series page should report:

- Series PV.
- Issue price.
- Aggregate yield.
- Weighted-average maturity.
- Weighted-average coupon.
- Debt-service schedule by date.
- Component PV, price, yield, duration, and convexity.
- Callable redemption diagnostics where applicable.
- Key-rate duration by maturity bucket.
- Annual debt-service table.

## 10. Validation Plan

Recommended validation:

1. One-row series should match the level coupon bond page.
2. Series PV should equal the sum of component PVs.
3. Aggregate cash-flow totals should equal component cash-flow totals.
4. Weighted-average maturity should move toward longer maturities as principal weights shift longer.
5. Positive rate shocks should reduce PV for ordinary positive cash-flow series.
6. Benchmark against known serial bond examples and manually calculated aggregate cash flows.

## 11. Current DerivaPro Status

Implemented now:

- Multi-row series input.
- Component level-coupon valuation.
- Series PV.
- Aggregate yield.
- Weighted-average maturity.
- Per-component diagnostics.
- Parallel rate scenarios.

Planned:

- Editable grid.
- Market price per maturity.
- Component yield/risk table.
- Callable series support.
- Debt-service calendar export.
- Portfolio integration.
