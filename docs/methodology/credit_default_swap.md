# Credit Default Swap Methodology

## 1. Scope and Product Definition

This document covers the DerivaPro **Credit Default Swap (CDS)** workflow under the Credit workspace (`/credit-derivatives/credit_default_swap`, backed by `CreditDefaultSwap` in `derivapro/models/mdls_credit.py`). A CDS is a bilateral contract in which the protection buyer pays a periodic premium (the spread) in exchange for compensation from the protection seller if a specified reference entity experiences a credit event (default). DerivaPro prices a single-name CDS using a piecewise-flat hazard-rate model bootstrapped from a single user-supplied spread quote, following the ISDA standard model conventions built into QuantLib.

## 2. Instrument Economics

The workflow accepts:

- Nominal (notional).
- Spread (the running premium, annualized).
- Recovery rate (the fraction of notional recovered in default).
- Risk-free rate (used to build a flat discounting curve).
- Payment frequency and coupon tenor (Annual, Semiannual, Quarterly, Monthly).
- Side: Buyer or Seller of protection.
- Entry (trade) date and end (maturity) date.
- Calendar convention: United States, TARGET, United Kingdom, or China.

The single spread quote is used both as the market input and as the flat coupon rate on the priced CDS leg, so the base-case valuation is expected to be close to zero (a freshly-issued at-market CDS) unless the trade parameters (recovery, risk-free rate, tenor) shift the priced NPV away from the entered spread.

## 3. Discounting Curve

A flat risk-free discount curve is built directly from the single risk-free rate input:

\[
DF(t) = e^{-r_f \cdot t}
\]

using `ql.FlatForward` under Actual/365 (Fixed) day counting. This is a simplification relative to a full term structure bootstrapped from OIS or Treasury instruments; it is adequate for illustrating CDS mechanics but should not be treated as a market-calibrated discount curve.

## 4. Hazard-Rate Bootstrap

DerivaPro bootstraps a piecewise-flat default-intensity (hazard-rate) curve from the single spread quote using `ql.PiecewiseFlatHazardRate` fed by a small ladder of `ql.SpreadCdsHelper` instruments (nominal tenors of 1 month, 3 months, 6 months, and the trade's own maturity in years), all quoting the same spread and recovery rate. Because every helper quotes an identical spread, the resulting hazard curve is effectively flat across all tenors; the multi-tenor ladder exists to exercise the same QuantLib bootstrapping machinery used for a full term-structure calibration, not to encode a term structure of credit risk from a single quote.

Each helper's maturity is generated using the CDS-standard `TwentiethIMM` date rule (rolling to the 20th of March/June/September/December). Because several of the short tenors can roll onto the *same* IMM date depending on how close the trade date is to an IMM roll, DerivaPro de-duplicates helpers that land on an identical pillar date before bootstrapping — otherwise the underlying date-generation collision (not a modelling choice) causes the bootstrap to reject the curve outright on certain trade dates.

The default probability to any date \(t\) implied by the bootstrapped hazard rate \(h\) is:

\[
Q(\text{survival to } t) = e^{-h t}, \qquad Q(\text{default by } t) = 1 - e^{-h t}
\]

## 5. Pricing Engine

The bootstrapped hazard curve and the flat discount curve are combined in QuantLib's `ql.IsdaCdsEngine`, which implements the ISDA standard CDS pricing model (the same model used industry-wide for CDS mark-to-market and upfront conversion). A `ql.CreditDefaultSwap` instrument is constructed with the trade's side, nominal, spread, and payment schedule, and priced through this engine to produce:

- **Net Present Value** — the mark-to-market value of the CDS position to the specified side.
- **Fair Spread** — the breakeven running spread at which the CDS would have zero NPV given the current hazard and discount curves.

## 6. Default Probability, Expected Loss, and Premium

- **Default Probability** reads the cumulative default probability off the bootstrapped hazard curve at the trade's coupon-period-implied maturity.
- **Expected Loss** is computed independently of the ISDA engine as a simple closed-form approximation:

\[
\text{Expected Loss} = N \times (1 - R) \times Q(\text{default})
\]

where \(N\) is notional and \(R\) is the recovery rate. This is a first-pass point estimate; it does not integrate loss-given-default over the full default-time distribution, so it will differ modestly from the loss implied by the full engine NPV for longer-dated or higher-hazard trades.

- **Premium Payment** is a simple per-period accrual estimate, \(N \times \text{spread} \times \alpha\), where \(\alpha\) is the coupon-period accrual fraction implied by the selected payment frequency — it is not read from the actual generated coupon schedule.

## 7. Sensitivity Analysis

The Sensitivity Analysis tab sweeps one input variable — **Spread**, **Recovery Rate**, or **Risk-Free Rate** — across a user-specified range and re-computes Expected Loss at each point, holding all other inputs fixed. The swept range is clamped to stay within economically valid bounds for the chosen variable (e.g. recovery rate is kept within [0%, 99%] and spread is kept strictly positive), because the boundary values themselves — an exact 0% spread, or a recovery rate at or extremely close to 100% — leave the ISDA hazard-rate bootstrap with no valid solution (the root-finding step used to imply the hazard rate has nothing to bracket).

Even within the clamped range, some parameter combinations can still land on a point the bootstrap cannot solve (this depends jointly on notional, recovery rate, tenor, and the swept variable itself, not on any single bound in isolation). DerivaPro treats this as expected numerical behavior rather than a workflow failure: the analysis skips any individual point that fails to converge, continues sweeping the rest of the range, and still returns and plots every point that priced successfully. The analysis only reports an error if *no* point in the requested range converges, in which case the user is asked to try a smaller range.

## 8. Assumptions in the Current Implementation

- A single spread quote drives the entire hazard-rate curve (flat term structure of credit risk); no multi-tenor CDS curve is bootstrapped from independently quoted tenor spreads.
- The discount curve is flat, built from a single risk-free rate rather than a bootstrapped term structure.
- Expected Loss uses a simplified closed-form point estimate rather than integrating over the ISDA engine's full loss distribution.
- Premium Payment is an accrual-fraction estimate, not read from the generated coupon schedule.
- Upfront payment is assumed to be zero (running-spread-only convention).
- Recovery rate and spread must be non-negative (recovery additionally bounded at 100%); the page validates this server-side and shows a corrective message rather than a raw pricing-library error.

## 9. Alternative Methodologies

### Multi-Tenor Curve Bootstrap

A production single-name CDS curve is typically bootstrapped from actual quotes at multiple tenors (1Y, 3Y, 5Y, 7Y, 10Y), producing a genuine term structure of hazard rates rather than a flat curve implied by one quote.

### Full Loss-Distribution Expected Loss

Rather than the closed-form point estimate used here, expected loss can be computed by integrating \((1-R)\) against the hazard-rate-implied default-time density over the life of the trade, which better captures the timing of potential default relative to discounting.

### Stochastic Hazard-Rate Models

More advanced credit models (e.g. CIR-based intensity processes) allow the hazard rate itself to be stochastic, which is required for pricing credit derivatives with optionality (e.g. CDS options, contingent CDS) but is not needed for a plain single-name CDS mark-to-market.

## 10. Validation Plan

Recommended validation cases:

1. At-market sanity check: with the priced spread equal to the fair spread implied by the same inputs, NPV should be close to zero.
2. Recovery rate monotonicity: holding spread fixed, expected loss should decrease as recovery rate increases.
3. Spread monotonicity: holding recovery fixed, both fair spread and NPV magnitude should move consistently as the quoted spread changes.
4. IMM-date robustness: run the same trade with entry dates on either side of an IMM roll date and confirm pricing succeeds in both cases (this specifically exercises the duplicate-pillar de-duplication).
5. Sensitivity sweep robustness: run a wide-range sweep on each of Spread, Recovery Rate, and Risk-Free Rate and confirm the analysis returns a usable curve (possibly with a few skipped edge points) rather than failing outright.
6. Negative/out-of-range inputs (negative nominal, negative spread, recovery rate outside [0, 1]) should be rejected with a validation message before reaching the pricing engine.

## 11. Current DerivaPro Status

Implemented now:

- Single-quote hazard-rate bootstrap via `ql.PiecewiseFlatHazardRate` with duplicate-pillar handling.
- ISDA-standard CDS pricing engine (NPV, fair spread).
- Default probability, expected loss, and premium payment estimates.
- Sensitivity analysis on Spread, Recovery Rate, and Risk-Free Rate with resilient per-point error handling.
- Server-side validation rejecting negative nominal/spread and out-of-range recovery rate.

Planned:

- Multi-tenor curve bootstrap from independently quoted spreads.
- Bootstrapped (non-flat) discount curve.
- Full loss-distribution expected-loss calculation.
- Coupon-schedule-based premium accrual instead of a period-fraction estimate.
