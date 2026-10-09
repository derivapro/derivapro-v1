# Callable / Putable Bond Methodology

## 1. Scope and Product Definition

This note covers the DerivaPro callable / putable fixed-rate bond workflow. A callable bond gives the issuer the right to redeem the bond before maturity at contractual call prices. A putable bond gives the investor the right to sell the bond back to the issuer before maturity at contractual put prices.

Economically:

\[
\begin{aligned}
V_{\text{callable}} &= V_{\text{straight}} - V_{\text{issuer call}} \\
V_{\text{putable}}  &= V_{\text{straight}} + V_{\text{investor put}}
\end{aligned}
\]

The methodology is aligned conceptually with `Math/Callablebond.html`, which discusses callable/putable bond fair value, cash-flow tables, yield-to-best/worst, price-to-best/worst, spread-to-best/worst, OAS, and term-structure model alternatives such as Hull-White and Black-Karasinski.

The current DerivaPro implementation is intentionally transparent and first-pass. It is not yet a calibrated production callable bond model.

## 2. Supported Product Terms

The current workflow supports:

- Valuation date.
- Maturity date.
- Face value.
- Coupon rate.
- Market clean price as a percentage of par.
- Coupon frequency: annual, semiannual, or quarterly.
- Day-count convention.
- Callable or putable option type.
- Flat exercise price as a percentage of par.
- First exercise year / lockout period.
- User-supplied short-rate volatility.
- User-supplied zero-rate discount curve.
- Parallel rate-shock scenario.

The exercise schedule is simplified: exercise is allowed on model coupon steps after the first exercise year, at one flat exercise price.

## 3. Straight-Bond Benchmark

The straight bond is valued as deterministic fixed cash flows:

\[
PV_{\text{straight}} =
\sum_i \text{Coupon}_i \cdot DF(t_i)
+ \text{Principal} \cdot DF(t_T)
\]

where:

\[
\text{Coupon}_i = N \cdot c \cdot \alpha_i
\]

This is the value of the bond without embedded issuer/investor optionality.

## 4. Short-Rate Lattice Approximation

The current DerivaPro option-adjusted value uses a simple recombining short-rate lattice. The final value at maturity is initialized from coupon plus principal. The model then works backward through time.

At each lattice node:

\[
V_{\text{cont}} =
\frac{1}{2}\left(V_{\text{up}} + V_{\text{down}}\right)e^{-r_{\text{node}}\Delta t}
+ \text{Coupon}_{\text{node}}
\]

The node short rate is a transparent approximation around the flat zero rate implied by the discount curve:

\[
r_{\text{node}} =
r_0 + \left(2j-k\right)\sigma_r\sqrt{\Delta t}
\]

where:

\[
\begin{aligned}
r_0 &= \text{flat rate implied by maturity discount factor} \\
\sigma_r &= \text{user-supplied short-rate volatility} \\
\Delta t &= \text{model time step}
\end{aligned}
\]

## 5. Exercise Decision

If exercise is allowed at the node, DerivaPro compares continuation value with exercise value.

Exercise value:

\[
V_{\text{exercise}} =
\frac{\text{exercise price pct}}{100}N
+ \text{Coupon}_{\text{node}}
\]

Callable bond:

\[
V_{\text{node}} = \min\left(V_{\text{cont}}, V_{\text{exercise}}\right)
\]

The issuer exercises when calling the bond is economically cheaper than leaving it outstanding.

Putable bond:

\[
V_{\text{node}} = \max\left(V_{\text{cont}}, V_{\text{exercise}}\right)
\]

The investor exercises when putting the bond is economically more valuable than holding it.

## 6. Embedded Option Value

The page reports:

\[
\begin{aligned}
V_{\text{call option}} &= PV_{\text{straight}} - PV_{\text{callable}} \\
V_{\text{put option}}  &= PV_{\text{putable}} - PV_{\text{straight}}
\end{aligned}
\]

Callable option value is usually positive from the issuer perspective and negative to the investor relative to a non-callable bond. Putable option value is usually positive to the investor.

## 7. Effective Duration

Effective duration is calculated from up/down shifted option-adjusted PVs:

\[
D_{\text{eff}} =
\frac{PV_{\text{down}} - PV_{\text{up}}}
{2 \cdot PV_{\text{base}} \cdot \Delta r}
\]

where:

\[
\Delta r = \frac{\text{shock}_{bp}}{10{,}000}
\]

Effective duration is more appropriate than simple Macaulay duration for callable/putable bonds because expected cash flows may change when rates change.

## 8. Yield-to-Best and Yield-to-Worst

DerivaPro calculates simplified yield diagnostics across maturity and eligible exercise-date cases.

For each candidate exercise date `k`, the page constructs cash flows up to that date:

\[
\begin{aligned}
CF_i &= \text{Coupon}_i, && i < k \\
CF_k &= \text{Coupon}_k + \text{Exercise Price} \cdot N, && \text{at exercise date } k
\end{aligned}
\]

It solves:

\[
\text{Market Clean Price} =
\sum_i \frac{CF_i}{(1+y_k)^{\tau_i}}
\]

The final maturity case is also included:

\[
CF_T = \text{Coupon}_T + \text{Principal}
\]

Then:

\[
\begin{aligned}
YTW &= \min_k(y_k) \\
YTB &= \max_k(y_k)
\end{aligned}
\]

For a callable investor view, yield-to-worst is commonly important because the issuer may call when it is disadvantageous to the investor. For a putable bond, the interpretation can differ because investor optionality can improve downside outcomes.

## 9. Price-to-Best / Worst and Spread-to-Best / Worst

The current app does not yet implement price-to-best/worst or spread-to-best/worst, but the methodology is important for the roadmap.

Price-to-worst evaluates each eligible redemption date and finds the price case that is least favorable under the selected convention. Spread-to-worst solves a spread over a reference curve such that the discounted cash flows to each candidate redemption date match market price, then selects the worst spread outcome.

Generic spread equation:

\[
\text{Market Price} =
\sum_i CF_i \cdot e^{-(z(t_i)+s)t_i}
\]

where `s` is the solved spread.

## 10. Option-Adjusted Spread

OAS is a model-implied spread added to the short-rate lattice or discount curve so that model value equals observed market price:

\[
\text{Market Price} =
V_{\text{model}}(\text{rates}, \sigma, \text{exercise schedule}, OAS)
\]

OAS is useful because it attempts to separate:

- Risk-free interest-rate optionality.
- Credit/liquidity compensation.
- Model and convention assumptions.

DerivaPro does not yet solve OAS on this page.

## 11. Production Model Alternatives

### Hull-White One-Factor Model

The Hull-White model represents the short rate with mean reversion and time-dependent drift calibrated to the initial curve:

\[
dr(t) = \left[\theta(t) - a r(t)\right]dt + \sigma dW_t
\]

It is widely used for interest-rate trees and Bermudan-style exercise valuation.

### Black-Karasinski Model

The Black-Karasinski model uses a lognormal short-rate process:

\[
d\ln r(t) =
\left[\theta(t) - a\ln r(t)\right]dt + \sigma dW_t
\]

It keeps rates positive under ordinary assumptions and is often used for callable bond and interest-rate option modeling.

### Monte Carlo Least-Squares Exercise

Callable bonds can also be valued through Monte Carlo with regression-based exercise decisions, though tree/lattice methods are usually more natural for one-factor callable bond valuation.

### Deterministic Yield-to-Worst

For simpler quote analytics, the instrument can be analyzed by deterministic yield-to-call / yield-to-maturity cases without a stochastic interest-rate model. That is useful for quoting but not a full fair-value model of the embedded option.

## 12. Assumptions in the Current Implementation

- Short-rate lattice is transparent and simplified, not calibrated Hull-White or Black-Karasinski.
- Exercise price is flat across all exercise dates.
- Exercise is allowed after first exercise year on model coupon steps.
- Notice periods are not modeled.
- Call/put schedules with date-specific prices are not yet supported.
- Accrued interest and clean/dirty price decomposition are simplified.
- Credit spreads and OAS are not modeled.
- Business-day adjustment and holiday calendars are simplified.
- The yield-to-best/worst calculation uses the supplied clean-price percentage and simplified accrued-interest handling.

## 13. Validation Plan

Recommended validation:

1. With exercise disabled or far out of the money, option-adjusted PV should approach straight-bond PV.
2. Callable bond PV should generally be less than or equal to straight-bond PV.
3. Putable bond PV should generally be greater than or equal to straight-bond PV.
4. Higher short-rate volatility should generally increase embedded option value.
5. Effective duration should shorten for callable bonds when rates fall enough to increase call likelihood.
6. Yield-to-maturity should match the level coupon bond page when no exercise candidates are active.
7. Exercise-date yield table should reconcile to cash-flow cases.
8. Compare against QuantLib callable bond examples and vendor benchmark cases.

## 14. Current DerivaPro Status

Implemented now:

- Straight-bond PV.
- Transparent short-rate lattice PV.
- Callable / putable exercise decision.
- Embedded option value.
- Effective duration from rate shocks.
- Yield-to-best and yield-to-worst diagnostics.
- Lattice-step cash-flow diagnostics.

Planned:

- Full call/put schedule table.
- Notice-period handling.
- Clean/dirty price and accrued interest.
- Hull-White and Black-Karasinski calibrated engines.
- OAS, spread duration, spread convexity.
- Key-rate duration.
- Callable bond series integration.
