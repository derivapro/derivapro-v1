# Callable and Putable Amortizing Bonds

## Scope and Reference Alignment

The calculator supports deterministic fixed coupons, step-up rates, bullet or amortizing principal, Bermudan dates, American exercise windows, and clean redemption prices expressed as percentages of outstanding principal. It uses a one-factor Hull-White or Black-Karasinski trinomial tree fitted to the supplied discount curve.

The local references reviewed are `Math/Callablebond.html`, `Math/TermStructureCalibration.html`, `Math/GenericBonds.html`, `Math/Daycount.html`, `Math/Dategen.html`, and the callable American/Bermudan amortizing bond guide under `how_to`. This document paraphrases the relevant principles and describes DerivaPro's implementation; the purchased documents are not redistributed.

| Reference topic | Implementation | Remaining boundary |
|---|---|---|
| Callable-bond valuation | Backward induction on a recombining trinomial tree | Numerical grid and exercise conventions may differ from another implementation |
| Initial term structure | Fit discount factors at every tree time using state prices | Does not estimate market volatility parameters |
| Model calibration | User supplies volatility and mean reversion | Calibration to cap/floor or swaption prices is a separate task |
| Dates and coupons | QuantLib schedule generation; distinct accrual and payment dates | Ex-dividend periods are not supported; the benchmark uses zero |
| Notice period | Decision before redemption, with conditional discounting of settlement cashflows | Calendar-day notice; already-issued notices are not modeled |
| Amortization | Principal roll-forward validated on the server | Accretion and payment-in-kind require additional contract rules |

Implementation: `callable_bond_schedule.py`, `short_rate_lattice.py`, and `callable_bond_tree.py`, coordinated by `price_callable_amortizing_bond` in `rates_fixed_income.py`.

## Input Definitions

| Input group | Inputs | Meaning |
|---|---|---|
| Contract | Effective, dated, settlement and maturity dates | Effective is contract start; dated is the interest accrual anchor; settlement is valuation date; maturity is final contractual accrual date |
| Coupon generation | Frequency, optional first and penultimate coupon dates | Backward generation from maturity, with optional stub controls |
| Coupon table | Date, opening principal, coupon rate, fixed payment | Interpretation is chosen explicitly as described below |
| Exercise | Rights, style, start/end dates, call/put prices | Zero price disables that right; Bermudan uses the end date; American samples the window |
| Calendar | Business-day convention and holidays | Weekends plus supplied holidays adjust payments; accrual dates remain contractual |
| Market | Dated discount factors or tenor/zero-rate pairs | Determines discounting at settlement |
| Model | Model name, sigma, mean reversion, refinement | Determines the risk-neutral rate distribution and numerical resolution |
| Quote | Market clean price in percent of original face | Target for OAS and deterministic yield diagnostics |

Coupon-table rates are decimals: 0.05 means 5%. Exercise and market prices are percentage points: 100 means par. Model volatility and mean reversion are displayed as percentages and divided by 100 internally, so 20.00% becomes \(\sigma=0.20\) and 0.50% becomes \(a=0.005\). In Hull-White, sigma is an absolute instantaneous rate volatility, not a relative change in the prevailing interest rate. In Black-Karasinski, sigma is the volatility of the log rate. The same displayed percentage therefore describes different model dynamics. The local usage guide describes 0.1%-3% as a typical Hull-White volatility range; the 20% default is retained only because it is the literal supplied benchmark input, not because it is a normal production calibration.

The benchmark defaults retain sigma = 0.20 and mean reversion = 0.005; the second experiment changes only sigma to 0.10. These inputs are deliberately retained for comparison, not recalibrated to improve agreement.

## Two Coupon Table Interpretations

### Coupon Terms through Cutoff Dates

This mode generates coupon dates from effective date, maturity, frequency and optional stubs. Each row covers dates strictly after the previous cutoff and through its own cutoff. Its notional and annual coupon rate apply to every generated coupon in that interval. Its fixed principal payment occurs once, on the last generated coupon in that interval.

A cutoff beyond maturity does not create a post-maturity payment. It completes the definition of terms applicable through maturity.

For the supplied example, the four rows are:

| Apply through | Opening principal | Annual coupon | Fixed principal |
|---|---:|---:|---:|
| 20-Dec-2020 | 100 | 5.0% | 0 |
| 20-Jun-2022 | 100 | 5.5% | 20 |
| 20-Dec-2022 | 80 | 5.5% | 0 |
| 20-Jun-2025 | 80 | 5.5% | 80 |

With effective/settlement 24-Sep-2019, maturity 24-Sep-2024 and semiannual frequency, the resulting payments are:

| Payment date | Opening principal | Coupon | Principal | Total |
|---|---:|---:|---:|---:|
| 24-Mar-2020 | 100 | 2.50 | 0 | 2.50 |
| 24-Sep-2020 | 100 | 2.50 | 0 | 2.50 |
| 24-Mar-2021 | 100 | 2.75 | 0 | 2.75 |
| 24-Sep-2021 | 100 | 2.75 | 0 | 2.75 |
| 24-Mar-2022 | 100 | 2.75 | 20 | 22.75 |
| 24-Sep-2022 | 80 | 2.20 | 0 | 2.20 |
| 24-Mar-2023 | 80 | 2.20 | 0 | 2.20 |
| 24-Sep-2023 | 80 | 2.20 | 0 | 2.20 |
| 24-Mar-2024 | 80 | 2.20 | 0 | 2.20 |
| 24-Sep-2024 | 80 | 2.20 | 80 | 82.20 |

This interpretation is supported by the reference's date-generation workflow and independently reproduces the supplied straight-bond PV: **101.74765444**, versus **101.7476542** externally. The detailed function-specific table-definition page was not present in the supplied folder, so this is documented as a tested interpretation rather than a claim about every external table format.

### Explicit Payment Dates

Every table row is an actual contractual coupon/payment date. The final row must equal maturity; after-maturity payments are rejected. Use this mode for fully specified irregular cashflows.

The interface derives opening principal from prior repayments. The server independently verifies the roll-forward, nonnegative finite amounts, unique dates, maturity coverage and full repayment. Changing the table interpretation does not silently rewrite the table.

## Coupons, Accrued Interest and Price

For opening principal \(N_i\), annual coupon \(c_i\), accrual fraction \(\alpha_i\), and principal repayment \(R_i\),

\[
C_i=N_i c_i\alpha_i,\qquad X_i=C_i+R_i,\qquad N_{i+1}=N_i-R_i.
\]

On this page the fixed-payment column is principal repayment. It is included once in the cashflow and once in the principal reduction. The lower-level API can separately represent additional fixed cashflows; those do not reduce principal.

Accrual fractions follow the chosen day-count rule. The local reference's **30/360 (ISDA)** changes a starting 31 to 30 and changes an ending 31 to 30 only when the adjusted starting day is 30. This maps to QuantLib **BondBasis**, not its differently defined Thirty360.ISDA/German convention. End-of-month schedule generation and day-count adjustments are separate operations.

For settlement \(s\) between coupon accrual dates \(a_i,b_i\),

\[
AI(s)=N_i c_i\,\tau(a_i,s),\qquad
P_{\rm dirty}(s)=\sum_{p_i>s}X_i D(s,p_i),\qquad
P_{\rm clean}=P_{\rm dirty}-AI.
\]

Actual/Actual ISMA calculations supply the nominal reference coupon period, including for stubs. Paid cashflows are excluded. Following and modified-following adjust payment dates using weekends and user holidays. Ex-dividend entitlement and pending adjusted coupons across settlement require further extension; use the no-adjustment benchmark for direct comparison.

## Curve Construction

Tree time and dated curve tenors use actual calendar days divided by 365. A discount-factor input \(D(0,T)\) is stored as an equivalent continuous zero rate,

\[
z(T)=-\log D(0,T)/T.
\]

The selected interpolation is then applied when evaluating the curve:

| Method | Interpolated quantity |
|---|---|
| Linear discount factor | \(D(0,T)\) |
| Exponential | \(\log D(0,T)\) |
| Linear zero rate | Continuous zero rate \(z(T)\) |
| Natural cubic spline | Continuous zero rates with natural endpoint conditions |

The valuation-date factor is one. Before the first future node and beyond the last node, the curve uses the nearest endpoint zero rate. Spline interpolation may overshoot; Black-Karasinski rejects grids with nonpositive implied forward rates instead of silently flooring them.

## Short-Rate Theory

The common state follows a risk-neutral Ornstein-Uhlenbeck process:

\[
dx_t=-a x_t\,dt+\sigma\,dW_t,\qquad x_0=0.
\]

Its conditional moments over step \(\Delta t\) are

\[
E[x_{t+\Delta t}\mid x_t]=x_t e^{-a\Delta t},\qquad
v^2=\frac{\sigma^2(1-e^{-2a\Delta t})}{2a}.
\]

For \(a=0\), the variance limit is \(\sigma^2\Delta t\). QuantLib supplies these moments. Volatility zero is represented by a deterministic one-state tree.

### Hull-White

\[
r_t=x_t+\phi(t).
\]

The time-dependent shift fits the initial discount curve. Negative short rates are permitted without an artificial floor.

### Black-Karasinski

\[
r_t=\exp(x_t+\phi(t)).
\]

The shift is solved numerically at each step. Positive short rates imply decreasing model discount factors. A nonpositive forward interval is incompatible with this unshifted lognormal model and produces an explicit validation error.

Both models assume one common rate factor, constant user-supplied \(a,\sigma\), a deterministic initial curve, deterministic contractual coupons/principal and rational exercise. Credit default, stochastic spreads, recovery and multiple curve factors are outside this implementation.

## Trinomial Construction and Curve Fit

Payment, notice and redemption times are mandatory grid points. QuantLib's TimeGrid inserts intermediate points using the requested refinement. The reported maximum time step is the actual resulting value; refinement is a resolution request rather than a promise of an identical proprietary grid.

At each transition, let \(m=x e^{-a\Delta t}\), \(h=\sqrt{3v^2}\), \(k\) be the nearest lattice index to \(m/h\), and \(e=m-kh\). The lower, middle and upper probabilities are

\[
p_d=\frac{1+e^2/v^2-e\sqrt{3}/v}{6},\quad
p_m=\frac{2-e^2/v^2}{3},\quad
p_u=\frac{1+e^2/v^2+e\sqrt{3}/v}{6}.
\]

They sum to one and match the OU mean and variance. Node locations recombine; probabilities are state dependent. The implementation rejects invalid probabilities and excessive grid/state sizes.

Let \(Q_{i,j}\) denote Arrow-Debreu state prices. Starting with \(Q_{0,0}=1\),

\[
Q_{i+1,k}=\sum_j Q_{i,j}e^{-r_{i,j}\Delta t_i}p_{j,k}.
\]

Curve fitting enforces

\[
\sum_j Q_{i,j}e^{-r_{i,j}\Delta t_i}=D(0,t_{i+1}).
\]

For Hull-White this gives

\[
\phi_i=\frac{\log\left(\sum_j Q_{i,j}e^{-x_{i,j}\Delta t_i}\right)
-\log D(0,t_{i+1})}{\Delta t_i}.
\]

Black-Karasinski solves the corresponding nonlinear equation using Brent's method. Curve fitting is distinct from estimating volatility and mean reversion from option quotes.

The application reports the maximum discount-factor fitting residual and the difference between a no-exercise tree price and direct discounted cashflows. Both should be near floating-point precision.

## Exercise and Notice Periods

Continuation is

\[
H_{i,j}=X_i+e^{-(r_{i,j}+s)\Delta t_i}
\sum_k p_{j,k}V_{i+1,k},
\]

where \(s\) is a continuous OAS, zero for the unspread price.

For redemption date \(e\) and calendar notice period \(n\), the decision date is \(d=e-n\). A clean exercise price \(K\), stated in percentage points, produces a settlement payoff consisting of accrued interest, contractual cashflows due on redemption, and \(KN_e/100\) on the principal remaining after scheduled repayment that day.

The exercise alternative at the notice date is the conditional discounted value of that settlement payoff plus all contractual payments during the notice period. This leg is rolled back through the same tree, retaining state dependence. Comparing an immediate undiscounted strike at the notice date would be incorrect.

The node value is

\[
V_{i,j}=\min\left(\max(H_{i,j},L^{\rm put}_{i,j}),L^{\rm call}_{i,j}\right),
\]

omitting any inactive right. For simultaneous conflicting rights, the current convention gives the issuer the final cap; contracts with different priority require explicit extension.

Bermudan uses each supplied end date as redemption. American windows are sampled in calendar-day increments controlled by refinement, including their endpoints; the corresponding notice dates enter the tree exactly. It is a discrete approximation to continuous exercise. Past notice dates are excluded; already-announced exercises are not inferred.

Exercise at contractual maturity is redundant and is not applied as a new option. Overlapping windows should be reviewed; multiple active call alternatives select the cheapest issuer settlement leg.

## Truncation and Probabilities

In last-callable-date mode, option rollback ends at the last relevant decision date. Later coupons and principal form a **state-dependent terminal continuation**, built by rolling them back on the full calibrated tail tree. This is not a deterministic discount-factor-ratio approximation. Tail construction still requires the full lattice, so this mode is not a guarantee of reduced tree-building memory.

Risk-neutral probability mass is propagated with the same transition probabilities. Paths stop at their first exercise decision; they cannot be counted again at later dates. Call probability, put probability and survival probability must sum to one.

Expected exercise time is conditional on exercise:

\[
E[\tau\mid{\rm exercise}]
=\frac{\sum_i t_{{\rm redemption},i}\Pr({\rm first\ exercise\ at}\ i)}
{\Pr({\rm exercise})}.
\]

This differs from an expected redemption time that assigns maturity to surviving paths. Probabilities are model probabilities, not forecasts of issuer behavior.

## OAS, Risk and Yield Diagnostics

OAS solves

\[
P_{\rm clean}(s)=P_{\rm market,clean}.
\]

The calibrated base dynamics are retained and a continuous spread is added to discounting, including notice settlement legs. Exercise is recomputed for each trial spread. The initial search is +/-1,000 bp, expanded up to +/-8,000 bp if needed. A failed bracket is reported explicitly; it is not described as OAS being inapplicable.

Rate scenarios shift input continuous zero rates, rebuild the interpolated curve, refit the tree and recompute exercise. For decimal shock \(h\),

\[
D_{\rm eff}=\frac{P(-h)-P(+h)}{2P(0)h},\qquad
C_{\rm eff}=\frac{P(-h)+P(+h)-2P(0)}{P(0)h^2}.
\]

These outputs use clean-price normalization. Another system may use dirty normalization; they coincide for this zero-accrued benchmark. BPV is a separate signed +1 bp repricing, irrespective of the chosen scenario shock.

Yield diagnostics solve deterministic maturity/exercise cashflows against market clean price plus accrued. They use annual-effective compounding; they are not the stochastic option-adjusted expected return. The currently displayed best/worst range includes feasible call and put scenarios; the legal investor/issuer strategy distinction should be considered for contracts containing both rights.

## Benchmark Status and Reproduction

Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_callable_bond.py
.venv/bin/python -m unittest tests.test_callable_amortizing_bond tests.test_structured_amortizing_bond_benchmark
```

Use coupon terms mode, custom table, settlement/effective 24-Sep-2019, maturity 24-Sep-2024, blank optional coupon dates, face 100, semiannual frequency, 30/360 ISDA, no adjustment, 30 calendar notice days, linear discount-factor interpolation and the supplied curve. The source leaves dated date blank; the form displays its contractual default, the effective date 24-Sep-2019, explicitly to avoid browser-generated placeholders.

The four exercise start dates visible in the source are 24-Sep-2019, 2020, 2021 and 2022, with call prices 100, 100, 102 and 105 and puts zero. Three end dates are obscured by “####” in the screenshot. Defaults retain **provisional** end dates 24-Sep-2020, 2021, 2022 and 2024 and the user's requested Bermudan style. Only the second end date is directly readable. Exact callable agreement cannot be asserted until the original workbook dates and exercise style are confirmed.

The straight price is reconciled independently. Callable outputs should be compared only after confirming exercise dates, style, notice conventions and output definitions, and checking refinement convergence. Do not fit volatility or alter contractual dates merely to force agreement.

## Alternatives and Validation

- A one-factor finite-difference PDE can use the same model, curve, notice and exercise conventions and provides an independent numerical check.
- A European option on a zero-coupon bond has a Hull-White analytical solution. The automated test compares this solution from QuantLib with the lattice.
- A standard fixed-rate callable bond can be checked with QuantLib's packaged tree engine when notice periods and amortization do not require custom handling.
- Monte Carlo with least-squares exercise is an alternative for richer state variables; it introduces regression and simulation errors.
- Multifactor models or stochastic volatility may improve market fit but require additional calibration instruments and validation.

Tests cover direct PV versus no-exercise tree PV, both rate models, zero volatility, probability conservation, notice timing, invalid schedules, analytical bond-option agreement, day-count mapping and the supplied straight-bond benchmark.

Implementation references: [QuantLib Hull-White curve fitting](https://github.com/lballabio/QuantLib/blob/master/ql/models/shortrate/onefactormodels/hullwhite.cpp) and [QuantLib trinomial tree](https://github.com/lballabio/QuantLib/blob/master/ql/methods/lattices/trinomialtree.cpp). The Python lattice uses standard moment-matching equations with QuantLib OU moments; it does not claim identical behavior to every version of QuantLib's C++ grid safeguards.
