# Interest Rate Cap / Floor Methodology

## Scope

This note covers the DerivaPro first-wave cap/floor workflow under Fixed Income Extensions. The workflow values a cap or floor as a strip of caplets or floorlets using user-supplied forward and discount curves and a user-entered volatility assumption.

## Product Description

An interest rate cap pays when a reference forward rate exceeds a strike rate. An interest rate floor pays when the reference forward rate falls below a strike rate. A cap or floor is decomposed into period-level options on forward rates.

Supported conventions in the current workflow:

- Cap or floor payoff.
- Annual, semiannual, or quarterly payment schedule.
- User-supplied notional.
- User-supplied strike rate.
- User-supplied flat forward-rate volatility.
- Separate discount and forward zero-rate curves.
- Rate and volatility scenario shocks.

## Pricing Framework

For each reset/payment period, the model computes:

\[
\begin{aligned}
\tau_i &= \text{year fraction}(\text{period start}_i,\text{ period end}_i) \\
F_i &= \frac{DF_{\text{fwd}}(T_{\text{start},i}) / DF_{\text{fwd}}(T_{\text{end},i}) - 1}{\tau_i}
\end{aligned}
\]

Each caplet is valued with a Black-style rate option formula:

\[
PV_{\text{caplet},i} =
N \cdot \tau_i \cdot DF_{\text{disc}}(T_{\text{pay},i})
\cdot BlackCall(F_i,K,\sigma,T_{\text{fix},i})
\]

Each floorlet uses the corresponding Black put formula:

\[
PV_{\text{floorlet},i} =
N \cdot \tau_i \cdot DF_{\text{disc}}(T_{\text{pay},i})
\cdot BlackPut(F_i,K,\sigma,T_{\text{fix},i})
\]

The cap or floor value is the sum of all caplet or floorlet present values.

## Outputs

The page reports:

- Present value.
- Strike rate.
- Volatility.
- Approximate vega.
- Base/up/down rate shock PV.
- Up/down volatility shock PV.
- Period-level forward rates, accruals, discount factors, and option PVs.

## Current Assumptions

- Flat volatility is used for all caplets or floorlets.
- Volatility is interpreted as lognormal Black volatility.
- Negative-rate treatment is not yet handled with shifted-lognormal or normal/Bachelier models.
- Calendars, fixing lags, holiday adjustment, in-arrears conventions, and market-standard cap volatility surface interpolation are not yet implemented.
- Curve inputs are interpreted as continuously compounded zero-rate curves.

## Recommended Validation

- Add Black, shifted-Black, and Bachelier model choices.
- Add cap/floor volatility-surface input by expiry and tenor.
- Compare against QuantLib cap/floor examples.
- Add caplet stripping and period-level Greeks.
- Add calibration controls and quote conventions.

## Detailed Methodology Notes

### Caplet and Floorlet Decomposition

An interest rate cap is a strip of caplets. Each caplet pays when the realized/index forward rate is above the strike. A floor is a strip of floorlets that pays when the forward rate is below the strike.

For period `i`:

\[
\begin{aligned}
T_i &= \text{option fixing time} \\
P_i &= \text{payment time} \\
\alpha_i &= \text{accrual factor} \\
F_i &= \text{projected forward rate} \\
K &= \text{strike} \\
\sigma_i &= \text{Black volatility} \\
DF_i &= \text{discount factor to payment date}
\end{aligned}
\]

Caplet PV:

\[
PV_{\text{caplet},i} =
N \cdot \alpha_i \cdot DF_i \cdot BlackCall(F_i,K,\sigma_i,T_i)
\]

Floorlet PV:

\[
PV_{\text{floorlet},i} =
N \cdot \alpha_i \cdot DF_i \cdot BlackPut(F_i,K,\sigma_i,T_i)
\]

Total cap/floor PV is the sum over all optionlets.

### Black Rate Option Formula

Under lognormal Black assumptions:

\[
\begin{aligned}
d_1 &= \frac{\ln(F/K) + \frac{1}{2}\sigma^2T}{\sigma\sqrt{T}} \\
d_2 &= d_1 - \sigma\sqrt{T}
\end{aligned}
\]

\[
\begin{aligned}
BlackCall &= F\Phi(d_1) - K\Phi(d_2) \\
BlackPut  &= K\Phi(-d_2) - F\Phi(-d_1)
\end{aligned}
\]

The current DerivaPro implementation uses this Black-style approximation with a user-supplied flat volatility.

### Cap/Floor Parity

For the same strike and schedule:

\[
Cap - Floor = PV_{\text{floating leg}} - PV_{\text{fixed leg at strike}}
\]

This relationship is useful for validation and for detecting sign or accrual errors.

### Alternative Methodologies

Production cap/floor analytics should support caplet-specific volatility by expiry and tenor, shifted Black for low or negative rate environments, Bachelier/normal model, SABR-smile calibrated volatility, multi-curve projection and OIS discounting, and in-arrears or compounding conventions.

### Additional Risk Measures

Recommended future analytics:

- Delta to forward curve.
- Vega by caplet expiry.
- Key-rate DV01.
- Volatility bucket risk.
- Strike ladder and moneyness diagnostics.
- Cap/floor parity test output.
