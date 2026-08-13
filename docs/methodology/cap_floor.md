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

```text
tau_i = year_fraction(period_start_i, period_end_i)
F_i = (DF_fwd(T_start_i) / DF_fwd(T_end_i) - 1) / tau_i
```

Each caplet is valued with a Black-style rate option formula:

```text
Caplet PV = Notional * tau_i * DF_disc(T_pay_i) * BlackCall(F_i, K, sigma, T_fix_i)
```

Each floorlet uses the corresponding Black put formula:

```text
Floorlet PV = Notional * tau_i * DF_disc(T_pay_i) * BlackPut(F_i, K, sigma, T_fix_i)
```

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
