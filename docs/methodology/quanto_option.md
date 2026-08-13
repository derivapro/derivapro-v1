# Quanto Option Methodology

## Scope

This note covers the first-wave DerivaPro quanto option workflow for foreign underlying options settled in domestic currency.

## Current Pricing Method

The current implementation uses a quanto-adjusted Black-Scholes formula. The foreign underlying drift is adjusted for equity-FX correlation:

```text
q_adjusted = foreign_yield + rho * equity_volatility * fx_volatility
```

The option is then priced as a domestic-currency Black-Scholes option using the adjusted carry term.

## Current Outputs

- Present value.
- Quanto-adjusted yield.
- Equity volatility and FX volatility.
- Correlation up/down scenario PV.

## Planned Enhancements

- Joint equity-FX Monte Carlo benchmark.
- FX rate and conversion-ratio controls.
- Quanto barrier and quanto Asian variants.
- Correlation sensitivity and stress templates.
