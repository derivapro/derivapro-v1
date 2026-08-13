# Cliquet / Ratchet Option Methodology

## Scope

This note covers the first-wave DerivaPro cliquet / ratchet option workflow.

## Current Pricing Method

The current implementation uses Monte Carlo reset simulation. The model simulates underlying levels over reset periods, computes local period returns, applies local floor/cap constraints, and then applies global floor/cap constraints to the accumulated return.

```text
R_i = S_i / S_{i-1} - 1
R_i,capped = min(max(R_i, local_floor), local_cap)
R_total = min(max(sum(R_i,capped), global_floor), global_cap)
Payoff = Notional * max(R_total, 0)
```

## Current Outputs

- Present value.
- Average capped return.
- Local/global cap and floor diagnostics.
- Volatility up/down scenario PV.

## Planned Enhancements

- Coupon-style cliquet variants.
- Forward-starting reset schedules.
- Stochastic volatility and realized volatility links.
- P&L attribution by reset period.
