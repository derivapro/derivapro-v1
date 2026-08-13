# Basket Option Methodology

## Scope

This note covers the first-wave DerivaPro basket option workflow for weighted multi-asset calls and puts.

## Current Pricing Method

The current implementation uses correlated Monte Carlo simulation. Each asset follows a geometric Brownian motion with user-supplied spot, volatility, and a constant pairwise correlation assumption. The terminal weighted basket is:

```text
B_T = sum(w_i * S_i,T)
```

The option payoff is:

```text
Call = max(B_T - K, 0)
Put  = max(K - B_T, 0)
```

## Current Outputs

- Present value.
- Initial basket level.
- Average terminal basket level.
- Basket up/down scenario PV.

## Planned Enhancements

- Full correlation matrix input.
- Worst-of, best-of, spread, and rainbow payoff variants.
- Local/stochastic volatility and copula alternatives.
- Basket Greeks and correlation sensitivity.
