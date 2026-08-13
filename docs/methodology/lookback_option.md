# Lookback Option Methodology

## Scope

This note covers the first-wave DerivaPro lookback option workflow for fixed-strike and floating-strike lookback calls and puts.

## Current Pricing Method

The current implementation uses Monte Carlo simulation of geometric Brownian motion paths. Payoffs reference either the path minimum or path maximum:

- Floating-strike call: `S_T - min(S_t)`
- Floating-strike put: `max(S_t) - S_T`
- Fixed-strike call: `max(max(S_t) - K, 0)`
- Fixed-strike put: `max(K - min(S_t), 0)`

The expected payoff is discounted at the risk-free rate.

## Current Outputs

- Present value.
- Monte Carlo standard error.
- Average simulated path minimum and maximum.
- Spot up/down scenario PV.

## Planned Enhancements

- Closed-form continuous-monitoring benchmark cases.
- Brownian bridge correction for discrete monitoring bias.
- Antithetic and Sobol simulation support.
- Greeks and convergence diagnostics.
