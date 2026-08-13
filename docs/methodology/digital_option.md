# Digital Option Methodology

## Scope

This note covers the first-wave DerivaPro digital option workflow for cash-or-nothing calls and puts.

## Current Pricing Method

The current implementation uses a closed-form Black-Scholes cash-or-nothing formula. For a call, the discounted payoff is:

```text
PV = exp(-rT) * Payout * N(d2)
```

For a put:

```text
PV = exp(-rT) * Payout * N(-d2)
```

where `d2` is the standard Black-Scholes terminal exercise threshold term.

## Current Outputs

- Present value.
- Risk-neutral exercise probability.
- Discount factor.
- Spot up/down scenario PV.

## Planned Enhancements

- Asset-or-nothing digital variant.
- PDE and binomial/tree validation.
- Greeks with smoothing near the discontinuous strike payoff.
- Barrier digital and double digital variants.
