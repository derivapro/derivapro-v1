# Enhanced Participation / Buffered Note Methodology

## Product Scope

This note covers the DerivaPro first-pass workflow for enhanced participation and buffered notes. The product provides leveraged upside participation up to a cap and absorbs initial downside through a buffer.

## Pricing Framework

The workflow simulates the terminal reference level and applies a piecewise payoff:

- Positive performance receives the stated participation multiplier.
- Upside return is capped at the stated maximum return.
- Downside losses inside the buffer are absorbed.
- Losses beyond the buffer are passed through using the downside participation rate.

The present value is the discounted expected payoff.

## Key Outputs

- Present value and PV as a percentage of notional.
- Buffer breach probability.
- Cap hit probability.
- Expected note return.
- Monte Carlo standard error.
- Final-level distribution diagnostics.

## Current Limitations

- Single-reference terminal payoff.
- Flat volatility, rate, and dividend assumptions.
- No issuer credit/funding spread adjustment.
- No explicit static option replication display yet.

## Planned Extensions

- Buffered notes with uncapped upside.
- Airbag/downside gearing variants.
- Put-spread replication diagnostics.
- Product-specific reporting and sensitivity surfaces.
