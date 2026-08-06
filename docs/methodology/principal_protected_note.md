# Principal-Protected Market-Linked Note Methodology

## Product Scope

This note covers the DerivaPro first-pass workflow for principal-protected market-linked notes. The structure protects a stated percentage of notional at maturity and provides upside participation in the reference asset, subject to an optional cap.

## Pricing Framework

The implementation simulates the terminal reference level under a flat risk-neutral equity process. The maturity payoff is:

- Protected principal amount.
- Plus positive reference performance multiplied by participation rate.
- Subject to the stated return cap.

The present value is the discounted expected payoff under the user-supplied market assumptions.

## Key Outputs

- Present value and PV as a percentage of notional.
- Protection floor.
- Upside participation probability.
- Cap hit probability.
- Expected note return.
- Final-level and discounted-payoff diagnostics.

## Current Limitations

- The workflow uses a simplified terminal payoff model.
- Flat volatility, rate, and dividend assumptions.
- No issuer credit spread or secondary-market liquidity adjustment.
- No explicit decomposition into zero-coupon bond plus listed/OTC option package yet.

## Planned Extensions

- Option-replication decomposition display.
- Callable and autocallable principal-protected variants.
- Basket/index-linked variants.
- Scenario grids for participation, cap, and protection levels.
