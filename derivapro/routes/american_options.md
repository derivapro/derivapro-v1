**Description:**
An American call or put option gives the holder the right, but not the obligation,
to buy or sell the underlying asset at a specified strike price at any time up
to and including the maturity date.

If exercised at time `t`, the payoff is:

* American call: `max(S_t - K, 0)`
* American put: `max(K - S_t, 0)`

where `S_t` is the underlying price at exercise and `K` is the strike price.

DerivaPro values the product-standard American option workflow with recombining
tree models and backward induction. At each node, the continuation value is
compared with immediate exercise value, which captures the core early-exercise
feature of American options.

**Methodology:**
The full methodology note is available in the repository:
`docs/methodology/american_option.md`.

**Instructions:**
Enter contract terms, market assumptions, and model settings. Market reference
data can be fetched separately and applied to the pricing assumptions only when
the user explicitly chooses to do so. Outputs are intended for research,
evaluation, and model-review support, not as investment advice.
