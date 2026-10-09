**Description:**
A European call or put option gives the holder the right, but not the obligation,
to buy or sell the underlying asset at a specified strike price on the maturity
date. The payoff is:

* European call: `max(S_T - K, 0)`
* European put: `max(K - S_T, 0)`

where `S_T` is the underlying price at maturity and `K` is the strike price.

DerivaPro prices this workflow with an analytical Black-Scholes-Merton benchmark
and, for registered users, a Monte Carlo comparison based on simulated terminal
prices. The page also provides Greeks, premium decomposition, first-pass driver
sensitivity, payoff views, market reference data, and a structured run summary.

**Methodology:**
The full methodology note is available in the repository:
`docs/methodology/european_option.md`.

**Instructions:**
Enter contract terms, market assumptions, and model settings. Market reference
data can be fetched separately and applied to the pricing assumptions only when
the user explicitly chooses to do so. Outputs are intended for research,
evaluation, and model-review support, not as investment advice.
