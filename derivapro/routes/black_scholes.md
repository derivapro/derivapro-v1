The Black-Scholes Model, developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, is one of the foundational models in financial mathematics used for pricing options and other derivatives.

The model provides a closed-form analytical solution for European-style option prices and has significantly influenced the development of modern derivatives markets.

## Key Assumptions

* Efficient markets: no transaction costs, taxes, or arbitrage opportunities.
* Constant risk-free rate over the option life.
* Lognormal stock dynamics via geometric Brownian motion.
* No dividends during the option life (in the original formulation).
* European exercise only (exercise at maturity).
* Continuous trading and dynamic hedging feasibility.

## Limitations of the Black-Scholes Model

Despite its widespread use, the Black-Scholes framework has important limitations:

* Constant volatility assumption: market volatility is often stochastic and may exhibit clustering or smile/skew effects.
* No-dividend assumption in the original model can limit direct applicability for dividend-paying underlyings.
* Lognormal return assumption may fail under jumps, fat tails, or stressed market regimes.
* Perfect market assumptions (no frictions) are unrealistic in practice due to spreads, funding, and execution constraints.

## Alternative Approaches

* Stochastic volatility models: Heston, SABR.
* Jump-diffusion models: Merton's jump-diffusion framework.
* Local volatility models: volatility varies by time and spot level.
* Binomial/trinomial lattice methods: useful for early-exercise features (for example, American options).