**Description:**
An Auto-Callable option is a structured product that typically involves a combination of options and is designed to be 
automatically "called" or exercised before the maturity date if certain pre-defined conditions are met.
The payoff structure differ based on whether the option is called early or if it expires at maturity without being called.

For an Auto-Callable Call Option: If the option is called early (i.e., the underlying asset price reaches or exceeds the barrier level at any time $t<T$), 
the payoff is the strike price plus the coupon (if applicable), or a fixed payout.

* $\text{Fixed Payout}$ if called early (price hits the barrier)
* $\max(S_T - K, 0)$ if not called early and held to expiration $T$

For an Auto-Callable Put Option: If the option is called early (i.e., the underlying asset price reaches or exceeds the barrier level at any time $t<T$), 
the payoff is the strike price minus the coupon (if applicable), or a fixed payout.

* $\text{Fixed Payout}$ if called early (price hits the barrier)
* $\max(K - S_T, 0)$ if not called early and held to expiration $T$

where $S_T$ is the value of the underlying asset at excerise $T$ and $K$ is the strike price.

Auto-Callable options provides the holder an opportunity for an early exit with a guaranteed return (the fixed payout), but if the option is not triggered early, 
the payoff behaves like a regular option based on the strike price and the final underlying asset price.

For the full model documentation on Exotic Auto-Callable Options, please refer to the [documentation](/exotic-options) here.

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Exotic option using real-time market data and customized user inputs. 
For any derivative and pricing method selected, the platform provides the Net Present Value (NPV), relevant Greeks, and supporting analyses.
