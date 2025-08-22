**Description:**
A European call (put) option grants the holder the right, but not the obligation, 
to buy (sell) the underlying asset at a specified price (strike) on the expiration date of the contract. 
If the option is cash-settled, the holder of a call (put) option receives a cash payoff equivalent to:

* for a European call: $\max(S_T - K, 0)$,
* for a European put: $\max(K - S_T, 0)$

where $S_T$ is the value of the underlying asset at expiration $T$ and $K$ is the strike price. 

These options are commonly used for hedging against price movements in the underlying asset, 
as well as for speculating on the future direction of the market. 
Vanilla European options are typically valued analytically using the Black-Scholes formula. 
For the full model documentation on Vanilla European Options, please refer to the [documentation](/vanilla-options) here.

**Instructions:**
The QDPTB (DerivaPro) enables users to price a vanilla option using real-time market data and customized user inputs. 
For any derivative and pricing method selected, the platform provides the Net Present Value (NPV), relevant Greeks, and supporting analyses.
