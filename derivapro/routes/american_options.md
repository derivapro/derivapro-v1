**Description:**
An American call (put) option grants the holder the right, but not the obligation, 
to buy (sell) the underlying asset at a specified price (strike price) at any time up until and including the expiration date of the contract. 

For cash-settled options, if an American call (put) option is exercised before expiration $t$:

* for an Amercican call: $\max(S_t - K, 0)$,
* for an American put: $\max(K - S_t, 0)$

where $S_t$ is the value of the underlying asset at exercise $t$ and $K$ is the strike price. 

For cash-settled options, if an American call (put) option is exercised at expiration $T$:

* for an Amercican call: $\max(S_T - K, 0)$,
* for an American put: $\max(K - S_T, 0)$

where $S_T$ is the value of the underlying asset at exercise $T$ and $K$ is the strike price. 

These options provide flexibility and can be used for hedging against price movements in 
underlying assets or speculating on future market direction throughout their life up until expiration. 
Due to their potential for early exercise, pricing American options involves complex valuation techniques 
that may include numerical methods, such as Cox Ross Rubinstein Tree, Jarrow Rudd Tree, or Trinomial Asset Pricing.

For the full model documentation on Vanilla American Options, please refer to the [documentation](/vanilla-options) here.

**Instructions:**
The QDPTB (DerivaPro) enables users to price a vanilla option using real-time market data and customized user inputs. 
For any derivative and pricing method selected, the platform provides the Net Present Value (NPV), relevant Greeks, and supporting analyses.
