**Description:**
An Asian call (put) option grants the holder the right, but not the obligation, 
to buy (sell) the underlying asset at a specified price (strike) on the expiration date of the contract.  Similar to European Options, 
the payoff is calculated based on the average price of the underlying asset over a specified period, noted as $S_{\text{average}}$.

If the option is cash-settled, the holder of a call (put) option receives a cash payoff equivalent to:

* for an Asian call: $\max(S_{\text{average}} - K, 0)$,
* for an Asian put: $\max(K - S_{\text{average}}, 0)$

where $S_{\text{average}}$ is the average value of the underlying asset over the specified period (up to expiration $T$) and $K$ is the strike price. 

Asian options provide the purchaser (or seller) a low volatility option because of its average price calculation
and are used by traders who are exposed to the undrelying assset over some period of time.


For the full model documentation on Exotic Asian Options, please refer to the [documentation](/exotic-options) here.

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Exotic option using real-time market data and customized user inputs. 
For any derivative and pricing method selected, the platform provides the Net Present Value (NPV), relevant Greeks, and supporting analyses.
