**Description:**
Barrier call (put) option is a type of derivative where the payoff depends on whether or not the underlying asset has reached or exceeded a pre-determined price.

Knock-out options expire worthless if the underlying asset breaches the barrier at any time before expiration $T$ (i.e., if the barrier is breached at any $t<T$). 
This limits potential profits for the option holder but also limits losses for the writer. There are two main types:

Up-and-out: The option ceases to exist if, at any time $t<T$, the underlying asset price moves above a barrier set above its initial price.

* Call Option Payoff at Expiration T: $\max(S_T - K, 0)$; 
* Put Option Payoff at Expiration T: $\max(K - S_T, 0)$,

where $S_T$ is the value of the underlying asset at excerise $T$ and $K$ is the strike price.

Down-and-out: The option ceases to exist if, at any time $t<T$, the underlying asset price moves below a barrier set below its initial price.

* Call Option Payoff at Expiration T: $\max(S_T - K, 0)$;
* Put Option Payoff at Expiration T: $\max(K - S_T, 0)$,

where $S_T$ is the value of the underlying asset at excerise $T$ and $K$ is the strike price.

For the full model documentation on Exotic Barrier Options, please refer to the [documentation](/exotic-options) here.

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Exotic option using real-time market data and customized user inputs. 
For any derivative and pricing method selected, the platform provides the Net Present Value (NPV), relevant Greeks, and supporting analyses.
