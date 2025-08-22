**Description:**
A forward contract is a derivative contract between two parties that agree to buy or sell 
an asset at a specified price on a future date. The price of a forward contract is represented as:

* For Risk-Adjusted Models:
$$ F = S_0 \cdot e^{(r - q)T} $$
* For Cost-Carry Models:
$$ F = S_0 \cdot e^{(r - q - c + s)T} $$


Unlike a Futures Contract that trades on an exchange, the Forward Contract is regarded as a Over-the-Counter (OTC) 
financial instrument. The decentralized nature of Forward Contracts allow the parties involve to customize the 
contract agreements, but face a greater level of default risk. 
Parties can enter either a Long or Short Forward Contract position. The basic payoff structure for a Long or Short position is represented as:

* For Long Forward Contracts: Payoff = $\( S_T - K \)$
* For Short Forward Contracts: Payoff = $\( K - S_T \)$

Where $S_T$  is the spot price of the underlying asset at maturity $T$

For the full model documentation on Forward Contracts, please refer to the here [documentation](/futures-forwards). 

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Forward Contract using real-time market data and customized user inputs.
To utilize this derivative pricing tool box, the user must provide the following parameters:

* Ticker (ex. GLD)
* Contract Fee ($)
* Risk-Free Rate (%, ex. 0.05)
* Continous Dividend Yield (%, ex. 0.02)
* Settlement Price ($)
* Num. of Contracts (#)
* Mutliplier (#)
* Entry Date (MM-DD-YYYY)
* Settlement Date (MM-DD-YYYY)
* Position (Long or Short Position)
* Model Selection (Risk Adjusted or Cost-Carry Model)
* Storage Cost (For Cost-Carry Model, represented in $)