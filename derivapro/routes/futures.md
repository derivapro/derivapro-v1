**Description:**
A Futures Contract is a derivative contract between two parties that agree to buy or sell 
an asset at a specified price on a future date. The price of a Futures Contract is represented as:

* For Risk-Adjusted Models:
$$ F = S_0 \cdot e^{(r - q)T} $$
* For Cost-Carry Models:
$$ F = S_0 \cdot e^{(r - q - c + s)T} $$


A Futures Contract is a standardized financial instrument that trades on an exchange, providing transparency, liquidity, and reduced counterparty risk 
through the involvement of a clearinghouse. This is in contrast to a Forward Contract, which is an over-the-counter (OTC) agreement, 
allowing the parties involved to fully customize the terms of the contract, such as the asset, quantity, and settlement date. 

Parties can enter either a Long or Short Futures Contract position. The basic payoff structure for a Long or Short position is represented as:

* For Long Futures Contracts: Payoff = $\( S_T - K \)$
* For Short Futures Contracts: Payoff = $\( K - S_T \)$

Where $S_T$  is the spot price of the underlying asset at maturity $T$

Note that the payoff for a Futures Contract is typically realized daily through mark-to-market adjustments, a feature supported by the DerivaPro Quant Toolbox.

For the full model documentation on Futures Contracts, please refer to the here [documentation](/futures-forwards). 

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Futures Contract using real-time market data and customized user inputs.
To utilize this derivative pricing tool box, the user must provide the following parameters:

* Ticker (ex. GLD)
* Risk-Free Rate (%, ex. 0.05)
* Continous Dividend Yield (%, ex. 0.02)
* Settlement Price ($)
* Contract Fee ($)
* Storage Cost (For Cost-Carry Model, represented in $)
* Num. of Contracts (#)
* Mutliplier (#)
* Initial Margin Requirement (%)
* Maintenance Margin (%)
* Entry Date (MM-DD-YYYY)
* Settlement Date (MM-DD-YYYY)
* Position (Long or Short Position)
* Model Selection (Risk Adjusted or Cost-Carry Model)
