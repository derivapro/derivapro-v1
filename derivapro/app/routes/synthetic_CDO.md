**Description:**
Synthetic Collateralized Debt Obligations (CDOs) are complex financial instruments that pool a variety of debt assets, such as bonds or loans, and divide them into tranches with different levels of risk and return. 
These tranches are typically categorized as Equity, Mezzanine, and Senior. 
Each tranche represents a different level of priority in terms of payments and default risk, which in turn affects the potential payoff for investors. 
The tranches allow investors to select investments based on their specific risk tolerance, offering tailored exposure to the underlying assets.

* Equity Tranche: The most junior and riskiest tranche, absorbing the first losses. It offers the highest potential return with higher yields but is the last to be paid after senior and mezzanine tranches.
* Mezzanine Tranche: Positioned between the equity and senior tranches, it carries moderate risk and return. Mezzanine investors are paid after senior tranches, but before equity investors, offering higher yields than senior tranches but lower than equity.
* Senior Tranche: The highest-ranking and least risky tranche, paid first from the cash flows. It offers the lowest yield but is protected from losses up to a certain level.

The DerivaPro Quant Toolbox utilizes the Credit Default Swap (CDS) framework to construct Synthetic CDOs, drawing from CDS's methodology to model the pool of debt assets. 
Using the QuantLib library, the toolbox generates key components such as the default schedule, hazard curve (modeled as a deterministic piecewise function), default probability, and risk-free curve. 
It also incorporates essential parameters, including the nominal value of the reference entity, spread rate, risk-free rate, and recovery rate, to accurately model the synthetic CDO structure.

For the full model documentation on Synthetic Collateralized Debt Obligations, please refer to the here [documentation](/credit-derivatives).

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Synthetic CDOs using Quant-Lib libraries and customized user inputs.
To utilize this derivative pricing tool box, the user must provide the following parameters:

For Tranches:

* Define the Tranche Bounaries for Equity, Mezzanine, and Senior Tranches

For CDS Contracts:

* Nominal value
* Spread
* Recovey Rate 
* Risk-Free Rate
* Calendar Type (i.e., United States, TARGET, United Kingdom, China)
* Side-Type (i.e., Seler/Buyer)
* Selected Tenor (i.e., Annual, Semianual, Quarterly, Monthly)
* Entry Date
* End Date

