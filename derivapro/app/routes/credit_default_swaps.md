**Description:**
A Credit Default Swap is a type of credit derivative financial instrument that allows investors to manage credit risk. 
It is a contract between two parties, the protection buyer and the protection seller. 
The protection buyer pays periodic premiums to the protection seller in exchange for protection against the default of a specific reference entity, such as a corporate bond or a loan. 

In the event of a default, the protection seller compensates the protection buyer for the loss incurred. 
These derivatives provide a way for market participants to transfer or hedge credit risk, thereby enhancing liquidity and enabling efficient risk management.

The DerivaPro Quant Toolbox leverages the Quant Lib library to create the default schedule, hazard curve (deterministic Piecewise Function), default probability, and risk-free curve.
It incorporates parameters such as the nominal value of the reference entity, spread rate, risk-free rate, and recovery rate. 

For the full model documentation on Credit Default Swaps, please refer to the here [documentation](/credit-derivatives).

**Instructions:**
The QDPTB (DerivaPro) enables users to price a Credit Default Swap using Quant-Lib libraries and customized user inputs.
To utilize this derivative pricing tool box, the user must provide the following parameters:

* Nominal value
* Spread
* Recovey Rate 
* Risk-Free Rate
* Calendar Type (i.e., United States, TARGET, United Kingdom, China)
* Side-Type (i.e., Seler/Buyer)
* Selected Tenor (i.e., Annual, Semianual, Quarterly, Monthly)
* Entry Date
* End Date