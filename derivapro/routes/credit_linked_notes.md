**Description:**
A Credit-Linked Note (CLN) is a structured financial product that combines a bond with a credit derivative. 
It allows investors to take exposure to the credit risk of a reference entity (such as a corporation or sovereign) without directly owning its debt. 

In a CLN, the issuer agrees to pay a periodic coupon to the investor, similar to a regular bond, but the principal repayment is contingent on the credit performance of the reference entity. If the reference entity defaults or experiences a credit event, the investor may lose some or all of the principal, depending on the terms of the CLN. CLNs provide investors with a way to gain exposure to credit risk while receiving higher yields compared to standard bonds, making them attractive to those seeking customized credit exposure.

The DerivaPro Quant Toolbox combines the Credit Default Swap (CDS) framework with fixed income products to build the pricing model for Credit Linked Notes (CLNs). 
The model calculates the bond price (whether fixed or floating, amortizing or non-amortizing) and then subtracts the premium payments from the CDS to derive the final price of the Credit Linked Note.

For the full model documentation on Credit Linked Notes, please refer to the here [documentation](/credit-derivatives).


**Instructions:**
The QDPTB (DerivaPro) enables users to price a Credit Linked Notes using Quant-Lib libraries and customized user inputs.
To utilize this derivative pricing tool box, the user must provide the following parameters:

##### For Bond Pricing:

* Evaluation Date
* Spot Dates
* Spot Rates
* Shocks (comma separated list, i.e. 0.02,0.03,0.04)
* Day Count Conversion,
* Interpolation Method
* Compounding Method
* Compounding Frequency
* Amortizing Bond Selection
* Index Date (for Floating Rate Bonds)
* Index Rate (for Floating Rate Bonds)
* Currency Value (for Floating Rate Bonds)

##### For CDS Contracts:

* Nominal value
* Spread
* Recovey Rate 
* Risk-Free Rate
* Calendar Type (i.e., United States, TARGET, United Kingdom, China)
* Side-Type (i.e., Seler/Buyer)
* Selected Tenor (i.e., Annual, Semianual, Quarterly, Monthly)
* Entry Date
* End Date