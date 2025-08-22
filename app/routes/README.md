# Quantitative Derivative Pricing Tool Box

Welcome to the Quantitative Derivative Pricing Tool Box (QDPTB)!

---

### Table of Contents

Users are encouraged to read this Markdown file before use of the tool. Included within the guide are the following:

1. [Purpose](#Purpose)
2. [Overview](#Overview)
3. [Derivatives](#Derivatives)
   - [Vanilla Options](#Vanilla-Options)
   - [Asian Options](#Asian-Options)
   - [Barrier Options](#Barrier-Options)
   - [ASR & VWAP+](#ASR-VWAP)
   
   
4. [Pricing Models Used](#Pricing-Models-Used)
   - [Monte Carlo Simulation](#Monte-Carlo-Simulation)
   - [Binomial and Trinomial Tree Lattice Models](#Binomial-and-Trinomial-Tree-Lattice-Models)
   
   
5. [How To Guide](#How-To-Guide)

---

### Purpose

* Design and develop a web-based dashboard with user-friendly options for derivatives selection, validation sets, and reporting.  
* Implement a suite of model valuation (i.e., Option Net Present Value) and risk management (e.g., Sensitivity or Risk-based P&L analyses) in the backend, coupled with an API manager to handle a Generative API (e.g., ChatGPT API) interactions.  
* Incorporate a back-end Market Data API that provides the input data into the derivatives pricing model.  
* Integrate the API to generate AI-driven interpretations and insights based on our analysis results.  
* Ensure data security and compliance with relevant regulations throughout the development process.  

---

### Overview

The QDPTB allows users to price a variety of derivatives using real-time market data. The program allows for customization of several factors, including the underlying derivative pricing method itself. For any derivative and pricing method chosen by the user, Net Present Value, relevant Greeks, and visualizations are provided. Specific toolbox functionalities for different derivative types can be found in more detail within the [How To Guide](#How-To-Guide).  

---

### Derivatives

#### <span style="color: blue;">Vanilla Options</span>

**Overview**: Vanilla options are the most common type of options and include European and American options.

**Descriptions**:  
- **European Options**: Can only be exercised at expiration.  
- **American Options**: Can be exercised at any time before expiration.  

**Mathematical Formulation**:  
- **European Call Option**:  
$$
C = e^{-rT} \mathbb{E}[(S_T - K)^+]
$$  
- **American Call Option**: Similar to European but requires considering the optimal stopping problem.

**Uses**:  
- Hedging against price movements in the underlying asset.
- Speculating on the future direction of the market.

**Notes**: American options are generally more expensive than European options due to the additional flexibility they offer.

**Greeks Calculation**:  
- **Delta**: Measures sensitivity to changes in the underlying asset's price.  
- **Gamma**: Measures the rate of change of Delta.  
- **Theta**: Measures sensitivity to time decay.  
- **Vega**: Measures sensitivity to volatility.  
- **Rho**: Measures sensitivity to interest rates.  

---

#### <span style="color: blue;">Asian Options</span>

**Overview**: Asian options are exotic options where the payoff depends on the average price of the underlying asset over a certain period.

**Descriptions**:   
- **Average Price Option**: Payoff is based on the average price of the asset over the option's life.  
- **Average Strike Option**: The strike price is the average price of the asset over the option's life.  

**Mathematical Formulation**:  
- **Arithmetic Average**:
$$
A = \frac{1}{n} \sum_{i=1}^{n} S_{t_i}
$$
- **Geometric Average**:
$$
G = \left( \prod_{i=1}^{n} S_{t_i} \right)^{\frac{1}{n}}
$$

**Uses**:  
- Reducing the risk of market manipulation at maturity.  
- Used in commodities markets where averaging is common.  

**Notes**: Less sensitive to extreme price movements, which can make them less expensive than standard options.

**Greeks Calculation**: Similar to vanilla options, but calculations are adjusted for the averaging process.

---

#### <span style="color: blue;">Barrier Options</span>

**Overview**: Barrier options are a type of exotic option where the payoff depends on whether the underlying asset reaches a certain price level.

**Descriptions**:   
- **Knock-In Options**: Only become active if the underlying reaches a certain barrier price.  
- **Knock-Out Options**: Become void if the underlying reaches a certain barrier price.

**Mathematical Formulation**:  
- **Up-and-In Call**:
$$
C = e^{-rT} \mathbb{E}[(S_T - K)^+ \mathbb{1}_{\{ \max S_t \geq B \}}]
$$
- **Down-and-Out Call**: Similar but the condition is
$$
\min S_t \leq B.
$$

**Uses**:   
- Cost-effective hedging strategies.  
- Tailoring exposure to specific price levels.

**Notes**: Typically cheaper than standard options because of the conditionality of their payoffs.

**Greeks Calculation**: Similar to vanilla options but with adjustments for the barrier condition.

---

#### <span style="color: blue;">ASR & VWAP+</span>

**Overview**: ASR (Accelerated Share Repurchase) and VWAP+ (Volume Weighted Average Price) are advanced strategies often used by institutional investors.

**Descriptions**:  
- **ASR**: Allows companies to buy back shares quickly.  
- **VWAP+**: Ensures execution of large orders at prices close to the volume-weighted average price.

**Mathematical Formulation**:  
- **VWAP Calculation**:
$$
\text{VWAP} = \frac{\sum_{i=1}^{n} P_i \times Q_i}{\sum_{i=1}^{n} Q_i}
$$

**Uses**:  
- Managing large trades with minimal market impact.  
- Strategic share buybacks.  

**Notes**: These strategies require sophisticated execution algorithms and are typically used in high-frequency trading environments.

---

### Pricing Models Used

#### <span style="color: blue;">Monte Carlo Simulation</span>

**Overview**: Monte Carlo Simulation is a statistical method used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables.

**Descriptions**: Uses random sampling and statistical modeling to estimate mathematical functions and mimic the operation of complex systems.

**Mathematical Formulation**:  
- **Basic Process**:
$$
S_{t+\Delta t} = S_t \exp \left( \left( \mu - \frac{\sigma^2}{2} \right) \Delta t + \sigma \sqrt{\Delta t} Z \right)
$$  
- **Payoff Calculation**: Average the discounted payoffs over a large number of simulated paths.

**Uses**:   
- Valuing complex derivatives with path-dependent features.  
- Risk management and scenario analysis.  

**Notes**: Computationally intensive but highly flexible.

**Steps**:  
1. Generate a large number of possible price paths for the underlying asset.  
2. Calculate the payoff for each path.  
3. Discount the payoffs to present value.  
4. Average the discounted payoffs to obtain the option price.  

**Greeks Calculation**: Can be estimated by finite difference methods on simulated paths.

---

#### <span style="color: blue;">Binomial and Trinomial Tree Lattice Models</span>

**Overview**: Tree lattice models are used to model the possible price paths that an underlying asset can take over the life of the derivative.

**Descriptions**:   
  
***Binomial Tree***: Models two possible price movements (up and down) for each time step.  
***Trinomial Tree***: Models three possible price movements (up, down, and stable) for each time step.  

**Mathematical Formulation**:  
  
***Binomial Tree***:   
  
  Up factor:
  $$
  u = e^{\sigma \sqrt{\Delta t}}
  $$
  Down factor:
  $$
  d = e^{-\sigma \sqrt{\Delta t}}
  $$
  Risk-neutral probability:
  $$
  p = \frac{e^{r \Delta t} - d}{u - d}
  $$

**Uses**:  
  
Pricing American options, where the possibility of early exercise must be considered.  
Providing a clear visual representation of the asset's price evolution.

**Notes**:  
  
Easier to implement than Monte Carlo simulations and useful for educational purposes.

**Steps**:  
  
1. Construct the tree by computing possible prices at each node.  
2. Calculate the option value at the final nodes.  
3. Work backward through the tree to calculate the option value at earlier nodes.  
4. Adjust for early exercise in the case of American options.  

**Greeks Calculation**:  
  
***Delta***: Difference between the values at adjacent nodes.  
***Gamma***: Calculated using central differences in the tree.  
***Theta***: Difference between option values at adjacent time steps.  

---

### How To Guide

This section provides a step-by-step guide on how to use the QDPTB for pricing various derivatives and utilizing different models. Detailed instructions on inputting data, selecting models, and interpreting results will be provided.

*To be developed further.*
