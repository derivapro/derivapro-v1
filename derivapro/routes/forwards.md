### Pricing Models

The risk-adjusted model uses financing and dividend-yield assumptions:

$$F = S_0 \cdot e^{(r-q)T}$$

The cost-of-carry model extends the calculation with convenience-yield and storage-cost assumptions:

$$F = S_0 \cdot e^{(r-q-c+s)T}$$

Here, $S_0$ is the current underlying level, $r$ is the risk-free rate, $q$ is dividend yield, $c$ is convenience yield, $s$ is storage cost, and $T$ is time to settlement.

### Position Payoff

- Long forward: $S_T-K$
- Short forward: $K-S_T$

$S_T$ is the underlying level at settlement and $K$ is the contractual settlement price. Position P&L also reflects contract count, multiplier, and contract fee.

### Contract Characteristics

A forward is a bilateral over-the-counter agreement. Its terms can be customized, but it generally carries more direct counterparty exposure than a centrally cleared futures contract. The workflow supports long and short positions, risk-adjusted and cost-of-carry models, valuation, settlement P&L, sensitivity analysis, scenarios, and risk-based P&L.