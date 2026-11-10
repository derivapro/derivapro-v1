### Pricing Models

The risk-adjusted model uses financing and dividend-yield assumptions:

$$F = S_0 \cdot e^{(r-q)T}$$

The cost-of-carry model extends the calculation with convenience-yield and storage-cost assumptions:

$$F = S_0 \cdot e^{(r-q-c+s)T}$$

Here, $S_0$ is the current underlying level, $r$ is the risk-free rate, $q$ is dividend yield, $c$ is convenience yield, $s$ is storage cost, and $T$ is time to settlement.

### Position Payoff

- Long futures: $S_T-K$
- Short futures: $K-S_T$

$S_T$ is the underlying level at settlement and $K$ is the contractual settlement price. Position P&L also reflects contract count, multiplier, and contract fee.

### Margin and Daily Settlement

Futures are standardized exchange-traded contracts supported by clearing and daily mark-to-market. This workflow calculates initial and maintenance margin requirements and simulates daily P&L, margin balances, and margin calls. It also supports sensitivity analysis, scenarios, and risk-based P&L.

Unlike a bilateral forward, futures gains and losses are generally realized through variation margin over the contract life rather than only at final settlement.
