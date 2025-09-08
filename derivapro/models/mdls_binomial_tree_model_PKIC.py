# Assuming no dividends

import numpy as np
from scipy.stats import norm
from market_data import StockData

class BinomialTreeEngineCRR:
    """
    Cox-Ross-Rubinstein (CRR) Binomial Tree Engine

    Methodology:
    - Forward diffusion (u, d) on the forward-to-maturity tree F
    - Deduce spot tree S from F using constant discount rate r
    - Backward induction with Black–Scholes initialization at step N-1
    - American early exercise at each node
    """

    def __init__(
        self,
        ticker,
        strike_price,
        start_date,
        end_date,
        risk_free_rate,
        volatility,
        num_steps=100,
        option_type="call",
        **kwargs,
    ):

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        # Market data
        stock_data = StockData(ticker, start_date, end_date)
        self.S0 = float(stock_data.get_closing_price())
        self.K = float(strike_price)

        # Times: in "no dividend" case, T_e = T_d = T
        # T_e: Time to option expiry in years between t₀ (valuation time/today) and t_m (option maturity)
        # T_d: Time to option delivery in years between t_(op.spot) (option spot date) and t_(op.expiry) (option expiry date)

        self.T = float(stock_data.get_years_difference())  # time to expiry in years
        self.r = float(risk_free_rate)
        self.sigma = float(volatility)
        self.N = int(num_steps)
        self.option_type = option_type.lower()

        # Internal params
        self._rebuild_params()
        self._validate_parameters()

    def diagnostics(self):
        """
        Print key inputs and derived parameters; check that S[0,0] ≈ S0 after forward->spot conversion.
        """
        print("[CRR Diagnostics] Inputs")
        print(f"  S0={self.S0:.6f}, K={self.K:.6f}, T={self.T:.6f}, r={self.r:.6f}, sigma={self.sigma:.6f}, N={self.N}")
        print("[CRR Diagnostics] Derived params")
        print(f"  dt={self.dt:.6f}, u={self.u:.6f}, d={self.d:.6f}, pu={self.pu:.6f}, pd={self.pd:.6f}, discount={self.discount:.6f}")
        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        print("[CRR Diagnostics] Spot tree checks")
        print(f"  S[0,0]={S[0,0]:.6f} (should equal S0)")
        print(f"  S[N,0]={S[self.N,0]:.6f}, S[N,N]={S[self.N,self.N]:.6f}")

    def _rebuild_params(self):
        """
        Recompute time step, up/down factors, risk-neutral probabilities, and discount factors.
        In the no-dividend case, T_e = T_d = T.
        """
        self.dt = self.T / self.N  # step size
        # CRR up/down
        self.u = np.exp(self.sigma * np.sqrt(self.dt))  # eq. (5)
        self.d = 1.0 / self.u                           # eq. (6)

        # FORWARD->SPOT CONSISTENCY:
        # Spot moves are u*e^{r dt} and d*e^{r dt} due to F->S construction.
        # Enforce E[S_{i+1}/S_i] = e^{r dt} => pu*u + (1-pu)*d = 1
        # => pu = (1 - d) / (u - d)
        self.pu = (1.0 - self.d) / (self.u - self.d)   # << CHANGED
        self.pd = 1.0 - self.pu

        self.discount = np.exp(-self.r * self.dt)
        self.epsilon = 1 if self.option_type == "call" else -1

    def _validate_parameters(self):
        if self.S0 <= 0:
            raise ValueError(f"Spot price must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        if self.N <= 0:
            raise ValueError(f"Number of steps must be positive, got {self.N}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"Option type must be 'call' or 'put', got {self.option_type}")
        if not (0.0 <= self.pu <= 1.0):
            raise ValueError(f"Risk-neutral probability pu = {self.pu:.6f} not in [0,1]. Check parameters.")

    def _setup_forward_tree(self):
        """
        Step 1: Forward diffusion (N steps) to construct forward-to-maturity tree F.

        In the no-dividend case, F contains forward-at-maturity values.
        To ensure S[0,0] = S0, initialize F[0,0] = S0 * exp(r * T),
        then diffuse with CRR up/down multipliers u and d.
        """
        F = np.zeros((self.N + 1, self.N + 1))
        # Initialize as forward to maturity
        F[0, 0] = self.S0 * np.exp(self.r * self.T)
        for i in range(1, self.N + 1):
            F[i, 0] = F[i - 1, 0] * self.d
            for j in range(1, i + 1):
                F[i, j] = F[i - 1, j - 1] * self.u
        return F

    def _setup_spot_tree(self, F):
        """
        Step 2: Deduce spot tree from forward tree (No dividends case).
        Using equation (7):
          S(i,j) = F(i,j) * exp(-r * (N - i) * T_d / N)
        In no-dividend case, T_d = T_e = T.
        """
        S = np.zeros_like(F)
        for i in range(self.N + 1):
            time_to_expiry = (self.N - i) * self.dt  # (N - i) * T/N
            discount_to_spot = np.exp(-self.r * time_to_expiry)
            for j in range(i + 1):
                S[i, j] = F[i, j] * discount_to_spot
        return S

    def _bs_step_forward(self, F_node, K, sigma, Te_over_N, r, Td_over_N, option_type):
        """
        Black–Scholes initialization at step N-1 using forward-at-maturity node value.

          d1 = [ ln(F/K) + 0.5 * σ^2 * (T_e/N) ] / [ σ * sqrt(T_e/N) ]
          d2 = d1 - σ * sqrt(T_e/N)

        For call:
          BS = exp(-r * T_d/N) * [ N(d1)*F - N(d2)*K ]         eq. (13)
        For put:
          BS = exp(-r * T_d/N) * [ N(-d2)*K - N(-d1)*F ]       eq. (14)
        """

        F_node = max(F_node, 1e-16)
        K = max(K, 1e-16)
        tau_e = max(Te_over_N, 1e-16)
        vol_sqrt_tau = sigma * np.sqrt(tau_e)

        d1 = (np.log(F_node / K) + 0.5 * sigma * sigma * tau_e) / vol_sqrt_tau
        d2 = d1 - vol_sqrt_tau
        disc = np.exp(-r * Td_over_N) # present value factor

        if option_type == "call":
            return disc * (norm.cdf(d1) * F_node - norm.cdf(d2) * K)
        else:
            return disc * (norm.cdf(-d2) * K - norm.cdf(-d1) * F_node)

    def _build_premium_tree_american(self, F, S):
        """
        Build premium tree for American option via backward induction with BS init.
        Returns the premium lattice C.
        """
        C = np.zeros_like(S)

        # Terminal payoffs at expiration: C[N, j] = max(ϵ(S[N,j] - K), 0)
        for j in range(self.N + 1):
            C[self.N, j] = max(self.epsilon * (S[self.N, j] - self.K), 0.0)

        # Initialization at N-1 using Black–Scholes on forward (better convergence)
        Te_over_N = self.T / self.N       # T_e/N
        Td_over_N = self.T / self.N       # T_d/N (no dividends => equal to T_e/N)
        for j in range(self.N):
            bs_price = self._bs_step_forward(
                F[self.N - 1, j], self.K, self.sigma, Te_over_N, self.r, Td_over_N, self.option_type
            )
            intrinsic = max(self.epsilon * (S[self.N - 1, j] - self.K), 0.0)
            C[self.N - 1, j] = max(bs_price, intrinsic)

        # Backward induction (American)
        for i in range(self.N - 2, -1, -1):
            for j in range(i + 1):
                continuation = (self.pu * C[i + 1, j + 1] + self.pd * C[i + 1, j]) * self.discount
                intrinsic = max(self.epsilon * (S[i, j] - self.K), 0.0)
                C[i, j] = max(continuation, intrinsic, 0.0)

        return C

    def price_american_option(self):
        """
        Modular American option pricer (no dividends).
        """
        F = self._setup_forward_tree()          # Forward-at-maturity price tree (CRR diffusion)
        S = self._setup_spot_tree(F)            # Spot price tree (discounted from F)
        C = self._build_premium_tree_american(F, S)  # Premium tree (backward induction with early exercise)

        return float(C[0, 0])

    def price(self):
        """
        Backward-compat wrapper. Currently routes to American pricer.
        """
        return self.price_american_option()

    # def price_european_option(self):
    #     """
    #     European option pricer (no early exercise) for comparison with American.
    #     Should equal American for calls on non-dividend stocks.
    #     """
    #     F = self._setup_forward_tree()
    #     S = self._setup_spot_tree(F)
    #     C = np.zeros_like(S)

    #     # Terminal payoffs at expiration
    #     for j in range(self.N + 1):
    #         C[self.N, j] = max(self.epsilon * (S[self.N, j] - self.K), 0.0)

    #     # === Black–Scholes initialization at N-1 (no early exercise for European) ===
    #     Te_over_N = self.T / self.N
    #     Td_over_N = self.T / self.N
    #     for j in range(self.N):
    #         bs_price = self._bs_step_forward(
    #             F[self.N - 1, j], self.K, self.sigma, Te_over_N, self.r, Td_over_N, self.option_type
    #         )
    #         C[self.N - 1, j] = bs_price

    #     # Backward induction WITHOUT early exercise (from N-2 down to 0)
    #     for i in range(self.N - 2, -1, -1):
    #         for j in range(i + 1):
    #             C[i, j] = (self.pu * C[i + 1, j + 1] + self.pd * C[i + 1, j]) * self.discount

    #     return float(C[0, 0])
    
    def get_greeks(self):
        """
        Finite-difference Greeks for the American option (no dividends).
        Returns:
        - theta: per-day decay (positive magnitude). For signed theta, flip sign as noted below.
        - vega: per 1% volatility change
        - rho:  per 1% interest-rate change
        """
        base_S0 = self.S0
        base_T = self.T
        base_r = self.r
        base_sigma = self.sigma

        base_price = self.price_american_option()

        # Delta
        ds = 0.01 * max(base_S0, 1.0)
        self.S0 = base_S0 + ds
        price_up = self.price_american_option()
        self.S0 = base_S0 - ds
        price_down = self.price_american_option()
        delta = (price_up - price_down) / (2.0 * ds)
        self.S0 = base_S0

        # Gamma
        gamma = (price_up - 2.0 * base_price + price_down) / (ds * ds)

        # Theta (per-day decay magnitude). If you prefer signed theta, use:
        # theta = (price_shorter - base_price) / dt_theta
        dt_theta = 1.0 / 365.0
        theta = 0.0
        if base_T - dt_theta > 0:
            self.T = base_T - dt_theta
            self._rebuild_params()
            price_shorter = self.price_american_option()
            theta = (base_price - price_shorter)  # per-day magnitude
        self.T = base_T
        self._rebuild_params()

        # Vega: slope per 1.00 vol unit, then scale to per 1%
        dvol = 0.01
        self.sigma = base_sigma + dvol
        self._rebuild_params()
        price_vega_bumped = self.price_american_option()
        vega_slope = (price_vega_bumped - base_price) / dvol
        vega = vega_slope * 0.01  # per 1% vol
        self.sigma = base_sigma
        self._rebuild_params()

        # Rho: slope per 1.00 rate unit, then scale to per 1%
        dr = 0.01
        self.r = base_r + dr
        self._rebuild_params()
        price_rho_bumped = self.price_american_option()
        rho_slope = (price_rho_bumped - base_price) / dr
        rho = rho_slope * 0.01  # per 1% rate
        self.r = base_r
        self._rebuild_params()

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
            "rho": float(rho),
        }

    def get_exercise_boundary(self):
        """
        Compute the optimal exercise boundary for American options (no dividends).
        Returns list of critical stock prices at each time step (from 0 to N-2).
        """

        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        C = np.zeros_like(S)
        boundary = []

        # Terminal
        for j in range(self.N + 1):
            C[self.N, j] = max(self.epsilon * (S[self.N, j] - self.K), 0.0)

        # N-1 init
        Te_over_N = self.T / self.N
        Td_over_N = self.T / self.N
        for j in range(self.N):
            bs_price = self._bs_step_forward(
                F[self.N - 1, j], self.K, self.sigma, Te_over_N, self.r, Td_over_N, self.option_type
            )
            intrinsic = max(self.epsilon * (S[self.N - 1, j] - self.K), 0.0)
            C[self.N - 1, j] = max(bs_price, intrinsic)

        # Backward with boundary scan
        for i in range(self.N - 2, -1, -1):
            critical = None
            for j in range(i + 1):
                continuation = (self.pu * C[i + 1, j + 1] + self.pd * C[i + 1, j]) * self.discount
                intrinsic = max(self.epsilon * (S[i, j] - self.K), 0.0)
                C[i, j] = max(continuation, intrinsic)
                # Mark boundary if early exercise is optimal at this node
                if abs(C[i, j] - intrinsic) < 1e-12 and intrinsic > 0.0:
                    critical = S[i, j]
            boundary.append(critical)

        return list(reversed(boundary))
    
# === MAIN (example) ===
if __name__ == "__main__":
    # Example usage
    engine = BinomialTreeEngineCRR(
        ticker="AAPL",
        strike_price=220,
        start_date="2025-09-03",
        end_date="2026-01-01",
        risk_free_rate=0.05,
        volatility=0.5,
        num_steps=252,
        option_type="call"
    )

    # Print S0 and run diagnostics
    print(f"S0 from MarketData: {engine.S0:.6f}")
    engine.diagnostics()

    american_price = engine.price_american_option()
    
    print(f"American {engine.option_type} price: ${american_price:.6f}")
    greeks = engine.get_greeks()

    print("\nGreeks:")
    print(f"  Delta: {greeks['delta']:.6f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Theta: {greeks['theta']:.6f} (per day)")
    print(f"  Vega:  {greeks['vega']:.6f} (per 1% vol)")
    print(f"  Rho:   {greeks['rho']:.6f} (per 1% rate)")
