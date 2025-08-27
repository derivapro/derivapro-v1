import numpy as np
from scipy.stats import norm
# from app.models.market_data import StockData # absolute improt for testing
# from ..models.market_data import StockData
from market_data import StockData

class BinomialTreeEngineCRR:
    def __init__(self, ticker, strike_price, start_date, end_date,
                 risk_free_rate, volatility, num_steps=100,
                 option_type='call', american=True, dividends=None, ex_div_dates=None, **kwargs):
        """
        Parameters (Inputs as per your UI):
            ticker:         Stock Ticker (string)
            strike_price:   Strike Price (K)
            start_date:     Option start date (YYYY-MM-DD)
            end_date:       Option end date (YYYY-MM-DD)
            risk_free_rate: r
            volatility:     sigma
            num_steps:      N (Binomial tree steps)
            option_type:    'call' or 'put'
            american:       True/False - for American/European option style
        """

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        stock_data = StockData(ticker, start_date, end_date)
        self.S0 = float(stock_data.get_closing_price())
        self.K = float(strike_price)
        self.T = float(stock_data.get_years_difference())
        self.r = float(risk_free_rate)
        self.sigma = float(volatility)
        self.N = int(num_steps)
        self.option_type = option_type.lower()
        self.american = american
        self.dividends = dividends if dividends is not None else []
        self.ex_div_dates = ex_div_dates if ex_div_dates is not None else []
        self.kwargs = kwargs

        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))      # (5) in doc
        self.d = 1 / self.u                                 # (6) in doc
        self.pu = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)  # (16)
        self.pd = 1 - self.pu
        self.discount = np.exp(-self.r * self.dt)
        self.epsilon = 1 if self.option_type == 'call' else -1

    def _setup_forward_tree(self):
        """
        Step 1: Construct the forward tree
        # Forward diffusion process (Eqn 5/6)
        """
        F = np.zeros((self.N + 1, self.N + 1))
        F[0, 0] = self.S0
        for i in range(1, self.N + 1):
            F[i, 0] = F[i - 1, 0] * self.d  # move down
            for j in range(1, i + 1):
                F[i, j] = F[i - 1, j - 1] * self.u  # move up
        return F

    def _setup_spot_tree(self, F):
        """
        Step 2: Deduce spot tree from forward tree (no dividends for now)
        # Eqn (7):   S(i, j) = F(i, j) * exp(-r * (T - i*dt))
        """
        S = np.zeros_like(F)
        for i in range(self.N + 1):
            for j in range(i + 1):
                S[i, j] = F[i, j] * np.exp(-self.r * (self.T - i * self.dt))
        return S

    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        """
        Step 3: Penultimate step Black-Scholes smoothing (Eqn 12-15)
        """
        if T <= 0:
            return max(self.epsilon * (S - K), 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def price(self):
        """
        Step 4: Backward computation of premium, applying early exercise at every step.
        # Eqn (20) for American options
        """
        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        C = np.zeros_like(S)

        # Step 1: Payoff at expiry
        for j in range(self.N + 1):
            C[self.N, j] = max(self.epsilon * (S[self.N, j] - self.K), 0)

        # Step 2: Black-Scholes smoothing at N-1 step
        for j in range(self.N):
            Tleft = self.dt
            bs_val = self._black_scholes_price(S[self.N - 1, j], self.K, Tleft, self.r, self.sigma, self.option_type)
            intrinsic = max(self.epsilon * (S[self.N - 1, j] - self.K), 0)
            if self.american:
                C[self.N - 1, j] = max(bs_val, intrinsic)
            else:
                C[self.N - 1, j] = bs_val

        # Step 3: Backward induction for value tree
        for i in range(self.N - 2, -1, -1):
            for j in range(i + 1):
                continuation = (self.pu * C[i+1, j+1] + self.pd * C[i+1, j]) * self.discount
                intrinsic = max(self.epsilon * (S[i, j] - self.K), 0)
                if self.american:
                    C[i, j] = max(continuation, intrinsic)
                else:
                    C[i, j] = continuation

        return C[0, 0]

    # def _setup_spot_tree_with_dividends(self, F):
    #     raise NotImplementedError("Dividend-adjusted spot trees not yet implemented.")

# === TESTING ===
if __name__ == "__main__":
    # Just update these variables to test different scenarios:
    ticker = "AAPL"
    strike_price = 220
    start_date = "2025-08-13"
    end_date = "2026-01-01"
    risk_free_rate = 0.05      
    volatility = 0.5    
    num_steps = 100   
    option_type = 'call'
    american = True

    # No need for manual input! Change values above for quick tests.
    engine = BinomialTreeEngineCRR(
        ticker=ticker,
        strike_price=strike_price,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        num_steps=num_steps,
        option_type=option_type,
        american=american
    )
    price = engine.price()
    print(f"\n[TEST] CRR {option_type.capitalize()} American Option Price: {price:.4f}")
    print(f"(Stock={ticker}, K={strike_price}, Start={start_date}, End={end_date}, r={risk_free_rate}, sigma={volatility}, N={num_steps})")