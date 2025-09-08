# Last updated Sep 08
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..models.market_data import StockData

class AutoMonteCarlo:
    """
    A class for pricing derivatives using Monte Carlo simulation.

    Parameters:
    - S0: Initial underlying asset price
    - K: Strike price
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - T: Time to maturity
    - q: COntinuous dividend yield
    - N: Number of time steps
    - M: Number of simulation paths
    """

    def __init__(self, ticker, K, r, sigma, T, q, N, M):
        self.ticker = ticker
        self.S0 = StockData(ticker).get_current_price()
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.N = N
        self.M = M
        self.dt = T / N

    def generate_paths(self, discretization="euler"):
        """
        Generate simulation paths using the specified discretization scheme,

        Parameters:
        - discretization: Discretization scheme to use (default: "euler")

        Returns:
        - paths: Array of shape (M, N+1) containing the simluated paths
        """
        paths = np.zeros((self.M, self.N + 1))
        paths[:, 0] = self.S0
        z = np.random.standard_normal((self.M, self.N))

        if discretization == "euler":
            paths[:, 1:] = self.S0 * np.exp(
                np.cumsum(
                    (self.r - self.q - 0.5 * self.sigma**2) * self.dt
                    + self.sigma * np.sqrt(self.dt) * z,
                    axis=1,
                )
            )

        elif discretization == "milstein":
            paths[:, 1:] = self.S0 * np.cumprod(
                1
                + (self.r - self.q) * self.dt
                + self.sigma * np.sqrt(self.dt) * z
                + 0.5 * self.sigma**2 * (z**2 - 1) * self.dt,
                axis=1,
            )
        else:
            raise ValueError(
                f"Unsupported discretization scheme: {discretization}. Supported schemes are 'euler' and 'milstein'."
            )

        self.paths = paths

        return paths

    def plot_paths(self, paths, title="Monte Carlo Paths", plotted_paths=200):
        num_paths = min(plotted_paths, paths.shape[0])
        selected_paths = paths[:num_paths]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=selected_paths.T, ax=ax, legend=False)

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Asset Price")
        ax.set_title(title)
        plt.show()

    def price_autocallable_option(
        self, discretization="euler", barrier_levels=None, coupon_rates=None
    ):
        if barrier_levels is None or coupon_rates is None:
            raise ValueError("Both barrier levels and coupon rates must be specified.")

        paths = self.generate_paths(discretization)

        # Ensure arrays align with the N observation steps (paths[:, 1:] has shape (M, N))
        if isinstance(barrier_levels, (int, float)):
            barrier_levels = np.full(self.N, barrier_levels)
        elif len(barrier_levels) != self.N:
            raise ValueError(f"Barrier levels must have length {self.N}.")

        if isinstance(coupon_rates, (int, float)):
            coupon_rates = np.full(self.N, coupon_rates)
        elif len(coupon_rates) != self.N:
            raise ValueError(f"Coupon rates must have length {self.N}.")

        # --- CHANGE 1: Proper autocall detection ---
        # Original: used np.argmax directly -> false positives when no autocall.
        autocall_check = paths[:, 1:] >= barrier_levels  # shape (M, N)
        autocalled_any = autocall_check.any(axis=1)      # True if path ever breaches
        first_idx_raw = np.argmax(autocall_check, axis=1)
        first_idx = np.where(autocalled_any, first_idx_raw, -1)  # -1 means no autocall

        # --- CHANGE 2: Event-date discounting ---
        # Original: discounted all payoffs with exp(-r * T).
        # Now: discount to actual payoff time (autocall date or maturity).
        event_time = np.where(autocalled_any, (first_idx + 1) * self.dt, self.T)
        discount_factors = np.exp(-self.r * event_time)

        # --- CHANGE 3: Safe coupon indexing ---
        # Original: coupon_rates[first_autocall] could index with invalid positions.
        # Now: only index coupon_rates for autocalled paths.
        autocall_payoffs = np.zeros(self.M)
        if np.any(autocalled_any):
            autocall_payoffs[autocalled_any] = (
                1.0 + coupon_rates[first_idx[autocalled_any]] * self.S0
            )

        # Payoff for non-autocall paths at maturity (kept same as your original logic)
        final_prices = paths[:, -1]
        non_autocall_payoffs = np.where(final_prices >= self.S0, self.S0, final_prices)

        # Combine autocall & non-autocall payoffs
        payoffs = np.where(autocalled_any, autocall_payoffs, non_autocall_payoffs)

        # --- CHANGE 4: Apply per-path discount factors ---
        # Original: exp(-r * T) * mean(payoffs)
        # Now: mean(payoffs * discount_factors)
        option_price = np.mean(payoffs * discount_factors)

        # Debug info (optional)
        if np.any(autocalled_any):
            avg_first_step = np.mean((first_idx[autocalled_any] + 1))
            print(f"[DEBUG Fixed] Num autocalled: {np.sum(autocalled_any)} / {self.M}")
            print(f"[DEBUG Fixed] Avg first autocall step: {avg_first_step:.2f} of {self.N}")
        else:
            print("[DEBUG Fixed] No paths autocalled.")

        return option_price


    def calculate_greeks(self, discretization, barrier_levels, coupon_rates, epsilon=1e-5):
        """
        Calculate the Greek values for the autocallable option.

        Parameters:
        - barrier_levels: Levels at which the autocall feature is activated
        - coupon_rates: Rates applied when the option is called
        - epsilon: Small change to use for finite difference approximations

        Returns:
        - greeks: Dictionary containing delta, gamma, vega, theta, and rho
        """
        # Calculate the option price for the base case
        price_base = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)

        # Delta
        self.S0 += epsilon
        price_up = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
        self.S0 -= epsilon  # Reset S0
        delta = (price_up - price_base) / epsilon

        # Gamma
        self.S0 -= epsilon
        price_down = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
        self.S0 += epsilon # reset S0
        gamma = (price_up - 2 * price_base + price_down) / (epsilon ** 2)

        # Vega
        self.sigma += epsilon
        price_up = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
        self.sigma -= epsilon  # Reset sigma
        vega = (price_up - price_base) / epsilon

        # Theta
        self.T -= epsilon
        price_up = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
        self.T += epsilon  # Reset T
        theta = (price_up - price_base) / epsilon

        # Rho
        self.r += epsilon
        price_up = self.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
        self.r -= epsilon  # Reset r
        rho = (price_up - price_base) / epsilon

        greeks = {
            'option_price': price_base,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

        return greeks

    def risk_pl_analysis(self, discretization, barrier_levels, coupon_rates, price_change=.01, vol_change=.01):
        price_initial = self.price_autocallable_option(discretization, barrier_levels, coupon_rates)

        original_spot_price = self.S0
        original_volatility = self.sigma

        # Adjust the spot price up and down by the specified percentage change
        self.S0 = original_spot_price * (1 + price_change)
        self.sigma = original_volatility * (1 + vol_change)

        price_bump = self.price_autocallable_option(discretization, barrier_levels, coupon_rates)
        delta_pl = self.calculate_greeks(discretization, barrier_levels, coupon_rates)['delta'] * (1 + price_change)
        gamma_pl = self.calculate_greeks(discretization, barrier_levels, coupon_rates)['gamma'] * .5 * (1 + price_change)
        vega_pl = self.calculate_greeks(discretization, barrier_levels, coupon_rates)['vega'] * (1 + vol_change)

        # Reset spot price and volatility to original
        self.S0 = original_spot_price
        self.sigma = original_volatility

        # Return P&L results
        return {
            'Initial Price': price_initial,
            'Bumped Price': price_bump,
            'Actual P&L': price_bump - price_initial,
            'Delta P&L': delta_pl,
            'Vega P&L': vega_pl,
            'Gamma P&L': gamma_pl,
            'Greek P&L Sum': (delta_pl + vega_pl + gamma_pl),
            'Difference': (price_bump - price_initial) - (delta_pl + vega_pl + gamma_pl)
        }

class AutocallableSmoothnessTest:
    def __init__(self, ticker, K, r, sigma, T, q, N, M, discretization='euler', barrier_levels=None, coupon_rates=None):
        self.ticker = ticker
        self.S = StockData(ticker).get_current_price()
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.N = N
        self.M = M
        self.discretization = discretization
        self.barrier_levels = barrier_levels
        self.coupon_rates = coupon_rates

    def generate_variable_range(self, variable, range_span, num_steps):
        if variable == 'strike_price':
            base = self.K
        elif variable == 'risk_free_rate':
            base = self.r
        elif variable == 'volatility':
            base = self.sigma
        else:
            raise ValueError("Unsupported variable type. Choose between 'strike_price', 'risk_free_rate', or 'volatility'.")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def calculate_greeks_over_range(self, variable, num_steps, range_span, target_variable):
        variable_values = self.generate_variable_range(variable, range_span, num_steps)
        greek_values = []

        for value in variable_values:
            if variable == 'strike_price':
                option = AutoMonteCarlo(self.ticker, self.S, value, self.r, self.sigma, self.q, self.N, self.M)
            elif variable == 'risk_free_rate':
                option = AutoMonteCarlo(self.ticker, self.S, self.K, value, self.sigma, self.q, self.N, self.M)
            elif variable == 'volatility':
                option = AutoMonteCarlo(self.ticker, self.S, self.K, self.r, value, self.q, self.N, self.M)
            else:
                raise ValueError("Unsupported variable type. Choose between 'strike_price', 'risk_free_rate', or 'volatility'.")

            greek_value = option.calculate_greeks(self.discretization, self.barrier_levels, self.coupon_rates)[target_variable]
            greek_values.append(greek_value)

        print(f'Variable values: {variable_values}')
        print(f'Greek values: {greek_values}')

        return variable_values, greek_values

    def plot_single_greek(self, variable_values, greek_values, target_variable, variable_name):
        plt.figure(figsize=(10, 6))
        plt.plot(variable_values, greek_values, label=target_variable.capitalize(), color='b')
        plt.title(f'{target_variable.capitalize()} vs {variable_name.capitalize()}')
        plt.xlabel(variable_name.capitalize())
        plt.ylabel(target_variable.capitalize())
        plt.tight_layout()
        
        
def auto_convergence_test(num_steps, max_sims, obs, pricer_class, mode, discretization, barrier_levels, coupon_rates, pricer_params, within_barrier=True, non_barrier_price=None):
    if mode == "steps":
        steps = list(np.linspace(0, num_steps, obs).round().astype(int))
        steps.pop(0)
        sims = max_sims
    elif mode == "simulations":
        steps = num_steps
        sims = list(np.linspace(0, max_sims, obs).round().astype(int))
        sims.pop(0)
    else:
        raise ValueError("Invalid mode. Choose 'steps' or simulations'.")

    results = []

    for param in steps if mode == "steps" else sims:
        N = param if mode == "steps" else steps
        M = sims if mode == "steps" else param
        

        current_params = pricer_params.copy()
        current_params["N"] = N
        current_params["M"] = M
        
        # Create dynamic barrier and coupon rates based on the time step.
        dynamic_barrier_levels = np.full(N, barrier_levels)
        dynamic_coupon_rates = np.full(N, coupon_rates)

        mc = pricer_class(**current_params)

        option_price = mc.price_autocallable_option(discretization, barrier_levels=dynamic_barrier_levels, coupon_rates=dynamic_coupon_rates)

        results.append((param, option_price))

    return results


def plot_convergence(results, mode):
    x, y = zip(*results)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.lineplot(x=x, y=y, marker="o")

    if mode == "steps":
        plt.xlabel("Number of Time Steps")
        plt.title("Option Price Convergence with Increasing Time Steps")
    elif mode == "simulations":
        plt.xlabel("Number of Simulations")
        plt.title("Option Price Convergence with Increasing Simulations")

    plt.ylabel("Option Price")
    plt.tight_layout()
    #plt.show()


'''
mc = MonteCarlo(S0=100, K=100, r=0.05, sigma=0.20, T=1, q=0, N=253, M=10000)

price = mc.price_autocallable_option(
    discretization="euler", barrier_levels=1.1, coupon_rates=0.05
)

print(f"Autocallable option price: {price:.2f}")

mc.plot_paths(mc.paths, title="Monte Carlo Paths for Autocallable Option")
'''
