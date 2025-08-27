import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from app.models.market_data import StockData

class MonteCarlo:
    required_params = {"ticker", "start_date", "end_date", "r", "sigma", "N", "M"}
    optional_params = {"K", "q", "barrier", "option_type", "barrier_type"}

    def __init__(self, **params):
        for key in self.required_params:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
            setattr(self, key, params[key])

        for key in self.optional_params:
            if key in params:
                setattr(self, key, params[key])

        self.T = StockData(self.ticker, self.start_date, self.end_date).get_years_difference()
        self.S0 = float(StockData(self.ticker, self.start_date, self.end_date).get_closing_price())
        self.dt = self.T / self.N

    def generate_paths(self, discretization="euler"):
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

    def price_barrier_option(self, discretization="euler"):
        barrier_params = {"K", "barrier", "option_type", "barrier_type"}
        missing_params = barrier_params - set(self.__dict__.keys())
        if missing_params:
            raise ValueError(
                f"Missing required parameters for barrier option pricing: {', '.join(missing_params)}"
            )

        paths = self.generate_paths(discretization=discretization)
        payoffs = np.zeros(self.M)
        discount_factor = np.exp(-self.r * self.T)

        if self.barrier_type == "up_and_out":
            knocked_out = np.any(paths > self.barrier, axis=1)
        elif self.barrier_type == "down_and_out":
            knocked_out = np.any(paths < self.barrier, axis=1)
        else:
            raise ValueError("Invalid barrier type.")

        print(f'[Original MC] OptionType={self.option_type}, BarrierType={self.barrier_type}, KnockedOutCount={np.sum(knocked_out)} / {self.M}')

        if self.option_type == "call":
            payoffs[~knocked_out] = np.maximum(paths[~knocked_out, -1] - self.K, 0)
        elif self.option_type == "put":
            payoffs[~knocked_out] = np.maximum(self.K - paths[~knocked_out, -1], 0)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        price = discount_factor * np.mean(payoffs)
        return price

    def plot_paths(self, paths, title="Monte Carlo Paths", plotted_paths=200):
        num_paths = min(plotted_paths, paths.shape[0])
        selected_paths = paths[:num_paths]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=selected_paths.T, ax=ax, legend=False)

        # ax.axhline(y=self.B, color="r", linestyle="-", linewidth=1.5, label="Barrier")

        ax.set_xlabel("Time steps")
        ax.set_ylabel("Asset Price")
        ax.set_title(title)
        #plt.show()

    def calculate_greeks(self, epsilon=1e-5):
        """
        Calculate Greek values for a barrier option using finite differences.

        Parameters:
        - epsilon: A small value for finite difference approximation.

        Returns:
        - A dictionary containing the values of Delta, Gamma, Vega, Theta, and Rho.
        """
        # Ensure necessary parameters for barrier option are set
        barrier_params = {"K", "barrier", "option_type", "barrier_type"}
        missing_params = barrier_params - set(self.__dict__.keys())
        if missing_params:
            raise ValueError(
                f"Missing required parameters for barrier option pricing: {', '.join(missing_params)}"
            )

        # Calculate the price for various perturbations
        base_price = self.price_barrier_option()

        # Delta
        self.S0 += epsilon
        price_up = self.price_barrier_option()
        self.S0 -= 2 * epsilon
        price_down = self.price_barrier_option()
        delta = (price_up - price_down) / (2 * epsilon)
        self.S0 += epsilon  # Restore original S0

        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)

        # Vega
        self.sigma += epsilon
        price_up = self.price_barrier_option()
        self.sigma -= 2 * epsilon
        price_down = self.price_barrier_option()
        vega = (price_up - price_down) / (2 * epsilon)
        self.sigma += epsilon  # Restore original sigma

        # Theta
        self.T -= epsilon
        price_up = self.price_barrier_option()
        self.T += epsilon  # Restore original T
        theta = (base_price - price_up) / epsilon

        # Rho
        self.r += epsilon
        price_up = self.price_barrier_option()
        self.r -= 2 * epsilon
        price_down = self.price_barrier_option()
        rho = (price_up - price_down) / (2 * epsilon)
        self.r -= epsilon  # Restore original r

        return {
            "option_price": self.price_barrier_option(),
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }

    def risk_pl_analysis(self, price_change=.01, vol_change=.01, discretization='euler'):
        price_initial = self.price_barrier_option(discretization)

        original_spot_price = self.S0
        original_volatility = self.sigma

        # Adjust the spot price up and down by the specified percentage change
        self.S0 = original_spot_price * (1 + price_change)
        self.sigma = original_volatility * (1 + vol_change)

        price_bump = self.price_barrier_option(discretization)
        delta_pl = self.calculate_greeks()['Delta'] * (1 + price_change)
        gamma_pl = self.calculate_greeks()['Gamma'] * .5 * (1 + price_change)
        vega_pl = self.calculate_greeks()['Vega'] * (1 + vol_change)

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

class MonteCarloSmoothnessTest:
    required_params = {"ticker", "start_date", "end_date", "r", "sigma", "N", "M"}
    optional_params = {"K", "q", "barrier", "option_type", "barrier_type"}

    def __init__(self, **params):
        for key in self.required_params:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
            setattr(self, key, params[key])

        for key in self.optional_params:
            if key in params:
                setattr(self, key, params[key])

        self.T = StockData(self.ticker, self.start_date, self.end_date).get_years_difference()
        self.S0 = float(StockData(self.ticker, self.start_date, self.end_date).get_closing_price())
        self.dt = self.T / self.N

    def generate_variable_range(self, variable, range_span, num_steps):
        if variable == 'strike_price':
            base = self.K
        elif variable == 'risk_free_rate':
            base = self.r
        elif variable == 'volatility':
            base = self.sigma
        else:
            raise ValueError("Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', or 'volatility'.")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def calculate_greeks_over_range(self, variable, num_steps, range_span, target_variable):
        variable_values = self.generate_variable_range(variable, num_steps, range_span)
        greek_values = []

        for value in variable_values:
            if variable == 'strike_price':
                input_params = {
                    'ticker': self.ticker, 'start_date': self.start_date, 'end_date': self.end_date, 'r': self.r,
                    'sigma': self.sigma, 'N': self.N, 'M': self.M, 'K': value, 'q': self.q, 'barrier': self.barrier,
                    'option_type': self.option_type, 'barrier_type': self.barrier_type
                }
                option = MonteCarlo(**input_params)
            elif variable == 'risk_free_rate':
                input_params = {
                    'ticker': self.ticker, 'start_date': self.start_date, 'end_date': self.end_date, 'r': value,
                    'sigma': self.sigma, 'N': self.N, 'M': self.M, 'K': self.K, 'q': self.q, 'barrier': self.barrier,
                    'option_type': self.option_type, 'barrier_type': self.barrier_type
                }
                option = MonteCarlo(**input_params)
            elif variable == 'volatility':
                input_params = {
                    'ticker': self.ticker, 'start_date': self.start_date, 'end_date': self.end_date, 'r': self.r,
                    'sigma': value, 'N': self.N, 'M': self.M, 'K': self.K, 'q': self.q, 'barrier': self.barrier,
                    'option_type': self.option_type, 'barrier_type': self.barrier_type
                }
                option = MonteCarlo(**input_params)
            else:
                raise ValueError("Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', or 'volatility'.")

            if target_variable == 'option_price':
                greek_value = option.calculate_greeks()[target_variable]
            else:
                greek_value = option.calculate_greeks()[target_variable[0].upper()+target_variable[1:]]
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

def convergence_test(max_steps, max_sims, obs, pricer_class, pricer_params, mode, within_barrier=True, non_barrier_price=None):
    if mode == "steps":
        steps = list(np.linspace(0, max_steps, obs).round().astype(int))
        steps.pop(0)
        sims = max_sims
    elif mode == "simulations":
        steps = max_steps
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

        mc = pricer_class(**current_params)
        # Generates paths using the provided path_generator function

        if within_barrier:
            option_price = mc.price_barrier_option()
        else:
            option_price = non_barrier_price

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
# Example Barrier Option Convergence Testing
barrier_params = {
    "ticker": 'tsla',
    "start_date": '2020-07-01',
    "end_date": '2024-07-30',
    "K": 80,
    "r": 0.05,
    "sigma": 0.2,
    "q": 0.02,
    "option_type": "call",
    "barrier_type": "up_and_out",
    "barrier": 120,
    "N": 253,
    "M": 1000,
}

print("Convergence testing for steps (Barrier Option):")
barrier_step_results = convergence_test(MonteCarlo, barrier_params, mode="steps")
for steps, price in barrier_step_results:
    print(f"Steps: {steps}, Price: {price:.2f}")

plot_convergence(barrier_step_results, mode="steps")

barrier_simulation_results = convergence_test(
    MonteCarlo, barrier_params, mode="simulations"
)
for simulations, price in barrier_simulation_results:
    print(f"Steps: {simulations}, Price: {price:.2f}")

plot_convergence(barrier_simulation_results, mode="simulations")


class MonteCarlo:
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

    def __init__(self, ticker, K, r, sigma, T, q, N, M, B, option_type, barrier_type):
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
        self.B = B
        self.option_type = option_type
        self.barrier_type = barrier_type

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

        return paths

    def price_barrier_option(self, discretization):
        paths = self.generate_paths(discretization=discretization)
        payoffs = np.zeros(self.M)

        if self.option_type == "call":
            payoffs = np.maximum(paths[:, -1] - self.K, 0)
        elif self.option_type == "put":
            payoffs = np.maximum(self.K - paths[:, -1], 0)
        else:
            raise ValueError("Invalid option type.")

        if self.barrier_type == "up_and_out":
            barrier_crossed = np.any(paths > self.B, axis=1)
        elif self.barrier_type == "down_and_out":
            barrier_crossed = np.any(paths < self.B, axis=1)
        else:
            raise ValueError("Invalid barrier type.")

        payoffs[barrier_crossed] = 0
        discount_factor = np.exp(-self.r * self.T)
        price = discount_factor * np.mean(payoffs)

        return price

    def plot_paths(self, paths, title="Monte Carlo Paths", plotted_paths=200):
        num_paths = min(plotted_paths, paths.shape[0])
        selected_paths = paths[:num_paths]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=selected_paths.T, ax=ax, legend=False)

        ax.axhline(y=self.B, color="r", linestyle="-", linewidth=1.5, label="Barrier")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Asset Price")
        ax.set_title(title)
        plt.show()

mc = MonteCarlo(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1,
    q=0.01,
    N=365,
    M=100000,
    B=120,
    option_type="call",
    barrier_type="up_and_out",
)

paths = mc.generate_paths(discretization="euler")
barrier_option_price = mc.price_barrier_option(discretization="euler")
print(f"Barrier Option price: {barrier_option_price:.2f}")
mc.plot_paths(paths, title="Monte Carlo Paths for Barrier Option")
'''


