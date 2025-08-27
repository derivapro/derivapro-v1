import QuantLib as ql
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from derivapro.app.models.market_data import StockData

class AsianOption:
    def __init__(self, ticker, K, sigma, r, q, T, averaging_dates, option_type="call", num_paths=100000, seed=42):
        self.ticker = ticker
        self.S = StockData(ticker).get_current_price()
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.T = T
        self.averaging_dates = averaging_dates
        self.option_type = option_type
        self.num_paths = num_paths
        self.seed = seed

    def price(self):
        print("---[DEBUG: Original/AsianOption Inputs]---")
        print(f"Spot (S): {self.S}")
        print(f"Strike (K): {self.K}")
        print(f"Volatility (sigma): {self.sigma}")
        print(f"Risk-free rate (r): {self.r}")
        print(f"Dividend yield (q): {self.q}")
        print(f"Expiry: {self.T}")
        print(f"Averaging Dates: {self.averaging_dates}")
        print(f"Option Type: {self.option_type}")
        print(f"Num paths: {self.num_paths}")
        print("------------------------------------------")

        print("[DEBUG] Averaging Dates to QuantLib QL.Date conversion:")
        for date in self.averaging_dates:
            print(f"  {date} (datetime) -> {date.day}-{date.month}-{date.year}")
        
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date

        #rng = ql.UniformRandomGenerator(self.seed)
        #seq = ql.UniformRandomSequenceGenerator(1, rng)
        bs_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(self.S)),
            ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, self.r, day_count)
            ),
            ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, self.q, day_count)
            ),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(calculation_date, calendar, self.sigma, day_count)
            ),
        )
        print(type(bs_process))
        engine = ql.MCDiscreteArithmeticAPEngine(
            bs_process, "PseudoRandom", requiredSamples=self.num_paths, seed=self.seed
        )

        averaging_dates = [
            ql.Date(date.day, date.month, date.year) for date in self.averaging_dates
        ]
        print("[DEBUG] QL.AveragingDates (as QL.Date and as year fraction from today):")
        for qd in averaging_dates:
            yf = ql.Actual365Fixed().yearFraction(ql.Date.todaysDate(), qd)
            print(f"  QL.Date: {qd}, year fraction: {yf:.6f}")

        expiry_date = ql.Date(self.T.day, self.T.month, self.T.year)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if self.option_type == "call" else ql.Option.Put, self.K
        )
        exercise = ql.EuropeanExercise(expiry_date)
        asian_option = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic, 0.0, 0, averaging_dates, payoff, exercise
        )

        asian_option.setPricingEngine(engine)
        price = asian_option.NPV()

        return price

    def calculate_greeks(self, epsilon=1e-5):
        """
        Calculate Greek values for an Asian option using finite differences.

        Parameters:
        - epsilon: A small value for finite difference approximation.

        Returns:
        - A dictionary containing the values of Delta, Gamma, Vega, Theta, and Rho.
        """
        base_price = self.price()

        # Delta
        self.S += epsilon
        price_up = self.price()
        self.S -= 2 * epsilon
        price_down = self.price()
        delta = (price_up - price_down) / (2 * epsilon)
        self.S += epsilon  # Restore original S

        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)

        # Vega
        self.sigma += epsilon
        price_up = self.price()
        self.sigma -= 2 * epsilon
        price_down = self.price()
        vega = (price_up - price_down) / (2 * epsilon)
        self.sigma += epsilon  # Restore original sigma

        # Theta
        original_t = self.T
        self.T -= pd.to_timedelta(arg=epsilon, unit='D')
        price_up = self.price()
        self.T = original_t  # Restore original T
        theta = (base_price - price_up) / epsilon

        # Rho
        self.r += epsilon
        price_up = self.price()
        self.r -= 2 * epsilon
        price_down = self.price()
        rho = (price_up - price_down) / (2 * epsilon)
        self.r -= epsilon  # Restore original r

        return {
            "option_price": self.price(),
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }

    def risk_pl_analysis(self, price_change=.01, vol_change=.01):
        price_initial = self.price()

        original_spot_price = self.S
        original_volatility = self.sigma

        # Adjust the spot price up and down by the specified percentage change
        self.S = original_spot_price * (1 + price_change)
        self.sigma = original_volatility * (1 + vol_change)

        price_bump = self.price()
        delta_pl = self.calculate_greeks()['Delta'] * (1 + price_change)
        gamma_pl = self.calculate_greeks()['Gamma'] * .5 * (1 + price_change)
        vega_pl = self.calculate_greeks()['Vega'] * (1 + vol_change)

        # Reset spot price and volatility to original
        self.S = original_spot_price
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

class AsianOptionSmoothnessTest:
    def __init__(self, ticker, K, sigma, r, q, T, averaging_dates, option_type="call", num_paths=100000, seed=42):
        self.ticker = ticker
        self.S = StockData(ticker).get_current_price()
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.T = T
        self.averaging_dates = averaging_dates
        self.option_type = option_type
        self.num_paths = num_paths
        self.seed = seed

    def generate_variable_range(self, variable, range_span, num_steps):
        if variable == 'strike_price':
            base = self.K
        elif variable == 'risk_free_rate':
            base = self.r
        elif variable == 'volatility':
            base = self.sigma
        else:
            raise ValueError("Unsupported variable type. Choose between 'strike_price, 'risk_free_rate', and 'volatility'.")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def calculate_greeks_over_range(self, variable, num_steps, range_span, target_variable):
        variable_values = self.generate_variable_range(variable, range_span, num_steps)
        greek_values = []

        for value in variable_values:
            if variable == 'strike_price':
                option = AsianOption(self.ticker, value, self.sigma, self.r, self.q, self.T, self.averaging_dates,
                                     self.option_type, self.num_paths, self.seed)
            elif variable == 'risk_free_rate':
                option = AsianOption(self.ticker, self.K, self.sigma, value, self.q, self.T, self.averaging_dates,
                                     self.option_type, self.num_paths, self.seed)
            elif variable == 'volatility':
                option = AsianOption(self.ticker, self.K, value, self.r, self.q, self.T, self.averaging_dates,
                                     self.option_type, self.num_paths, self.seed)
            else:
                raise ValueError("Unsupported variable type. Choose between 'strike_price', 'risk_free_rate', or 'volatility'.")

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

def lattice_convergence_test(max_steps, max_sims, obs, pricer_class, pricer_params, mode='steps'):
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
        #N = param if mode == "steps" else steps

        current_params = pricer_params.copy()
        if mode == "steps":
            current_params["num_paths"] = int(param)  # Update time steps
        elif mode == "simulations":
            current_params["num_paths"] = int(param)
        
       
        lattice = pricer_class(**current_params)

        option_price = lattice.price()
        
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
S = 100.0
K = 100.0
sigma = 0.20
r = 0.05
q = 0.01
T = datetime(2024, 12, 31)
averaging_dates = [datetime(2024, 6, 30), datetime(2024, 9, 30), datetime(2024, 12, 31)]
option_type = "call"
num_paths = 2**16

asian_option_test = AsianOption(
    S, K, sigma, r, q, T, averaging_dates, option_type, num_paths
)
price = asian_option_test.price()

print(f"Test Asian Option Price: ${round(price, 2)}")

'''

