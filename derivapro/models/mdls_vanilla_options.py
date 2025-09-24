import numpy as np
from scipy.stats import norm
#import os
import matplotlib.pyplot as plt
from ..models.market_data import StockData
import matplotlib
matplotlib.use('Agg')   # Use the Agg backend for non-interactive plotting
import importlib.util
import sys
import os

# Import the Monte Carlo module with space in filename
monte_carlo_path = os.path.join(os.path.dirname(__file__), 'mdls_monte_carlo_NEW.py')
spec = importlib.util.spec_from_file_location("monte_carlo_module", monte_carlo_path)
if spec is not None:
    monte_carlo_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(monte_carlo_module)
else:
    raise ImportError(f"Could not load Monte Carlo module from {monte_carlo_path}")

class BlackScholes:
    def __init__(self, ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type='call'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.option_type = option_type
        self.time_to_expiry = StockData(ticker, start_date, end_date).get_years_difference()
        self.spot_price = float(StockData(ticker, start_date, end_date).get_closing_price())
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def d1(self):
        d1_numerator = np.log(self.spot_price / self.strike_price) + (self.risk_free_rate + (self.volatility ** 2) / 2) * self.time_to_expiry
        d1_denominator = self.volatility * np.sqrt(self.time_to_expiry)
        return d1_numerator / d1_denominator

    def d2(self):
        return self.d1() - self.volatility * np.sqrt(self.time_to_expiry)

    def call_price(self):
        d1_value = self.d1()
        d2_value = self.d2()
        return self.spot_price * norm.cdf(d1_value) - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2_value)

    def put_price(self):
        d1_value = self.d1()
        d2_value = self.d2()
        return self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(-d2_value) - self.spot_price * norm.cdf(-d1_value)

    def delta(self):
        if self.option_type == 'call':
            return norm.cdf(self.d1())
        elif self.option_type == 'put':
            return norm.cdf(self.d1()) - 1

    def gamma(self):
        d1_value = self.d1()
        return norm.pdf(d1_value) / (self.spot_price * self.volatility * np.sqrt(self.time_to_expiry))

    def vega(self):
        d1_value = self.d1()
        return self.spot_price * norm.pdf(d1_value) * np.sqrt(self.time_to_expiry)

    def theta(self):
        d1_value = self.d1()
        d2_value = self.d2()
        term1 = -(self.spot_price * norm.pdf(d1_value) * self.volatility) / (2 * np.sqrt(self.time_to_expiry))
        if self.option_type == 'call':
            term2 = self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2_value)
            return term1 - term2
        elif self.option_type == 'put':
            term2 = self.risk_free_rate * self.strike_price * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(-d2_value)
            return term1 + term2
    
    def rho(self):
        d2_value = self.d2()
        if self.option_type == 'call':
            return self.strike_price * self.time_to_expiry * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(d2_value)
        elif self.option_type == 'put':
            return -self.strike_price * self.time_to_expiry * np.exp(-self.risk_free_rate * self.time_to_expiry) * norm.cdf(-d2_value)

class SmoothnessTest:
    def __init__(self, ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type='call'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.option_type = option_type
        self.time_to_expiry = StockData(ticker, start_date, end_date).get_years_difference()
        self.spot_price = float(StockData(ticker, start_date, end_date).get_closing_price())
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def generate_variable_range(self, variable, num_steps, range_span):
        """Generate a range of values for the specified variable"""
        if variable == 'strike_price':
            base_value = self.strike_price
        elif variable == 'risk_free_rate':
            base_value = self.risk_free_rate
        elif variable == 'volatility':
            base_value = self.volatility
        else:
            raise ValueError(f"Unsupported variable: {variable}")

        return np.linspace(base_value * (1 - range_span), base_value * (1 + range_span), num_steps)

    def calculate_single_greek(self, option, target_variable):
        """Calculate a single Greek for the given option"""
        if target_variable == 'option_price':
            return option.call_price() if self.option_type == 'call' else option.put_price()
        elif target_variable == 'delta':
            return option.delta()
        elif target_variable == 'gamma':
            return option.gamma()
        elif target_variable == 'vega':
            return option.vega()
        elif target_variable == 'theta':
            return option.theta()
        elif target_variable == 'rho':
            return option.rho()
        else:
            raise ValueError(f"Unsupported target variable: {target_variable}")

    def calculate_greeks_over_range(self, variable, num_steps, range_span, target_variable):
        """Calculate Greeks over a range of variable values"""
        values = self.generate_variable_range(variable, num_steps, range_span)
        greek_values = []

        for val in values:
            if variable == 'strike_price':
                option = BlackScholes(self.ticker, val, self.start_date, self.end_date, self.risk_free_rate, self.volatility, self.option_type)
            elif variable == 'risk_free_rate':
                option = BlackScholes(self.ticker, self.strike_price, self.start_date, self.end_date, val, self.volatility, self.option_type)
            elif variable == 'volatility':
                option = BlackScholes(self.ticker, self.strike_price, self.start_date, self.end_date, self.risk_free_rate, val, self.option_type)
            else:
                raise ValueError(f"Unsupported variable: {variable}")
            
            greek_values.append(self.calculate_single_greek(option, target_variable))

        return values, greek_values

    def plot_single_greek(self, variable_values, greek_values, target_variable, variable_name):
        """Plot a single Greek over variable range"""
        plt.figure(figsize=(10, 6))
        plt.plot(variable_values, greek_values, 'b-', linewidth=2)
        plt.xlabel(variable_name.replace('_', ' ').title())
        plt.ylabel(target_variable.replace('_', ' ').title())
        plt.title(f'{target_variable.replace("_", " ").title()} vs {variable_name.replace("_", " ").title()}')
        plt.grid(True)
        plt.tight_layout()
