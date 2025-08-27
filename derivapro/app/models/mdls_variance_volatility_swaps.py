import QuantLib as ql
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay
from dateutil.relativedelta import relativedelta
from derivapro.app.models.market_data import StockData


# class StockData:
#     def __init__(self, ticker, start_date=None, end_date=None):
#         self.ticker = ticker
#         if end_date is None:
#             self.end_date = datetime.today().strftime('%Y-%m-%d')
#         else:
#             self.end_date = str(end_date)
#         if start_date is None:
#             self.start_date = self.get_previous_market_day(self.end_date)
#         else:
#             self.start_date = str(start_date)
      
#     def get_years_difference(self):
#         """
#         Calculate the difference between two dates in years.
        
#         Args:
#         - start_date (str): Start date in 'YYYY-MM-DD' format.
#         - end_date (str): End date in 'YYYY-MM-DD' format.
        
#         Returns:
#         - float: Difference between the dates in years.
#         """
#         start = datetime.strptime(self.start_date, '%Y-%m-%d')
#         end = datetime.strptime(self.end_date, '%Y-%m-%d')
#         difference = relativedelta(end, start)
        
#         # Calculate the difference in years as a float
#         years_difference = difference.years + difference.months / 12 + difference.days / 365.25
#         return years_difference

#     def get_previous_market_day(self, date):
#         return (pd.to_datetime(date) - BDay(1)).strftime('%Y-%m-%d')
    
#     def get_stock_data(self):
#         """
#         Retrieve stock data for the given ticker.
        
#         Args:
#         - start_date (str): Start date in 'YYYY-MM-DD' format. Default is the previous market day.
#         - end_date (str): End date in 'YYYY-MM-DD' format. Default is today's date.
        
#         Returns:
#         - DataFrame: Pandas DataFrame containing stock data.

#         if self.end_date is None:
#             end_date = datetime.today().strftime('%Y-%m-%d')
        
#         if start_date is None:
#             start_date = self.get_previous_market_day(end_date)
#         """
#         stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
#         return stock_data
    
#     def get_current_price(self):
#         """
#         Retrieve the current price of the stock.
#         """
#         stock_info = yf.Ticker(self.ticker)
#         current_price = stock_info.history(period="1d").iloc[-1]['Close']
#         return current_price
    
#     def get_closing_price(self):
#         """
#         Retrieve the closing price of the stock given an input date.
#         """               
#         prev_date = (pd.to_datetime(self.start_date) - BDay(1)).strftime('%Y-%m-%d')

#         stock = yf.Ticker(self.ticker)
#         hist = stock.history(start=prev_date, end=self.start_date)
        
#         closing_price = hist['Close'].iloc[-1]
#         return closing_price

#     def get_implied_volatility(self, expiry_date=None, strike=None, option_type='call'):
#         """
#         Get the implied volatility of the nearest available option contract to the specified expiry date and strike price.
        
#         Args:
#         - expiry_date (str or datetime): Expiry date of the option in 'YYYY-MM-DD' format or as a datetime object. Default is the next available expiry.
#         - strike (float): Strike price of the option.
#         - option_type (str): Type of the option, 'call' or 'put'. Default is 'call'.
        
#         Returns:
#         - dict or str: Dictionary containing the nearest available contract parameters and the estimated implied volatility, 
#                     or a message indicating that the strike price was not found exactly on any contract.
#         """
#         try:
#             # Set default expiry date if not provided
#             if expiry_date is None:
#                 expiry_date = datetime.today() + timedelta(days=30)  # Default to next month's expiry
                
#             # Convert string expiry_date to datetime if needed
#             if isinstance(expiry_date, str):
#                 expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
            
#             # Fetch option chain data for the given expiry date
#             ticker_info = yf.Ticker(self.ticker)
#             options = ticker_info.option_chain(expiry_date.strftime('%Y-%m-%d'))
            
#             # Set default strike price if not provided
#             if strike is None:
#                 if option_type == 'call':
#                     strike_options = options.calls['strike']
#                 else:
#                     strike_options = options.puts['strike']
                    
#                 # Calculate default strike price as the one closest to 100
#                 strike = strike_options.iloc[(strike_options - 100).abs().idxmin()]
            
#             # Select option chain based on option type
#             option_chain = options.calls if option_type == 'call' else options.puts
            
#             # Initialize variables for storing results
#             exact_strike_found = False
#             nearest_strike = None
#             min_diff = float('inf')
#             implied_volatility = None
            
#             # Iterate over option chain data to find nearest strike price
#             for _, option_data in option_chain.iterrows():
#                 diff = abs(option_data['strike'] - strike)
#                 if diff == 0:  # Strike found exactly
#                     exact_strike_found = True
#                     nearest_strike = option_data['strike']
#                     implied_volatility = option_data['impliedVolatility']
#                     break
#                 elif diff < min_diff:
#                     min_diff = diff
#                     nearest_strike = option_data['strike']
#                     implied_volatility = option_data['impliedVolatility']
            
#             # Format expiry date
#             nearest_expiry = expiry_date.strftime('%Y-%m-%d')

#             # If exact strike price not found, return with a note
#             if not exact_strike_found:
#                 return {
#                     'expiry_date': nearest_expiry,
#                     'strike_price': nearest_strike,
#                     'implied_volatility': implied_volatility,
#                     'NOTE': "Implied volatility has been estimated for the nearest available option contract."
#                 }
#             else:  # If exact strike price found, return without a note
#                 return {
#                     'expiry_date': nearest_expiry,
#                     'strike_price': nearest_strike,
#                     'implied_volatility': implied_volatility
#                 }
#         except Exception as e:  # Catch any exceptions and return as string
#             return str(e)

class varianceSwaps:
    def __init__(self, ticker, start_date, end_date, as_of_date, strike_vol, new_strike_vol, vega_notional, risk_free_rate, position,
                rho, kappa, theta, sigma, calendar):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.as_of_date = as_of_date
        self.strike_vol = strike_vol
        self.new_strike_vol = new_strike_vol 
        self.risk_free_rate = risk_free_rate
        self.position = position  # Position in the swap (long or short)
        self.time_to_expiry = StockData(ticker, start_date, end_date).get_years_difference()
        self.vega_notional = vega_notional
        self.rho = rho  # Correlation between asset returns and variance
        self.kappa = kappa  # Rate of mean reversion in the variance process
        self.theta = theta  # Long-term mean of variance process
        self.sigma = sigma  # Volatility of volatility (vol of vol) in the variance process
        self.calendar = calendar  # QuantLib calendar to use for business day calculation
        # Initialize StockData object with both start and end dates
        self.stock_data = StockData(ticker=self.ticker, start_date=self.start_date, end_date=self.end_date)

        # Get the last known asset price (no date argument needed here)
        self.S0 = self.stock_data.get_closing_price()  # This returns the first element in case it's a Series
        if isinstance(self.S0, pd.Series):
            self.S0 = round(self.S0.item(), 4)
        self.v0 = self.realized_variance()  # Realized variance based on self.stock_data
        if isinstance(self.v0, pd.Series):
            self.v0 = round(self.v0.item(), 4)

        # Setting the "as_of_date" to today if not provided
        if as_of_date is None:
            self.as_of_date = datetime.today().date()  # Use datetime.today() directly
        else:
            self.as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()

    def get_price_on_date(self, date):
        """
        Obtain closing price of ticker.
        """
        self.stock_data.start_date = date
        try:
            price = self.stock_data.get_closing_price()
            if isinstance(price, pd.Series):
                price = price.item()
            return price
        except Exception as e:
            raise ValueError(f"Failed to retrieve price for {self.ticker} on {date}: {e}")
    
    def calculate_trading_days(self, start_date, end_date, calendar):
        # Convert start and end dates to QuantLib Date objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        start_qldate = ql.Date(start_date.day, start_date.month, start_date.year)
        end_qldate = ql.Date(end_date.day, end_date.month, end_date.year)
        
        # Calculate the number of business days (trading days) between start and end date
        trading_days = calendar.businessDaysBetween(start_qldate, end_qldate)
        
        return trading_days

    def variance_notional(self):
        strike_vol = float(self.strike_vol) * 100
        var_notional = float(self.vega_notional) / float((strike_vol * 2))
        return round(var_notional, 4)    

    def realized_variance(self):
        # Use the already fetched stock data
        stock_data = self.stock_data.get_stock_data()
        # Calculate log returns
        log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
        
        # Calculate trading days (using the calendar method)
        trading_days = self.calculate_trading_days(self.start_date, self.end_date, self.calendar)
        
        # Calculate realized variance
        real_variance = np.var(log_returns) * trading_days
        if isinstance(real_variance, pd.Series):
            real_variance = real_variance.item()
        return round(real_variance, 4)
    
    def settlement_amount(self, position):
        # Use the stored realized variance value
        real_variance = self.v0  # Already calculated in __init__
        if isinstance(real_variance, pd.Series):
            real_variance = real_variance.item()
        strike_vol = float(self.strike_vol) * 100
        var_notional = self.variance_notional()
        if position == 'long':
            settlement_amount = (real_variance - strike_vol ** 2) * var_notional
        elif position == 'short':
            settlement_amount = (strike_vol ** 2 - real_variance) * var_notional

        if isinstance(settlement_amount, pd.Series):
            settlement_amount = settlement_amount.item()
        return round(settlement_amount, 4)

    def heston_model_sim(self, S0, v0, rho, kappa, theta, sigma, start_date, end_date, calendar, M=1000):
        trading_days = self.calculate_trading_days(start_date, end_date, calendar)
        theta = float(theta)
        kappa = float(kappa)
        sigma = float(sigma)
        delta_t = trading_days / 252.0  # Total time in years
        N = trading_days               # Number of time steps
        dt = delta_t / N              # Time step in years

        # Initialize arrays
        S = np.full((N + 1, M), S0)
        v = np.full((N + 1, M), v0)  # Use variance, not volatility

        # Mean and covariance for correlated Brownian motions
        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])
        
        Z = np.random.multivariate_normal(mu, cov, size=(N, M))

        for i in range(1, N + 1):
            # Ensure variance stays positive (full truncation Euler)
            v_prev = np.maximum(v[i - 1], 0)
            
            # Asset price update using variance
            S[i] = S[i - 1] * np.exp((float(self.risk_free_rate) - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z[i - 1, :, 0])
            
            # Variance update using CIR process (Heston)
            v[i] = np.maximum(
                v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * Z[i - 1, :, 1],
                0
            )

        return S, v

    def simulated_settlement_amount(self, position):
        # Use the stored realized variance value
        real_variance = self.v0  # Already calculated in __init__
        if isinstance(real_variance, pd.Series):
            real_variance = real_variance.item()
        strike_vol = float(self.strike_vol) * 100
        # Simulate volatilities using the Heston model
        S, v = self.heston_model_sim(self.S0, self.v0, self.rho, self.kappa, self.theta, self.sigma,
                                    self.start_date, self.end_date, self.calendar)
        expected_real_variance = np.mean(v)
        
        combined_variance = 0.5 * (real_variance + expected_real_variance)

        var_notional = self.variance_notional()
        if position == 'long':
            simulated_settlement_amount = (combined_variance - strike_vol ** 2) * var_notional
        elif position == 'short':
            simulated_settlement_amount = (strike_vol ** 2 - combined_variance) * var_notional

        if isinstance(simulated_settlement_amount, pd.Series):
            simulated_settlement_amount = simulated_settlement_amount.item()
        return round(simulated_settlement_amount, 4), round(expected_real_variance, 4)

    def current_value(self, position):
        # Use the stored realized variance value
        real_variance = self.v0  # Already calculated in __init__
        if isinstance(real_variance, pd.Series):
            real_variance = real_variance.item()
        strike_vol = float(self.strike_vol) * 100

        # Ensure the 'as_of_date' is today or before
        if self.as_of_date > datetime.today().date():  # Compare with today's date
            raise ValueError("The 'As of Date' must be today or earlier.")
        
        # Calculate total trading days and elapsed trading days
        total_days = self.calculate_trading_days(self.start_date, self.end_date, self.calendar)
        elapsed_days = self.calculate_trading_days(self.start_date, self.as_of_date, self.calendar)
        remaining_days = total_days - elapsed_days

        # Compute weights
        weight_past = elapsed_days / total_days
        weight_remaining = remaining_days / total_days

        if self.new_strike_vol is not None:
            strike_vol_to_use = self.new_strike_vol
        else:
            strike_vol_to_use = strike_vol
        
        # Compute weighted variances
        realized_var = real_variance * weight_past
        simulated_var = (float(strike_vol_to_use) ** 2) * weight_remaining

        current_realized_var = realized_var + simulated_var

        if position == 'long':
            expected_payoff_at_expiration = (current_realized_var - float(self.strike_vol) ** 2) * self.variance_notional()
            current_swap_value = expected_payoff_at_expiration / (1 + (float(self.risk_free_rate) * remaining_days))
        elif position == 'short':
            expected_payoff_at_expiration = (float(self.strike_vol) ** 2 - current_realized_var) * self.variance_notional()
            current_swap_value = expected_payoff_at_expiration / (1 + (float(self.risk_free_rate) * remaining_days))
        
        if isinstance(current_swap_value, pd.Series):
            current_swap_value = current_swap_value.item()
        return round(current_swap_value, 4)


# # Assuming all other imports and StockData are defined
# ticker = 'AAPL'
# start_date = '2025-01-01'  
# end_date = '2026-05-01'
# as_of_date = '2025-05-10'  # Current date for the swap valuation 
# strike_vol = 0.2 # this needs to be interpolated from the volatility surface later
# new_strike_vol = None  # New strike volatility for the swap valuation
# vega_notional = 50000  
# risk_free_rate = 0.01
# position = 'short'  # Position in the swap (long or short)
# rho = -0.5  # Correlation
# kappa = 2.0  # Rate of mean reversion
# theta = 0.03  # Long-term mean variance
# sigma = 0.3  # Volatility of volatility
# calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
# need the daily, weekly, 
# # Instantiate the varianceSwaps class
# variance_swap = varianceSwaps(
#     ticker, start_date, end_date, as_of_date, strike_vol, new_strike_vol, vega_notional, risk_free_rate, position, rho, kappa, theta, sigma, calendar
# )
# # Get variance notional and print with label
# variance_notional = variance_swap.variance_notional()
# print(f"Variance Notional: {variance_notional}")

# # Get realized variance and print with label
# realized_variance = variance_swap.realized_variance()
# print(f"Realized Variance: {realized_variance}")

# # Get settlement amount and print with label
# settlement_amount = variance_swap.settlement_amount(position)
# print(f"Settlement Amount: {settlement_amount}")

# # Get simulated settlement amount and expected real variance, and print with labels
# simulated_settlement_amount, expected_real_variance = variance_swap.simulated_settlement_amount(position)
# print(f"Simulated Settlement Amount: {simulated_settlement_amount}, Expected Real Variance: {expected_real_variance}")

# # Get current value and print with label
# current_value = variance_swap.current_value(position)
# print(f"Current Value: {current_value}")




