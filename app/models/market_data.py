import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay
from dateutil.relativedelta import relativedelta

class StockData:
    def __init__(self, ticker, start_date=None, end_date=None):
        self.ticker = ticker
        if end_date is None:
            self.end_date = datetime.today().strftime('%Y-%m-%d')
        else:
            self.end_date = str(end_date)
        if start_date is None:
            self.start_date = self.get_previous_market_day(self.end_date)
        else:
            self.start_date = str(start_date)
      
    def get_years_difference(self):
        """
        Calculate the difference between two dates in years.
        
        Args:
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
        - float: Difference between the dates in years.
        """
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        difference = relativedelta(end, start)
        
        # Calculate the difference in years as a float
        years_difference = difference.years + difference.months / 12 + difference.days / 365.25
        return years_difference

    def get_previous_market_day(self, date):
        return (pd.to_datetime(date) - BDay(1)).strftime('%Y-%m-%d')
    
    def get_stock_data(self):
        """
        Retrieve stock data for the given ticker.
        
        Args:
        - start_date (str): Start date in 'YYYY-MM-DD' format. Default is the previous market day.
        - end_date (str): End date in 'YYYY-MM-DD' format. Default is today's date.
        
        Returns:
        - DataFrame: Pandas DataFrame containing stock data.

        if self.end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = self.get_previous_market_day(end_date)
        """
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return stock_data
    
    def get_current_price(self):
        """
        Retrieve the current price of the stock.
        """
        stock_info = yf.Ticker(self.ticker)
        current_price = stock_info.history(period="1d").iloc[-1]['Close']
        return current_price
    
    def get_closing_price(self):
        """
        Retrieve the closing price of the stock given an input date.
        """               
        prev_date = (pd.to_datetime(self.start_date) - BDay(1)).strftime('%Y-%m-%d')

        stock = yf.Ticker(self.ticker)
        hist = stock.history(start=prev_date, end=self.start_date)
        
        closing_price = hist['Close'].iloc[-1]
        return closing_price

    def get_implied_volatility(self, expiry_date=None, strike=None, option_type='call'):
        """
        Get the implied volatility of the nearest available option contract to the specified expiry date and strike price.
        
        Args:
        - expiry_date (str or datetime): Expiry date of the option in 'YYYY-MM-DD' format or as a datetime object. Default is the next available expiry.
        - strike (float): Strike price of the option.
        - option_type (str): Type of the option, 'call' or 'put'. Default is 'call'.
        
        Returns:
        - dict or str: Dictionary containing the nearest available contract parameters and the estimated implied volatility, 
                    or a message indicating that the strike price was not found exactly on any contract.
        """
        try:
            # Set default expiry date if not provided
            if expiry_date is None:
                expiry_date = datetime.today() + timedelta(days=30)  # Default to next month's expiry
                
            # Convert string expiry_date to datetime if needed
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
            
            # Fetch option chain data for the given expiry date
            ticker_info = yf.Ticker(self.ticker)
            options = ticker_info.option_chain(expiry_date.strftime('%Y-%m-%d'))
            
            # Set default strike price if not provided
            if strike is None:
                if option_type == 'call':
                    strike_options = options.calls['strike']
                else:
                    strike_options = options.puts['strike']
                    
                # Calculate default strike price as the one closest to 100
                strike = strike_options.iloc[(strike_options - 100).abs().idxmin()]
            
            # Select option chain based on option type
            option_chain = options.calls if option_type == 'call' else options.puts
            
            # Initialize variables for storing results
            exact_strike_found = False
            nearest_strike = None
            min_diff = float('inf')
            implied_volatility = None
            
            # Iterate over option chain data to find nearest strike price
            for _, option_data in option_chain.iterrows():
                diff = abs(option_data['strike'] - strike)
                if diff == 0:  # Strike found exactly
                    exact_strike_found = True
                    nearest_strike = option_data['strike']
                    implied_volatility = option_data['impliedVolatility']
                    break
                elif diff < min_diff:
                    min_diff = diff
                    nearest_strike = option_data['strike']
                    implied_volatility = option_data['impliedVolatility']
            
            # Format expiry date
            nearest_expiry = expiry_date.strftime('%Y-%m-%d')

            # If exact strike price not found, return with a note
            if not exact_strike_found:
                return {
                    'expiry_date': nearest_expiry,
                    'strike_price': nearest_strike,
                    'implied_volatility': implied_volatility,
                    'NOTE': "Implied volatility has been estimated for the nearest available option contract."
                }
            else:  # If exact strike price found, return without a note
                return {
                    'expiry_date': nearest_expiry,
                    'strike_price': nearest_strike,
                    'implied_volatility': implied_volatility
                }
        except Exception as e:  # Catch any exceptions and return as string
            return str(e)

'''
ticker = 'f'  # Apple Inc.
date = '2023-01-02'  # Specific date
end_date = '2023-01-04'  # Specific date
closing_price = StockData(ticker, date, end_date).get_closing_price()
print(closing_price)

# Example usage:
ticker = "AAPL" 
s_data = StockData(ticker)

# Get current price
current_price = s_data.get_current_price()
print("Current price:", current_price)

# Get historical data for the previous market day
stock_data = s_data.get_stock_data()
print(stock_data)

# Get difference in years between two dates
start_date = "2020-01-01"
end_date = "2024-07-01"
years_diff = s_data.get_years_difference(start_date, end_date)
print(years_diff)


print("\nImplied Volatility:")
result1 = s_data.get_implied_volatility()
print(result1)

# Example 2: Both parameters specified for standard values
print("\nExample 2:")
result2 = s_data.get_implied_volatility(expiry_date="2024-08-17", strike=1500)
print(result2)

# Example 3: Both parameters specified for wacky values
print("\nExample 3:")
result3 = s_data.get_implied_volatility(expiry_date="2025-06-30", strike=200)
print(result3)
'''
