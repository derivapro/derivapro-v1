import QuantLib as ql
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
from app.models.market_data import StockData

class Forwards:
    def __init__(self, ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date, 
                 settlement_date, settlement_price, num_contracts, multiplier, 
                 position, contract_fee, model_selection, storage_cost):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.convenience_yield = convenience_yield
        self.storage_cost = storage_cost
        self.settlement_price = settlement_price
        self.num_contracts = num_contracts
        self.multiplier = multiplier
        self.position = position  # "long" or "short"
        self.entry_date = entry_date
        self.settlement_date = settlement_date
        self.time_to_maturity = self.calculate_time_to_maturity(entry_date, settlement_date)
        self.stock_data = StockData(ticker=self.ticker, start_date=entry_date)
        self.spot_price_entry = self.get_price_on_date(entry_date)
        self.contract_fee = contract_fee  # Transaction cost per contract
        self.model_selection = model_selection

    def calculate_time_to_maturity(self, entry_date, settlement_date):
        """
        Calculate time to maturity in years using QuantLib.
        """
        entry_date = ql.Date(int(entry_date.split('-')[2]), int(entry_date.split('-')[1]), int(entry_date.split('-')[0]))
        settlement_date = ql.Date(int(settlement_date.split('-')[2]), int(settlement_date.split('-')[1]), int(settlement_date.split('-')[0]))
        time_to_maturity = ql.Actual365Fixed().yearFraction(entry_date, settlement_date)
        return time_to_maturity
    
    def get_price_on_date(self, date):
        """
        Obtain closing price of ticker.
        """
        self.stock_data.start_date = date
        try:
            return self.stock_data.get_closing_price()
        except Exception as e:
            raise ValueError(f"Failed to retrieve price for {self.ticker} on {date}: {e}")
    
    def forward_price(self):
        """
        Calculate the forwards price using the formula:
        F = S * exp((r - q) * T) for Risk Adjusted
        F = S * exp((r + s - c) * T) for Cost Carry (i.e., Commodities)
        r = risk free
        q = continous dividend yield
        c = convience yield
        s = storage cost
        """
        if self.model_selection == 'cost_carry_model':
            storage_cost = self.storage_cost / self.spot_price_entry
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate
                                                            - self.convenience_yield + storage_cost - self.dividend_yield) * self.time_to_maturity), 4) 
            print(storage_cost)
        else:
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity), 4)
        return price
    
    def calculate_profit_loss(self):
        adjusted_entry_price = self.forward_price() + self.contract_fee
        adjusted_settlement_price = self.settlement_price - self.contract_fee

        if self.position.lower() == "long":
            profit_loss = round((adjusted_settlement_price - adjusted_entry_price) * self.multiplier * self.num_contracts,4)
        elif self.position.lower() == "short":
            profit_loss = round((adjusted_entry_price - adjusted_settlement_price) * self.multiplier * self.num_contracts,4)
        else:
            raise ValueError("Position must be 'long' or 'short'")

        return profit_loss
    
    def risk_pl_analysis(self, price_change):
        price_initial = self.forward_price()
        original_spot_price = self.spot_price_entry
        
        self.spot_price_entry = original_spot_price * (1+price_change)
        price_bump = self.forward_price()
        self.spot_price_entry = original_spot_price
        return {
            'Initial Price': "${:,.4f}".format(price_initial),
            'Bumped Price': "${:,.4f}".format(price_bump),
            'Actual P&L': "${:,.4f}".format(price_bump-price_initial)
        }

class ForwardsAnalysis:
    def __init__(self, ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date, 
                 settlement_date, settlement_price, num_contracts, multiplier, 
                 position, contract_fee,num_steps,step_range, model_selection, storage_cost):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.convenience_yield = convenience_yield
        self.storage_cost = storage_cost
        self.settlement_price = settlement_price
        self.num_contracts = num_contracts
        self.multiplier = multiplier
        self.position = position  # "long" or "short"
        self.entry_date = entry_date
        self.settlement_date = settlement_date
        self.time_to_maturity = self.calculate_time_to_maturity(entry_date, settlement_date)
        self.stock_data = StockData(ticker=self.ticker, start_date=entry_date)
        self.spot_price_entry = self.get_price_on_date(entry_date)
        self.contract_fee = contract_fee  # Transaction cost per contract
        self.num_steps = num_steps
        self.step_range = step_range
        self.model_selection = model_selection
    
    def calculate_time_to_maturity(self, entry_date, settlement_date):
        """
        Calculate time to maturity in years using QuantLib.
        """
        entry_date = ql.Date(entry_date.day, entry_date.month, entry_date.year)
        settlement_date = ql.Date(settlement_date.day, settlement_date.month, settlement_date.year)
    
        time_to_maturity = ql.Actual365Fixed().yearFraction(entry_date, settlement_date)
        return time_to_maturity
    def get_price_on_date(self, date):
        """
        Obtain closing price of ticker.
        """
        self.stock_data.start_date = date
        try:
            return self.stock_data.get_closing_price()
        except Exception as e:
            raise ValueError(f"Failed to retrieve price for {self.ticker} on {date}: {e}")
    
    def forward_price(self):
        """
        Calculate the forwards price using the formula:
        F = S * exp((r - q) * T) for Risk Adjusted
        F = S * exp((r + s - c - q) * T) for Cost Carry (i.e., Commodities)
        r = risk free
        q = continous dividend yield
        c = convience yield
        s = storage cost
        """
        if self.model_selection == 'cost_carry_model':
            storage_cost = self.storage_cost/self.spot_price_entry
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate
                                                            - self.convenience_yield + storage_cost - self.dividend_yield) * self.time_to_maturity), 4)
        else:
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity), 4)
        return price
    
    def generate_variable_range(self, variable, range_span, num_steps):
        """
        Generate variable ranges for Risk-Free rate, Dividend Yield, and Convenience Yield.
        """
        if variable == 'risk_free_rate':
            base = self.risk_free_rate
        elif variable == 'dividend_yield':
            base = self.dividend_yield
        elif variable == 'convenience_yield':
            base = self.convenience_yield            
        else:
            raise ValueError(f"Unsupported or missing variable: {variable}")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def analyze_variable_sensitivity(self,variable, range_span, num_steps):
        """
        Sensitize the variables (Risk-Free Rate, Dividend Yield, Convenience Yield)
        """
        variable_range = self.generate_variable_range(variable, range_span, num_steps)
        forward_sensitivity_analysis_results = []

        for value in variable_range:
            if variable == 'risk_free_rate':
                self.risk_free_rate = value
            elif variable == 'dividend_yield':
                self.dividend_yield = value
            elif variable == 'convenience_yield':
                self.convenience_yield = value
            
            self.spot_price_entry = self.get_price_on_date(self.entry_date)
            forward_price = self.forward_price()
            forward_sensitivity_analysis_results.append((value, forward_price))

        return forward_sensitivity_analysis_results

    def plot_sensitivity_analysis(self, variable, range_span, num_steps):
        """
        Plot forward prices based on sensitivity analysis for a given variable.
        """
        sensitivity__analysis_results = self.analyze_variable_sensitivity(variable, range_span, num_steps)
        variable_values, forward_prices = zip(*sensitivity__analysis_results)
        
        # Convert to percentage if the variable is risk-free rate
        variable_values = [val * 100 for val in variable_values]
        x_label = f'{variable.replace("_", " ").title()} (%)'

        plt.figure(figsize=(10, 6))
        plt.plot(variable_values, forward_prices, marker='o', linestyle='-', color='b')
        plt.title(f'Sensitivity Analysis: Forward Prices vs {variable.replace("_", " ").title()}')
        plt.xlabel(x_label)
        plt.ylabel('Forward Price ($)')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    
class Futures:
    def __init__(self, ticker, risk_free_rate,  dividend_yield, entry_date, 
                 settlement_date, settlement_price, num_contracts, multiplier, 
                 position, storage_cost, contract_fee, model_selection, initial_margin_pct, maintenance_margin_pct, convenience_yield, daily_prices=None):
        self.ticker = ticker
        self.stock_data = StockData(ticker=self.ticker, start_date=entry_date, end_date=settlement_date)
        self.entry_date = entry_date
        self.settlement_date = settlement_date
        self.time_to_maturity = self.calculate_time_to_maturity(entry_date, settlement_date)
        self.spot_price_entry = self.get_price_on_date(entry_date)
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.settlement_price = settlement_price
        self.num_contracts = num_contracts
        self.multiplier = multiplier
        self.position = position  # "long" or "short"
        self.storage_cost = storage_cost
        self.initial_margin_pct = initial_margin_pct
        self.maintenance_margin_pct = maintenance_margin_pct
        self.contract_fee = contract_fee
        self.convenience_yield = convenience_yield
        self.model_selection = model_selection
        self.initial_margin, self.maintenance_margin = self.calculate_margin_requirement(initial_margin_pct, maintenance_margin_pct)
        self.initial_margin_cc, self.maintenance_margin_cc = self.calculate_margin_requirement(initial_margin_pct, maintenance_margin_pct)
        self.daily_prices = daily_prices or []  # List of daily settlement prices
    
    def calculate_time_to_maturity(self, entry_date, settlement_date):
        """
        Calculate time to maturity in years using QuantLib.
        """
        entry_date = ql.Date(int(entry_date.split('-')[2]), int(entry_date.split('-')[1]), int(entry_date.split('-')[0]))
        settlement_date = ql.Date(int(settlement_date.split('-')[2]), int(settlement_date.split('-')[1]), int(settlement_date.split('-')[0]))
        time_to_maturity = ql.Actual365Fixed().yearFraction(entry_date, settlement_date)
        return time_to_maturity

    def get_price_on_date(self, date):
        self.stock_data.start_date = date
        try:
            return self.stock_data.get_closing_price()
        except Exception as e:
            raise ValueError(f"Failed to retrieve price for {self.ticker} on {date}: {e}")

    def generate_daily_prices(self):
        """
        Generate a list of daily prices and their corresponding dates between the entry date and settlement date.
        This list will be used in the mark_to_market function.
    
        Updates:
            - self.daily_prices: List of daily closing prices.
            - self.daily_dates: List of corresponding dates.
        """
        try:
            self.stock_data = StockData(self.ticker, self.entry_date, self.settlement_date)
            stock_data = self.stock_data.get_stock_data()
            
            if stock_data.empty:
                raise ValueError("No Data retrieved for the specified date range")
            
            close_column_name = ('Close', self.ticker)
            if close_column_name not in stock_data.columns:
                raise ValueError("No 'Close' column found in the stock data")
            close_column = stock_data[close_column_name]
            
            self.daily_prices = close_column.tolist()
            self.daily_dates = stock_data.index.tolist()
            
            if not self.daily_prices:
                raise ValueError("No daily prices retrieved for the specified date range.")
            
        except Exception as e:
            raise ValueError(f"Failed to generate daily prices: {e}")
            
    def futures_price(self):
        """
        Calculate the forwards price using the formula:
        F = S * exp((r - q) * T) for Risk Adjusted
        F = S * exp((r + s - c - q) * T) for Cost Carry (i.e., Commodities)
        r = risk free
        q = continous dividend yield
        c = convience yield
        s = storage cost
        """        
        if self.model_selection == 'cost_carry_model':
            storage_cost = self.storage_cost / self.spot_price_entry
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate + storage_cost - self.convenience_yield - self.dividend_yield) * self.time_to_maturity), 4) 
        else:
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity), 4)
        return price
    
    def calculate_margin_requirement(self, initial_margin_pct, maintenance_margin_pct):
        """
        Calculate initial and maintenance margin based on Futures Price. It must be 25% of the contract value
        """
        self.initial_margin = self.futures_price() * initial_margin_pct * self.num_contracts * self.multiplier
        self.maintenance_margin = self.initial_margin * maintenance_margin_pct
        
        margin_dict = {
            "Initial Margin:": "${:,.4f}".format(self.initial_margin),
            "Maintenance Margin:": "${:,.4f}".format(self.maintenance_margin)
        }
        return margin_dict

    def calculate_profit_loss(self):
        """
        Calculate profit/loss at settlement.
        """
        adjusted_entry_price = self.futures_price() + self.contract_fee
        adjusted_settlement_price = self.settlement_price - self.contract_fee

        if self.position.lower() == "long":
            profit_loss = round((adjusted_settlement_price - adjusted_entry_price) * self.multiplier * self.num_contracts, 4)
        elif self.position.lower() == "short":
            profit_loss = round((adjusted_entry_price - adjusted_settlement_price) * self.multiplier * self.num_contracts, 4)
        else:
            raise ValueError("Position must be 'long' or 'short'")

        return profit_loss
    
    def mark_to_market(self):
        """
        Perform daily mark-to-market calculations for futures.
        """
        if not self.daily_prices:
            raise ValueError("Daily prices are required for mark-to-market calculations.")
    
        # Initialize margin balance with initial margin
        margin_balance = self.initial_margin # Use the initial margin as the starting point
        pnl_list = []
        margin_balance_list = []
        margin_call_list = []
        margin_call_dates = []

        # Loop through daily prices and calculate daily P&L, margin balance, and margin calls
        for i in range(1, len(self.daily_prices)):
            previous_storage_cost = 0
            current_storage_cost = 0
            previous_price = self.daily_prices[i - 1]
            current_price = self.daily_prices[i]

            if isinstance(previous_price, list) or isinstance(current_price, list):
                raise TypeError(f"Expected a numeric value, but found a list: previous_price={previous_price}, current_price={current_price}")

            # Ensure prices are floats
            previous_price = float(previous_price)  # Convert previous price to float
            current_price = float(current_price)  # Convert current price to float

            if self.model_selection == 'cost_carry_model':
                previous_storage_cost = self.storage_cost / previous_price
                current_storage_cost = self.storage_cost / current_price
                previous_futures_price = round(previous_price * math.exp((self.risk_free_rate + previous_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
                current_futures_price = round(current_price * math.exp((self.risk_free_rate + current_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
            else:
                previous_futures_price = round(previous_price * math.exp((self.risk_free_rate + previous_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
                current_futures_price = round(current_price * math.exp((self.risk_free_rate + current_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 

            adjusted_previous_price = previous_futures_price + self.contract_fee
            adjusted_current_price = current_futures_price - self.contract_fee

            if self.position.lower() == "long":
                daily_pnl = round((adjusted_current_price - adjusted_previous_price) * self.multiplier * self.num_contracts, 4)
            elif self.position.lower() == "short":
                daily_pnl = round((adjusted_previous_price - adjusted_current_price) * self.multiplier * self.num_contracts, 4)
            else:
                raise ValueError("Position must be 'long' or 'short'")

            pnl_list.append(daily_pnl)
            margin_balance += daily_pnl
            margin_balance_list.append(margin_balance)

            current_date = self.daily_dates[i]  # Get the trading date for the current price

            # Check for margin call
            if margin_balance < self.maintenance_margin:
                margin_call_list.append(True)
                margin_call_dates.append(current_date.strftime('%Y-%m-%d'))  # Store the date when margin call is triggered
                margin_balance = self.maintenance_margin  # Top up to initial margin
            else:
                margin_call_list.append(False)
                margin_call_dates.append(current_date.strftime('%Y-%m-%d'))  # Store the date with no margin call

        # Format results
        pnl_list = ["${:,.4f}".format(num) for num in pnl_list]
        margin_balance_list = ["${:,.4f}".format(num) for num in margin_balance_list]
        return {
            "daily_pnl": pnl_list,
            "margin_balance": margin_balance_list,
            "margin_calls": margin_call_list,
            "margin_call_dates": margin_call_dates  # Include all dates for margin calls
        }
        
    def risk_pl_analysis(self, price_change):
        price_initial = self.futures_price()
        original_spot_price = self.spot_price_entry
        
        self.spot_price_entry = original_spot_price * (1+price_change)
        price_bump = self.futures_price()
        
        self.spot_price_entry = original_spot_price
        return {
            'Initial Price': "${:,.4f}".format(price_initial),
            'Bumped Price': "${:,.4f}".format(price_bump),
            'Actual P&L': "${:,.4f}".format(price_bump-price_initial)    
        }

class FuturesAnalysis:
    def __init__(self, ticker, risk_free_rate, dividend_yield, entry_date, 
                 settlement_date, settlement_price, num_contracts, multiplier, 
                 position, contract_fee, num_steps, step_range, storage_cost, 
                 initial_margin_pct, maintenance_margin_pct, model_selection, convenience_yield, daily_prices=None):
        self.ticker = ticker
        self.stock_data = StockData(ticker=self.ticker, start_date=entry_date, end_date=settlement_date)
        self.entry_date = entry_date
        self.storage_cost = storage_cost
        self.model_selection = model_selection
        self.settlement_date = settlement_date
        self.convenience_yield = convenience_yield
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.settlement_price = settlement_price
        self.num_contracts = num_contracts
        self.multiplier = multiplier
        self.position = position  # "long" or "short"
        self.time_to_maturity = self.calculate_time_to_maturity(entry_date, settlement_date)
        self.spot_price_entry = self.get_price_on_date(entry_date)
        self.contract_fee = contract_fee
        self.num_steps = num_steps
        self.step_range = step_range 
        self.daily_prices = daily_prices or []  # List of daily settlement prices
        self.initial_margin_pct =initial_margin_pct
        self.maintenance_margin_pct=maintenance_margin_pct
        self.initial_margin, self.maintenance_margin = self.calculate_margin_requirement(initial_margin_pct, maintenance_margin_pct)        


    def calculate_time_to_maturity(self, entry_date, settlement_date):
        """
        Calculate time to maturity in years using QuantLib.
        """
        entry_date = ql.Date(entry_date.day, entry_date.month, entry_date.year)
        settlement_date = ql.Date(settlement_date.day, settlement_date.month, settlement_date.year)
    
        time_to_maturity = ql.Actual365Fixed().yearFraction(entry_date, settlement_date)
        return time_to_maturity

    def get_price_on_date(self, date):
        self.stock_data.start_date = date  # Assume StockData fetch logic
        try:
            return self.stock_data.get_closing_price()
        except Exception as e:
            raise ValueError(f"Failed to retrieve price for {self.ticker} on {date}: {e}")

    def futures_price(self):
        """
        Calculate the forwards price using the formula:
        F = S * exp((r - q) * T) for Risk Adjusted
        F = S * exp((r + s - c - q) * T) for Cost Carry (i.e., Commodities)
        r = risk free
        q = continous dividend yield
        c = convience yield
        s = storage cost
        """        
        if self.model_selection == 'cost_carry_model':
            storage_cost = self.storage_cost / self.spot_price_entry
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate + storage_cost - self.convenience_yield - self.dividend_yield) * self.time_to_maturity), 4) 
        else:
            price = round(self.spot_price_entry * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity), 4)
        return price

    def generate_variable_range(self, variable, range_span, num_steps):
        if variable == 'risk_free_rate':
            base = self.risk_free_rate
        elif variable == 'dividend_yield':
            base = self.dividend_yield
        elif variable == 'convenience_yield':
            base = self.convenience_yield
        else:
            raise ValueError(f"Unsupported or missing variable: {variable}")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def analyze_variable_sensitivity(self, variable, range_span, num_steps):
        variable_range = self.generate_variable_range(variable, range_span, num_steps)
        future_sensitivity_analysis_results = []

        for value in variable_range:
            if variable == 'risk_free_rate':
                self.risk_free_rate = value
            elif variable == 'dividend_yield':
                self.dividend_yield = value
            elif variable == 'convenience_yield':
                self.convenience_yield = value
            
            self.spot_price_entry = self.get_price_on_date(self.entry_date)
            futures_price = self.futures_price()
            future_sensitivity_analysis_results.append((value, futures_price))

        return future_sensitivity_analysis_results

    def plot_sensitivity_analysis(self, variable, range_span, num_steps):
        """
        Plot future prices based on sensitivity analysis for a given variable.
        """
        sensitivity_analysis_results = self.analyze_variable_sensitivity(variable, range_span, num_steps)
        variable_values, futures_price = zip(*sensitivity_analysis_results)
        
        
        variable_values = [val * 100 for val in variable_values]
        x_label = f'{variable.replace("_", " ").title()} (%)'

        plt.figure(figsize=(10,6))
        plt.plot(variable_values, futures_price, marker='o', linestyle='-', color='b')
        plt.title(f'Sensitivity Analysis: Futures Prices vs {variable.replace("_", " ").title()}')
        plt.xlabel(x_label)
        plt.ylabel('Futures Price ($)')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def calculate_margin_requirement(self, initial_margin_pct, maintenance_margin_pct):
        """
        Calculate initial and maintenance margin based on Futures Price. It must be 25% of the contract value
        """
        self.initial_margin = self.futures_price() * initial_margin_pct
        self.maintenance_margin = self.futures_price() * maintenance_margin_pct * self.num_contracts * self.multiplier
        
        margin_dict = {
            "Initial Margin:": "${:,.4f}".format(self.initial_margin),
            "Maintenance Margin:": "${:,.4f}".format(self.maintenance_margin)
        }
        return margin_dict

    def generate_daily_prices(self):
        """
        Generate a list of daily prices and their corresponding dates between the entry date and settlement date.
        This list will be used in the mark_to_market function.
    
        Updates:
            - self.daily_prices: List of daily closing prices.
            - self.daily_dates: List of corresponding dates.
        """
        try:
            self.stock_data = StockData(self.ticker, self.entry_date, self.settlement_date)
            stock_data = self.stock_data.get_stock_data()
            
            if stock_data.empty:
                raise ValueError("No Data retrieved for the specified date range")
            
            close_column_name = ('Close', self.ticker)
            if close_column_name not in stock_data.columns:
                raise ValueError("No 'Close' column found in the stock data")
            close_column = stock_data[close_column_name]
            
            self.daily_prices = close_column.tolist()
            self.daily_dates = stock_data.index.tolist()
            
            if not self.daily_prices:
                raise ValueError("No daily prices retrieved for the specified date range.")
            
        except Exception as e:
            raise ValueError(f"Failed to generate daily prices: {e}")

    def mark_to_market(self):
        """
        Perform daily mark-to-market calculations for futures.
        """
        if not self.daily_prices:
            raise ValueError("Daily prices are required for mark-to-market calculations.")
    
        # Initialize margin balance with initial margin
        margin_balance = self.maintenance_margin  # Use the initial margin as the starting point
        pnl_list = []
        margin_balance_list = []
        margin_call_list = []
        margin_call_dates = []

        # Loop through daily prices and calculate daily P&L, margin balance, and margin calls
        for i in range(1, len(self.daily_prices)):
            previous_storage_cost = 0
            current_storage_cost = 0
            previous_price = self.daily_prices[i - 1]
            current_price = self.daily_prices[i]

            if isinstance(previous_price, list) or isinstance(current_price, list):
                raise TypeError(f"Expected a numeric value, but found a list: previous_price={previous_price}, current_price={current_price}")

            # Ensure prices are floats
            previous_price = float(previous_price)  # Convert previous price to float
            current_price = float(current_price)  # Convert current price to float

            if self.model_selection == 'cost_carry_model':
                previous_storage_cost = self.storage_cost / previous_price
                current_storage_cost = self.storage_cost / current_price
                previous_futures_price = round(previous_price * math.exp((self.risk_free_rate + previous_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
                current_futures_price = round(current_price * math.exp((self.risk_free_rate + current_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
            else:
                previous_futures_price = round(previous_price * math.exp((self.risk_free_rate + previous_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 
                current_futures_price = round(current_price * math.exp((self.risk_free_rate + current_storage_cost - self.convenience_yield) * self.time_to_maturity), 4) 

            adjusted_previous_price = previous_futures_price + self.contract_fee
            adjusted_current_price = current_futures_price - self.contract_fee

            if self.position.lower() == "long":
                daily_pnl = round((adjusted_current_price - adjusted_previous_price) * self.multiplier * self.num_contracts, 4)
            elif self.position.lower() == "short":
                daily_pnl = round((adjusted_previous_price - adjusted_current_price) * self.multiplier * self.num_contracts, 4)
            else:
                raise ValueError("Position must be 'long' or 'short'")

            pnl_list.append(daily_pnl)
            margin_balance += daily_pnl
            margin_balance_list.append(margin_balance)

            current_date = self.daily_dates[i]  # Get the trading date for the current price

            # Check for margin call
            if margin_balance < self.maintenance_margin:
                margin_call_list.append(True)
                margin_call_dates.append(current_date.strftime('%Y-%m-%d'))  # Store the date when margin call is triggered
                margin_balance = self.maintenance_margin  # Top up to initial margin
                print(f"Margin call triggered on {current_date.strftime('%Y-%m-%d')}")
            else:
                margin_call_list.append(False)
                margin_call_dates.append(current_date.strftime('%Y-%m-%d'))  # Store the date with no margin call

        # Format results
        pnl_list = ["${:,.4f}".format(num) for num in pnl_list]
        margin_balance_list = ["${:,.4f}".format(num) for num in margin_balance_list]
        return {
            "daily_pnl": pnl_list,
            "margin_balance": margin_balance_list,
            "margin_calls": margin_call_list,
            "margin_call_dates": margin_call_dates  # Include all dates for margin calls
        }


    


