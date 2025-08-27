# Note: last updated on Aug 06

import math
#from cgitb import small
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from app.models.market_data import StockData

class LatticeModel:

    def __init__(self, ticker, strike_price, start_date, end_date, risk_free_rate, volatility):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strike_price = strike_price
        self.time_to_expiry = StockData(ticker, start_date, end_date).get_years_difference()
        self.spot_price = float(StockData(ticker, start_date, end_date).get_closing_price())
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        
    def Cox_Ross_Rubinstein_Tree(self, option_type='call', steps=100, plot_vis='no', greeks=False):
        print(f"[DEBUG] Running Cox Ross Rubinstein Tree: steps={steps}, option_type={option_type}")

        # The factor by which the price rises (assuming it rises) = u
        # The factor by which the price falls (assuming it falls) = d
        # The probability of a price rise = pu
        # The probability of a price fall = pd
        # discount rate = disc
        
        u=math.exp(self.volatility*math.sqrt(self.time_to_expiry/steps))
        d=math.exp(-self.volatility*math.sqrt(self.time_to_expiry/steps))
        pu=((math.exp(self.risk_free_rate*self.time_to_expiry/steps))-d)/(u-d)
        pd=1-pu
        disc=math.exp(-self.risk_free_rate*self.time_to_expiry/steps)

        St = [0] * (steps+1)
        C = [0] * (steps+1)
        
        St[0]=self.spot_price*d**steps
        
        for j in range(1, steps+1): 
            St[j] = St[j-1] * u/d
        
        for j in range(1, steps+1):
            if option_type.lower() == 'put':
                C[j] = max(self.strike_price-St[j],0)
            elif option_type.lower() == 'call':
                C[j] = max(St[j]-self.strike_price,0)
        
        for i in range(steps, 0, -1):
            for j in range(0, i):
                C[j] = disc*(pu*C[j+1]+pd*C[j])

        # Calculate Greeks
        delta = (C[1] - C[0]) / (St[1] - St[0]) if steps > 0 else 0
        gamma = ((C[2] - C[1]) / (St[2] - St[1]) - (C[1] - C[0]) / (St[1] - St[0])) / ((St[1] - St[0]) / 2) if steps > 1 else 0

        if plot_vis.lower() == 'yes':
            stock_prices = [self.spot_price * (u ** i) for i in range(steps + 1)]
            plt.plot(stock_prices, C, label='Option Price')
            plt.xlabel('Stock Price')
            plt.ylabel('Option Price')
            plt.title('Option Price vs. Stock Price')
            plt.legend()
            plt.grid(True)
            plt.show()

        if greeks:
            return {
                'Delta': delta,
                'Gamma': gamma,
                'option_price': C[0]
            }
        else:
            return C[0]

    def CRRGreeks(self, option_type, steps):
        delta = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=True)['Delta']
        gamma = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=True)['Gamma']
        option_price = self.Cox_Ross_Rubinstein_Tree(option_type, steps)

        # Calculate Theta (using a small change in time)
        small_change = 1e-4  # A small change in time
        price_up = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.time_to_expiry -= small_change
        price_down = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        theta = (price_down - price_up) / (2 * small_change)
        self.time_to_expiry += small_change

        # Calculate Vega (using finite difference for volatility)
        small_vol_change = 0.01  # A small change in volatility
        price_up_vol = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.volatility += small_vol_change
        price_down_vol = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        vega = (price_down_vol - price_up_vol) / (2 * small_vol_change)
        self.volatility -= small_vol_change

        # Calculate Rho (using finite difference for interest rate)
        small_rate_change = 0.01  # A small change in risk-free rate
        price_up_rate = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        rho = (price_down_rate - price_up_rate) / (2 * small_rate_change)
        self.risk_free_rate -= small_rate_change

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho,
            'option_price': option_price
        }

    
    def Jarrow_Rudd_Tree(self, option_type='call', steps=100, plot_vis='no', greeks=False):
        print(f"[DEBUG] Running Jarrow Rudd Tree: steps={steps}, option_type={option_type}")
       
        # The factor by which the price rises (assuming it rises) = u
        # The factor by which the price falls (assuming it falls) = d
        # The probability of a price rise = pu
        # The probability of a price fall = pd
        # discount rate = disc
            
        u=math.exp((self.risk_free_rate-(self.volatility**2/2))*self.time_to_expiry/steps+self.volatility*math.sqrt(self.time_to_expiry/steps))
        d=math.exp((self.risk_free_rate-(self.volatility**2/2))*self.time_to_expiry/steps-self.volatility*math.sqrt(self.time_to_expiry/steps))
        pu=0.5
        pd=1-pu
        disc=math.exp(-self.risk_free_rate*self.time_to_expiry/steps)

        St = [0] * (steps+1)
        C = [0] * (steps+1)
        
        St[0]=self.spot_price*d**steps
        
        for j in range(1, steps+1): 
            St[j] = St[j-1] * u/d
        
        for j in range(1, steps+1):
            if option_type.lower() == 'put':
                C[j] = max(self.strike_price-St[j],0)
            elif option_type.lower() == 'call':
                C[j] = max(St[j]-self.strike_price,0)
        
        for i in range(steps, 0, -1):
            for j in range(0, i):
                C[j] = disc*(pu*C[j+1]+pd*C[j])

        # Calculate Greeks
        delta = (C[1] - C[0]) / (St[1] - St[0]) if steps > 0 else 0
        gamma = ((C[2] - C[1]) / (St[2] - St[1]) - (C[1] - C[0]) / (St[1] - St[0])) / ((St[1] - St[0]) / 2) if steps > 1 else 0

        if plot_vis.lower() == 'yes':
            stock_prices = [self.spot_price * (u ** i) for i in range(steps + 1)]
            plt.plot(stock_prices, C, label='Option Price')
            plt.xlabel('Stock Price')
            plt.ylabel('Option Price')
            plt.title('Option Price vs. Stock Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        if greeks:
            return {
                'Delta': delta,
                'Gamma': gamma,
                'option_price': C[0]
            }
        else:
            return C[0]

    def JRTGreeks(self, option_type, steps):
        delta = self.Jarrow_Rudd_Tree(option_type, steps, greeks=True)['Delta']
        gamma = self.Jarrow_Rudd_Tree(option_type, steps, greeks=True)['Gamma']
        # Calculate Theta (using a small change in time)
        small_change = 1e-4  # A small change in time
        price_up = self.Jarrow_Rudd_Tree(option_type, steps)
        self.time_to_expiry -= small_change
        price_down = self.Jarrow_Rudd_Tree(option_type, steps)
        theta = (price_down - price_up) / (2 * small_change)
        self.time_to_expiry += small_change
        # Calculate Vega (using finite difference for volatility)
        small_vol_change = 0.01  # A small change in volatility
        price_up_vol = self.Jarrow_Rudd_Tree(option_type, steps)
        self.volatility += small_vol_change
        price_down_vol = self.Jarrow_Rudd_Tree(option_type, steps)
        vega = (price_down_vol - price_up_vol) / (2 * small_vol_change)
        self.volatility -= small_vol_change
        # Calculate Rho (using finite difference for interest rate)
        small_rate_change = 0.01  # A small change in risk-free rate
        price_up_rate = self.Jarrow_Rudd_Tree(option_type, steps)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Jarrow_Rudd_Tree(option_type, steps)
        rho = (price_down_rate - price_up_rate) / (2 * small_rate_change)
        self.risk_free_rate -= small_rate_change

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }


    ## define a calculator to see optimal number of steps
    def step_optimization(self, option_type='call', start=10, step=50, limit=1000):
        runs1 = list(range(start,limit,step))
        CRR1 = []
        JR1 = []

        for i in runs1:
            CRR1.append(LatticeModel(self.ticker, self.strike_price, self.start_date, self.end_date, self.risk_free_rate, self.volatility).Cox_Ross_Rubinstein_Tree(option_type=option_type, steps=i))
            JR1.append(LatticeModel(self.ticker, self.strike_price, self.start_date, self.end_date, self.risk_free_rate, self.volatility).Jarrow_Rudd_Tree(option_type=option_type, steps=i))

        plt.plot(runs1, CRR1, label='Cox Ross Rubinstein')
        plt.plot(runs1, JR1, label='Jarrow Rudd')
        plt.legend(loc='upper right')
        plt.show()

    def Trinomial_Asset_Pricing(self, option_type='call', steps=100, plot_vis='no'):
        print(f"[DEBUG] Running Trinomial Asset Pricing: steps={steps}, option_type={option_type}")
        
        '''
        Trinomial Asset Pricing model for American Call and Put options
                
        Returns the option price estimated by the trinomial tree
        '''
        deltaT = self.time_to_expiry/steps  
        deltaX = np.sqrt(deltaT*(self.volatility**2) + ((self.risk_free_rate-0.5*self.volatility**2)**2)*(deltaT**2)) 
        u = np.exp(self.volatility*np.sqrt(3*deltaT))
        d=1/u
        D = self.risk_free_rate-(0.5*self.volatility**2)
        
        #check for convergence
        if deltaX < self.volatility*np.sqrt(3*deltaT):
            deltaX = self.volatility*np.sqrt(3*deltaT)
        
        pu = 0.5*(((self.volatility**2*deltaT +D**2*deltaT**2)/deltaX**2) + (deltaT*D/deltaX))
        pm = 1 - ((deltaT*self.volatility**2 + D**2*deltaT**2)/deltaX**2)
        pd = 0.5*(((self.volatility**2*deltaT +D**2*deltaT**2)/deltaX**2) - (deltaT*D/deltaX))
        
        underlying = np.zeros((steps+1,steps+1,steps+1))
        underlying[0,0,0] = self.spot_price
        
        for i in range(1,steps+1):
            underlying[i,0,0] = underlying[i-1,0,0]
            
            for j in range(1,i+1):
                underlying[i,j,0] = underlying[i-1,j-1,0]*u
                
                for k in range(1,j+1):
                    underlying[i,j,k] = underlying[i-1,j-1,k-1]*d
        
        optionval = np.zeros((steps+1,steps+1,steps+1))   
        
        for i in range(steps+1):
            for j in range(i+1):

                if option_type.lower() == 'call':
                    optionval[steps,i,j] = max(0, underlying[steps,i,j] - self.strike_price)
            
                elif option_type.lower() == 'put':
                    optionval[steps,i,j] = max(0, self.strike_price - underlying[steps,i,j])
        
        for i in range(steps-1,-1,-1):
            for j in range(i+1):
                for k in range(j+1):

                    if option_type.lower() == 'call':
                        optionval[i,j,k] = max(0, underlying[i,j,k]-self.strike_price, np.exp(-self.risk_free_rate*deltaT)*(pu*optionval[i+1,j+1,k]+pm*optionval[i+1,j,k]+pd*optionval[i+1,j+1,k+1]))               

                    elif option_type.lower() == 'put':
                        optionval[i,j,k] = max(0, self.strike_price-underlying[i,j,k], np.exp(-self.risk_free_rate*deltaT)*(pu*optionval[i+1,j+1,k]+pm*optionval[i+1,j,k]+pd*optionval[i+1,j+1,k+1]))

        option_price = optionval[0, 0, 0]

        return option_price

    def TAPGreeks(self, option_type, steps):
        option_price = self.Trinomial_Asset_Pricing(option_type, steps)

        original_spot_price = self.spot_price
        self.spot_price *= 1.01
        price_up = self.Trinomial_Asset_Pricing(option_type, steps)
        self.spot_price = original_spot_price * .99
        price_down = self.Trinomial_Asset_Pricing(option_type, steps)
        delta = (price_up - price_down) / (self.spot_price * 0.02)
        self.spot_price = original_spot_price

        # Gamma
        gamma = (price_up - 2 * option_price + price_down) / (self.spot_price * 0.01 ** 2)
        self.spot_price = original_spot_price

        # Theta
        small_change = 1e-4
        price_now = self.Trinomial_Asset_Pricing(option_type, steps)
        self.time_to_expiry -= small_change
        price_down = self.Trinomial_Asset_Pricing(option_type, steps)
        theta = (price_down - price_now) / small_change
        self.time_to_expiry += small_change

        # Vega
        small_vol_change = 0.01
        price_up_vol = self.Trinomial_Asset_Pricing(option_type, steps)
        self.volatility += small_vol_change
        price_down_vol = self.Trinomial_Asset_Pricing(option_type, steps)
        vega = (price_down_vol - price_up_vol) / small_vol_change
        self.volatility -= small_vol_change

        # Rho
        small_rate_change = 0.01
        price_up_rate = self.Trinomial_Asset_Pricing(option_type, steps)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Trinomial_Asset_Pricing(option_type, steps)
        rho = (price_down_rate - price_up_rate) / small_rate_change
        self.risk_free_rate -= small_rate_change

        return {
            'option_price': option_price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    
    def risk_pl_analysis(self, option_type='call', steps=100, price_change=0.01, vol_change=0.01, model='CRR'):
        """
        Risk-Based P&L Analysis for American Options.

        :param option_type: 'call' or 'put'
        :param steps: number of steps in the binomial/trinomial tree
        :param price_change: percentage change in the spot price for the P&L analysis
        :param model: tree model to use ('CRR', 'JR', 'TAP')
        :return: P&L results based on model and sensitivity analysis
        """
        if model == 'CRR':
            price_initial = self.Cox_Ross_Rubinstein_Tree(option_type, steps)
        elif model == 'JRT':
            price_initial = self.Jarrow_Rudd_Tree(option_type, steps)
        elif model == 'TAP':
            price_initial = self.Trinomial_Asset_Pricing(option_type, steps)
        else:
            raise ValueError("Invalid model selection. Choose 'CRR', 'JRT', or 'TAP'.")
        
        original_spot_price = self.spot_price
        original_volatility = self.volatility

        # Adjust the spot price up and down by the specified percentage change
        self.spot_price = original_spot_price * (1 + price_change)
        self.volatility = original_volatility * (1 + vol_change)
        if model == 'CRR':
            price_bump = self.Cox_Ross_Rubinstein_Tree(option_type, steps)
            delta_pl = self.CRRGreeks(option_type, steps)['Delta'] * (1 + price_change)
            gamma_pl = self.CRRGreeks(option_type, steps)['Gamma'] * 0.5 * (1 + price_change)**2
            vega_pl = self.CRRGreeks(option_type, steps) ['Vega'] * (1 + vol_change)          
        elif model == 'JRT':

            price_bump = self.Jarrow_Rudd_Tree(option_type, steps)
            delta_pl = self.JRTGreeks(option_type, steps)['Delta'] * (1 + price_change)
            gamma_pl = self.JRTGreeks(option_type, steps)['Gamma'] * 0.5 * (1 + price_change)**2
            vega_pl = self.JRTGreeks(option_type, steps)['Vega'] * (1 + vol_change)
        elif model == 'TAP':
            price_bump = self.Trinomial_Asset_Pricing(option_type, steps)
            delta_pl = self.TAPGreeks(option_type, steps)['Delta'] * (1 + price_change)
            gamma_pl = self.TAPGreeks(option_type, steps)['Gamma'] * 0.5 * (1 + price_change)**2
            vega_pl = self.TAPGreeks(option_type, steps)['Vega'] * (1 + vol_change)

        # Reset spot price to original
        self.spot_price = original_spot_price
        self.volatility = original_volatility

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

class AmericanOptionSmoothnessTest:
    def __init__(self, ticker, strike_price, start_date, end_date, risk_free_rate, volatility, model, option_type, num_steps):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strike_price = strike_price
        self.time_to_expiry = StockData(ticker, start_date, end_date).get_years_difference()
        self.spot_price = float(StockData(ticker, start_date, end_date).get_closing_price())
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.model = model
        self.option_type = option_type
        self.num_steps = num_steps

    def generate_variable_range(self, variable, num_steps, range_span):
        if variable == 'strike_price':
            base = self.strike_price
        elif variable == 'risk_free_rate':
            base = self.risk_free_rate
        elif variable == 'volatility':
            base = self.volatility
        else:
            raise ValueError("Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', 'volatility'.")

        return np.linspace(base - range_span, base + range_span, num_steps)

    def calculate_single_greek(self, option, target_variable):
        if self.model == 'CRR':
            if target_variable == 'option_price':
                return option.CRRGreeks(self.option_type, self.num_steps)[target_variable]
            else:
                return option.CRRGreeks(self.option_type, self.num_steps)[target_variable[0].upper()+target_variable[1:]]
        elif self.model == 'JRT':
            if target_variable == 'option_price':
                return option.JRTGreeks(self.option_type, self.num_steps)[target_variable]
            else:
                return option.JRTGreeks(self.option_type, self.num_steps)[target_variable[0].upper()+target_variable[1:]]
        else:
            if target_variable == 'option_price':
                return option.TAPGreeks(self.option_type, self.num_steps)[target_variable]
            else:
                return option.TAPGreeks(self.option_type, self.num_steps)[target_variable[0].upper()+target_variable[1:]]

    def calculate_greeks_over_range(self, variable, num_steps, range_span, target_variable):
        variable_values = self.generate_variable_range(variable, num_steps, range_span)
        greek_values = []

        for value in variable_values:
            if variable == 'strike_price':
                option = LatticeModel(self.ticker, value, self.start_date, self.end_date, self.risk_free_rate, self.volatility)
            elif variable == 'risk_free_rate':
                option = LatticeModel(self.ticker, self.strike_price, self.start_date, self.end_date, value, self.volatility)
            elif variable == 'volatility':
                option = LatticeModel(self.ticker, self.strike_price, self.start_date, self.end_date, self.risk_free_rate, value)
            else:
                raise ValueError("Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', 'volatility'.")

            greek_value = self.calculate_single_greek(option, target_variable)
            greek_values.append(greek_value)

        print(f"Variable values: {variable_values}")
        print(f"Greek values: {greek_values}")

        return variable_values, greek_values

    def plot_single_greek(self, variable_values, greek_values, target_variable, variable_name):
        plt.figure(figsize=(10, 6))
        plt.plot(variable_values, greek_values, label=target_variable.capitalize(), color='b')
        plt.title(f'{target_variable.capitalize()} vs {variable_name.capitalize()}')
        plt.xlabel(variable_name.capitalize())
        plt.ylabel(target_variable.capitalize())
        plt.tight_layout()
    
def lattice_convergence_test(max_steps, max_sims, obs, pricer_class, pricer_params, model, option_type, mode='steps'):
    print(f"[DEBUG] lattice_convergence_test called with model: {repr(model)}")

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

        current_params = pricer_params.copy()
        current_params['risk_free_rate'] = current_params.pop('r')
        current_params['volatility'] = current_params.pop('sigma')
        
        del current_params['max_steps']
        del current_params['option_type']
        del current_params['num_steps']
        del current_params['model']
        del current_params['obs']
        del current_params['mode']
        
        for key in ['num_paths', 'mc_steps']:
            if key in current_params:
                del current_params[key]

        lattice = pricer_class(**current_params)

        # Generates paths using the provided path_generator function
        if model == 'Cox Ross Rubinstein Tree':
            option_price = lattice.Cox_Ross_Rubinstein_Tree(option_type, steps=N)
        elif model == 'Jarrow Rudd Tree':
            option_price = lattice.Jarrow_Rudd_Tree(option_type, steps=N)
        elif model == 'Trinomial Asset Pricing':
            option_price = lattice.Trinomial_Asset_Pricing(option_type, steps=N)
        else:
            raise ValueError("Invalid model. Choose 'Cox Ross Rubinstein Tree', 'Jarrow Rudd Tree', or 'Trinomial Asset Pricing'.")

        # results.append((param, option_price))
        results.append((int(param), float(option_price)))
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
option = LatticeModel(ticker='TSLA', spot_price=250, strike_price=251, time_to_expiry=0.5, risk_free_rate=0.05, volatility=0.2)
print(option.Cox_Ross_Rubinstein_Tree(steps=1000))
print(option.Cox_Ross_Rubinstein_Tree(option_type='put'))
print(option.Jarrow_Rudd_Tree(option_type='call', plot_vis='yes'))
print(option.Trinomial_Asset_Pricing(option_type='call'))

option = LatticeModel('AAPL', 80, '2024-01-01', '2025-01-01', 0.05, 0.2)
rbpl = option.risk_pl_analysis(option_type = 'call', steps=100, price_change=0.001, vol_change=0.001, model='CRR')

print(rbpl)
'''

