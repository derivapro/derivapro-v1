import QuantLib as ql
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ..models.mdls_bonds import NCFixedBonds, NCFloatingBonds

class CreditDefaultSwap:
    def __init__(self, nominal, spread, recovery_rate, risk_free,
             payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, variable, range_span, num_steps):
        self.nominal = nominal
        self.spread = spread
        self.recovery_rate = recovery_rate
        self.risk_free = risk_free
        self.calendar = calendar
        self.side = side
        self.selected_period = selected_period
        self.payment_frequency_ql = payment_frequency_ql
        self.period_fraction = period_fraction
        self.entry_date = self._convert_to_quantlib_date(entry_date)  # Convert "yyyy-mm-dd" to QuantLib.Date
        self.end_date = self._convert_to_quantlib_date(end_date)  # Convert "yyyy-mm-dd" to QuantLib.Date
        self.T = int(ql.Actual365Fixed().yearFraction(self.entry_date, self.end_date))
        self.todaysDate = self._convert_to_quantlib_date(entry_date)
        self.variable = variable
        self.range_span = range_span
        self.num_steps = num_steps
        ql.Settings.instance().evaluationDate = self.todaysDate  # Set evaluation date

    def _convert_to_quantlib_date(self, date_input):
        """Helper function to convert date string, datetime.date, or QuantLib Date to QuantLib Date format."""
        if isinstance(date_input, str):
            return ql.Date(*[int(i) for i in date_input.split('-')[::-1]])  # Assumes date format is 'YYYY-MM-DD' 
        elif isinstance(date_input, datetime.date):
            return ql.Date(date_input.day, date_input.month, date_input.year)
        elif isinstance(date_input, ql.Date):
            return date_input       
        else:
            raise ValueError("Date format not recognized. Use 'YYYY-MM-DD' string, datetime.date, or QuantLib Date.")

    def cds_schedule(self):
        cdsSchedule = ql.MakeSchedule(
            self.entry_date, self.end_date, self.selected_period,
            self.payment_frequency_ql, self.calendar,
            ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM
        )
        return cdsSchedule

    def spread_instruments(self):
        # Define the risk-free curve
        risk_free_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todaysDate, self.risk_free, ql.Actual365Fixed())
        )
        
        # Calculate the time to maturity in years
        T = int(ql.Actual365Fixed().yearFraction(self.entry_date, self.end_date))
    
        # Define the tenors (annual, semiannual, quarterly, monthly)
        tenors = [
            ql.Period(T, ql.Years),  # Annual
            ql.Period(6, ql.Months),  # Semiannual
            ql.Period(3, ql.Months),  # Quarterly
            ql.Period(1, ql.Months),  # Monthly
        ]
    
        # Use the single spread input for all instruments
        instruments = [
            ql.SpreadCdsHelper(
                ql.QuoteHandle(ql.SimpleQuote(self.spread)),
                tenor,
                0,  # Upfront (assume no upfront payment)
                self.calendar,
                self.payment_frequency_ql,
                ql.Following,
                ql.DateGeneration.TwentiethIMM,
                ql.Actual365Fixed(),
                self.recovery_rate,
                risk_free_curve
            ) for tenor in tenors
        ]
        
        return instruments

    def hazard_curve_function(self):
        instruments = self.spread_instruments()
        hazard_curve = ql.PiecewiseFlatHazardRate(self.todaysDate, instruments, ql.Actual365Fixed())
        return hazard_curve

    def pricing_engine(self):
        hazard_curve = self.hazard_curve_function()
        default_probability = ql.DefaultProbabilityTermStructureHandle(hazard_curve)
        risk_free_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todaysDate, self.risk_free, ql.Actual365Fixed())
        )
        engine = ql.IsdaCdsEngine(default_probability, self.recovery_rate, risk_free_curve)
        return engine

    def net_present_value(self):
        schedule = self.cds_schedule()
        engine = self.pricing_engine()
        cds = ql.CreditDefaultSwap(self.side, self.nominal, self.spread, schedule, ql.Following, ql.Actual365Fixed())
        cds.setPricingEngine(engine)
        npv = cds.NPV()
        return "${:,.4f}".format(npv)
    
    def fair_spread(self):
        schedule = self.cds_schedule()
        engine = self.pricing_engine()
        cds = ql.CreditDefaultSwap(self.side, self.nominal, self.spread, schedule, ql.Following, ql.Actual365Fixed())
        cds.setPricingEngine(engine)
        fair_spread = cds.fairSpread()*100
        return "{:,.4f}%".format(fair_spread)

    def default_probability(self):
        maturity_date = self.calendar.adjust(self.entry_date + self.selected_period, ql.Following)
        hazard_curve = self.hazard_curve_function()
        default_prob = hazard_curve.defaultProbability(maturity_date)*100
        return "{:,.4f}%".format(default_prob)
    def expected_loss(self):
        default_prob = self.default_probability().replace('%', '').strip()
        default_prob = float(default_prob)/100
        expected_loss = self.nominal * (1 - self.recovery_rate) * default_prob
        return "${:,.4f}".format(expected_loss)
    
    def premium_payment(self):
        premium_payment = self.nominal * self.spread * self.period_fraction
        return "${:,.4f}".format(premium_payment)
    def cds_results(self):
        results = {
            "Net Present Value": self.net_present_value(),
            "Fair Spread": self.fair_spread(),
            "Default Probability": self.default_probability(),
            "Expected Loss": self.expected_loss(),
            "Premium Payment": self.premium_payment()
        }
        return results
    
    def generate_variable_range(self, variable, range_span, num_steps):
        if variable == 'recovery_rate':
            base = self.recovery_rate
        elif variable == 'risk_free':
            base = self.risk_free
        elif variable == 'spread':
            base = self.spread            
        else:
            raise ValueError(f"Unsupported or missing variable: {variable}")
        return np.linspace(base - range_span, base + range_span, num_steps)
    
    def analyze_variable_sensitivity(self,variable, range_span, num_steps):
        variable_range = self.generate_variable_range(variable, range_span, num_steps)
        cds_analysis_results = []

        for value in variable_range:
            if variable == 'recovery_rate':
                self.recovery_rate = value
            elif variable == 'risk_free':
                self.risk_free = value
            elif variable == 'spread':
                self.spread = value
            
            cds_expectedLoss = self.expected_loss()
            cds_analysis_results.append((value, cds_expectedLoss))   

        return cds_analysis_results

    def plot_sensitivity_analysis(self, variable, range_span, num_steps):
        sensitivity__analysis_results = self.analyze_variable_sensitivity(variable, range_span, num_steps)
        variable_values, cds_expectedLoss = zip(*sensitivity__analysis_results)
        
        variable_values = [val * 100 for val in variable_values]
        x_label = f'{variable.replace("_", " ").title()} (%)'

        plt.figure(figsize=(10, 6))
        plt.plot(variable_values, cds_expectedLoss, marker='o', linestyle='-', color='b')
        plt.title(f'Sensitivity Analysis: Expected Loss vs {variable.replace("_", " ").title()}')
        plt.xlabel(x_label)
        plt.ylabel('Expected Loss ($)')
        plt.tight_layout()
        plt.grid(True)
        # plt.show()


class SyntheticCDO:
    def __init__(self, cds_list, tranches):
        self.cds_list = cds_list
        self.tranches = tranches

    def tranche_cashflows(self):
        """ Calculate the cashflows for each tranche based on CDS portfolio. """
        tranche_cashflows = {}

        for tranche_name, lower_bound, upper_bound in self.tranches:
            tranche_cashflows[tranche_name] = {"expected_loss": 0, "default_probability": 0, "premium_payment": 0}

            for cds in self.cds_list:
                # Convert notional percentage to dollar value
                lower_bound_value = lower_bound * cds.nominal
                upper_bound_value = upper_bound * cds.nominal
                tranche_exposure = upper_bound_value - lower_bound_value  # Notional amount for the tranche

                # Get expected loss, default probability, and premium as numeric values
                expected_loss = float(cds.expected_loss().replace('$', '').replace(',', '').strip())
                default_prob = float(cds.default_probability().replace('%', '').replace(',', '').strip())
                premium = float(cds.premium_payment().replace('$', '').replace(',', '').strip())

                # Initialize remaining loss to be allocated
                remaining_loss = expected_loss  
                allocated_loss = 0  

                # Allocate losses **cascading down** through tranches
                if remaining_loss > lower_bound_value:  # Only absorb losses **if losses reach this tranche**
                    tranche_loss = min(remaining_loss, upper_bound_value) - lower_bound_value
                    allocated_loss = max(0, tranche_loss)
                    remaining_loss -= allocated_loss  # Reduce remaining losses for next tranche

                # Allocate premium cascading down through tranches (only if tranche absorbs losses)
                tranche_premium = (allocated_loss / cds.nominal) * premium if allocated_loss > 0 else 0
                tranche_cashflows[tranche_name]["premium_payment"] += tranche_premium

                # Update tranche cashflows
                tranche_cashflows[tranche_name]["expected_loss"] += allocated_loss
                tranche_cashflows[tranche_name]["default_probability"] += default_prob * (allocated_loss / expected_loss if expected_loss > 0 else 0)
                tranche_cashflows[tranche_name]["premium_payment"] += tranche_premium

            # Format results
            tranche_cashflows[tranche_name]["expected_loss"] = f"${tranche_cashflows[tranche_name]['expected_loss']:.4f}"
            tranche_cashflows[tranche_name]["default_probability"] = f"{tranche_cashflows[tranche_name]['default_probability']:.4f}%"
            tranche_cashflows[tranche_name]["premium_payment"] = f"${tranche_cashflows[tranche_name]['premium_payment']:.4f}"

        return tranche_cashflows


    def tranche_npv(self):
        """ Calculate the NPV for each tranche in the CDO. """
        tranche_values = {}

        for tranche_name, lower_bound, upper_bound in self.tranches:
            tranche_values[tranche_name] = {"npv": 0}

            for cds in self.cds_list:
                lower_bound_value = lower_bound 
                upper_bound_value = upper_bound 

                # Get NPV value as numeric
                npv = float(cds.net_present_value().replace('$', '').replace(',', '').strip())  # Convert to float

                # Allocate NPV to the tranche based on attachment/detachment points
                tranche_npv = max(0, min(npv, upper_bound_value) - lower_bound_value)
                tranche_values[tranche_name]["npv"] += tranche_npv
            tranche_values[tranche_name]["npv"] = f"${tranche_values[tranche_name]['npv']:.4f}"
        return tranche_values

    def calculate_synthetic_cdo(self):
        """ Run the pricing calculations for the synthetic CDO. """
        tranche_cashflows = self.tranche_cashflows()
        tranche_npv = self.tranche_npv()

        return {
            "tranche_cashflows": tranche_cashflows,
            "tranche_npv": tranche_npv
        }
    
    
    def generate_variable_range(self, variable, range_span, num_steps):
        """ Generate the variable range for each CDS instance """
        ranges = {}
    
        for cds in self.cds_list:
            # Determine the base value for the specified variable from each CDS instance
            if variable == 'recovery_rate':
                base = cds.recovery_rate
            elif variable == 'risk_free':
                base = cds.risk_free
            elif variable == 'spread':
                base = cds.spread
            else:
                raise ValueError(f"Unsupported or missing variable: {variable}")
            
            # Generate the range of values for the current CDS instance
            ranges[cds] = np.linspace(base - range_span, base + range_span, num_steps)
        
        return ranges
    
    def analyze_variable_sensitivity(self, variable, range_span, num_steps):
        """ Analyze the sensitivity of the synthetic CDO to a specific variable across all CDS instances for each tranche. """
        # Generate the variable ranges for all CDS instances
        variable_ranges = self.generate_variable_range(variable, range_span, num_steps)
    
        # Dictionary to store the sensitivity results for each tranche and CDS
        sensitivity_results_by_cds = {
            cds: {tranche_name: [] for tranche_name, _, _ in self.tranches} for cds in self.cds_list
        }
    
        # Iterate through the variable range steps
        for step in range(num_steps):
            for cds in self.cds_list:
                # Apply the current step value of the variable for this CDS
                current_value = variable_ranges[cds][step]
                if variable == 'risk_free':
                    cds.risk_free = current_value
                elif variable == 'recovery_rate':
                    cds.recovery_rate = current_value
                elif variable == 'spread':
                    cds.spread = current_value
                
                # Calculate cashflows for the synthetic CDO after updating the current CDS
                cdo_cashflows = self.tranche_cashflows()
                
                # Store the results for each tranche for the current CDS
                for tranche_name in sensitivity_results_by_cds[cds].keys():
                    expectedLoss = float(cdo_cashflows[tranche_name]['expected_loss'].replace('$', '').replace(',', '').strip())
                    sensitivity_results_by_cds[cds][tranche_name].append((current_value, expectedLoss))
        return sensitivity_results_by_cds



    def plot_sensitivity_analysis(self, variable, range_span, num_steps):
        """Plot the sensitivity analysis results for each tranche across all CDS instances in separate subplots."""
        # Get the results of the sensitivity analysis by CDS and tranche
        sensitivity_results_by_cds = self.analyze_variable_sensitivity(variable, range_span, num_steps)
        
        # Create a dictionary to hold results for each tranche
        tranche_results = {}
        
        # Organize results by tranche
        for cds, results in sensitivity_results_by_cds.items():
            for tranche_name, data in results.items():
                if tranche_name not in tranche_results:
                    tranche_results[tranche_name] = {}
                if cds not in tranche_results[tranche_name]:
                    tranche_results[tranche_name][cds] = []
                tranche_results[tranche_name][cds].extend(data)

        # Create subplots for each tranche
        num_tranches = len(tranche_results)
        fig, axes = plt.subplots(num_tranches, 1, figsize=(12, 8), sharex=True)  # Create subplots

        # Generate a colormap with enough colors
        num_lines = sum(len(cds_data) for cds_data in tranche_results.values())
        colors = plt.cm.viridis(np.linspace(0, 1, num_lines))  # Use the viridis colormap

        color_index = 0  # Initialize color index

        for ax, (tranche_name, cds_data) in zip(axes, tranche_results.items()):
            for cds, data in cds_data.items():
                if data:  # Check if we have results for this CDS in the current tranche
                    variable_values, total_cashflow = zip(*data)
                    variable_values = [val * 100 for val in variable_values]
                    ax.plot(variable_values, total_cashflow, marker='o', linestyle='-', 
                            color=colors[color_index],  # Use the next color from the colormap
                            label=f'CDS {self.cds_list.index(cds) + 1}')  # Label by CDS instance
                    color_index += 1  # Move to the next color
            ax.set_title(f'Sensitivity Analysis for {tranche_name}')  # Title for each subplot
            ax.grid(True)  # Add grid to each subplot
            ax.set_ylabel('Expected Loss')
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.4f}'))  # Format as currency
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.4f}%'))  # Format as percentage
            ax.legend(loc='upper right')  # Legend for the current subplot

        variable_values = [val * 100 for val in variable_values]
        x_label = f'{variable.replace("_", " ").title()} (%)'
        # Set common labels
        plt.xlabel(x_label)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()  # Show the plot
        
    
    
class CLNPricingFixed:
    def __init__(self, bond_params, cds_instance):
        self.bond_fixed = NCFixedBonds(**bond_params)
        self.cds = cds_instance

    def calculate_cln_price_fixed(self, issue_date, maturity_date, tenor, coupon_rate, face_value, notionals, amortization):
        # Calculate bond NPV using fixed_rate from NCFixedBonds  
        if amortization == "No":
            bond_results = self.bond_fixed.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, face_value)
        elif amortization == "Yes":
            bond_results = self.bond_fixed.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals)          
            
        # Access NPVs for all shocks (e.g., 10, 20, 30 years)
        shock_npvs = {}
        for period, data in bond_results.items():
            if isinstance(period, int):
                shock_npvs[period] = data['Price']
        
        # Get CDS Premium Payments dynamically
        cds_prem_pay = self.cds.premium_payment()
        cds_prem_pay_value = float(cds_prem_pay.replace("$", "").replace(",", ""))  # Convert formatted string to float

        # Get CDS Expected Loss Payments dynamically
        cds_expected_loss = self.cds.expected_loss()
        cds_expected_loss_value = float(cds_expected_loss.replace("$", "").replace(",", ""))  # Convert formatted string to float
        
        # CLN price for each shock period: Bond NPV - CDS NPV
        cln_prices = {}
        for period, bond_npv in shock_npvs.items():
            cln_prices[period] = bond_npv - cds_prem_pay_value - cds_expected_loss_value
        
        # print("CLN Prices for all shocks:", cln_prices)
        return cln_prices

    
class CLNPricingFloat:
    def __init__(self, bond_params_float, cds_instance):
        self.bond_float = NCFloatingBonds(**bond_params_float)
        self.cds = cds_instance

    def calculate_cln_price_float(self, shocks, issueDate, maturityDate, tenor, spread, holding, dayCount, notional, notional_dates, notionals, amortization):
        # Calculate bond NPV using fixed_rate from NCFixedBonds  
        if amortization == "No":
            bond_results = self.bond_float.price_floating(shocks, issueDate, maturityDate, tenor, spread, notional, dayCount)
        elif amortization == "Yes":
            bond_results = self.bond_float.price_amortizing_floating(shocks, issueDate, maturityDate, tenor, spread, notionals,
                                              notional_dates, dayCount)
                     
        # Access NPVs for all shocks (e.g., 10, 20, 30 years)
        shock_npvs = {}
        for period, data in bond_results.items():
            if isinstance(period, int):
                shock_npvs[period] = data['Price']
        
        # Get CDS Premium Payments dynamically
        cds_prem_pay = self.cds.premium_payment()
        cds_prem_pay_value = float(cds_prem_pay.replace("$", "").replace(",", ""))  # Convert formatted string to float
        
        # Get CDS Expected Loss Payments dynamically
        cds_expected_loss = self.cds.expected_loss()
        cds_expected_loss_value = float(cds_expected_loss.replace("$", "").replace(",", ""))  # Convert formatted string to float

        # CLN price for each shock period: Bond NPV - CDS NPV
        cln_prices = {}
        for period, bond_npv in shock_npvs.items():
            cln_prices[period] = bond_npv - cds_prem_pay_value - cds_expected_loss_value
        
        return cln_prices



class CLNSensitivityAnalysis:
    def __init__(self, bond_params, cds_instance, bond_results):
        self.bond_params = bond_params
        self.cds = cds_instance
        self.bond_results = bond_results  # Store the bond results directly

    def generate_variable_range(self, variable, range_span, num_steps):
        """Generate a range of values for sensitivity analysis."""
        if variable == 'recovery_rate':
            base = self.cds.recovery_rate
        elif variable == 'risk_free':
            base = self.cds.risk_free
        elif variable == 'spread':
            base = self.cds.spread            
        else:
            raise ValueError(f"Unsupported or missing variable: {variable}")
        return np.linspace(base - range_span, base + range_span, num_steps)
        
    def analyze_variable_sensitivity(self, variable, range_span, num_steps):
        """Analyze how changes in the variable affect the CLN prices using provided bond prices."""
        variable_range = self.generate_variable_range(variable, range_span, num_steps)
        
        # Get periods directly from bond_results keys, excluding 'Values' if present
        periods = [period for period in self.bond_results.keys() if isinstance(period, (int, float))]
        cln_analysis_results = {period: [] for period in periods}  # Initialize results for each period
        
        for value in variable_range:
            # Update the CDS instance with the new value
            if variable == 'recovery_rate':
                self.cds.recovery_rate = value
            elif variable == 'risk_free':
                self.cds.risk_free = value
            elif variable == 'spread':
                self.cds.spread = value
            
            # Get CDS premium payment
            cds_premium = float(self.cds.premium_payment().replace('$', '').replace(',', ''))
            cds_expected_loss = float(self.cds.expected_loss().replace('$', '').replace(',', ''))
            # Use provided bond prices for each period
            for period in periods:
                # Convert bond price to float if it's not already
                bond_price = float(self.bond_results[period]['Price']) if isinstance(self.bond_results[period]['Price'], str) else self.bond_results[period]['Price']
                cln_price = bond_price - cds_premium - cds_expected_loss
                cln_analysis_results[period].append((value, cln_price))

        return cln_analysis_results

    def plot_sensitivity_analysis(self, variable, range_span, num_steps):
        """Plot the sensitivity analysis results for each shock in separate subplots."""
        sensitivity_analysis_results = self.analyze_variable_sensitivity(variable, range_span, num_steps)
        
        num_shocks = len(sensitivity_analysis_results)
        fig, axes = plt.subplots(num_shocks, 1, figsize=(12, 8), sharex=True)  # Create subplots

        colors = ['b', 'g', 'r']  # Extend colors if needed

        for ax, (shock, color) in zip(axes, zip(sensitivity_analysis_results.keys(), colors)):
            if sensitivity_analysis_results[shock]:  # Check if we have results for this shock
                values, cln_prices = zip(*sensitivity_analysis_results[shock])
                values = [val * 100 for val in values]

                ax.plot(values, cln_prices, marker='o', linestyle='-', color=color, 
                        label=f'{shock}bp Shock')
                ax.set_title(f'{shock}bp Shock')  # Title for each subplot
                ax.grid(True)  # Add grid to each subplot
                ax.set_ylabel('CLN Price')
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.4f}'))  # Format as currency
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.4f}%'))  # Format as Percentage
        # Set common labels
        plt.xlabel(f'{variable.replace("_", " ").title()}')
        plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.legend(loc='upper right')  # Legend for the last subplot
        plt.show()  # Show the plot

