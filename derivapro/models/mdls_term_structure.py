import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime
from ..models.yieldterm_market_data import MarketRateProvider, TreasuryRateProvider, SOFRRateProvider, FREDSwapRatesProvider
from scipy.optimize import least_squares

class YieldTermStructure:
    def __init__(self):
        self.market_rates = []  # list of (tenor, rate, source)

    def _normalize_tenor(self, tenor):
        # If already a QuantLib Period, just return
        if isinstance(tenor, ql.Period):
            return tenor
        
        # If string like '1Y', '6M', convert to ql.Period
        if isinstance(tenor, str):
            try:
                length = int(tenor[:-1])
                unit = tenor[-1].upper()
                if unit == 'Y':
                    return ql.Period(length, ql.Years)
                elif unit == 'M':
                    return ql.Period(length, ql.Months)
                elif unit == 'D':
                    return ql.Period(length, ql.Days)
                else:
                    raise ValueError(f"Unknown tenor unit '{unit}' in {tenor}")
            except Exception as e:
                raise ValueError(f"Failed to parse tenor string '{tenor}': {e}")

        # If it's an integer or float (assumed years)
        if isinstance(tenor, (int, float)):
            return ql.Period(int(tenor), ql.Years)

        # Otherwise, fail explicitly
        raise TypeError(f"Unsupported tenor type: {type(tenor)}")

    def append_market_rates(self, market_rates, source="generic"):
        for tenor, rate in market_rates:
            try:
                tenor = self._normalize_tenor(tenor)
            except Exception as e:
                print(f"Skipping invalid tenor {tenor}: {e}")
                continue
            
            if source.lower() == "swap":
                # Check for known invalid swap data (e.g., placeholder -999 or missing rate)
                if rate is None or rate < 0 or np.isnan(rate):
                    print(f"Skipping invalid swap rate for tenor {tenor}")
                    continue
            self.market_rates.append((tenor, rate, source))

    def create_rate_helpers(self, start_date):
        calendar = ql.TARGET()
        self.rate_helpers = []

        for tenor, rate, source in self.market_rates:
            if source.lower() == "swap" and (rate is None or rate < 0 or np.isnan(rate)):
                print(f"Skipping invalid swap rate for tenor {tenor}")
                continue
            quote = ql.QuoteHandle(ql.SimpleQuote(rate))
            source_lc = source.lower()

            if source_lc == "treasury":
                if (tenor.units() == ql.Months) or (tenor.units() == ql.Years and tenor.length() <= 1):
                    helper = ql.DepositRateHelper(
                        quote, tenor, 3, calendar,
                        ql.ModifiedFollowing, False,
                        ql.Actual360()
                    )
                else:
                    helper = ql.SwapRateHelper(
                        quote,
                        tenor,
                        calendar,
                        ql.Annual,
                        ql.Unadjusted,
                        ql.Thirty360(ql.Thirty360.USA),
                        ql.USDLibor(ql.Period(3, ql.Months))
                    )
            elif source_lc == "sofr":
                helper = ql.DepositRateHelper(
                    quote, tenor, 0, calendar,
                    ql.ModifiedFollowing, False,
                    ql.Actual360()
                )
            
            elif source_lc == 'swap':
                helper = ql.SwapRateHelper(
                    quote,
                    tenor,
                    calendar,
                    ql.Annual,
                    ql.Unadjusted,
                    ql.Thirty360(ql.Thirty360.USA),
                    ql.USDLibor(ql.Period(3, ql.Months))
                )
            
            else:
                raise ValueError(f"Unknown source '{source}' for tenor {tenor}.")
            self.rate_helpers.append(helper)

        return self.rate_helpers
    
    def average_duplicate_rates(self, start_date):
        from collections import defaultdict
        date_map = defaultdict(list)
        
        # Group rates by maturity date
        for tenor, rate, source in self.market_rates:
            maturity = start_date + tenor
            date_map[maturity].append((tenor, rate, source))
        
        # Average rates for duplicate dates
        averaged_market_rates = []
        for maturity, instruments in date_map.items():
            if len(instruments) == 1:
                averaged_market_rates.append(instruments[0])  # just one instrument
            else:
                avg_rate = sum([r[1] for r in instruments]) / len(instruments)
                tenor = instruments[0][0]
                averaged_market_rates.append((tenor, avg_rate, instruments[0][2]))  # keep first source
        
        self.market_rates = averaged_market_rates

    def bootstrap_curve(self, start_date, method_name):
        ql.Settings.instance().evaluationDate = start_date

        rate_helpers = self.create_rate_helpers(start_date)
        methods = {
            "PiecewiseFlatForward": ql.PiecewiseFlatForward,
            "LogLinearDiscount": ql.PiecewiseLogLinearDiscount,
            "LogCubicDiscount": ql.PiecewiseLogCubicDiscount,
            "LinearZero": ql.PiecewiseLinearZero,
            "CubicZero": ql.PiecewiseCubicZero,
            "LinearForward": ql.PiecewiseLinearForward,
            "SplineCubicDiscount": ql.PiecewiseSplineCubicDiscount
        }
        method_constructor = methods.get(method_name)
        if method_constructor is None:
            raise ValueError(f"Unknown curve construction method '{method_name}'.")

        curve = method_constructor(start_date, rate_helpers, ql.Actual360())
        curve.enableExtrapolation()
        return curve

    def calculate_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def fit_nelson_siegel_curve(self, start_date, curve):
        def ns_yield(t, beta0, beta1, beta2, tau):
            if t == 0: 
                return beta0
            term1 = beta0
            term2 = beta1 * ((1 - np.exp(-t / tau)) / (t / tau))
            term3 = beta2 * (((1 - np.exp(-t / tau)) / (t / tau)) - np.exp(-t / tau))
            return term1 + term2 + term3

        # Pillar dates and zero rates
        pillar_dates = [start_date + tenor for tenor, _, _ in self.market_rates]
        ttm = [ql.Actual365Fixed().yearFraction(start_date, d) for d in pillar_dates]
        zero_rates = [curve.zeroRate(d, ql.Actual360(), ql.Continuous).rate() for d in pillar_dates]

        ttm = np.array(ttm)
        zero_rates = np.array(zero_rates)

        # Initial guess for params
        x0 = [0.02, -0.01, 0.01, 2.0]

        def error_fn(params):
            beta0, beta1, beta2, tau = params
            modeled = np.array([ns_yield(t, beta0, beta1, beta2, tau) for t in ttm])
            return modeled - zero_rates

        result = least_squares(error_fn, x0, bounds=([-1, -1, -1, 0.01], [1, 1, 1, 10]))
        beta0, beta1, beta2, tau = result.x

        def zero_curve(t): 
            return ns_yield(t, beta0, beta1, beta2, tau)

        # Calculate modeled zero rates
        modeled_zero = np.array([zero_curve(t) for t in ttm])

        # Calculate discount factors from zero rates
        df_true = np.exp(-zero_rates * ttm)
        df_modeled = np.exp(-modeled_zero * ttm)

        # Calculate forward rates from discount factors
        fwd_true = -(np.log(df_true[1:]) - np.log(df_true[:-1])) / (ttm[1:] - ttm[:-1])
        fwd_modeled = -(np.log(df_modeled[1:]) - np.log(df_modeled[:-1])) / (ttm[1:] - ttm[:-1])

        # Calculate R² values
        r2_zero = self.calculate_r2(zero_rates, modeled_zero)
        r2_df = self.calculate_r2(df_true, df_modeled)
        r2_fwd = self.calculate_r2(fwd_true, fwd_modeled)

        return zero_curve, ttm, zero_rates, r2_zero, r2_df, r2_fwd
    
    def fit_svensson_curve(self, start_date, curve):
        def svensson_yield(t, beta0, beta1, beta2, beta3, tau1, tau2):
            if t == 0:
                return beta0
            term1 = beta0
            term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
            term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
            term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
            return term1 + term2 + term3 + term4

        pillar_dates = [start_date + tenor for tenor, _, _ in self.market_rates]
        ttm = [ql.Actual365Fixed().yearFraction(start_date, d) for d in pillar_dates]
        zero_rates = [curve.zeroRate(d, ql.Actual360(), ql.Continuous).rate() for d in pillar_dates]

        ttm = np.array(ttm)
        zero_rates = np.array(zero_rates)

        x0 = [0.02, -0.01, 0.01, 0.01, 2.0, 0.5]  # initial guesses

        def error_fn(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            modeled = np.array([svensson_yield(t, beta0, beta1, beta2, beta3, tau1, tau2) for t in ttm])
            return modeled - zero_rates

        result = least_squares(error_fn, x0, bounds=([-1, -1, -1, -1, 0.01, 0.01], [1, 1, 1, 1, 10, 10]))
        beta0, beta1, beta2, beta3, tau1, tau2 = result.x

        def zero_curve(t):
            return svensson_yield(t, beta0, beta1, beta2, beta3, tau1, tau2)
        
        # Calculate discount factors from zero rates
        modeled_zero = np.array([zero_curve(t) for t in ttm])

        df_true = np.exp(-zero_rates * ttm)
        df_modeled = np.exp(-modeled_zero * ttm)

        # Calculate forward rates from discount factors
        fwd_true = -(np.log(df_true[1:]) - np.log(df_true[:-1])) / (ttm[1:] - ttm[:-1])
        fwd_modeled = -(np.log(df_modeled[1:]) - np.log(df_modeled[:-1])) / (ttm[1:] - ttm[:-1])

        # Calculate R² values
        r2_zero = self.calculate_r2(zero_rates, modeled_zero)
        r2_df = self.calculate_r2(df_true, df_modeled)
        r2_fwd = self.calculate_r2(fwd_true, fwd_modeled)

        return zero_curve, ttm, zero_rates, r2_zero, r2_df, r2_fwd
    
    def yield_curve(self, start_date, fit_selection, method_name, forward_tenor):
        forward_tenor_str = str(forward_tenor)  # For plotting labels, but we use forward_tenor (Period) in calculations

        if fit_selection.lower() == "yes":
            # 1) Bootstrap piecewise curve first
            piecewise_curve = self.bootstrap_curve(start_date, method_name)

            # 2) Fit Nelson-Siegel to bootstrapped zero rates
            ns_zero, ttm, bootstrapped_zero_rates, r2_zero, r2_df, r2_fwd  = self.fit_nelson_siegel_curve(start_date, piecewise_curve)

            # 2.1) Fit Svensson to bootstrapped zero rates
            svensson_zero = self.fit_svensson_curve(start_date, piecewise_curve)[0]  # only grab the fitted curve function

            # 3) Prepare forward rates and discount factors from NS fitted curve
            ttm_grid = np.linspace(0.01, 30, 360)
            zero_rates_ns = [ns_zero(t) for t in ttm_grid]
            discount_factors_ns = [np.exp(-r * t) for r, t in zip(zero_rates_ns, ttm_grid)]

            fwd_len = ql.Actual365Fixed().yearFraction(start_date, start_date + forward_tenor)
            forward_rates_ns = [
                (ns_zero(t + fwd_len) * (t + fwd_len) - ns_zero(t) * t) / fwd_len
                for t in ttm_grid if (t + fwd_len) <= 30
            ]
            forward_ttm_ns = [t for t in ttm_grid if (t + fwd_len) <= 30]

            # 3.1) Prepare forward rates and discount factors from Svensson fitted curve
            zero_rates_sv = [svensson_zero(t) for t in ttm_grid]
            discount_factors_sv = [np.exp(-r * t) for r, t in zip(zero_rates_sv, ttm_grid)]  

            forward_rates_sv = [
                (svensson_zero(t + fwd_len) * (t + fwd_len) - svensson_zero(t) * t) / fwd_len
                for t in ttm_grid if (t + fwd_len) <= 30
            ]
            forward_ttm_sv = [t for t in ttm_grid if (t + fwd_len) <= 30]

            # 4) Get piecewise bootstrapped zero rates for comparison
            max_date = piecewise_curve.maxDate()
            dates = [start_date + ql.Period(i, ql.Months) for i in range(0, 360)]
            dates = [d for d in dates if d <= max_date]
            ttm_pw = np.array([ql.Actual365Fixed().yearFraction(start_date, d) for d in dates])
            zero_rates_pw = [piecewise_curve.zeroRate(d, ql.Actual360(), ql.Continuous).rate() for d in dates]
            discount_factors_pw = [piecewise_curve.discount(d) for d in dates]
            forward_rates_pw = []

            fwd_shift = forward_tenor.length()

            for i in range(len(dates) - fwd_shift):
                fwd = piecewise_curve.forwardRate(dates[i], dates[i + fwd_shift], ql.Actual360(), ql.Continuous).rate()
                forward_rates_pw.append(fwd)
            forward_ttm_pw = (ttm_pw[:-fwd_shift] + ttm_pw[fwd_shift:]) / 2

            # --- Combine all plots into one figure with 3 subplots ---
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # Zero Rate plot
            axs[0].plot(ttm_pw, zero_rates_pw, label="Bootstrapped Zero Rates", marker='o')
            axs[0].plot(ttm_grid, zero_rates_ns, label="Nelson-Siegel Fit", marker='x')
            axs[0].plot(ttm_grid, zero_rates_sv, label="Svensson Fit", marker='+')
            axs[0].set_xlabel("Time to Maturity (years)")
            axs[0].set_ylabel("Zero Rate")
            axs[0].set_title("Zero Rate Curve")
            axs[0].legend()
            axs[0].grid(True)

            # Discount Factor plot
            axs[1].plot(ttm_pw, discount_factors_pw, label="Bootstrapped Discount Factors", marker='o', color='orange')
            axs[1].plot(ttm_grid, discount_factors_ns, label="Nelson-Siegel Discount Factors", marker='x', color='red')
            axs[1].plot(ttm_grid, discount_factors_sv, label="Svensson Discount Factors", marker='+', color='blue')
            axs[1].set_xlabel("Time to Maturity (years)")
            axs[1].set_ylabel("Discount Factor")
            axs[1].set_title("Discount Factor Curve")
            axs[1].legend()
            axs[1].grid(True)

            # Forward Rate plot
            axs[2].plot(forward_ttm_pw, forward_rates_pw, label="Bootstrapped Forward Rates", marker='o', color='green')
            axs[2].plot(forward_ttm_ns, forward_rates_ns, label="Nelson-Siegel Forward Rates", marker='x', color='darkgreen')
            axs[2].plot(forward_ttm_sv, forward_rates_sv, label="Svensson Forward Rates", marker='+', color='purple')
            axs[2].set_xlabel("Time to Start (years)")
            axs[2].set_ylabel("Forward Rate")
            axs[2].set_title(f"Forward Curve ({forward_tenor_str})")
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()
            return fig  # Return the figure instead of showing it

        else:
            # For other methods, just bootstrap and plot
            curve = self.bootstrap_curve(start_date, method_name)

            max_date = curve.maxDate()
            dates = [start_date + ql.Period(i, ql.Months) for i in range(0, 360)]
            dates = [d for d in dates if d <= max_date]

            ttm = np.array([ql.Actual365Fixed().yearFraction(start_date, d) for d in dates])
            zero_rates = [curve.zeroRate(d, ql.Actual360(), ql.Continuous).rate() for d in dates]
            discount_factors = [curve.discount(d) for d in dates]

            fwd_len = forward_tenor.length()
            forward_rates = []
            for i in range(len(dates) - fwd_len):
                fwd = curve.forwardRate(dates[i], dates[i + fwd_len], ql.Actual360(), ql.Continuous).rate()
                forward_rates.append(fwd)
            forward_ttm = (ttm[:-fwd_len] + ttm[fwd_len:]) / 2

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
            plt.subplot(1, 3, 1)
            plt.plot(ttm, zero_rates, marker='o', label='Zero Rate (Yield Curve)')
            plt.xlabel('Time to Maturity (years)')
            plt.ylabel('Zero Rate')
            plt.title(f'Zero Curve - {method_name}')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(ttm, discount_factors, marker='o', color='orange', label='Discount Factor')
            plt.xlabel('Time to Maturity (years)')
            plt.ylabel('Discount Factor')
            plt.title('Discount Curve')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(forward_ttm, forward_rates, marker='o', color='green', label=f'{forward_tenor_str} Forward Rate')
            plt.xlabel('Time to Start (years)')
            plt.ylabel('Forward Rate')
            plt.title(f'Forward Curve ({forward_tenor_str} Forward Rates)')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            return fig


# # ----------- USAGE -------------------

# api_key = "a7a1a9c282ee0093003008999c337857"
# start_date = ql.Date(24, 2, 2016)
# forward_tenor = ql.Period(3, ql.Months)
# method_name = "PiecewiseFlatForward"
# fit_selection = "yes"  

# treasury_provider = TreasuryRateProvider(api_key)
# sofr_provider = SOFRRateProvider()
# swap_provider = FREDSwapRatesProvider(api_key)


# treasury_rates = treasury_provider.get_market_rates(start_date=start_date)
# sofr_rates = sofr_provider.get_market_rates(startDate=start_date)
# swap_rates = swap_provider.get_market_rates(start_date=start_date)

# ts = YieldTermStructure()
# ts.append_market_rates(treasury_rates, source="treasury")
# ts.append_market_rates(sofr_rates, source="sofr")
# ts.append_market_rates(swap_rates, source="swap")

# ts.average_duplicate_rates(start_date)
# ts.yield_curve(start_date, fit_selection, method_name, forward_tenor)
# curve = ts.bootstrap_curve(start_date, method_name)

# # Unpack all returns: (zero_curve, ttm, zero_rates, r2_zero, r2_df, r2_fwd)
# _, _, _, r2_zero_ns, r2_df_ns, r2_fwd_ns = ts.fit_nelson_siegel_curve(start_date, curve)
# _, _, _, r2_zero_sv, r2_df_sv, r2_fwd_sv = ts.fit_svensson_curve(start_date, curve)

# print(f"Nelson-Siegel R² (Zero Rates): {r2_zero_ns:.4f}")
# print(f"Nelson-Siegel R² (Discount Factors): {r2_df_ns:.4f}")
# print(f"Nelson-Siegel R² (Forward Rates): {r2_fwd_ns:.4f}")

# print(f"Svensson R² (Zero Rates): {r2_zero_sv:.4f}")
# print(f"Svensson R² (Discount Factors): {r2_df_sv:.4f}")
# print(f"Svensson R² (Forward Rates): {r2_fwd_sv:.4f}")