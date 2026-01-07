import QuantLib as ql
import requests
from fredapi import Fred
from datetime import datetime
import pandas as pd
import math

# --- Base Provider Interface ---
class MarketRateProvider:
    def get_market_rates(self, start_date=None):
        raise NotImplementedError("Subclasses must implement this method")

def to_pd_timestamp(date_input):
    """Convert input to normalized pandas Timestamp (date only)."""
    if isinstance(date_input, pd.Timestamp):
        return date_input.normalize()
    if isinstance(date_input, datetime):
        return pd.Timestamp(date_input).normalize()
    if hasattr(date_input, "year") and hasattr(date_input, "month") and hasattr(date_input, "dayOfMonth"):
        # QuantLib.Date
        dt = datetime(date_input.year(), date_input.month(), date_input.dayOfMonth())
        return pd.Timestamp(dt).normalize()
    if isinstance(date_input, str):
        return pd.Timestamp(date_input).normalize()
    raise TypeError(f"Unsupported date type: {type(date_input)}")

class FREDSwapRatesProvider(MarketRateProvider):
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.swap_ids = {
            '1Y': 'DSWP1',
            '2Y': 'WSWP2',
            '3Y': 'DSWP3',
            '4Y': 'DSWP4',
            '5Y': 'DSWP5',
            '7Y': 'WSWP7',
            '30Y': 'DSWP30'
        }

    def get_market_rates(self, start_date=None):
        latest_rates = {}
        for label, series_id in self.swap_ids.items():
            data = self.fred.get_series(series_id)
            if start_date is not None:
                py_start_date = to_pd_timestamp(start_date)
                data = data[data.index >= py_start_date]
            if data.empty:
                # print(f"No swap data available for {label} after {start_date}")
                latest_value = None
            else:
                latest_value = data.dropna().iloc[-1]
            latest_rates[label] = (latest_value / 100) if latest_value is not None else None
        
        # Dynamically create list, skip None values
        def tenor_key(label):
            return int(''.join(filter(str.isdigit, label)))
        
        return [
            (ql.Period(tenor_key(label), ql.Years), latest_rates[label])
            for label in sorted(latest_rates.keys(), key=tenor_key)
            if latest_rates[label] is not None
        ]

class TreasuryRateProvider(MarketRateProvider):
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.series_ids = {
            '1M': 'GS1M', '3M': 'GS3M', '6M': 'GS6M',
            '1Y': 'GS1', '2Y': 'GS2', '5Y': 'GS5', '7Y': 'GS7',
            '10Y': 'GS10', '30Y': 'GS30'
        }

    def get_market_rates(self, start_date=None):
        latest_rates = {}
        end_date = None
        if start_date is not None:
            end_date = to_pd_timestamp(start_date)

        for label, series_id in self.series_ids.items():
            data = self.fred.get_series(series_id, observation_end=end_date)
            data = data.dropna()
            if data.empty:
                raise ValueError(f"No treasury data available for {label} on or before {end_date}")
            latest_value = data.iloc[-1]
            latest_rates[label] = latest_value / 100  # Convert from % to decimal

        def tenor_key(label):
            return int(''.join(filter(str.isdigit, label)))

        return [
            (ql.Period(tenor_key(label), ql.Months if 'M' in label else ql.Years), latest_rates[label])
            for label in sorted(latest_rates.keys(), key=tenor_key)
        ]

class SOFRRateProvider(MarketRateProvider):
    def sofr_operations(self, rateType: str = 'sofr', startDate: str = None, format: str = 'json', data_type: str = 'rate'):
        if startDate is not None:
            if isinstance(startDate, ql.Date):
                startDate = startDate.to_date().isoformat()  # 'YYYY-MM-DD'
            elif isinstance(startDate, str):
                # assume already in correct format
                pass
            else:
                raise ValueError("startDate must be QuantLib Date or string in 'YYYY-MM-DD' format")
        else:
            startDate = "2025-06-24"

        url = f"https://markets.newyorkfed.org/api/rates/secured/{rateType}/search.{format}?startDate={startDate}&type={data_type}"
        return url

    def get_sofr_data(self, startDate=None):
        url = self.sofr_operations(startDate=startDate)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None

    def get_market_rates(self, startDate=None):
        data = self.get_sofr_data(startDate=startDate)
        if data and "refRates" in data:
            rates = []
            for entry in data["refRates"]:
                tenor = ql.Period(1, ql.Days)  # SOFR usually daily rates
                rate = entry["percentRate"] / 100.0
                rates.append((tenor, rate))
            return rates[-1:] #Pull only the last overnight rate that represents today's value
        else:
            return []

# class SOFRCompoundedRateCalculator:
#     def __init__(self, rates, day_count=None, calendar=None):
#         """
#         rates: list of (date, overnight_rate), rate as decimal (e.g. 0.053)
#         day_count: optional QuantLib day count convention (default Actual/360)
#         calendar: optional QuantLib calendar (default US GovBond)
#         """

#         self.rates = sorted(rates, key=lambda x: x[0])
#         self.day_count = day_count if day_count is not None else ql.Actual360()
#         self.calendar = calendar if calendar is not None else ql.UnitedStates(ql.UnitedStates.GovernmentBond)


#     def _filter_rates_for_tenor(self, end_date: ql.Date, tenor: ql.Period):
#         start_date = self.calendar.advance(end_date, -tenor)
#         return [
#             (d, r) for d, r in self.rates
#             if start_date < d <= end_date
#         ]

#     def compound(self, end_date, tenor, method="compounded"):
#         rates = self._filter_rates_for_tenor(end_date, tenor)
#         if len(rates) < 2:
#             raise ValueError("Not enough rate observations for selected tenor")

#         accruals = []
#         for i in range(len(rates) - 1):
#             d0, r = rates[i]
#             d1, _ = rates[i + 1]
#             dt = self.day_count.yearFraction(d0, d1)
#             accruals.append((r, dt))

#         if method == "simple":
#             num = sum(r * dt for r, dt in accruals)
#             den = sum(dt for _, dt in accruals)
#             return num / den

#         elif method == "compounded":
#             compound_factor = 1.0
#             for r, dt in accruals:
#                 compound_factor *= (1 + r * dt)
#             return compound_factor - 1.0

#         elif method == "continuous":
#             return math.exp(sum(r * dt for r, dt in accruals)) - 1.0

#         else:
#             raise ValueError(f"Unsupported compounding method: {method}")

class SOFRCompoundedRateCalculator:
    def __init__(self, rates, day_count=ql.Actual360(), compounding=ql.Compounded, compounding_frequency=ql.Daily):
        """
        rates: list of tuples (ql.Date, rate as decimal)
        """
        self.rates = sorted(rates, key=lambda x: x[0])
        self.day_count = day_count
        self.compounding = compounding
        self.compounding_frequency = compounding_frequency

    def compound(self, end_date, tenor):
        start_date = end_date - ql.Period(tenor.length(), tenor.units())
        filtered_rates = [(d, r) for d, r in self.rates if start_date <= d <= end_date]
        if len(filtered_rates) < 1:
            raise ValueError("Not enough rate data to compute compounded rate")

        total_factor = 1.0
        for d, r in filtered_rates:
            dt = self.day_count.yearFraction(d, d+1)
            ir = ql.InterestRate(r, self.day_count, self.compounding, self.compounding_frequency)
            total_factor *= ir.compoundFactor(dt)

        # Convert total factor back to rate
        compounded_rate = total_factor - 1.0
        return compounded_rate




        
# class ShockedSOFRRateProvider(MarketRateProvider):

#     def sofr_operations(self, rateType: str = 'sofr', startDate: str = None, format: str = 'json', data_type: str = 'rate'):
#         # Convert QuantLib Date to string 'YYYY-MM-DD' if startDate is QuantLib Date
#         if startDate is not None:
#             if isinstance(startDate, ql.Date):
#                 startDate = startDate.to_date().isoformat()  # 'YYYY-MM-DD'
#             elif isinstance(startDate, str):
#                 # assume already in correct format
#                 pass
#             else:
#                 raise ValueError("startDate must be QuantLib Date or string in 'YYYY-MM-DD' format")
#         else:
#             # default if no date passed
#             startDate = "2025-06-24"

#         url = f"https://markets.newyorkfed.org/api/rates/secured/{rateType}/search.{format}?startDate={startDate}&type={data_type}"
#         return url

#     def get_sofr_data(self, startDate=None):
#         url = self.sofr_operations(startDate=startDate)
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             return data
#         else:
#             print(f"Request failed with status code: {response.status_code}")
#             return None

#     def get_market_rates(self, startDate=None):
#         data = self.get_sofr_data(startDate=startDate)
#         if data and "refRates" in data:
#             rates = []
#             for entry in data["refRates"]:
#                 tenor = ql.Period(1, ql.Days)
#                 rate = entry["percentRate"] / 100.0
#                 rates.append((tenor, rate))
#             return rates
#         else:
#             return []