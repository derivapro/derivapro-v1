import QuantLib as ql
import requests
from fredapi import Fred
from datetime import datetime
from typing import Any, Optional, Union
import pandas as pd
import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy cache helper – mirrors the one in market_data.py
# ---------------------------------------------------------------------------


def _memoize(timeout: int):
    """Return a Flask-Caching memoize decorator, or a no-op if unavailable."""
    try:
        from ..extensions import cache as _c

        return _c.memoize(timeout=timeout)
    except Exception:

        def _noop(fn):
            return fn

        return _noop


# --- Base Provider Interface ---
class MarketRateProvider:
    def get_market_rates(self, start_date: Optional[Any] = None) -> list:
        raise NotImplementedError("Subclasses must implement this method")


def to_pd_timestamp(date_input: Any) -> pd.Timestamp:
    """Convert input to a normalized pandas Timestamp (date only)."""
    if isinstance(date_input, pd.Timestamp):
        return date_input.normalize()
    if isinstance(date_input, datetime):
        return pd.Timestamp(date_input).normalize()
    if (
        hasattr(date_input, "year")
        and hasattr(date_input, "month")
        and hasattr(date_input, "dayOfMonth")
    ):
        # QuantLib.Date
        dt = datetime(date_input.year(), date_input.month(), date_input.dayOfMonth())
        return pd.Timestamp(dt).normalize()
    if isinstance(date_input, str):
        return pd.Timestamp(date_input).normalize()
    raise TypeError(f"Unsupported date type: {type(date_input)}")


# ---------------------------------------------------------------------------
# Module-level cached FRED / SOFR API calls
# TTL = 3600 s (1 hour) – rates change at most once per business day.
# ---------------------------------------------------------------------------


@_memoize(timeout=3600)
def _fred_get_series(
    api_key: str, series_id: str, observation_end_str: Optional[str]
) -> pd.Series:
    """Fetch a single FRED time-series (cached 1 hour)."""
    fred = Fred(api_key=api_key)
    return fred.get_series(series_id, observation_end=observation_end_str)


@_memoize(timeout=3600)
def _sofr_fetch(start_date_str: str) -> Optional[dict]:
    """Fetch SOFR rate data from the NY Fed API (cached 1 hour)."""
    url = (
        f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        f"?startDate={start_date_str}&type=rate"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning("SOFR request failed: HTTP %s for %s", response.status_code, url)
    except Exception as exc:
        logger.warning("SOFR request error: %s", exc)
    return None


class FREDSwapRatesProvider(MarketRateProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.swap_ids: dict[str, str] = {
            "1Y": "DSWP1",
            "2Y": "WSWP2",
            "3Y": "DSWP3",
            "4Y": "DSWP4",
            "5Y": "DSWP5",
            "7Y": "WSWP7",
            "30Y": "DSWP30",
        }

    def get_market_rates(self, start_date: Optional[Any] = None) -> list:
        py_start: Optional[str] = (
            to_pd_timestamp(start_date).strftime("%Y-%m-%d")
            if start_date is not None
            else None
        )

        latest_rates: dict[str, Optional[float]] = {}
        for label, series_id in self.swap_ids.items():
            data = _fred_get_series(self.api_key, series_id, None)  # fetch full series
            if py_start is not None:
                data = data[data.index >= py_start]
            latest_value = data.dropna().iloc[-1] if not data.empty else None
            latest_rates[label] = (
                (float(latest_value) / 100) if latest_value is not None else None
            )

        def tenor_key(label: str) -> int:
            return int("".join(filter(str.isdigit, label)))

        return [
            (ql.Period(tenor_key(label), ql.Years), latest_rates[label])
            for label in sorted(latest_rates.keys(), key=tenor_key)
            if latest_rates[label] is not None
        ]


class TreasuryRateProvider(MarketRateProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.series_ids: dict[str, str] = {
            "1M": "GS1M",
            "3M": "GS3M",
            "6M": "GS6M",
            "1Y": "GS1",
            "2Y": "GS2",
            "5Y": "GS5",
            "7Y": "GS7",
            "10Y": "GS10",
            "30Y": "GS30",
        }

    def get_market_rates(self, start_date: Optional[Any] = None) -> list:
        end_date_str: Optional[str] = (
            to_pd_timestamp(start_date).strftime("%Y-%m-%d")
            if start_date is not None
            else None
        )

        latest_rates: dict[str, float] = {}
        for label, series_id in self.series_ids.items():
            data = _fred_get_series(self.api_key, series_id, end_date_str)
            data = data.dropna()
            if data.empty:
                raise ValueError(
                    f"No treasury data available for {label} on or before {end_date_str}"
                )
            latest_rates[label] = float(data.iloc[-1]) / 100

        def tenor_key(label: str) -> int:
            return int("".join(filter(str.isdigit, label)))

        return [
            (
                ql.Period(tenor_key(label), ql.Months if "M" in label else ql.Years),
                latest_rates[label],
            )
            for label in sorted(latest_rates.keys(), key=tenor_key)
        ]


class SOFRRateProvider(MarketRateProvider):
    def sofr_operations(
        self,
        rateType: str = "sofr",
        startDate: Optional[Union[str, Any]] = None,
        format: str = "json",
        data_type: str = "rate",
    ) -> str:
        if startDate is not None:
            if hasattr(startDate, "to_date"):
                # QuantLib.Date
                startDate = startDate.to_date().isoformat()
            elif not isinstance(startDate, str):
                raise ValueError(
                    "startDate must be a QuantLib Date or string in 'YYYY-MM-DD' format"
                )
        else:
            startDate = "2025-06-24"

        return (
            f"https://markets.newyorkfed.org/api/rates/secured/{rateType}/search.{format}"
            f"?startDate={startDate}&type={data_type}"
        )

    def get_sofr_data(
        self, startDate: Optional[Union[str, Any]] = None
    ) -> Optional[dict]:
        """Fetch SOFR data via the NY Fed API (cached 1 hour)."""
        # Normalise startDate to a plain string for the cache key
        if startDate is None:
            start_str = "2025-06-24"
        elif hasattr(startDate, "to_date"):
            start_str = startDate.to_date().isoformat()
        else:
            start_str = str(startDate)
        return _sofr_fetch(start_str)

    def get_market_rates(self, startDate=None):
        data = self.get_sofr_data(startDate=startDate)
        if data and "refRates" in data:
            rates: list = []
            for entry in data["refRates"]:
                tenor = ql.Period(1, ql.Days)  # SOFR overnight rate
                rate = entry["percentRate"] / 100.0
                rates.append((tenor, rate))
            return rates[-1:]  # Return only the latest overnight rate
        return []


class SOFRCompoundedRateCalculator:
    def __init__(
        self,
        rates: list,
        day_count: Any = None,
        compounding: Any = None,
        compounding_frequency: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        rates : list of (ql.Date, float) tuples – rate as a decimal.
        """
        self.rates = sorted(rates, key=lambda x: x[0])
        self.day_count = day_count if day_count is not None else ql.Actual360()
        self.compounding = compounding if compounding is not None else ql.Compounded
        self.compounding_frequency = (
            compounding_frequency if compounding_frequency is not None else ql.Daily
        )

    def compound(self, end_date: Any, tenor: Any) -> float:
        start_date = end_date - ql.Period(tenor.length(), tenor.units())
        filtered_rates = [(d, r) for d, r in self.rates if start_date <= d <= end_date]
        if len(filtered_rates) < 1:
            raise ValueError("Not enough rate data to compute compounded rate")

        total_factor = 1.0
        for d, r in filtered_rates:
            dt = self.day_count.yearFraction(d, d + 1)
            ir = ql.InterestRate(
                r, self.day_count, self.compounding, self.compounding_frequency
            )
            total_factor *= ir.compoundFactor(dt)

        return total_factor - 1.0
