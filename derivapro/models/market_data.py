"""market_data.py — yfinance market-data fetcher with Flask-Caching support.

All network I/O is isolated in module-level *cached* helper functions so that
the cache key is the minimal set of primitive parameters (ticker, dates) rather
than a StockData instance.  StockData methods delegate to those helpers,
keeping the public API unchanged while eliminating duplicate API calls within
a single request (e.g. sensitivity-analysis loops).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy cache helpers – avoids circular imports; gracefully degrades when the
# Flask app hasn't been initialized (tests, management scripts, etc.).
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


# ---------------------------------------------------------------------------
# Module-level cached API wrappers – keyed on explicit primitive parameters
# ---------------------------------------------------------------------------


@_memoize(timeout=300)
def _fetch_closing_price(ticker: str, prev_date: str, start_date: str) -> float:
    """Fetch the last closing price from yfinance (cached 5 min by default)."""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=prev_date, end=start_date)
    return float(hist["Close"].iloc[-1])


@_memoize(timeout=300)
def _fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV history from yfinance (cached 5 min by default)."""
    return yf.download(ticker, start=start_date, end=end_date)


@_memoize(timeout=60)
def _fetch_current_price(ticker: str) -> float:
    """Fetch the most-recent trading price for *ticker* (cached 1 min)."""
    stock_info = yf.Ticker(ticker)
    return float(stock_info.history(period="1d").iloc[-1]["Close"])


# ---------------------------------------------------------------------------
# StockData – public API (fully type-annotated)
# ---------------------------------------------------------------------------


class StockData:
    def __init__(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        self.ticker: str = ticker
        self.end_date: str = (
            datetime.today().strftime("%Y-%m-%d") if end_date is None else str(end_date)
        )
        self.start_date: str = (
            self.get_previous_market_day(self.end_date)
            if start_date is None
            else str(start_date)
        )

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    def get_years_difference(self) -> float:
        """Return the calendar difference between start_date and end_date in years."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        diff = relativedelta(end, start)
        return diff.years + diff.months / 12 + diff.days / 365.25

    @staticmethod
    def get_previous_market_day(date: str) -> str:
        """Return the previous business day relative to *date* (YYYY-MM-DD)."""
        return (pd.to_datetime(date) - BDay(1)).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Market-data fetchers (delegate to cached module-level functions)
    # ------------------------------------------------------------------

    def get_stock_data(self) -> pd.DataFrame:
        """Return historical OHLCV data for the configured ticker and date range."""
        return _fetch_stock_data(self.ticker, self.start_date, self.end_date)

    def get_current_price(self) -> float:
        """Return the most-recent closing price for the configured ticker."""
        return _fetch_current_price(self.ticker)

    def get_closing_price(self) -> float:
        """Return the closing price on (or just before) *start_date*."""
        prev_date: str = (pd.to_datetime(self.start_date) - BDay(1)).strftime(
            "%Y-%m-%d"
        )
        return _fetch_closing_price(self.ticker, prev_date, self.start_date)

    def get_implied_volatility(
        self,
        expiry_date: Optional[Any] = None,
        strike: Optional[float] = None,
        option_type: str = "call",
    ) -> dict[str, Any]:
        """
        Return implied volatility for the nearest available option contract.

        Parameters
        ----------
        expiry_date : str | datetime | None
            Target expiry date (default: 30 days from today).
        strike : float | None
            Target strike price.
        option_type : str
            ``"call"`` or ``"put"``.

        Returns
        -------
        dict
            Contract parameters and estimated implied volatility.
        """
        try:
            if expiry_date is None:
                expiry_date = datetime.today() + timedelta(days=30)

            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")

            ticker_info = yf.Ticker(self.ticker)
            options = ticker_info.option_chain(expiry_date.strftime("%Y-%m-%d"))

            chain = options.calls if option_type.lower() == "call" else options.puts

            if strike is None:
                nearest = chain.iloc[
                    (chain["strike"] - chain["strike"].mean()).abs().argsort()[:1]
                ]
            else:
                nearest = chain.iloc[(chain["strike"] - strike).abs().argsort()[:1]]

            return {
                "ticker": self.ticker,
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "strike": float(nearest["strike"].values[0]),
                "option_type": option_type,
                "implied_volatility": float(nearest["impliedVolatility"].values[0]),
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch implied volatility for %s: %s", self.ticker, exc
            )
            return {
                "ticker": self.ticker,
                "expiry_date": str(expiry_date),
                "strike": strike,
                "option_type": option_type,
                "implied_volatility": None,
                "error": str(exc),
            }
