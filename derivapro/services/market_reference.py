from __future__ import annotations

import math
import base64
import io
import warnings
from datetime import datetime
from typing import Any

from ..utils.lazy_imports import LazyImport


pd = LazyImport("pandas")
np = LazyImport("numpy")
plt = LazyImport("matplotlib.pyplot")
yf = LazyImport("yfinance")


PERIOD_OPTIONS = {
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
    "2y": "2 Years",
}


def build_equity_market_reference(
    symbol: str,
    period: str = "6mo",
    strike: float | None = None,
    maturity_date: str | None = None,
    option_type: str = "call",
    visual_mode: str = "none",
) -> dict[str, Any]:
    """Build a compact equity market reference snapshot from yfinance.

    The output is intended as user-facing context for pricing assumptions, not
    as a controlled production market-data source.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        raise ValueError("Enter a ticker symbol before fetching market reference.")

    period = period if period in PERIOD_OPTIONS else "6mo"
    ticker = yf.Ticker(symbol)
    history = _fetch_history(ticker, symbol, period)
    close = _select_close_series(history)

    if close.empty:
        raise ValueError(f"No price history was returned for {symbol}.")

    latest_close = float(close.iloc[-1])
    first_close = float(close.iloc[0])
    period_return = (latest_close / first_close - 1.0) if first_close > 0 else None
    returns = close.pct_change().dropna()
    realized_vol = (
        float(returns.std() * math.sqrt(252.0)) if len(returns) >= 2 else None
    )

    high = _series_value(history, "High", "max")
    low = _series_value(history, "Low", "min")
    avg_volume = _series_value(history, "Volume", "mean")
    latest_date = _date_label(close.index[-1])
    option_reference = _fetch_option_reference(
        ticker,
        latest_close,
        strike,
        maturity_date,
        option_type,
    )
    visual_mode = _normalize_visual_mode(visual_mode)
    price_chart = (
        _build_price_chart(close)
        if visual_mode in {"price_chart", "all"}
        else None
    )
    iv_surface = (
        _fetch_iv_surface_snapshot(
            ticker,
            latest_close,
            strike,
            option_type,
        )
        if visual_mode in {"iv_smile", "iv_surface_3d", "all"}
        else None
    )
    iv_surface_plot = (
        _build_iv_surface_plot(iv_surface)
        if iv_surface and visual_mode in {"iv_surface_3d", "all"}
        else None
    )

    return {
        "symbol": symbol,
        "provider": "Yahoo Finance via yfinance",
        "period": period,
        "period_label": PERIOD_OPTIONS[period],
        "latest_close": latest_close,
        "latest_date": latest_date,
        "first_close": first_close,
        "period_return": period_return,
        "realized_volatility": realized_vol,
        "high": high,
        "low": low,
        "average_volume": avg_volume,
        "observations": int(len(close)),
        "option_reference": option_reference,
        "visual_mode": visual_mode,
        "price_chart": price_chart,
        "iv_surface": iv_surface,
        "iv_surface_plot": iv_surface_plot,
        "disclaimer": (
            "Market reference data is fetched on demand from a free public-data "
            "provider and should be validated before production or commercial use."
        ),
    }


def _normalize_visual_mode(visual_mode: Any) -> str:
    if visual_mode is True:
        return "all"
    if visual_mode is False or visual_mode is None:
        return "none"

    visual_mode = str(visual_mode)
    if visual_mode in {"price_chart", "iv_smile", "iv_surface_3d", "all"}:
        return visual_mode
    return "none"


def _build_price_chart(close: Any) -> dict[str, Any] | None:
    if close.empty:
        return None

    max_points = 80
    if len(close) > max_points:
        step = max(1, math.ceil(len(close) / max_points))
        close = close.iloc[::step]

    min_price = float(close.min())
    max_price = float(close.max())
    price_range = max(max_price - min_price, 1e-9)
    denominator = max(len(close) - 1, 1)
    points = []

    for idx, price in enumerate(close):
        x_value = (idx / denominator) * 100.0
        y_value = 100.0 - ((float(price) - min_price) / price_range) * 100.0
        points.append(f"{x_value:.2f},{y_value:.2f}")

    return {
        "points": " ".join(points),
        "start_date": _date_label(close.index[0]),
        "end_date": _date_label(close.index[-1]),
        "min_price": min_price,
        "max_price": max_price,
        "start_price": float(close.iloc[0]),
        "end_price": float(close.iloc[-1]),
        "point_count": int(len(close)),
    }


def _fetch_iv_surface_snapshot(
    ticker: Any,
    spot_price: float,
    strike: float | None,
    option_type: str,
) -> dict[str, Any] | None:
    try:
        expirations = list(ticker.options or [])[:4]
    except Exception:
        return None

    if not expirations:
        return None

    target_strike = float(strike) if strike and strike > 0 else spot_price
    panels = []
    all_ivs = []
    surface_points = []

    for expiry_index, expiry in enumerate(expirations):
        try:
            option_chain = ticker.option_chain(expiry)
            chain = option_chain.calls if option_type == "call" else option_chain.puts
        except Exception:
            continue

        if chain.empty or "strike" not in chain.columns:
            continue

        chain = chain.copy()
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain["impliedVolatility"] = pd.to_numeric(
            chain.get("impliedVolatility"), errors="coerce"
        )
        chain = chain.dropna(subset=["strike", "impliedVolatility"])
        chain = chain[chain["impliedVolatility"] > 0]
        if chain.empty:
            continue

        nearby = chain.iloc[(chain["strike"] - target_strike).abs().argsort()[:7]]
        nearby = nearby.sort_values("strike")
        rows = []
        for _, row in nearby.iterrows():
            iv_value = float(row["impliedVolatility"])
            strike_value = float(row["strike"])
            moneyness = strike_value / spot_price
            all_ivs.append(iv_value)
            surface_points.append(
                {
                    "expiry_index": expiry_index,
                    "expiration": expiry,
                    "strike": strike_value,
                    "moneyness": moneyness,
                    "implied_volatility": iv_value,
                }
            )
            rows.append(
                {
                    "strike": strike_value,
                    "moneyness": moneyness,
                    "implied_volatility": iv_value,
                    "width": min(max(iv_value * 100.0, 4.0), 100.0),
                }
            )

        panels.append({"expiration": expiry, "rows": rows})

    if not panels:
        return None

    return {
        "option_type": option_type,
        "target_strike": target_strike,
        "expirations": panels,
        "surface_points": surface_points,
        "min_iv": min(all_ivs),
        "max_iv": max(all_ivs),
    }


def _build_iv_surface_plot(iv_surface: dict[str, Any] | None) -> str | None:
    if not iv_surface or not iv_surface.get("surface_points"):
        return None

    points = iv_surface["surface_points"]
    if len(points) < 3:
        return None

    x = np.array([point["expiry_index"] for point in points], dtype=float)
    y = np.array([point["moneyness"] for point in points], dtype=float)
    z = np.array([point["implied_volatility"] for point in points], dtype=float)

    fig = plt.figure(figsize=(8.5, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfaf7")

    try:
        surface = ax.plot_trisurf(
            x,
            y,
            z,
            cmap="cividis",
            linewidth=0.25,
            antialiased=True,
            alpha=0.92,
        )
        fig.colorbar(surface, ax=ax, shrink=0.62, aspect=12, pad=0.08)
    except Exception:
        ax.scatter(x, y, z, c=z, cmap="cividis", s=42)

    expirations = [
        panel["expiration"] for panel in iv_surface.get("expirations", [])
    ]
    ax.set_xticks(range(len(expirations)))
    ax.set_xticklabels(expirations, rotation=18, ha="right", fontsize=8)
    ax.set_xlabel("Expiration")
    ax.set_ylabel("Strike / Spot")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Listed Option Implied Volatility Surface Snapshot", pad=16)
    ax.view_init(elev=24, azim=-132)
    ax.grid(True, alpha=0.24)

    img = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*constrained_layout.*",
            category=UserWarning,
        )
        fig.savefig(img, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode("ascii")


def _fetch_history(ticker: Any, symbol: str, period: str) -> Any:
    history = ticker.history(period=period, interval="1d", auto_adjust=False)
    if not history.empty:
        return history

    downloaded = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if hasattr(downloaded, "columns") and getattr(downloaded.columns, "nlevels", 1) > 1:
        downloaded = downloaded.xs(symbol, axis=1, level=-1, drop_level=True)
    return downloaded


def _select_close_series(history: Any) -> Any:
    close_column = "Adj Close" if "Adj Close" in history.columns else "Close"
    close = pd.to_numeric(history[close_column], errors="coerce").dropna()
    return close


def _series_value(history: Any, column: str, method: str) -> float | None:
    if column not in history.columns:
        return None

    series = pd.to_numeric(history[column], errors="coerce").dropna()
    if series.empty:
        return None

    if method == "max":
        return float(series.max())
    if method == "min":
        return float(series.min())
    if method == "mean":
        return float(series.mean())
    return None


def _date_label(value: Any) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _fetch_option_reference(
    ticker: Any,
    spot_price: float,
    strike: float | None,
    maturity_date: str | None,
    option_type: str,
) -> dict[str, Any] | None:
    try:
        expirations = list(ticker.options or [])
    except Exception:
        return None

    if not expirations:
        return None

    expiry = _nearest_expiration(expirations, maturity_date)
    try:
        option_chain = ticker.option_chain(expiry)
        chain = option_chain.calls if option_type == "call" else option_chain.puts
    except Exception:
        return None

    if chain.empty or "strike" not in chain.columns:
        return None

    chain = chain.copy()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
    chain["impliedVolatility"] = pd.to_numeric(
        chain.get("impliedVolatility"), errors="coerce"
    )
    chain = chain.dropna(subset=["strike", "impliedVolatility"])
    chain = chain[chain["impliedVolatility"] > 0]
    if chain.empty:
        return None

    target_strike = float(strike) if strike and strike > 0 else spot_price
    nearest_idx = (chain["strike"] - target_strike).abs().idxmin()
    nearest = chain.loc[nearest_idx]

    nearby = chain.iloc[(chain["strike"] - target_strike).abs().argsort()[:5]]
    nearby = nearby.sort_values("strike")

    return {
        "expiration": expiry,
        "option_type": option_type,
        "target_strike": target_strike,
        "nearest_strike": float(nearest["strike"]),
        "implied_volatility": float(nearest["impliedVolatility"]),
        "last_price": _optional_float(nearest.get("lastPrice")),
        "bid": _optional_float(nearest.get("bid")),
        "ask": _optional_float(nearest.get("ask")),
        "open_interest": _optional_float(nearest.get("openInterest")),
        "sample_smile": [
            {
                "strike": float(row["strike"]),
                "implied_volatility": float(row["impliedVolatility"]),
            }
            for _, row in nearby.iterrows()
        ],
    }


def _nearest_expiration(expirations: list[str], maturity_date: str | None) -> str:
    if not maturity_date:
        return expirations[0]

    try:
        target = datetime.strptime(maturity_date, "%Y-%m-%d").date()
    except ValueError:
        return expirations[0]

    def distance(expiry: str) -> int:
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            return abs((expiry_date - target).days)
        except ValueError:
            return 10**9

    return min(expirations, key=distance)


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
