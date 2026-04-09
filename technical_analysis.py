"""
technical_analysis.py
======================
Gofi AI — Technical Analysis Engine
All functions here are callable by the chatbot's function-calling layer
when users ask questions like:
  • "What is the trend for ZANACO?"
  • "Show me RSI for CEC"
  • "Is AIRTEL Zambia in a support zone?"

Each function returns a structured dict so the LLM can narrate the result
in plain Zambian-English.
"""

import pandas as pd
import numpy as np
from typing import Optional
from api_Data import get_historical_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ohlcv(ticker: str, start_date: str = "2023-01-01") -> pd.DataFrame:
    """
    Loads historical OHLCV data for a LuSE ticker via the EODHD client.
    Returns a clean DataFrame with datetime index.
    """
    df = get_historical_data(ticker, start_date=start_date)

    # Normalise column names (EODHD can return mixed casing)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df[["Open", "High", "Low", "Close", "Volume"]] = (
        df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["Close"])


# ---------------------------------------------------------------------------
# 1. Moving Averages (SMA & EMA)
# ---------------------------------------------------------------------------

def moving_averages(ticker: str, short: int = 20, long: int = 50) -> dict:
    """
    Computes short-term and long-term Simple Moving Averages.
    A 'Golden Cross' (short crosses above long) is a bullish signal;
    a 'Death Cross' is bearish.

    Returns: dict with SMA values, signal, and plain-text summary.
    """
    df = _load_ohlcv(ticker)
    df[f"SMA_{short}"] = df["Close"].rolling(short).mean()
    df[f"SMA_{long}"]  = df["Close"].rolling(long).mean()
    df[f"EMA_{short}"] = df["Close"].ewm(span=short, adjust=False).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    sma_short = round(latest[f"SMA_{short}"], 4)
    sma_long  = round(latest[f"SMA_{long}"],  4)
    ema_short = round(latest[f"EMA_{short}"], 4)
    price     = round(latest["Close"], 4)

    # Crossover detection
    if prev[f"SMA_{short}"] < prev[f"SMA_{long}"] and sma_short > sma_long:
        signal = "GOLDEN CROSS — Bullish momentum building 🟢"
    elif prev[f"SMA_{short}"] > prev[f"SMA_{long}"] and sma_short < sma_long:
        signal = "DEATH CROSS — Bearish pressure detected 🔴"
    elif sma_short > sma_long:
        signal = "Uptrend — SMA{} above SMA{} 🟢".format(short, long)
    else:
        signal = "Downtrend — SMA{} below SMA{} 🔴".format(short, long)

    return {
        "ticker":    ticker,
        "price":     price,
        f"SMA_{short}": sma_short,
        f"SMA_{long}":  sma_long,
        f"EMA_{short}": ema_short,
        "signal":    signal,
        "summary": (
            f"{ticker} is trading at ZMW {price}. "
            f"The {short}-day SMA is {sma_short} and the {long}-day SMA is {sma_long}. "
            f"Signal: {signal}."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Relative Strength Index (RSI)
# ---------------------------------------------------------------------------

def rsi(ticker: str, period: int = 14) -> dict:
    """
    Computes RSI to identify overbought/oversold conditions.
    RSI > 70: overbought (potential sell zone).
    RSI < 30: oversold (potential buy zone).
    """
    df = _load_ohlcv(ticker)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest_rsi = round(df["RSI"].iloc[-1], 2)
    price      = round(df["Close"].iloc[-1], 4)

    if latest_rsi >= 70:
        zone    = "Overbought"
        outlook = "Consider taking profits — stock may be due for a pullback."
    elif latest_rsi <= 30:
        zone    = "Oversold"
        outlook = "Potential buying opportunity — look for a reversal confirmation."
    else:
        zone    = "Neutral"
        outlook = "No extreme condition. Monitor for trend confirmation."

    return {
        "ticker":  ticker,
        "price":   price,
        "RSI":     latest_rsi,
        "zone":    zone,
        "summary": (
            f"{ticker} RSI({period}) = {latest_rsi} → {zone}. {outlook}"
        ),
    }


# ---------------------------------------------------------------------------
# 3. MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------------

def macd(ticker: str, fast: int = 12, slow: int = 26, signal_period: int = 9) -> dict:
    """
    Computes MACD line, Signal line, and Histogram.
    MACD > Signal line: bullish. MACD < Signal: bearish.
    """
    df = _load_ohlcv(ticker)
    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram  = macd_line - signal_line

    latest_macd   = round(macd_line.iloc[-1], 4)
    latest_signal = round(signal_line.iloc[-1], 4)
    latest_hist   = round(histogram.iloc[-1], 4)
    price         = round(df["Close"].iloc[-1], 4)

    if latest_macd > latest_signal and histogram.iloc[-1] > histogram.iloc[-2]:
        signal = "Bullish — MACD above signal and histogram expanding 🟢"
    elif latest_macd < latest_signal and histogram.iloc[-1] < histogram.iloc[-2]:
        signal = "Bearish — MACD below signal and histogram contracting 🔴"
    else:
        signal = "Neutral / Crossover in progress ⚪"

    return {
        "ticker":       ticker,
        "price":        price,
        "MACD":         latest_macd,
        "signal_line":  latest_signal,
        "histogram":    latest_hist,
        "signal":       signal,
        "summary": (
            f"{ticker} MACD = {latest_macd}, Signal = {latest_signal}, "
            f"Histogram = {latest_hist}. {signal}"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(ticker: str, period: int = 20, std_dev: float = 2.0) -> dict:
    """
    Computes Bollinger Bands.
    Price near upper band → overbought stretch.
    Price near lower band → potential support / oversold.
    """
    df = _load_ohlcv(ticker)
    df["SMA"]   = df["Close"].rolling(period).mean()
    df["STD"]   = df["Close"].rolling(period).std()
    df["Upper"] = df["SMA"] + std_dev * df["STD"]
    df["Lower"] = df["SMA"] - std_dev * df["STD"]

    latest = df.iloc[-1]
    price  = round(latest["Close"], 4)
    upper  = round(latest["Upper"], 4)
    lower  = round(latest["Lower"], 4)
    mid    = round(latest["SMA"],   4)

    band_width = round(upper - lower, 4)
    pct_b      = round((price - lower) / (upper - lower) * 100, 1)  # 0-100

    if price >= upper:
        position = "At/Above Upper Band — Overbought territory"
    elif price <= lower:
        position = "At/Below Lower Band — Potential support zone"
    else:
        position = f"Inside bands ({pct_b}% of band width from lower)"

    return {
        "ticker":     ticker,
        "price":      price,
        "upper_band": upper,
        "mid_band":   mid,
        "lower_band": lower,
        "band_width": band_width,
        "%B":         pct_b,
        "position":   position,
        "summary": (
            f"{ticker} Bollinger Bands({period}, {std_dev}σ): "
            f"Lower={lower}, Mid={mid}, Upper={upper}. "
            f"Current price {price} → {position}."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Support & Resistance Levels
# ---------------------------------------------------------------------------

def support_resistance(ticker: str, lookback: int = 90) -> dict:
    """
    Identifies key support and resistance price levels using local
    highs/lows over the lookback window (default 90 trading days).
    """
    df = _load_ohlcv(ticker).tail(lookback)
    price = round(df["Close"].iloc[-1], 4)

    # Simple pivot-point based approach
    pivot_high = df["High"].rolling(5, center=True).max()
    pivot_low  = df["Low"].rolling(5, center=True).min()

    resistance_levels = sorted(
        df["High"][df["High"] == pivot_high].dropna().unique(), reverse=True
    )[:3]
    support_levels = sorted(
        df["Low"][df["Low"] == pivot_low].dropna().unique(), reverse=False
    )[-3:]

    nearest_resistance = min(
        (r for r in resistance_levels if r > price), default=None
    )
    nearest_support = max(
        (s for s in support_levels if s < price), default=None
    )

    return {
        "ticker":             ticker,
        "price":              price,
        "resistance_levels":  [round(r, 4) for r in resistance_levels],
        "support_levels":     [round(s, 4) for s in support_levels],
        "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else "N/A",
        "nearest_support":    round(nearest_support, 4) if nearest_support else "N/A",
        "summary": (
            f"{ticker} key levels (last {lookback} days): "
            f"Nearest Support = ZMW {nearest_support}, "
            f"Nearest Resistance = ZMW {nearest_resistance}. "
            f"Current price: ZMW {price}."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Full Technical Report (aggregates all signals)
# ---------------------------------------------------------------------------

def full_technical_report(ticker: str) -> dict:
    """
    Master function: runs all TA tools and returns a unified report dict.
    Called by the chatbot when a user asks for a 'full analysis'.
    """
    print(f"[TA] Generating full technical report for {ticker}...")
    report = {
        "ticker":             ticker,
        "moving_averages":    moving_averages(ticker),
        "rsi":                rsi(ticker),
        "macd":               macd(ticker),
        "bollinger_bands":    bollinger_bands(ticker),
        "support_resistance": support_resistance(ticker),
    }

    # Build a plain-English summary for the LLM to narrate
    report["narrative"] = (
        f"Technical Analysis Report for {ticker}:\n"
        f"1. Trend: {report['moving_averages']['signal']}\n"
        f"2. Momentum (RSI): {report['rsi']['summary']}\n"
        f"3. MACD: {report['macd']['signal']}\n"
        f"4. Volatility (BB): {report['bollinger_bands']['position']}\n"
        f"5. Levels: {report['support_resistance']['summary']}"
    )
    return report


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = full_technical_report("ZNCO.LUSE")
    print("\n" + "="*60)
    print(report["narrative"])
