import os
import pandas as pd
import requests
from eodhd import APIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize with your API Key
API_KEY = os.getenv("EODHD_API_KEY")
if not API_KEY:
    print("Warning: EODHD_API_KEY not found in environment.")

api = APIClient(API_KEY)

def get_luse_stocks():
    """Fetches all active tickers on the Lusaka Securities Exchange"""
    print("Fetching LuSE Tickers...")
    try:
        # Exchange code for Lusaka is 'LUSE'
        symbols = api.get_exchange_symbols("LUSE")
        return pd.DataFrame(symbols)
    except Exception as e:
        print(f"Error fetching LuSE stocks: {e}")
        return pd.DataFrame()

def get_live_price(ticker):
    """Fetches 15-min delayed live price for a specific stock (e.g., 'ZNCO.LUSE')"""
    try:
        # Note: Ticker must include the .LUSE suffix
        resp = api.get_live_stock_prices(ticker)
        if isinstance(resp, list):
            return pd.DataFrame(resp)
        return pd.DataFrame([resp])
    except Exception as e:
        print(f"Error fetching live price for {ticker}: {e}")
        return pd.DataFrame()

def get_historical_data(ticker, start_date="2024-01-01"):
    """Fetches EOD historical data via raw REST API to bypass wrapper bug."""
    try:
        url = f"https://eodhd.com/api/eod/{ticker}?api_token={API_KEY}&fmt=json&from={start_date}"
        resp = requests.get(url)
        resp.raise_for_status()
        # The JSON is a list of dictionaries with open, high, low, close, volume, etc.
        return pd.DataFrame(resp.json())
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

# --- Execution Example ---
if __name__ == "__main__":
    # 1. List all available Zambian stocks
    luse_list = get_luse_stocks()
    print("\n--- Active Tickers on LuSE ---")
    print(luse_list[['Code', 'Name', 'Currency']].head())

    # 2. Get live data for ZANACO (Zambian National Commercial Bank)
    # Ticker for ZANACO is ZNCO
    zanaco_live = get_live_price("ZNCO.LUSE")
    print("\n--- ZANACO Live Data ---")
    print(zanaco_live[['code', 'close', 'change_p']])

    # 3. Get historical data for CEC (Copperbelt Energy Corporation)
    cec_hist = get_historical_data("CECZ.LUSE")
    print("\n--- CEC Historical (Last 5 days) ---")
    print(cec_hist.tail())