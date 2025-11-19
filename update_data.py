import yfinance as yf
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from fredapi import Fred

# ============================
# GOOGLE SHEETS AUTH
# ============================
creds_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
creds = Credentials.from_service_account_info(
    creds_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)

gc = gspread.authorize(creds)

SHEET_ID = "1QkEUXSVxPqBgEhNxtoKY7sra5yc1515WqydeFSg7ibQ"
sheet = gc.open_by_key(SHEET_ID).sheet1

# ============================
# FRED API
# ============================
fred_api_key = os.environ["FRED_API_KEY"]
fred = Fred(api_key=fred_api_key)

def safe_fred(series_id):
    """Fetch latest value from FRED safely with fill-forward."""
    try:
        data = fred.get_series(series_id).ffill()
        return float(data.iloc[-1])
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return 0.0

# ============================
# FETCH BTC PRICE
# ============================
btc = yf.download("BTC-USD", period="7d", interval="1d")["Close"].ffill().iloc[-1]

# ============================
# FETCH GLOBAL M2 COMPONENTS
# ============================
us_m2 = safe_fred("M2SL")                   # US
eu_m2 = safe_fred("MYAGM2EZM196N")          # Euro Area
jp_m2 = safe_fred("MYAGM2JPM189S")          # Japan
cn_m2 = safe_fred("MYAGM2CNM189N")          # China

# ============================
# FETCH FX RATES FROM FRED
# ============================
usd_per_eur = safe_fred("CCUSMA02EZM618N")  # USD per EUR
usd_per_jpy = safe_fred("CCUSMA02JPM618N")  # USD per JPY
usd_per_cnh = safe_fred("CCUSMA02CNM618N")  # USD per CNH

# Convert FX into typical price format
eurusd = 1 / usd_per_eur if usd_per_eur else 0  # EURUSD
usdjpy = usd_per_jpy if usd_per_jpy else 0       # USDJPY (already correct)
usdcnh = usd_per_cnh if usd_per_cnh else 0       # USDCNH (already correct)

# ============================
# SYNTHETIC GLOBAL LIQUIDITY M2
# ============================
synthetic_m2 = (us_m2 + eu_m2 * eurusd + jp_m2 / usdjpy + cn_m2 / usdcnh) / 1e12

# ============================
# APPEND TO GOOGLE SHEET
# ============================
now = datetime.now().strftime("%Y-%m-%d")
row = [now, float(btc), float(synthetic_m2)]

sheet.append_row(row)

print("SUCCESS: Google Sheet updated with BTC + Synthetic M2")
