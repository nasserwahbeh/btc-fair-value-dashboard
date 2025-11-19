# update_data.py

import os
import json
from datetime import datetime

import yfinance as yf
from fredapi import Fred
import gspread
from google.oauth2.service_account import Credentials

# ==========================
# GOOGLE AUTH
# ==========================
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

# ==========================
# FRED AUTH
# ==========================
fred = Fred(api_key=os.environ["FRED_API_KEY"])


def last_value(series_id: str) -> float:
    """Get latest available value from a FRED series."""
    s = fred.get_series(series_id)
    s = s.dropna()
    return float(s.iloc[-1])


# ==========================
# FETCH BTC (daily close)
# ==========================
btc_df = yf.download("BTC-USD", period="7d", interval="1d")
btc_close = float(btc_df["Close"].ffill().iloc[-1])

# ==========================
# FETCH M2 LEVELS (monthly)
# ==========================

# US M2
us_m2 = last_value("M2SL")

# Euro Area M2
eu_m2 = last_value("MYAGM2EZM196N")

# Japan M2
jp_m2 = last_value("MYAGM2JPM189S")

# China M2
cn_m2 = last_value("MYAGM2CNM189N")

# ==========================
# FX RATES (daily) – invert as you specified
# ==========================

# DEXUSEU: U.S. Dollars per Euro  -> EURUSD = 1 / DEXUSEU
dexuseu = last_value("DEXUSEU")
eurusd = 1.0 / dexuseu if dexuseu != 0 else 0.0

# DEXJPUS: Japanese Yen per U.S. Dollar -> JPYUSD = 1 / DEXJPUS
dexjpus = last_value("DEXJPUS")
jpyusd = 1.0 / dexjpus if dexjpus != 0 else 0.0

# DEXCHUS: Chinese Yuan per U.S. Dollar -> CNHUSD ≈ 1 / DEXCHUS
dexchus = last_value("DEXCHUS")
cnhusd = 1.0 / dexchus if dexchus != 0 else 0.0

# ==========================
# SYNTHETIC GLOBAL M2 (TRILLIONS)
# ==========================
synthetic_m2_trillions = (
    us_m2
    + eu_m2 * eurusd
    + jp_m2 * jpyusd
    + cn_m2 * cnhusd
) / 1e12

# ==========================
# APPEND ROW TO SHEET
# ==========================
today_str = datetime.utcnow().strftime("%Y-%m-%d")

row = [today_str, btc_close, synthetic_m2_trillions]

# write as normal user-entered values
sheet.append_row(row, value_input_option="USER_ENTERED")

print("Updated successfully:", row)
