import yfinance as yf
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred
import json
import os
import numpy as np

# ================================
# GOOGLE AUTH
# ================================
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

# ================================
# API KEYS
# ================================
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# ================================
# FETCH BTC CLOSE
# ================================
btc = yf.download("BTC-USD", period="7d", interval="1d")["Close"].ffill().iloc[-1]

# ================================
# FETCH M2 COMPONENTS
# (Daily data may contain NaN → forward fill)
# ================================
us_m2 = fred.get_series("M2SL").ffill().iloc[-1]        # USA
eu_m2 = fred.get_series("M2REAL")[-1] if "M2REAL" in fred.series else 0  # Placeholder EU M2 if no exact series
jp_m2 = fred.get_series("BOJMBASEW").ffill().iloc[-1]  # Japan Monetary base weekly
cn_m2 = fred.get_series("MABMM201S")[-1] if "MABMM201S" in fred.series else 0  # China M2 (placeholder)

# ================================
# FX RATES
# ================================
eurusd = yf.download("EURUSD=X", period="7d", interval="1d")["Close"].ffill().iloc[-1]
usdjpy = yf.download("JPY=X", period="7d", interval="1d")["Close"].ffill().iloc[-1]
usd_cnh = yf.download("CNH=X", period="7d", interval="1d")["Close"].ffill().iloc[-1]

# ================================
# SYNTHETIC GLOBAL M2 FORMULA
# ================================
synthetic_m2 = (us_m2 + eu_m2 * eurusd + jp_m2 / usdjpy + cn_m2 / usd_cnh) / 1e12

# ================================
# APPEND ROW TO SHEET
# ================================
today = datetime.now().strftime("%Y-%m-%d")
row = [today, float(btc), float(synthetic_m2)]

sheet.append_row(row)
print("Updated successfully")
