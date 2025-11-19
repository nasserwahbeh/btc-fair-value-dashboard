import yfinance as yf
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import json
import os

# === GOOGLE AUTH ===
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

# === FETCH DATA ===

# BTC close
btc = yf.download("BTC-USD", period="7d", interval="1d")["Close"].iloc[-1]

# US M2 from Yahoo
m2sl = yf.download("M2SL", period="7d", interval="1d")["Close"].iloc[-1]

# FX rates
eurusd = yf.download("EURUSD=X", period="7d", interval="1d")["Close"].iloc[-1]
usdjpy = yf.download("JPY=X", period="7d", interval="1d")["Close"].iloc[-1]
usd_cnh = yf.download("CNH=X", period="7d", interval="1d")["Close"].iloc[-1]

# TODO – real values once located; currently placeholders
eu_m2 = 0
jp_m2 = 0
cn_m2 = 0

synthetic_m2 = (m2sl + eu_m2 * eurusd + jp_m2 / usdjpy + cn_m2 / usd_cnh) / 1e12

# Append row (date, BTC price, synthetic M2)
now = datetime.now().strftime("%Y-%m-%d")
row = [now, float(btc), float(synthetic_m2)]
sheet.append_row(row)

print("Updated successfully")
