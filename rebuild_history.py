import yfinance as yf
import pandas as pd
from datetime import datetime
from fredapi import Fred
import gspread
from google.oauth2.service_account import Credentials
import json
import os

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
# FRED AUTH
# ============================
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# ============================
# HISTORICAL RANGE
# ============================
start_date = "2017-01-01"

# ============================
# FETCH BTC PRICE (Daily)
# ============================
btc = (
    yf.download("BTC-USD", start=start_date, interval="1d")["Close"]
    .rename("BTC")
)

# ============================
# FETCH M2 COMPONENTS
# ============================
# M2
us_m2 = fred.get_series("M2SL").rename("US_M2")
eu_m2 = fred.get_series("MYAGM2EZM196N").rename("EU_M2")
jp_m2 = fred.get_series("MYAGM2JPM189S").rename("JP_M2")
cn_m2 = fred.get_series("MYAGM2CNM189N").rename("CN_M2")

# FX Rates (inverse tickers from FRED)
eurusd = (1 / fred.get_series("DEXUSEU")).rename("EURUSD")  # USD/EUR -> EUR/USD
jpyusd = (1 / fred.get_series("DEXJPUS")).rename("JPYUSD")  # USD/JPY -> JPY/USD
cnhusd = (1 / fred.get_series("DEXCHUS")).rename("CNHUSD")  # USD/CNH -> CNH/USD

# ============================
# ALIGN FREQUENCIES + FILL GAPS
# ============================
df = pd.concat([btc, us_m2, eu_m2, jp_m2, cn_m2, eurusd, jpyusd, cnhusd], axis=1)
df = df.ffill().bfill()

# ============================
# Compute Global M2 Liquidity
# ============================
df["Global_M2"] = (
    df["US_M2"]
    + df["EU_M2"] * df["EURUSD"]
    + df["JP_M2"] * df["JPYUSD"]
    + df["CN_M2"] * df["CNHUSD"]
)

# Convert to trillions
df["Global_M2_trn"] = df["Global_M2"] / 1e12
df["US_M2_trn"] = df["US_M2"] / 1e12
df["EU_M2_trn"] = df["EU_M2"] / 1e12
df["JP_M2_trn"] = df["JP_M2"] / 1e12
df["CN_M2_trn"] = df["CN_M2"] / 1e12

# ============================
# FINAL EXPORT FORMAT
# ============================
df = df.reset_index()
df["index"] = df["index"].dt.strftime("%Y-%m-%d")
df = df[["index", "BTC", "Global_M2_trn", "US_M2_trn", "EU_M2_trn", "JP_M2_trn", "CN_M2_trn"]]

# ============================
# WRITE TO GOOGLE SHEET
# ============================
sheet.clear()
sheet.append_row(df.columns.tolist())

values = df.values.tolist()
sheet.append_rows(values)

print("Historical rebuild complete ✔")
