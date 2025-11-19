import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf
import gspread
from fredapi import Fred
from google.oauth2.service_account import Credentials

# === CONFIG ===
SHEET_URL = "https://docs.google.com/spreadsheets/d/1QkEUXSVxPqBgEhNxtoKY7sra5yc1515WqydeFSg7ibQ/edit#gid=0"

GCP_SERVICE_ACCOUNT = os.environ["GCP_SERVICE_ACCOUNT"]
FRED_API_KEY = os.environ["FRED_API_KEY"]

# === AUTH ===
creds_info = json.loads(GCP_SERVICE_ACCOUNT)
creds = Credentials.from_service_account_info(
    creds_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
sheet = gc.open_by_url(SHEET_URL).sheet1

fred = Fred(api_key=FRED_API_KEY)

# === DOWNLOAD FULL DATASETS (FROM JAN 1 2017) ===
start_date = "2017-01-01"

btc = yf.download("BTC-USD", start=start_date, interval="1d")["Close"]

us_m2 = fred.get_series("M2SL").loc[start_date:]
eu_m2 = fred.get_series("MYAGM2EZM196N").loc[start_date:]
jp_m2 = fred.get_series("MYAGM2JPM189S").loc[start_date:]
cn_m2 = fred.get_series("MYAGM2CNM189N").loc[start_date:]

eurusd = fred.get_series("DEXUSEU").loc[start_date:]
usdjpy = fred.get_series("DEXJPUS").loc[start_date:]
usdcny = fred.get_series("DEXCHUS").loc[start_date:]

# === MERGE ALL DATA ON DATE ===
df = pd.DataFrame({
    "US_M2": us_m2,
    "EU_M2": eu_m2,
    "JP_M2": jp_m2,
    "CN_M2": cn_m2,
    "EURUSD": eurusd,
    "USDJPY": usdjpy,
    "USDCNY": usdcny
}).ffill()

# === CONVERT TO TRILLIONS USD ===
df["US_M2_trn"] = df["US_M2"] / 1e3
df["EU_M2_trn"] = (df["EU_M2"] * df["EURUSD"]) / 1e12
df["JP_M2_trn"] = (df["JP_M2"] / df["USDJPY"]) / 1e12
df["CN_M2_trn"] = (df["CN_M2"] / df["USDCNY"]) / 1e12

df["Global_M2_trn"] = (
    df["US_M2_trn"]
    + df["EU_M2_trn"]
    + df["JP_M2_trn"]
    + df["CN_M2_trn"]
)

# === MERGE BTC PRICES ===
df = df.merge(btc, left_index=True, right_index=True, how="left").ffill()
df.rename(columns={"Close": "BTC"}, inplace=True)

df = df[["BTC", "Global_M2_trn", "US_M2_trn", "EU_M2_trn", "JP_M2_trn", "CN_M2_trn"]]

df.reset_index(inplace=True)
df.rename(columns={"index": "Date"}, inplace=True)

# === WRITE TO GOOGLE SHEET (REPLACE EVERYTHING) ===
sheet.clear()
sheet.update([df.columns.tolist()] + df.values.tolist())

print("Historical rebuild complete! Starting from 2017-01-01")
