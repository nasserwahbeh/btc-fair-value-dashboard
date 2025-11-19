import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import datetime

# --------------------------- PAGE CONFIG --------------------------------
st.set_page_config(page_title="ML Bitcoin Fair Value Model", layout="wide")

# --------------------------- LOAD DATA ----------------------------------
sheet_id = "PUT_YOUR_SHEET_ID_HERE"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df = pd.read_csv(url)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert unix time to datetime
df["time"] = pd.to_datetime(df["time"], unit="s")

# Rename lag column just for safety
df.rename(columns={"lag_0": "lag0"}, inplace=True)

# ------------------------ MODEL BUILDING -------------------------------
df["m2"] = df["lag0"]  # rename for clarity
df = df.sort_values("time")

X = df["m2"].values.reshape(-1, 1)
y = df["close"].values.reshape(-1, 1)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

df["fair_value"] = model.predict(X_poly)

# Z-Score
residuals = df["close"] - df["fair_value"].iloc[:, 0]
df["zscore"] = (residuals - residuals.mean()) / residuals.std()

latest = df.iloc[-1]

# ----------------------- MODEL METRICS --------------------------------
preds = model.predict(X_poly)
r2 = r2_score(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
mae = mean_absolute_error(y, preds)
pv_error_pct = ((latest["close"] - latest["fair_value"]) / latest["fair_value"]) * 100

# ----------------------- UI HEADER -------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:white;'>ML Bitcoin Fair Value Model</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h4 style='text-align:center; color:#bbbbbb;'>Machine Learning fair value analysis via M2 Money Supply Polynomial Regression</h4>",
    unsafe_allow_html=True,
)

# ----------------------- DISPLAY RANGE SLIDER --------------------------
display_years = st.slider("Display Range (Years)", 1, 10, 5)
cutoff_date = df["time"].max() - pd.DateOffset(years=display_years)
df_range = df[df["time"] >= cutoff_date]

# ----------------------- MAIN CHART -----------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_range["time"],
    y=df_range["close"],
    mode="lines",
    name="Bitcoin Price",
    line=dict(color="white", width=2),
))

fig.add_trace(go.Scatter(
    x=df_range["time"],
    y=df_range["fair_value"],
    mode="lines",
    name="Fair Value",
    line=dict(color="#FFD700", width=2),
))

# Confidence bands
for i, c in zip([1, 2], ["#FFD70020", "#FFD70010"]):
    fig.add_trace(go.Scatter(
        x=np.concatenate([df_range["time"], df_range["time"][::-1]]),
        y=np.concatenate([
            (df_range["fair_value"] + i * residuals.std()),
            (df_range["fair_value"] - i * residuals.std())[::-1],
        ]),
        fill="toself",
        fillcolor=c,
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±{i}σ",
        showlegend=True,
    ))

fig.update_layout(
    template="plotly_dark",
    height=600,
    yaxis=dict(type="log", tickvals=[10000, 50000, 100000, 250000, 500000, 1000000]),
    margin=dict(l=10, r=10, t=10, b=10),
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------- METRIC CARDS --------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("BTC Price", f"${latest['close']:,.0f}")
col2.metric("Fair Value", f"${latest['fair_value']:,.0f}")
col3.metric("Z-Score", f"{latest['zscore']:.2f}σ")

# MODEL METRICS
st.markdown("<h3 style='text-align:center; color:white;'>Model Performance Metrics</h3>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R² Score", f"{r2:.4f}")
m2.metric("RMSE", f"${rmse:,.0f}")
m3.metric("MAE", f"${mae:,.0f}")
m4.metric("Residual σ", f"{residuals.std():.3f}")
m5.metric("FV Error %", f"{pv_error_pct:.2f}%")

st.markdown(
    f"<p style='text-align:center; color:grey;'>Last Updated: <b>{latest['time'].strftime('%Y-%m-%d %H:%M:%S')}</b></p>",
    unsafe_allow_html=True
)

# ----------------------- Z-SCORE OSCILLATOR ---------------------------
st.markdown("<h3 style='text-align:center; color:white;'>Z-Score Oscillator (Deviation from Fair Value)</h3>", unsafe_allow_html=True)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df_range["time"], y=df_range["zscore"],
    mode="lines", line=dict(color="white"), name="Z-Score"
))

for level, col in zip([1, 2, -1, -2], ["red", "red", "red", "red"]):
    fig2.add_hline(y=level, line=dict(color=col, dash="dot"))

fig2.update_layout(template="plotly_dark", height=350, yaxis_title="Z-Score")
st.plotly_chart(fig2, use_container_width=True)
