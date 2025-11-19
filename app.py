import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="ML BTC Fair Value Model", layout="wide")

# ===== UI STYLING =====
st.markdown("""
    <style>
        .big-metric { font-size: 32px !important; font-weight: 700 !important; }
        .sub-metric { font-size: 22px !important; font-weight: 600 !important; color:#ddd !important; }
        .section-title { font-size: 20px !important; font-weight: 700 !important; text-align:center; padding-top:20px; }
    </style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
SHEET_URL = "https://docs.google.com/spreadsheets/d/1LqsFETHwFXzDmekdC4LVk9tOukd_jLKj8p7p10NfL_U/export?format=csv"
df = pd.read_csv(SHEET_URL)

df.columns = [c.strip().lower() for c in df.columns]
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('time')
df = df.sort_index()

latest = df.iloc[-1]
last_updated = latest.name.strftime("%Y-%m-%d %H:%M UTC")

# ===== MODEL =====
df_daily = df.copy()
poly = PolynomialFeatures(degree=8)
X_poly = poly.fit_transform(np.arange(len(df_daily)).reshape(-1,1))
y_train = np.log(df_daily["lag 0"].values)

model = LinearRegression().fit(X_poly, y_train)
preds = model.predict(X_poly)
current_r2 = model.score(X_poly, y_train)

df_daily['fair_value'] = np.exp(preds)

# Z-Score Calculation
df_daily["z_score"] = (df_daily["close"] - df_daily["fair_value"]) / df_daily["close"].rolling(365).std()

# ===== FILTER RANGE =====
st.title("💠 ML Bitcoin Fair Value Model")
st.markdown("### Machine Learning fair value analysis via M2 Money Supply Polynomial Regression")

display_years = st.slider("Display Range (Years)", 1, 7, 5)
df_range = df_daily[df_daily.index >= (df_daily.index[-1] - timedelta(days=display_years * 365))]

# ===== R² + LAST UPDATED =====
left, space, right = st.columns([3,5,3])
with left:
    st.markdown(f"**Last Updated:** {last_updated}")
with right:
    st.markdown(f"<p style='text-align:right;'>**R² Score:** {current_r2:.3f}</p>", unsafe_allow_html=True)

# ===== MAIN FAIR VALUE CHART =====
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_range.index, y=df_range["close"],
    mode="lines", name="Bitcoin Price", line=dict(color="#FFFFFF", width=2)
))

fig.add_trace(go.Scatter(
    x=df_range.index, y=df_range["fair_value"],
    mode="lines", name="Fair Value", line=dict(color="#E4C441", width=2)
))

# ±1σ / ±2σ Confidence Bands
z1pos, z1neg = df_daily["fair_value"]*1.1, df_daily["fair_value"]*0.9
z2pos, z2neg = df_daily["fair_value"]*1.2, df_daily["fair_value"]*0.8

fig.add_trace(go.Scatter(x=df_range.index, y=z2pos, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_range.index, y=z1pos, line=dict(width=0), fill="tonexty", fillcolor="rgba(255,215,0,0.15)", showlegend=False))
fig.add_trace(go.Scatter(x=df_range.index, y=z1neg, line=dict(width=0), fill="tonexty", fillcolor="rgba(255,215,0,0.15)", showlegend=False))
fig.add_trace(go.Scatter(x=df_range.index, y=z2neg, line=dict(width=0), fill="tonexty", fillcolor="rgba(255,215,0,0.15)", showlegend=False))

# Clean Y-axis
fig.update_yaxes(
    tickvals=[10000, 50000, 100000, 200000, 500000, 1000000],
    ticktext=["10K","50K","100K","200K","500K","1M"],
    type="log",
    color="white"
)
fig.update_layout(
    height=550,
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ===== METRICS =====
m1, m2, m3 = st.columns(3)
m1.metric("BTC Price", f"${latest['close']:,.0f}")
m2.metric("Fair Value", f"${latest['fair_value']:,.0f}")
m3.metric("Z-Score", f"{latest['z_score']:.2f}σ")

# ===== Z-SCORE OSCILLATOR =====
st.markdown("<div class='section-title'>Z-Score Oscillator (Deviation from Fair Value)</div>", unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_range.index, y=df_range["z_score"], mode="lines",
    line=dict(color="#FFFFFF", width=2)
))

fig2.update_yaxes(title_text="Z-Score (σ)", color="white")
fig2.update_layout(
    height=300,
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white")
)

st.plotly_chart(fig2, use_container_width=True)
