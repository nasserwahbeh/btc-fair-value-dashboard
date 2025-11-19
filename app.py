import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

# ======================================
# STREAMLIT CONFIG
# ======================================
st.set_page_config(
    page_title="Bitcoin Macro Fair Value Model",
    layout="wide"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #111111;
            padding: 25px;
            border-radius: 16px;
            border: 1px solid #333333;
            text-align: center;
            font-size: 22px;
            font-weight: 600;
        }
        .metric-value {font-size: 38px; font-weight: 700; color: white;}
        .stApp {background-color: #0d0d0d;}
    </style>
""", unsafe_allow_html=True)

# ======================================
# LOAD GOOGLE SHEET CSV
# ======================================
sheet_url = "https://docs.google.com/spreadsheets/d/1QkEUXSVxPqBgEhNxtoKY7sra5yc1515WqydeFSg7ibQ/export?format=csv"
df = pd.read_csv(sheet_url)

df.columns = df.columns.str.strip()
df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
df = df.dropna(subset=["time", "close", "Lag 0"])
df = df.sort_values("time").set_index("time")
df = df.rename(columns={"Lag 0": "M2"})

# ======================================
# FEATURE ENGINEERING
# ======================================
df["log_BTC"] = np.log(df["close"])
df["log_M2"] = np.log(df["M2"])
df["fair_value"] = np.nan
df["residual_std"] = np.nan

min_training_samples = 365
update_frequency = 7
poly = PolynomialFeatures(degree=2)

# ======================================
# REGRESSION + FAIR VALUE MODEL
# ======================================
for i in range(min_training_samples, len(df), update_frequency):
    train = df.iloc[:i][["log_BTC", "log_M2"]].dropna()
    if len(train) < min_training_samples:
        continue

    X_poly = poly.fit_transform(train[["log_M2"]].values)
    y_train = train["log_BTC"].values

    model = LinearRegression().fit(X_poly, y_train)
    preds = model.predict(X_poly)
    residual_std = (y_train - preds).std()

    if residual_std < 1e-4:
        residual_std = 1e-4

    end = min(i + update_frequency, len(df))
    for j in range(i, end):
        val = df.iloc[j]["log_M2"]
        fv = np.exp(model.predict(poly.transform([[val]]))[0])
        fv = np.clip(fv, 1, 1_000_000)
        df.iloc[j, df.columns.get_loc("fair_value")] = fv
        df.iloc[j, df.columns.get_loc("residual_std")] = residual_std

# ======================================
# Z-SCORE & BANDS
# ======================================
df["fair_value_log"] = np.log(df["fair_value"])
df["upper_1std"] = np.exp(df["fair_value_log"] + df["residual_std"])
df["lower_1std"] = np.exp(df["fair_value_log"] - df["residual_std"])
df["upper_2std"] = np.exp(df["fair_value_log"] + 2 * df["residual_std"])
df["lower_2std"] = np.exp(df["fair_value_log"] - 2 * df["residual_std"])
df["z_score"] = (df["log_BTC"] - df["fair_value_log"]) / df["residual_std"]

df_plot = df[df["fair_value"].notna()].copy()
latest = df_plot.iloc[-1]

# ======================================
# TITLE + SLIDER
# ======================================
st.markdown("<h1 style='text-align:center; color:white;'>Bitcoin Fair Value Model</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#999;'>Macro Valuation via M2 Polynomial Regression</h3>", unsafe_allow_html=True)

colA, colB, colC = st.columns([2,6,2])
with colA:
    st.caption("Display Range (Years)")
    display_years = st.slider("", 1, 7, 7)

df_zoom = df_plot.last(f"{display_years}Y")

# ======================================
# MAIN PRICE + FAIR VALUE CHART
# ======================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["upper_2std"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["lower_2std"], fill="tonexty",
                         fillcolor="rgba(212,175,55,0.07)", line=dict(width=0), name="±2σ"))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["upper_1std"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["lower_1std"], fill="tonexty",
                         fillcolor="rgba(212,175,55,0.17)", line=dict(width=0), name="±1σ"))

fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["close"], name="Bitcoin Price",
                         line=dict(color="white", width=2.5)))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["fair_value"], name="Fair Value",
                         line=dict(color="#D4AF37", width=3.5)))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
    font=dict(color="white", size=14),
    margin=dict(l=30,r=30,t=10,b=10),
    height=520)

fig.update_yaxes(type="log",
                 gridcolor="rgba(255,255,255,0.08)")

st.plotly_chart(fig, use_container_width=True)

# ======================================
# METRIC CARDS
# ======================================
m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card'>BTC Price<br><div class='metric-value'>${latest['close']:,.0f}</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'>Fair Value<br><div class='metric-value'>${latest['fair_value']:,.0f}</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'>Z-Score<br><div class='metric-value'>{latest['z_score']:.2f}σ</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================
# Z-SCORE OSCILLATOR
# ======================================
st.markdown("<h2 style='text-align:center; color:white;'>Z-Score Oscillator (Deviation from Fair Value)</h2>",
            unsafe_allow_html=True)

osc = go.Figure()
osc.add_hline(y=0, line=dict(color="white", width=1))
osc.add_hline(y=1, line=dict(color="#D4AF37", width=1, dash="dot"))
osc.add_hline(y=-1, line=dict(color="#D4AF37", width=1, dash="dot"))
osc.add_hline(y=2, line=dict(color="red", width=1, dash="dot"))
osc.add_hline(y=-2, line=dict(color="red", width=1, dash="dot"))

z = df_zoom["z_score"].values
times = df_zoom.index

for i in range(len(z)-1):
    osc.add_trace(go.Scatter(
        x=[times[i], times[i+1]],
        y=[z[i], z[i+1]],
        mode="lines",
        line=dict(
            width=3,
            color="#00ff88" if z[i] <= -2 else
                  "#66ffbb" if z[i] <= -1 else
                  "#ffffff" if z[i] <= 1 else
                  "#ffb347" if z[i] <= 2 else
                  "#ff0055"
        ),
        showlegend=False
    ))

osc.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
    font=dict(color="white"), height=300,
    margin=dict(l=20, r=20, t=10, b=20)
)

st.plotly_chart(osc, use_container_width=True)
