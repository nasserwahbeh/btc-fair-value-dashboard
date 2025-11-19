import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ================================
# STREAMLIT PAGE CONFIG
# ================================
st.set_page_config(
    page_title="ML Bitcoin Fair Value Model",
    layout="wide",
)

# ================================
# CUSTOM CSS
# ================================
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
    .metric-value {
        font-size: 38px;
        font-weight: 700;
        color: white;
    }
    .stApp {
        background-color: #0d0d0d;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD DATA
# ================================
sheet_url = "https://docs.google.com/spreadsheets/d/1QkEUXSVxPqBgEhNxtoKY7sra5yc1515WqydeFSg7ibQ/export?format=csv"
df = pd.read_csv(sheet_url)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert UNIX timestamp
df["time"] = pd.to_datetime(df["time"], unit="s")

df = df.sort_values("time")
df = df.set_index("time").dropna()

# Rename for consistency
df_daily = df.rename(columns={"lag 0": "m2"})
df_daily = df_daily[["close", "m2"]].dropna()

# Log transforms
df_daily["log_BTC"] = np.log(df_daily["close"])
df_daily["log_M2"] = np.log(df_daily["m2"])

# ================================
# MODEL SETUP
# ================================
min_training_samples = 365
update_frequency = 7

poly = PolynomialFeatures(degree=2)

df_daily["fair_value"] = np.nan
df_daily["residual_std"] = np.nan

for i in range(min_training_samples, len(df_daily), update_frequency):
    train = df_daily.iloc[:i][["log_BTC", "log_M2"]].dropna()
    if len(train) < min_training_samples:
        continue

    X_poly = poly.fit_transform(train[["log_M2"]].values)
    y_train = train["log_BTC"].values

    model = LinearRegression().fit(X_poly, y_train)
    preds = model.predict(X_poly)
    
    residual_std = (y_train - preds).std()


    r2 = r2_score(y_train, preds)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    mae = mean_absolute_error(y_train, preds)

latest_error_pct = ((latest["close"] - latest["fair_value"]) / latest["fair_value"]) * 100

    end = min(i + update_frequency, len(df_daily))
    for j in range(i, end):
        val = df_daily.iloc[j]["log_M2"]
        df_daily.iloc[j, df_daily.columns.get_loc("fair_value")] = np.exp(model.predict(poly.transform([[val]]))[0])
        df_daily.iloc[j, df_daily.columns.get_loc("residual_std")] = residual_std

# ================================
# BANDS + Z-SCORE
# ================================
df_daily["fair_log"] = np.log(df_daily["fair_value"])
df_daily["upper_1"] = np.exp(df_daily["fair_log"] + df_daily["residual_std"])
df_daily["lower_1"] = np.exp(df_daily["fair_log"] - df_daily["residual_std"])
df_daily["upper_2"] = np.exp(df_daily["fair_log"] + 2 * df_daily["residual_std"])
df_daily["lower_2"] = np.exp(df_daily["fair_log"] - 2 * df_daily["residual_std"])
df_daily["z_score"] = (df_daily["log_BTC"] - df_daily["fair_log"]) / df_daily["residual_std"]

df_plot = df_daily[df_daily["fair_value"].notna()]
latest = df_plot.iloc[-1]

# ================================
# TITLE + SLIDER
# ================================
st.markdown("<h1 style='text-align:center; color:white;'>ML Bitcoin Fair Value Model</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#999;'>Machine Learning fair value analysis via M2 Money Supply Polynomial Regression</h3>", unsafe_allow_html=True)

colA, colB, colC = st.columns([2,6,2])
with colA:
    st.caption("Display Range (Years)")
    display_years = st.slider("", 1, 7, 5)  # default=5 years

df_zoom = df_plot.last(f"{display_years}Y")

# ================================
# MAIN CHART
# ================================
fig = go.Figure()

# Shaded bands
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["upper_2"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["lower_2"], fill="tonexty",
                         fillcolor="rgba(212,175,55,0.07)", line=dict(width=0), name="±2σ"))

fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["upper_1"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["lower_1"], fill="tonexty",
                         fillcolor="rgba(212,175,55,0.17)", line=dict(width=0), name="±1σ"))

# Price + Fair Value lines
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["close"], name="Bitcoin Price", line=dict(color="white", width=2.5)))
fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["fair_value"], name="Fair Value", line=dict(color="#D4AF37", width=3.2)))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="white", size=14),
    margin=dict(l=30, r=30, t=20, b=10),
    height=550
)

fig.update_yaxes(
    type="log",
    tickvals=[10000,20000,30000,50000,70000,100000,150000,200000],
    ticktext=["10k","20k","30k","50k","70k","100k","150k","200k"],
    gridcolor="rgba(255,255,255,0.08)"
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("<h3 style='text-align:center; color:white;'>Model Performance Metrics</h3>", unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("R² Score", f"{r2:.4f}")
m2.metric("RMSE", f"${rmse:,.0f}")
m3.metric("MAE", f"${mae:,.0f}")
m4.metric("Residual σ", f"{latest['residual_std']:.3f}")
m5.metric("FV Error %", f"{latest_error_pct:.2f}%")
m6.metric("Update Cycle", "7 Days")

# ================================
# METRIC CARDS
# ================================
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'>BTC Price<br><div class='metric-value'>${latest['close']:,.0f}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>Fair Value<br><div class='metric-value'>${latest['fair_value']:,.0f}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>Z-Score<br><div class='metric-value'>{latest['z_score']:.2f}σ</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================
# Z-SCORE OSCILLATOR
# ================================
st.markdown("<h2 style='text-align:center; color:white; font-weight:700;'>Z-Score Oscillator (Deviation from Fair Value)</h2>", unsafe_allow_html=True)

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
            width=2.5,
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
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="white"),
    margin=dict(l=20, r=20, t=20, b=30),
    height=300
)

osc.update_yaxes(title_text="Z-Score (σ)")

st.plotly_chart(osc, use_container_width=True)
