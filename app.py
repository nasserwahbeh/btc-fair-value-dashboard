import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ================================
# STREAMLIT PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Bitcoin Macro Fair Value",
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
# LOAD + CLEAN DATA
# ================================
sheet_url = "https://docs.google.com/spreadsheets/d/1QkEUXSVxPqBgEhNxtoKY7sra5yc1515WqydeFSg7ibQ/export?format=csv"
df = pd.read_csv(sheet_url)

# CLEAN FIRST
df = df.dropna(axis=0, how="all")                  # Remove completely empty rows
df.columns = df.columns.str.strip()                # Strip trailing spaces
df = df[df["time"].notna()]                        # Remove rows missing time

# Debug view
st.write("COLUMNS:", list(df.columns))
st.write("HEAD:", df.head())

# Convert UNIX time to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('time').sort_index()

# Ensure numeric types
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['Lag 0'] = pd.to_numeric(df['Lag 0'], errors='coerce')

df = df.dropna()


# ================================
# SETUP DAILY DF
# ================================
df_daily = df[['close', 'Lag 0']].copy()
df_daily['log_BTC'] = np.log(df_daily['close'])
df_daily['log_M2'] = np.log(df_daily['Lag 0'])

# MODEL VARS
min_training_samples = 365
update_frequency = 7
poly = PolynomialFeatures(degree=2)

# Reset computed columns
df_daily['fair_value'] = np.nan
df_daily['residual_std'] = np.nan

# ================================
# TRAIN / COMPUTE FAIR VALUE
# ================================
for i in range(min_training_samples, len(df_daily), update_frequency):
    train = df_daily.iloc[:i][['log_BTC', 'log_M2']].dropna()
    if len(train) < min_training_samples:
        continue

    X_poly = poly.fit_transform(train[['log_M2']])
    y_train = train['log_BTC'].values

    model = LinearRegression().fit(X_poly, y_train)
    preds = model.predict(X_poly)
    residual_std = (y_train - preds).std()

    end = min(i + update_frequency, len(df_daily))
    for j in range(i, end):
        val = df_daily.iloc[j]['log_M2']
        df_daily.iloc[j, df_daily.columns.get_loc('fair_value')] = np.exp(model.predict(poly.transform([[val]]))[0])
        df_daily.iloc[j, df_daily.columns.get_loc('residual_std')] = residual_std


# ================================
# BANDS + ZSCORE
# ================================
df_daily['fair_value_log'] = np.log(df_daily['fair_value'])
df_daily['upper_1std'] = np.exp(df_daily['fair_value_log'] + df_daily['residual_std'])
df_daily['lower_1std'] = np.exp(df_daily['fair_value_log'] - df_daily['residual_std'])
df_daily['upper_2std'] = np.exp(df_daily['fair_value_log'] + 2*df_daily['residual_std'])
df_daily['lower_2std'] = np.exp(df_daily['fair_value_log'] - 2*df_daily['residual_std'])
df_daily['z_score'] = (df_daily['log_BTC'] - df_daily['fair_value_log']) / df_daily['residual_std']

df_plot = df_daily[df_daily['fair_value'].notna()]
latest = df_plot.iloc[-1]


# ================================
# TITLE + SLIDER
# ================================
st.markdown("<h1 style='text-align:center;'>Bitcoin Fair Value Model</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#999;'>Macro Valuation via M2 Polynomial Regression</h3>", unsafe_allow_html=True)

toolbar = st.container()
with toolbar:
    a,b,c = st.columns([2,6,2])
    with a:
        st.caption("Display Range (Years)")
        display_years = st.slider("", 1, 7, 7)


df_zoom = df_plot.last(f"{display_years}Y")


# ================================
# MAIN PRICE CHART
# ================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom['close'], name="Bitcoin Price",
                         line=dict(color="white", width=2.5)))

fig.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom['fair_value'], name="Fair Value",
                         line=dict(color="#D4AF37", width=3)))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(color="white", size=14),
    height=480
)

fig.update_yaxes(type="log")
st.plotly_chart(fig, use_container_width=True)


# ================================
# METRICS
# ================================
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'>BTC Price<br><div class='metric-value'>${latest['close']:,.0f}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>Fair Value<br><div class='metric-value'>${latest['fair_value']:,.0f}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>Z-Score<br><div class='metric-value'>{latest['z_score']:.2f}σ</div></div>", unsafe_allow_html=True)
