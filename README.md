# 🧠 ML Bitcoin Fair Value Model  
### Machine Learning Fair-Value Insight Using Global M2 Money Supply (Polynomial Regression)

A production-grade, real-time Bitcoin fair-value dashboard that models Bitcoin’s intrinsic value using M2 global liquidity trends.  
Built with **Python, Streamlit, Scikit-Learn, Plotly, Google Sheets, and TradingView Webhooks** — deployed fully serverless and self-updating.

🔗 **Live App:** https://btc-fair-value-model.streamlit.app/  
📈 **Data Auto-Updated Daily via TradingView Alerts → Google Sheets → Streamlit**

---

## 📌 **Project Overview**
This project estimates the fair value of Bitcoin based on **global monetary supply expansion**, compared against current market price. Using a **Polynomial Regression model (degree 2)** and **expanding walk-forward retraining**, it avoids look-ahead bias and recalculates fair value weekly while using real-time liquidity data.

The model outputs:
- Fair Value price of Bitcoin
- ±1σ and ±2σ deviation valuation bands
- Macro deviation Z-Score oscillator (overvalued / undervalued signals)
- Interactive visualization on log scale
- Adjustable time window (1–7 years)
- High-end UI with institutional dashboard styling

---

## 🚀 **Key Features**
| Feature | Description |
|--------|-------------|
| **Daily automated data ingestion** | TradingView Webhook → Google Apps Script → Google Sheets |
| **ML polynomial regression model** | Expanding window retraining every 7 days |
| **Real-time dashboard** | Streamlit interface with responsive plotting |
| **Deviation Z-Score oscillator** | Helps identify historically extreme valuation conditions |
| **Serverless Deployment** | Streamlit Cloud + GitHub CI |
| **No look-ahead / repainting** | Uses walk-forward retraining methodology |
| **Log-scaled price chart** | Cleaner macro visualization |

---

## 🧠 **Model Methodology**
### **Expanding Window Training**
```python
for i in range(min_training_samples, len(df_daily), update_frequency):
    train = df_daily.iloc[:i][["log_BTC", "log_M2"]]
```
- Begins training after first year (365 samples)
- Retrains weekly using all past historical data
- Avoids peeking into future values

## Z-Score Interpretation

| Z | Meaning |
|---|---------|
| z < -2 | Extremely undervalued |
| -1 < z < 1 | Fair-value region |
| z > 2 | Extremely overvalued |

# 📊 Architecture
```mathematica
TradingView Pine Script → Webhook → Google Apps Script → Google Sheets CSV → 
Streamlit → Scikit-Learn Model → Interactive Dashboard
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Frontend / UI | Streamlit + Plotly |
| ML Engine | Scikit-Learn Polynomial Regression |
| Hosting | Streamlit Cloud |
| Data Pipeline | Google Sheets + Webhook |
| Programming | Python |
| SCM & CI/CD | GitHub |

## 📁 Repository Structure
```bash
/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
```

## 🔧 Local Installation
```bash
git clone https://github.com/nasserwahbeh/BTC-Fair-Value-Dashboard.git
cd BTC-Fair-Value-Dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Visual Outputs
### Main Fair-Value Chart
- Bitcoin price vs. ML-estimated fair value
- ±1σ and ±2σ uncertainty bands
- Automated real-time updating pipeline

### Z-Score Oscillator
- Highlights extreme deviations from fair value
- Helps identify optimal buy / sell macro timing

## 🔄 Auto-Updating Workflow

| Source | Frequency |
|--------|-----------|
| TradingView alert | Daily close |
| Google Sheet update | Real-time |
| Streamlit App | Refresh on load |

## 📋 Example Use Cases

- Crypto portfolio macro allocation
- Hedging / selling zones detection
- Timing long-term DCA entries & exits
- Liquidity-driven quant research

## 🤝 Contributions

PRs and feature requests are welcome — feel free to open an issue or reach out.

## 📬 Contact

**Nasser Wahbeh**  
Quantitative Investment Systems & Automation Engineering

📧 nasserwahbeh198@gmail.com

🔗 https://linkedin.com/in/nasserwahbeh](https://www.linkedin.com/in/nasser-wahbeh-1315501b6/

---

⭐ **If you find this useful, please star the repo — it helps a lot!**
