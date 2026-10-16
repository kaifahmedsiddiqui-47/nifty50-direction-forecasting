%%writefile app.py
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NIFTY 50 Volatility Forecast", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("ridge_vol.joblib")

@st.cache_data
def load_data():
    cfg = json.load(open("app_config.json"))
    df = pd.read_csv("app_data.csv", parse_dates=["Date"])
    return cfg, df

model = load_model()
cfg, df = load_data()
features, target_vol = cfg["features"], cfg["target_vol"]

st.title("NIFTY 50 volatility forecast")
st.caption("Educational project. Not investment advice.")

# Latest row: forecast of volatility over the next 5 trading days
latest = df.iloc[[-1]]
pred_vol = float(model.predict(latest[features])[0])
exposure = min(max(target_vol / max(pred_vol, 5), 0), 1)

st.subheader(f"As of {latest['Date'].iloc[0].date()}")
c1, c2, c3 = st.columns(3)
c1.metric("NIFTY 50 close", f"{latest['Close'].iloc[0]:,.0f}")
c2.metric("India VIX", f"{latest['vix'].iloc[0]:.1f}")
c3.metric("Predicted 5-day volatility", f"{pred_vol:.1f}%")

st.metric("Suggested exposure (volatility target rule)", f"{exposure * 100:.0f}% invested")
st.write(
    f"The rule holds `target volatility ({target_vol:.1f}) / predicted volatility`, "
    "capped at 100%. Higher forecast volatility means a smaller position."
)