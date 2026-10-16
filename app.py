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
    eq = pd.read_csv("backtest_equity_test.csv", parse_dates=["Date"])
    return cfg, df, eq

model = load_model()
cfg, df, eq = load_data()
features, target_vol = cfg["features"], cfg["target_vol"]

st.title("NIFTY 50 volatility forecast")
st.caption("Educational project. Not investment advice.")

# ---- Latest forecast ----
latest = df.iloc[[-1]]
pred_vol = float(model.predict(latest[features])[0])
exposure = min(max(target_vol / max(pred_vol, 5), 0), 1)

st.subheader(f"As of {latest['Date'].iloc[0].date()}")
c1, c2, c3 = st.columns(3)
c1.metric("NIFTY 50 close", f"{latest['Close'].iloc[0]:,.0f}")
c2.metric("India VIX", f"{latest['vix'].iloc[0]:.1f}")
c3.metric("Predicted 5-day volatility", f"{pred_vol:.1f}%")
st.metric("Illustrative exposure (volatility target rule)", f"{exposure * 100:.0f}% invested")
st.write(
    f"The rule holds `target volatility ({target_vol:.1f}) / predicted volatility`, "
    "capped at 100%. Higher forecast volatility means a smaller position."
)

# ---- History charts ----
st.divider()
st.subheader("Market history")
window = st.selectbox("Time window", ["Last 1 year", "Last 3 years", "Since 2010"], index=1)
years = {"Last 1 year": 1, "Last 3 years": 3}.get(window)
recent = df if years is None else df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=years)]

st.write("**NIFTY 50 close**")
st.line_chart(recent.set_index("Date")["Close"])

st.write("**India VIX (implied) vs 21-day realised volatility**")
st.line_chart(recent.set_index("Date")[["vix", "vol_21d"]].rename(
    columns={"vix": "India VIX", "vol_21d": "21-day realised volatility"}))

# ---- Backtest ----
st.divider()
st.subheader("Backtest: volatility targeting")
st.write("**Test period (2023 to Oct 2026), growth of 1**")
st.line_chart(eq.set_index("Date").rename(columns={"Vol target (Ridge forecast)": "Vol target (Ridge)"}))

results = pd.DataFrame(
    {
        "Period": ["2020-2022"] * 3 + ["2023-Oct 2026"] * 3,
        "Strategy": ["Buy & hold", "Vol target (21d vol)", "Vol target (Ridge)"] * 2,
        "Ann. return %": [14.34, 11.34, 13.16, 6.02, 3.09, 3.96],
        "Ann. volatility %": [22.58, 14.07, 14.72, 12.62, 11.68, 11.81],
        "Sharpe": [0.64, 0.81, 0.89, 0.48, 0.26, 0.33],
        "Max drawdown %": [-38.44, -22.11, -20.44, -15.77, -18.82, -17.81],
        "Avg exposure %": [100.0, 81.1, 87.1, 100.0, 95.9, 98.0],
    }
)
st.dataframe(results, hide_index=True)
st.caption("Long-only, maximum 100% invested, 0.05% trading cost. Sharpe ignores the risk-free rate.")

# ---- Findings and limits ----
with st.expander("What this project found, and its limits"):
    st.markdown(
        """
- **Direction is not predictable from daily features.** Same-day US, crude and FX moves predict the *opening gap* (67% accuracy vs 61% baseline on 2023+ data), but that information only arrives after NIFTY has closed. The open-to-close move showed no signal (AUC about 0.47).
- **Volatility is predictable.** A Ridge model forecasting next-5-day volatility had R² 0.25 on the test period, against 0.09 for India VIX and -0.23 for 21-day volatility.
- **Volatility targeting helps in crashes, not calm markets.** In 2020-2022 it cut max drawdown from -38% to -20%. In 2023-2026 it stayed about 98% invested and buy-and-hold did better. The worst drawdown there (Sep 2024-Mar 2025) was a slow decline at moderate volatility, which this rule cannot detect.
- **Limits:** daily data only, one market, the model was trained before 2023 and not refit, and the data shown is a snapshot, not a live feed. Not investment advice.
"""
    )