# NIFTY 50: Direction and Volatility Forecasting

An end-to-end data science project on India's NIFTY 50 index: data collection and cleaning, exploratory analysis, direction and volatility models with time-based validation, a volatility-targeting backtest, and a small Streamlit dashboard.

> Educational project. Nothing here is investment advice.

## Summary of findings

1. **Next-day direction is hard to predict.** Same-day moves in the S&P 500, crude oil and USD/INR make a model look strong (63.5% accuracy on validation), but most of that comes from the overnight opening gap, using information that only arrives after NIFTY has closed. Split into its two parts, the *opening gap* is predictable and the *open-to-close move* is not.
2. **Volatility is predictable.** A simple Ridge regression forecasts next-5-day realised volatility better than using India VIX directly or recent realised volatility.
3. **Volatility targeting helps in crashes, not in calm markets.** In 2020-2022 (which includes COVID) it cut maximum drawdown from -38% to -20%. In the calm 2023-2026 test period it stayed about 98% invested and buy-and-hold did better.

## Data

| Series | Ticker (Yahoo Finance) | Use |
|---|---|---|
| NIFTY 50 | `^NSEI` | Target and price features |
| India VIX | `^INDIAVIX` | Volatility feature |
| USD/INR | `INR=X` | Overnight feature |
| S&P 500 | `^GSPC` | Overnight feature |
| WTI crude futures | `CL=F` | Overnight feature |

- Daily data from January 2010 to October 2026, downloaded with `yfinance`. Yahoo Finance is an unofficial source, so values should be spot-checked against niftyindices.com.
- The NIFTY `Volume` column is zero or unreliable for an index, so it was dropped.
- Gaps from different market holidays were forward-filled (longest run of repeated values: 4 days). No backward fill was used.
- WTI crude traded **negative on 20 April 2020**, which breaks percentage change. Those days were set to 0 and 9 other days with moves beyond ±20% (all March-May 2020) were clipped at ±20%.
- After feature engineering the modelling table has 4,071 rows (17 March 2010 to 12 October 2026).

## Method

**Validation is strictly time-based**, never random:

| Split | Period |
|---|---|
| Train | 2010 - 2019 |
| Validation | 2020 - 2022 |
| Test (evaluated once) | 2023 - October 2026 |

Walk-forward evaluation by year was also used for the direction models. All features use only information available at the time of prediction, and target windows were separated from the validation period to avoid overlap.

**Features:** lagged returns, 10- and 50-day moving-average ratios, 5- and 21-day realised volatility, RSI(14), India VIX level and change, day of week, and (for direction models only) same-day changes in USD/INR, S&P 500 and crude.

## Results

### 1. Direction

Share of up days overall: about 53%. A model that always predicts "up" is the baseline, and the larger class share is used as the baseline in each period.

| Model / feature set (validation 2020-2022) | Accuracy | AUC |
|---|---|---|
| Always up | 54.8% | - |
| Logistic regression, with overnight features | 63.5% | 0.674 |
| Logistic regression, without overnight features | 54.8% | 0.529 |

The 63.5% is misleading: S&P 500, crude and FX values for day *t* are only known after NIFTY closes on day *t*, so the model was partly predicting the next morning's gap. The target was therefore split in two.

| Target | Baseline | Validation (accuracy / AUC) | **Test, 2023+ (accuracy / AUC)** |
|---|---|---|---|
| Opening gap up (next open above today's close) | 61.1% (test) | 77.1% / 0.790 | **67.1% / 0.732** |
| Open-to-close move up | 53.0% (majority class, test) | 50.7% / 0.508 | **50.8% / 0.472** |

- US overnight moves, crude and USD/INR predict the opening gap, and the result held on unseen data.
- Nothing predicts the open-to-close move, which is the part a trader could act on. This is consistent with an efficient market: information available before the open is already in the opening price.

### 2. Volatility (next 5 trading days, annualised %)

Features use only information known at NIFTY's close (the overnight S&P 500, crude and FX moves are excluded).

| Model | Validation MAE / R² | **Test MAE / R²** |
|---|---|---|
| Baseline: current 21-day volatility | 7.93 / 0.235 | 4.66 / -0.233 |
| Baseline: current India VIX | 8.51 / 0.423 | 4.84 / 0.086 |
| **Ridge regression** | **6.38 / 0.518** | **3.72 / 0.251** |
| Gradient boosting | 6.85 / 0.318 | 3.95 / 0.195 |
| Random forest | 6.95 / 0.244 | - |

Ridge was chosen on the validation window before the test period was scored. Tree models did worse partly because the validation window contains the 2020 crash, which lay outside their training range. Test R² is lower than validation because 2023-2026 was a calm period with little volatility to explain.

### 3. Backtest: volatility targeting

Rule: hold `target volatility / predicted volatility` of the index, capped at 100% (long-only, no leverage). Target volatility is 14.3, the median 21-day volatility before 2023 (a pre-2020 value was used for the validation run). Positions are set at the close of day *t* and earn the return of day *t+1*. Cost: 0.05% per unit of position change. The model is not refit during the test period.

| Period | Strategy | Ann. return | Ann. vol | Sharpe* | Max drawdown | Avg exposure |
|---|---|---|---|---|---|---|
| 2020-2022 | Buy & hold | 14.34% | 22.58% | 0.64 | -38.44% | 100% |
| 2020-2022 | Vol target (21d vol) | 11.34% | 14.07% | 0.81 | -22.11% | 81.1% |
| 2020-2022 | **Vol target (Ridge)** | 13.16% | 14.72% | **0.89** | **-20.44%** | 87.1% |
| 2023-Oct 2026 | **Buy & hold** | **6.02%** | 12.62% | **0.48** | **-15.77%** | 100% |
| 2023-Oct 2026 | Vol target (21d vol) | 3.09% | 11.68% | 0.26 | -18.82% | 95.9% |
| 2023-Oct 2026 | Vol target (Ridge) | 3.96% | 11.81% | 0.33 | -17.81% | 98.0% |

\*Sharpe here ignores the risk-free rate (about 6-7% in India), so read it as return per unit of risk.

- The Ridge forecast beat the naive 21-day rule in both periods, but only beat buy-and-hold in the crash window.
- The worst test-period drawdown (September 2024 to March 2025) was a slow decline at moderate volatility. The strategy was still 98% invested, because predicted volatility never exceeded its target.

## Limitations

- Daily data, one index, one market regime per period. Three or four market episodes drive most of the result.
- The validation window was used to choose Ridge, so validation numbers are optimistic. Only the test period is an unbiased check, and it was evaluated once.
- The backtest uses a flat 0.05% cost and ignores taxes, slippage, and the risk-free rate. A sensitivity grid over target volatility and cost can be added.
- Yahoo Finance data is unofficial and may contain gaps.
- The dashboard shows a snapshot as of the last data date, not a live feed.

## Project structure

Adjust the names below to match your folder.

```
nifty50-direction-forecasting/
├── 1_nifty50_eda.ipynb        # data checks, returns, volatility, election days, correlations
├── 4_models.ipynb             # direction models, baselines, walk-forward, gap/intraday split
├── (volatility and backtest notebook)
├── app.py                     # Streamlit dashboard
├── nifty50_raw.csv            # raw NIFTY 50 download
├── nifty50_features.csv       # modelling table
├── ridge_vol.joblib           # trained volatility model
├── app_config.json            # feature list and target volatility
├── app_data.csv               # data read by the app
├── backtest_equity_test.csv   # test-period equity curves
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/kaifahmedsiddiqui-47/nifty50-direction-forecasting.git
cd nifty50-direction-forecasting
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

jupyter lab                     # open the notebooks in order
python -m streamlit run app.py  # launch the dashboard at localhost:8501
```

The saved model was trained with the scikit-learn version in your environment, so use the same environment when running the app.

## Possible extensions

- Add a trend filter to the backtest, designed on validation only, then tested once.
- Compare with GARCH and HAR volatility models.
- Add a sensitivity table (target volatility and costs) to the backtest.
- Refresh the data automatically in the app.

## Author

Md Kaif Ahmed Siddiqui, B.E. in Artificial Intelligence and Data Science. GitHub: [kaifahmedsiddiqui-47](https://github.com/kaifahmedsiddiqui-47)