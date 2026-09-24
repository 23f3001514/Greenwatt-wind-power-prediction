# Predictive Modeling for Wind Turbine Power Output at GreenWatt Energy Solutions

A Random Forest regression model that predicts wind turbine power output from sensor and environmental data, so grid scheduling and maintenance can be optimized in advance.

## Problem

GreenWatt Energy Solutions operates a fleet of wind turbines but lacks an accurate way to forecast power output. This unpredictability leads to inefficient grid scheduling, reduced profitability, and higher maintenance costs. This project builds a data-driven pipeline to predict turbine power output in advance.

## Dataset

`data/train.csv` contains **329,539 turbine records** across 16 columns: timestamp, active/reactive power readings, ambient temperature, generator speed, generator winding temperature, nacelle and internal temperatures, wind speed, wind direction, wind turbulence, turbine_id, and the target **Target** (power output).

Six columns contain a single NaN each — total of 1 row lost during cleaning. No duplicates. Target is approximately normally distributed (mean = 46.33, std = 2.62).

## Approach

1. **Data quality checks** — missing values, duplicates, constant columns, zero-value analysis
2. **Exploratory analysis** — distributions, boxplots, power curve visualization, correlation heatmap, temporal patterns
3. **Timestamp feature engineering** — Hour, Day, Month, DayOfWeek, Quarter, IsWeekend + cyclical sin/cos encodings
4. **Outlier detection & treatment** — IQR-based detection, winsorization (capping) to preserve genuine gust events
5. **Feature & target separation** — handled inf values, dropped single-NaN row
6. **Stratified 80/20 train-test split** with `random_state=42`
7. **Imputation** (`SimpleImputer`, median) and **scaling** (`StandardScaler`) — fitted on training data only
8. **Model training** — Linear, Ridge, Lasso, Random Forest, Hist Gradient Boosting
9. **Evaluation** — MSE, RMSE, MAE, R², MAPE + residual analysis
10. **Data leakage check** — retrained without leaky power columns for honest evaluation
11. **Time-series lag features** — Target_lag1/2/3 + rolling mean/std
12. **Chronological split & recursive 5-step forecasting**

## Results (65,908-record test set)

| Model | RMSE | MAE | R² | MAPE (%) |
|---|---|---|---|---|
| **Random Forest** | **0.985** | **0.609** | **0.858** | **1.309** |
| Hist Gradient Boosting | 1.332 | 0.921 | 0.740 | 1.981 |
| Linear Regression | 1.944 | 1.371 | 0.447 | 2.935 |
| Ridge Regression | 1.944 | 1.371 | 0.447 | 2.935 |
| Lasso Regression | 1.946 | 1.368 | 0.446 | 2.930 |

**Winner:** Random Forest — nearly **2× better R²** than linear baselines, confirming the power curve is fundamentally non-linear.

## Key findings

- **Wind speed and generator speed** are the strongest legitimate drivers of power output.
- **Nacelle and internal temperatures** add secondary signal — thermal efficiency correlates with output.
- **Time-of-day and seasonal features** add marginal signal — instantaneous wind conditions dominate.
- **Data leakage matters** — `active_power_raw` and `active_power_calculated_by_converter` are essentially alternate measurements of the target.
- **Outliers in wind data are real** — gusts and peak power events, not errors. Winsorization is preferred over deletion.
- **Recursive forecasting** supports 5-step-ahead power prediction for grid scheduling and market bidding.

## Limitations

- Random split may overestimate performance — chronological split is the honest evaluation for time-series.
- Leaky power columns inflate R² — the no-leakage model is the deployable one.
- `turbine_id` was excluded — per-turbine models are a future improvement.
- Long-horizon forecasts accumulate error.

## Repository structure
├── data/train.csv
├── notebooks/major_project.ipynb
├── reports/GreenWatt_Wind_Power_Project_Report.pdf
├── requirements.txt
└── README.md


## How to run

```bash
git clone https://github.com/23f3001514/greenwatt-wind-power-prediction.git
cd greenwatt-wind-power-prediction
pip install -r requirements.txt
jupyter notebook notebooks/major_project.ipynb

Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter
