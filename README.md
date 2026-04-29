# Agent 20 — German Electricity Price Forecasting

Day-ahead hourly electricity price forecasting for the German bidding zone (DE-LU),
using ENTSO-E market data, DWD weather data, and a progression of models from
classical baselines to gradient boosting to time-series foundation models.

## Status

## Status

**Week 1 / Day 3 complete.** Master modeling dataset assembled.

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `data/processed/dataset.parquet` | 55,392 | 37 | Joined hourly data, Jan 2020 - Apr 2026, ML-ready |

Features include: day-ahead price (target), actual + forecast load, day-ahead
solar/wind forecasts, per-fuel generation actuals (14 fuels + 3 aggregates),
population-weighted national weather (temp/humidity/pressure/sunshine/cloud),
and per-city wind speeds for 7 German cities.

## Data summary

| Series              | Rows   | Columns | Description |
|---------------------|--------|---------|-------------|
| prices              | 55,392 | 1       | DE-LU day-ahead prices, EUR/MWh, hourly avg |
| load_actual         | 55,391 | 1       | Actual demand, MW |
| load_forecast       | 55,391 | 1       | Day-ahead load forecast, MW |
| generation_actual   | 55,391 | 25      | Per-fuel-type generation incl. wind/solar/nuclear/coal/gas |
| wind_solar_forecast | 55,391 | 3       | Day-ahead forecasts: Wind Onshore, Wind Offshore, Solar |

## Goals

1. Beat naive seasonal baselines by at least 15 percent MAE with classical ML
2. Compare against deep learning (TFT/N-BEATS) and Chronos-2 in Week 2
3. Wrap with full MLOps: DVC + MLflow + Streamlit dashboard in Week 5

## Stack

Python 3.11 · LightGBM · pandas · entsoe-py · meteostat · WSL2 Ubuntu

## Author

Allen — M.Sc. Digital Engineering and Management, RWTH Aachen
GitHub: allencharly04
