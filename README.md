# Agent 20 — German Electricity Price Forecasting

Day-ahead hourly electricity price forecasting for the German bidding zone (DE-LU),
using ENTSO-E market data, DWD weather data, and a progression of models from
classical baselines to gradient boosting to time-series foundation models.

## Status

Week 1 in progress — data ingestion + LightGBM baseline.

## Goals

1. Beat naive seasonal baselines by at least 15 percent MAE with classical ML
2. Compare against deep learning (TFT/N-BEATS) and Chronos-2 in Week 2
3. Wrap with full MLOps: DVC + MLflow + Streamlit dashboard in Week 5

## Stack

Python 3.11 · LightGBM · pandas · entsoe-py · meteostat · WSL2 Ubuntu

## Author

Allen — M.Sc. Digital Engineering and Management, RWTH Aachen
GitHub: allencharly04
