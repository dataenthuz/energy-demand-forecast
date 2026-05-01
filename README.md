# energy-demand-forecast

Predicting hourly electricity demand using time series models. Started with naive baselines and worked up to LightGBM with lag features and calendar variables.

## Dataset

PJM Hourly Energy Consumption (PJME region) from Kaggle - 145k+ hourly readings from 2002 to 2018.

Get the data:

```bash
# one-time setup: put kaggle.json in ~/.kaggle/ (kaggle.com > Account > API > Create New Token)
pip install kaggle
python data/download.py
```

This downloads `PJME_hourly.csv` into the `data/` folder.

## Models compared

- Naive baseline - same hour last week (lag 168h)
- Linear regression with calendar features
- XGBoost
- LightGBM

LightGBM is winning so far on MAE. Prophet is close but takes way longer to tune.

## Feature engineering

The biggest gains came from adding:
- Lag features at 24, 48, and 168 hours (same hour yesterday, day before, last week)
- Rolling mean and std over 24h and 168h windows
- Hour of day, day of week, month, is_weekend

## Run it

```bash
pip install -r requirements.txt
python data/download.py   # first time only
python forecast.py
```

## Stack

Python - pandas - LightGBM - XGBoost - scikit-learn - matplotlib - kaggle
