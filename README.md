# energy-demand-forecast

Predicting hourly electricity demand for the PJM grid using time series feature engineering and gradient boosting models.

## Dataset

PJM Hourly Energy Consumption from Kaggle - PJME region, ~145,000 hourly readings from 2002 to 2018.

## Get the data

```bash
# one-time setup: put kaggle.json in ~/.kaggle/ (kaggle.com > Account > Settings > API > Create New Token)
pip install kaggle
python data/download.py
```

This downloads `PJME_hourly.csv` into the `data/` folder.

## Approach

Standard time series features: hour of day, day of week, month, quarter, is_weekend.

Lag features: same hour 24h ago, 48h ago, 168h ago (same hour last week).

Rolling stats: 24h mean/std, 168h mean.

Models compared: naive baseline, linear regression, XGBoost, LightGBM.

Train on data before 2017, test on 2017 and beyond.

## Run it

```bash
pip install -r requirements.txt
python data/download.py       # first time only
python forecast.py            # train all models, print results, save plots
```

## Notebook

Full walkthrough with plots and analysis: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Files

```
data/
  download.py           # fetches PJME_hourly.csv from Kaggle
  PJME_hourly.csv       # gitignored
notebooks/
  analysis.ipynb        # EDA, feature engineering, model comparison
forecast.py             # main script: feature engineering + all models + plots
requirements.txt
```

## Outputs

Running `forecast.py` saves:
- `forecast_sample.png` - 2-week forecast vs actual (July 2017)
- `feature_importance.png` - top features from LightGBM

## Stack

Python - pandas - scikit-learn - XGBoost - LightGBM - matplotlib
