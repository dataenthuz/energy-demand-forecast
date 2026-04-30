# energy-demand-forecast

Predicting hourly electricity demand using time series models. Started with naive baselines and worked up to LightGBM with lag features and calendar variables.

## Dataset

PJM Hourly Energy Consumption data from Kaggle - years of hourly load data from a US grid operator. Real data with seasonality, weather effects, and holidays baked in.

## Models

- **Naive baseline** - previous week's same hour (surprisingly hard to beat)
- **Linear regression** - with hour, day-of-week, month as features
- **XGBoost / LightGBM** - with lag features (24h, 48h, 168h), rolling stats, and calendar variables
- **Prophet** - Facebook's time series library, good for catching seasonality out of the box

LightGBM is winning so far on MAE. Prophet is close but takes way longer to tune.

## Feature engineering

The biggest gains came from adding:
- Lag features at 24, 48, and 168 hours (same hour yesterday, day before, last week)
- Rolling mean and std over 24h and 168h windows
- Hour of day, day of week, month, is_weekend, is_holiday

## Files

```
notebooks/
  01_eda.ipynb           # demand patterns, seasonality decomposition
  02_baseline.ipynb      # naive + linear models
  03_tree_models.ipynb   # XGBoost and LightGBM
  04_prophet.ipynb       # Prophet model
src/
  features.py            # lag features, rolling stats, calendar vars
  evaluate.py            # MAE, RMSE, MAPE
requirements.txt
```

## Stack

Python · pandas · LightGBM · XGBoost · Prophet · scikit-learn · matplotlib
