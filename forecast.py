import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Dataset: PJM Hourly Energy Consumption (Kaggle)
# Run: python data/download.py
DATA_PATH = 'data/PJME_hourly.csv'

print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
df.columns = ['MW']
df = df.sort_index()

print(f"Loaded {len(df):,} hourly readings")
print(f"Range: {df.index.min()} to {df.index.max()}")
print(f"MW  min={df['MW'].min():,.0f}  max={df['MW'].max():,.0f}  mean={df['MW'].mean():,.0f}")


def add_features(df):
    """Add calendar and lag features for time series modeling."""
    df = df.copy()
    # calendar
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    # lag features - same hour yesterday, 2 days ago, last week
    df['lag_24h'] = df['MW'].shift(24)
    df['lag_48h'] = df['MW'].shift(48)
    df['lag_168h'] = df['MW'].shift(168)
    # rolling stats over previous 24h and 168h
    df['rolling_mean_24h'] = df['MW'].shift(1).rolling(24).mean()
    df['rolling_std_24h'] = df['MW'].shift(1).rolling(24).std()
    df['rolling_mean_168h'] = df['MW'].shift(1).rolling(168).mean()
    return df


print("\nEngineering features...")
df_feat = add_features(df).dropna()

SPLIT = '2017-01-01'
train = df_feat[df_feat.index < SPLIT]
test = df_feat[df_feat.index >= SPLIT]

FEATURE_COLS = [c for c in df_feat.columns if c != 'MW']
X_train, y_train = train[FEATURE_COLS], train['MW']
X_test, y_test = test[FEATURE_COLS], test['MW']

print(f"Train: {len(train):,} rows  ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test:  {len(test):,} rows  ({test.index.min().date()} to {test.index.max().date()})")


def score(name, preds, actuals):
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    print(f"  {name:<30} MAE={mae:7,.0f} MW   RMSE={rmse:7,.0f} MW   MAPE={mape:.2f}%")
    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


results = []
print("\n--- Results ---")

# 1. Naive baseline: same hour last week
naive_pred = X_test['lag_168h']
results.append(score('Naive (lag 168h)', naive_pred, y_test))

# 2. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append(score('Linear Regression', lr_pred, y_test))

# 3. XGBoost
print("  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results.append(score('XGBoost', xgb_pred, y_test))

# 4. LightGBM
print("  Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
results.append(score('LightGBM', lgb_pred, y_test))

# Summary table
print()
results_df = pd.DataFrame(results).set_index('name').round(2)
print(results_df.to_string())

# --- Plot: 2-week forecast sample ---
mask = (test.index >= '2017-07-01') & (test.index < '2017-07-15')
idx = test.index[mask]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(idx, y_test[idx], label='Actual', color='black', linewidth=1.2)
ax.plot(idx, lgb_pred[mask], label='LightGBM', color='steelblue', linewidth=1.0, alpha=0.85)
ax.plot(idx, naive_pred[idx], label='Naive', color='tomato', linestyle='--', linewidth=1.0, alpha=0.7)
ax.set_title('Forecast vs Actual - July 2017 (2 weeks)')
ax.set_ylabel('MW')
ax.legend()
plt.tight_layout()
plt.savefig('forecast_sample.png', dpi=150)
print("\nSaved forecast_sample.png")

# --- Plot: Feature importance ---
imp = pd.Series(lgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\n--- Top Features (LightGBM) ---")
print(imp.head(8).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
imp.head(8).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('LightGBM Feature Importance (top 8)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("Saved feature_importance.png")
