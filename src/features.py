import numpy as np
import pandas as pd

LOOKBACK = 30
HORIZON = 14

def fill_from_train(train: pd.Series, values: pd.Series):
    med = float(train.median())
    if not np.isfinite(med):
        med = 0.0
    return values.fillna(med), med

def make_windows(values, lookback=LOOKBACK, horizon=HORIZON):
    X, y = [], []
    arr = np.asarray(values, dtype=float)
    for end in range(lookback, len(arr) - horizon + 1):
        X.append(arr[end-lookback:end])
        y.append(arr[end:end+horizon])
    return np.asarray(X), np.asarray(y)

def xgb_row(history, next_date):
    h = np.asarray(history, dtype=float)
    row = {}
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        row[f"lag_{lag}"] = float(h[-lag]) if len(h) >= lag else float(h[-1])
    for w in [3, 7, 14, 30]:
        tail = h[-w:] if len(h) >= w else h
        row[f"mean_{w}"] = float(np.mean(tail))
        row[f"std_{w}"] = float(np.std(tail))
    row["dayofweek"] = next_date.dayofweek
    row["month"] = next_date.month
    row["dayofyear_sin"] = np.sin(2*np.pi*next_date.dayofyear/365.25)
    row["dayofyear_cos"] = np.cos(2*np.pi*next_date.dayofyear/365.25)
    return pd.DataFrame([row])

def xgb_training_frame(series):
    rows, targets = [], []
    s = series.astype(float)
    for i in range(30, len(s)):
        hist = s.iloc[:i].values
        rows.append(xgb_row(hist, s.index[i]))
        targets.append(float(s.iloc[i]))
    if not rows:
        raise ValueError("Not enough training observations for XGBoost (need >30 days).")
    return pd.concat(rows, ignore_index=True), np.asarray(targets)
