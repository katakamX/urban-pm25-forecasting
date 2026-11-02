import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HORIZONS = {1: [0], 7: list(range(7)), 14: list(range(14))}

def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0, 2*np.abs(y_pred-y_true)/denom))*100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return {"MAE": mae, "RMSE": rmse, "sMAPE_pct": smape, "R2": r2}

def aggregate_metrics(records):
    rows = []
    df = pd.DataFrame(records)
    for (city, model, horizon), group in df.groupby(["city", "model", "horizon"]):
        m = score(group["actual"], group["prediction"])
        rows.append({"city": city, "model": model, "horizon_days": horizon,
                     "n_forecasts": len(group), **m})
    return pd.DataFrame(rows).sort_values(["city", "horizon_days", "MAE"])
