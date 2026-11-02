import warnings
import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from .features import LOOKBACK, HORIZON, make_windows, xgb_row, xgb_training_frame

def _nonnegative(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def arima_forecast(train, steps=14):
    y = np.asarray(train, dtype=float)
    # Small, predeclared search space; avoid expensive unrestricted auto-ARIMA.
    candidates = [(1,0,1), (2,0,1), (1,1,1), (2,1,1), (3,1,0), (1,1,2), (2,1,2)]
    best_fit, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidates:
            try:
                fit = ARIMA(y, order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_fit, best_aic = fit, fit.aic
            except Exception:
                continue
    if best_fit is None:
        return np.repeat(float(y[-1]), steps), None
    return _nonnegative(best_fit.forecast(steps=steps)), best_fit

def train_xgb(train_series, seed=42):
    X, y = xgb_training_frame(train_series)
    model = XGBRegressor(
        n_estimators=450, max_depth=3, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.9,
        reg_lambda=2.0, objective="reg:squarederror",
        random_state=seed, n_jobs=4
    )
    model.fit(X, y, verbose=False)
    return model

def xgb_recursive_forecast(model, history, start_date, steps=14):
    hist = list(np.asarray(history, dtype=float))
    out = []
    for step in range(steps):
        date = pd.Timestamp(start_date) + pd.Timedelta(days=step)
        pred = float(model.predict(xgb_row(hist, date))[0])
        pred = max(0.0, pred)
        out.append(pred)
        hist.append(pred)
    return np.asarray(out)

def build_sequence_model(kind, lookback=LOOKBACK, horizon=HORIZON):
    model = Sequential([Input(shape=(lookback, 1))])
    if kind == "GRU":
        model.add(GRU(48, return_sequences=False))
    elif kind == "LSTM":
        model.add(LSTM(48, return_sequences=False))
    elif kind == "BiLSTM":
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
    else:
        raise ValueError(kind)
    model.add(Dropout(0.20))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(horizon))
    model.compile(optimizer="adam", loss="mae")
    return model

def train_neural(train_series, kind, epochs=60, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    values = np.asarray(train_series, dtype=float).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).ravel()
    X, y = make_windows(scaled, LOOKBACK, HORIZON)
    if len(X) < 25:
        raise ValueError(f"Not enough windows to train {kind}: {len(X)} (need at least 25).")
    X = X.reshape((-1, LOOKBACK, 1))
    # Chronological validation tail; no shuffle.
    split = max(1, int(len(X) * 0.85))
    model = build_sequence_model(kind)
    early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
              epochs=epochs, batch_size=32, shuffle=False,
              callbacks=[early], verbose=0)
    return model, scaler

def neural_forecast(model, scaler, history, steps=14):
    hist = np.asarray(history, dtype=float)
    scaled = scaler.transform(hist.reshape(-1, 1)).ravel()
    window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    pred_scaled = model.predict(window, verbose=0)[0][:steps]
    return _nonnegative(scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel())
