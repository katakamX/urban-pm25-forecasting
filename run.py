import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data import load_daily_data, make_complete_series
from src.features import fill_from_train
from src.models import (arima_forecast, train_xgb, xgb_recursive_forecast,
                        train_neural, neural_forecast)
from src.evaluation import aggregate_metrics
from src.reporting import make_figures, write_diagnostics

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Daily PM2.5 forecasting: Bengaluru vs Delhi")
    parser.add_argument("--data", default="data/raw/city_day.csv")
    parser.add_argument("--cities", nargs="+", default=["Bengaluru", "Delhi"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-origins", type=int, default=45,
                        help="Maximum rolling origins per city; evenly spaced in test period.")
    args = parser.parse_args()

    out = Path("outputs"); models_dir = Path("models")
    out.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True)
    daily = load_daily_data(args.data, tuple(args.cities))
    series_by_city, diag = make_complete_series(daily)
    daily_export = pd.concat(
        [s.rename("PM2.5").rename_axis("Date").reset_index().assign(City=city)
         for city, s in series_by_city.items()], ignore_index=True)
    daily_export.to_csv(out / "clean_daily.csv", index=False)
    write_diagnostics(diag, out / "model_diagnostics.md")

    records, examples = [], []
    for city, raw in series_by_city.items():
        # Split based on time index, not random sampling.
        split = int(len(raw) * 0.80)
        if split < 120 or len(raw) - split < 20:
            raise ValueError(f"{city}: insufficient data for robust train/test split.")
        # The model's available history at an origin is filled using only values
        # available up to that origin and a median calculated on that prefix.
        test_origins = np.arange(split, len(raw)-14+1)
        if len(test_origins) == 0:
            raise ValueError(f"{city}: no complete 14-day test windows.")
        if len(test_origins) > args.max_origins:
            positions = np.linspace(0, len(test_origins)-1, args.max_origins).round().astype(int)
            test_origins = test_origins[positions]

        for origin_idx in test_origins:
            final_neural_preds = {}
            prefix_raw = raw.iloc[:origin_idx]
            filled_prefix, train_median = fill_from_train(prefix_raw, prefix_raw)
            # No target values after origin are used by any model.
            history = filled_prefix.values.astype(float)
            future_dates = raw.index[origin_idx:origin_idx+14]
            actual = raw.iloc[origin_idx:origin_idx+14].values.astype(float)
            # Score only days where original target was observed.
            valid = np.isfinite(actual)

            # Baselines
            persistence = np.repeat(history[-1], 14)
            seasonal = np.array([history[-7 + (i % 7)] for i in range(14)], dtype=float)
            # The seasonal-naive forecast for day h repeats the most recent
            # observation from the same weekday (weekly seasonal lag).
            for name, pred in [("Persistence", persistence), ("SeasonalNaive7", seasonal)]:
                for h in (1,7,14):
                    for j in range(h):
                        if valid[j]:
                            records.append({"city":city,"model":name,"horizon":h,
                                            "origin":raw.index[origin_idx],"lead":j+1,
                                            "actual":actual[j],"prediction":pred[j]})

            # ARIMA
            pred, arima_fit = arima_forecast(history[-min(len(history), 900):], 14)
            for h in (1,7,14):
                for j in range(h):
                    if valid[j]:
                        records.append({"city":city,"model":"ARIMA","horizon":h,
                                        "origin":raw.index[origin_idx],"lead":j+1,
                                        "actual":actual[j],"prediction":pred[j]})

            # XGBoost
            xgb = train_xgb(pd.Series(history, index=raw.index[:origin_idx]))
            xpred = xgb_recursive_forecast(xgb, history, future_dates[0], 14)
            for h in (1,7,14):
                for j in range(h):
                    if valid[j]:
                        records.append({"city":city,"model":"XGBoost","horizon":h,
                                        "origin":raw.index[origin_idx],"lead":j+1,
                                        "actual":actual[j],"prediction":xpred[j]})

            # Neural models. Refit per origin to preserve strict rolling-origin protocol.
            for kind in ("GRU","LSTM","BiLSTM"):
                try:
                    net, scaler = train_neural(pd.Series(history), kind, epochs=args.epochs)
                    npred = neural_forecast(net, scaler, history, 14)
                    if origin_idx == test_origins[-1]:
                        final_neural_preds[kind] = npred.copy()
                    for h in (1,7,14):
                        for j in range(h):
                            if valid[j]:
                                records.append({"city":city,"model":kind,"horizon":h,
                                                "origin":raw.index[origin_idx],"lead":j+1,
                                                "actual":actual[j],"prediction":npred[j]})
                    if origin_idx == test_origins[-1]:
                        net.save(models_dir / f"{city.lower()}_{kind.lower()}.keras")
                        joblib.dump(scaler, models_dir / f"{city.lower()}_{kind.lower()}_scaler.joblib")
                except Exception as exc:
                    print(f"[WARN] {city} {kind} skipped at {raw.index[origin_idx].date()}: {exc}")

            if origin_idx == test_origins[-1]:
                for j, date in enumerate(future_dates):
                    if np.isfinite(actual[j]):
                        examples.append({"city":city,"model":"ACTUAL","date":date,"value":actual[j]})
                    for name, pred_values in [("ARIMA",pred),("XGBoost",xpred)]:
                        examples.append({"city":city,"model":name,"date":date,"value":pred_values[j]})
                    # Neural predictions are captured from the final-origin fit above.
                    for kind, p in final_neural_preds.items():
                        examples.append({"city":city,"model":kind,"date":date,"value":p[j]})

        print(f"[INFO] Completed rolling origins for {city}: {len(test_origins)}")

    metrics = aggregate_metrics(records)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(records).to_csv(out / "rolling_predictions.csv", index=False)
    ex = pd.DataFrame(examples)
    ex.to_csv(out / "forecast_examples.csv", index=False)
    make_figures(series_by_city, metrics, ex, out)

    # Machine-readable run metadata.
    metadata = {
        "data_file": args.data, "cities": list(series_by_city.keys()),
        "split": "chronological 80/20; rolling-origin test",
        "max_origins_per_city": args.max_origins, "epochs": args.epochs,
        "target": "daily PM2.5", "units": "µg/m³",
        "horizons_days": [1,7,14],
        "python_note": "Record Python and package versions for final submission."
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\n=== MODEL COMPARISON (lower MAE/RMSE is better) ===")
    print(metrics.to_string(index=False))
    print(f"\nResults saved under: {out.resolve()}")

if __name__ == "__main__":
    main()
