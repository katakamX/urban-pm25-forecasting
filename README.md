# Urban PM2.5 Forecasting Across Indian Cities
### Bengaluru vs Delhi | ARIMA, GRU, LSTM, BiLSTM and XGBoost

**Submission project | CSE-AIML**

This repository builds a leakage-aware daily PM2.5 forecasting experiment for Bengaluru and Delhi using the Kaggle *Air Quality Data in India (2015–2020)* dataset. It compares five model families and evaluates forecasts at 1, 7 and 14 days ahead.

> **Important reproducibility note:** the Kaggle dataset is not redistributed in this ZIP. Download `city_day.csv` from the source below and place it at `data/raw/city_day.csv`. The pipeline generates all scores and plots from the actual downloaded data; no performance numbers are fabricated in this submission.

## 1. Research question

How accurately can daily city-level PM2.5 be forecast one, seven and fourteen days ahead, and how do performance and failure modes differ between Bengaluru (southern peninsular city) and Delhi (north Indian city with pronounced post-monsoon/winter pollution episodes)?

## 2. Dataset

- **Dataset:** Air Quality Data in India (2015–2020), by Rohan Rao on Kaggle.
- **Primary file:** `city_day.csv` (daily city aggregates).
- **Source:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Coverage described by the source:** 26 cities, 2015–2020; the daily file contains city/date pollutant summaries.
- **Target:** `PM2.5` (µg/m³).
- **Selected cities:** Bengaluru and Delhi.

The dataset is historical and ends in 2020. It is appropriate for a reproducible academic comparison, but it is **not a live/current air-quality feed**. The project deliberately does not mix datasets with different measurement and aggregation protocols.

### Download

Option A — Kaggle CLI:

```bash
pip install kaggle
# Configure your Kaggle API token as described at https://github.com/Kaggle/kaggle-api
kaggle datasets download -d rohanrao/air-quality-data-in-india -p data/raw --unzip
```

Option B — browser: download `city_day.csv` from the Kaggle dataset page and copy it to `data/raw/city_day.csv`.

## 3. Environment

Python 3.10 or 3.11 is recommended. TensorFlow wheels may not be available for every Python/platform combination; use a supported Python version and install the pinned-compatible requirements.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the complete experiment

```bash
python run.py
```

Optional configuration:

```bash
python run.py --data data/raw/city_day.csv --cities Bengaluru Delhi --epochs 60
```

Outputs:

- `outputs/clean_daily.csv` — city-specific daily target series after conservative gap handling.
- `outputs/metrics.csv` — MAE, RMSE, sMAPE and R² for each city/model/horizon.
- `outputs/forecast_examples.csv` — forecast trajectories from the final available forecast origin.
- `outputs/figures/` — city histories, seasonal profiles, model comparison and forecast plots.
- `outputs/model_diagnostics.md` — data coverage and diagnostic notes.
- `models/` — fitted model artifacts and scalers where applicable.

## 5. Experimental design

1. Parse dates and normalize city labels.
2. Retain only Bengaluru and Delhi.
3. Aggregate duplicate city-date rows using the median, then create a complete daily index.
4. Do **not** fill long gaps by interpolation. Short gaps (up to 3 consecutive days) are time-interpolated; remaining gaps are filled with the training-period median during model preparation. Missingness counts are reported.
5. Use a chronological holdout: the final 20% of each city's observed time span is the test period. No random shuffling.
6. Generate multiple rolling forecast origins in the test period. At each origin, all models predict the next 14 days using only observations available at that origin.
7. Report horizons at 1, 7 and 14 days. MAE and RMSE are primary metrics; sMAPE and R² are supplementary.
8. Compare results city-by-city and horizon-by-horizon. A single overall score can conceal meaningful regional and seasonal differences.

### Model implementation

- **ARIMA:** statsmodels ARIMA( p,d,q ) with a small, predeclared order search on the training prefix for each origin. AIC chooses among candidate orders; forecast is capped at zero.
- **GRU / LSTM / BiLSTM:** sequence models trained on rolling windows of past 30 days and predicting the next 14 daily values. Inputs are scaled using training data only. BiLSTM is used as an encoder over the historical input window; it does not receive future observations.
- **XGBoost:** autoregressive one-step regressor using lag, rolling and calendar features. Future predictions are recursive, so later steps use earlier predictions rather than unavailable actual values.

The neural models are intentionally trained separately per city. The two cities have different distributions and seasonal structures; pooling without a city-aware architecture could obscure those differences.

## 6. Avoiding common forecasting mistakes

- Never randomly split a time series.
- Never use future PM2.5 values to construct lag/rolling features.
- Fit scalers only on the training prefix.
- For multi-step forecasts, do not feed actual test targets into later forecast steps.
- Do not impute using information from the future.
- Do not claim that the historical Kaggle dataset represents current CPCB conditions.
- Compare with simple baselines before interpreting complex-model wins.

## 7. Baselines and interpretation

The experiment includes a seasonal-naive baseline (same day one week earlier) and a persistence baseline (last observed value). Complex models should be judged against these. If XGBoost or a neural model does not beat the baseline, that is a valid scientific result.

## 8. How to discuss model failure

Use the generated forecast plots and diagnostics to discuss:

- **Delhi winter episodes:** abrupt changes associated with regional transport, stagnant boundary layers, temperature inversions, crop-residue burning, fireworks and local emissions are not fully encoded in a univariate PM2.5 history. A model may underpredict sudden peaks.
- **Bengaluru variability:** rainfall, wind, traffic, construction and local activity can cause short-lived changes that a univariate model cannot anticipate.
- **Long horizons:** uncertainty compounds; recursive XGBoost can drift, while neural networks tend to regress toward the training mean.
- **ARIMA:** linear dependence and stationary-difference assumptions may fail around regime shifts and extreme pollution events.
- **GRU/LSTM/BiLSTM:** limited data, missingness, nonstationarity and overfitting can make deep models unstable. BiLSTM does not magically see the future; it reads the historical input sequence in both directions.
- **Dataset limits:** city-day averages hide station-level spatial variation; missing readings are not necessarily random; 2015–2020 data do not capture later policy, fleet or climate changes.

Do not state that stubble burning *caused* a particular prediction error unless the project includes independent fire/meteorological covariates and a causal design. Phrase this as a plausible unobserved driver.

## 9. Project structure

```text
urban_pm25_forecasting/
├── data/raw/                  # place city_day.csv here
├── outputs/                   # generated metrics, forecasts and figures
├── models/                    # generated model artifacts
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── reporting.py
├── run.py
├── requirements.txt
├── REPORT.md
└── references.md
```

## 10. Reproducibility checklist

- [ ] Dataset downloaded from the cited Kaggle source.
- [ ] Dataset filename and date coverage checked.
- [ ] `python run.py` completed without errors.
- [ ] `outputs/metrics.csv` inspected.
- [ ] Forecast plots inspected for both cities.
- [ ] Report's results table populated from generated metrics (do not invent scores).
- [ ] Python/package versions recorded if submitting on a different machine.

## 11. Academic integrity and scope

This is a complete runnable implementation and report framework. The final numerical findings are intentionally generated from the dataset at execution time. Replace the clearly marked results placeholders in `REPORT.md` with values from `outputs/metrics.csv` after running the experiment. This prevents fabricated experimental claims.
