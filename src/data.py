from pathlib import Path
import numpy as np
import pandas as pd

CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "delhi": "Delhi",
}

def load_daily_data(path: str, cities=("Bengaluru", "Delhi")):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload city_day.csv from "
            "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india "
            "and place it at data/raw/city_day.csv"
        )
    df = pd.read_csv(path)
    required = {"City", "Date", "PM2.5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["City", "Date", "PM2.5"]].copy()
    df["City"] = (df["City"].astype(str).str.strip().str.lower()
                  .map(CITY_ALIASES).fillna(df["City"].astype(str).str.strip()))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["City"].isin(cities)]
    # Negative particulate concentrations are physically invalid.
    df.loc[df["PM2.5"] < 0, "PM2.5"] = np.nan
    # Median aggregation is robust to duplicate city/date records.
    daily = (df.groupby(["City", "Date"], as_index=False)["PM2.5"]
               .median()
               .sort_values(["City", "Date"]))
    if daily.empty:
        raise ValueError("No Bengaluru/Delhi records found. Check City labels and input file.")
    return daily

def make_complete_series(daily: pd.DataFrame, max_short_gap=3):
    series_by_city = {}
    diagnostics = []
    for city, group in daily.groupby("City"):
        s = group.set_index("Date")["PM2.5"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(full_idx)
        original_missing = int(s.isna().sum())
        # Interpolate only short interior gaps; do not bridge long outages.
        s = s.interpolate(method="time", limit=max_short_gap, limit_area="inside")
        remaining_missing = int(s.isna().sum())
        # Preserve chronology and avoid dropping dates. The training-only median is
        # applied later in each split; this marker is retained for diagnostics.
        series_by_city[city] = s.astype(float)
        diagnostics.append({
            "city": city, "start": str(s.index.min().date()),
            "end": str(s.index.max().date()), "calendar_days": len(s),
            "missing_after_short_interpolation": remaining_missing,
            "missing_before_interpolation": original_missing,
            "observed_fraction": round(float(s.notna().mean()), 4),
        })
    return series_by_city, pd.DataFrame(diagnostics)
