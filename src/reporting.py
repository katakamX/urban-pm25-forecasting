from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_figures(series_by_city, metrics, examples, outdir):
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for city, s in series_by_city.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(s.index, s.values, linewidth=0.8)
        ax.set(title=f"{city}: daily PM2.5 history", xlabel="Date", ylabel="PM2.5 (µg/m³)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(figdir / f"{city.lower()}_history.png", dpi=160)
        plt.close(fig)

        monthly = s.groupby(s.index.month).median()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set(title=f"{city}: median PM2.5 by calendar month",
               xlabel="Month", ylabel="Median PM2.5 (µg/m³)", xticks=range(1,13))
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(figdir / f"{city.lower()}_monthly_profile.png", dpi=160)
        plt.close(fig)

    if not metrics.empty:
        for city in metrics.city.unique():
            sub = metrics[metrics.city == city]
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in sub.model.unique():
                m = sub[sub.model == model].sort_values("horizon_days")
                ax.plot(m.horizon_days, m.MAE, marker="o", label=model)
            ax.set(title=f"{city}: MAE by forecast horizon",
                   xlabel="Forecast horizon (days)", ylabel="MAE (µg/m³)",
                   xticks=[1,7,14])
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(figdir / f"{city.lower()}_model_comparison.png", dpi=160)
            plt.close(fig)

    if not examples.empty:
        for city in examples.city.unique():
            sub = examples[examples.city == city]
            fig, ax = plt.subplots(figsize=(11, 5))
            actual = sub[sub.model == "ACTUAL"]
            if not actual.empty:
                ax.plot(actual.date, actual.value, marker="o", label="Actual")
            for model in sub.model.unique():
                if model == "ACTUAL":
                    continue
                m = sub[sub.model == model]
                ax.plot(m.date, m.value, marker=".", label=model)
            ax.set(title=f"{city}: final-origin 14-day forecast",
                   xlabel="Date", ylabel="PM2.5 (µg/m³)")
            ax.legend()
            ax.grid(alpha=.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(figdir / f"{city.lower()}_forecast_example.png", dpi=160)
            plt.close(fig)

def write_diagnostics(diag, out_path):
    lines = ["# Data diagnostics", "",
             "The following coverage and missingness statistics are computed from the supplied CSV.",
             "Short interior gaps of at most three consecutive days are interpolated; longer gaps remain missing",
             "until each chronological training split applies its own training-only median fill.", "",
             diag.to_markdown(index=False), "",
             "## Interpretation cautions", "",
             "- City-day aggregates hide within-city station variability.",
             "- Missingness may be informative (monitor downtime or episodic data availability).",
             "- A univariate PM2.5 series cannot directly observe fire counts, wind, rainfall, mixing height,",
             "  traffic, policy changes or transported pollution.",
             "- The historical dataset ends in 2020; forecasts are an offline benchmark, not a live advisory.",
             ""]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
