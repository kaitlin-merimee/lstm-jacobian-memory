"""Collect the nb30 summaries (one row per basin and model version).   python src/backup/nb30_collect.py   [--date 04-01]"""
import argparse, pandas as pd
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--date", default="03-01"); a = ap.parse_args(); SUF = "" if a.date == "03-01" else f"_d{a.date}"
D = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in sorted((ROOT / "analysis/read_directions").glob(f"nb30_summary_*_seed?{SUF}.csv"))], ignore_index=True)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); print(D.round(2).to_string(index=False))
print("\nmedian over runs:"); print(D.drop(columns=["basin", "seed", "unit"]).median(numeric_only=True).round(2).to_string())
D.to_csv(ROOT / f"analysis/read_directions/nb30_all_runs{SUF}.csv", index=False)
