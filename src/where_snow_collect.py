"""Collect the where_snow_patching summaries (one row per basin and model version) into one table.
  python src/where_snow_collect.py                 # 1 March runs
  python src/where_snow_collect.py --date 04-01    # runs made with --date 04-01
"""
import argparse, pandas as pd
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--date", default="03-01"); a = ap.parse_args(); SUF = "" if a.date == "03-01" else f"_d{a.date}"
D = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in sorted((ROOT / "analysis/read_directions").glob(f"nb27_summary_*_seed?{SUF}.csv"))], ignore_index=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
keep = ["basin", "seed", "n_snow_cells", "n80_read", "n95_read", "n80_pca", "probe_r2", "probe_inside_top_read", "probe_cos_melt_grad"] + [c for c in D if c.startswith("rec_")]
T = D[keep].rename(columns=lambda c: c.replace("rec_", "recovered: ")); print(T.round(2).to_string(index=False))
print("\nmedian over runs:"); print(T.drop(columns=["basin", "seed"]).median(numeric_only=True).round(2).to_string())
T.to_csv(ROOT / f"analysis/read_directions/nb27_all_runs{SUF}.csv", index=False)
