"""memory_per_cell_collect: gather the memory_per_cell runs (5 snow basins x 3 model versions) into one table and print the medians by start day.

  PYTHONPATH=src python src/memory_per_cell_collect.py            # login node, seconds
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
D = (ROOT if not a.toy else Path("toyroot")) / "analysis/cell_memory"; pd.set_option("display.width", 220)
A = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in sorted(D.glob("nb32_memory_*_seed?.csv"))], ignore_index=True)
A["kind2"] = np.where(A.kind == "read", A.item, A.kind); A.to_csv(D / "nb32_all_runs.csv", index=False)
ORDER = ["10-01", "11-01", "12-01", "01-01", "02-01", "03-01", "04-01", "05-01", "06-01", "07-01", "08-01", "09-01"]
run = A.groupby(["basin", "seed", "start", "kind2"]).agg(memory_d=("memory_d", "median"), flow_abs_mm=("flow_abs_mm", "median"), gate_tau_d=("gate_tau_d", "median")).reset_index()
S = run.groupby(["start", "kind2"]).agg(memory_d=("memory_d", "median"), mem_lo=("memory_d", lambda v: v.quantile(.25)), mem_hi=("memory_d", lambda v: v.quantile(.75)),
                                        flow_abs_mm=("flow_abs_mm", "median"), gate_tau_d=("gate_tau_d", "median"), runs=("memory_d", "size")).reset_index()
S.to_csv(D / "nb32_by_start.csv", index=False)
print(f"{A.groupby(['basin', 'seed']).ngroups} runs ({A.basin.nunique()} basins x {A.seed.nunique()} model versions); horizon {A.horizon.max()} d (memory equal to it = at least that long)")
for k in ("memory_d", "flow_abs_mm"):
    print(f"\n{k}, median over runs of each run's median (rows: kind; columns: start day)")
    print(S.pivot(index="kind2", columns="start", values=k)[[o for o in ORDER if o in S.start.values]].round(2).to_string())
c = A[A.cell >= 0]
rho = c.groupby(["basin", "seed", "start", "wy"])[["memory_d", "flow_abs_mm"]].apply(lambda g: spearmanr(g.memory_d, g.flow_abs_mm)[0]).groupby("start").median()
print("\nacross cells, memory against flow effect (rank correlation, median over runs and years):"); print(rho.reindex([o for o in ORDER if o in rho.index]).round(2).to_string())
print("\ncell memory (coupled, this test) against each cell's forget gate alone, median days by start: "
      + ", ".join(f"{s} {g.memory_d.median():.0f} vs {g.gate_tau_d.median():.0f}" for s, g in c.groupby("start")))

# ---- when does the stored memory end? start day + memory of the direction flow reads most, against observed melt timing (whole_model_snow_lag) ----
A["start_date"] = pd.to_datetime([f"{w - 1 if int(s[:2]) >= 10 else w}-{s}" for w, s in zip(A.wy, A.start)]); r1 = A[A.item == "read1"].copy()
r1["end_date"] = r1.start_date + pd.to_timedelta(r1.memory_d, "D"); r1["end_dowy"] = (r1.end_date - pd.to_datetime((r1.wy - 1).astype(str) + "-10-01")).dt.days
r1[["basin", "seed", "wy", "start", "memory_d", "memory_censored", "end_date", "end_dowy"]].to_csv(D / "nb32_read1_end_dates.csv", index=False)
WIN = ["10-01", "11-01", "12-01", "01-01", "02-01", "03-01", "04-01", "05-01"]
e_ = r1[r1.start.isin(WIN)].groupby("start").agg(end_dowy=("end_dowy", "median"), censored=("memory_censored", "mean")).reindex(WIN)
fmt = lambda dw: (pd.Timestamp("2001-10-01") + pd.Timedelta(days=float(dw))).strftime("%d %b") if np.isfinite(dw) else "-"
print("\nthe direction flow reads most: date its nudge has faded to 37%, by start day (median over runs and years; share censored at the horizon)")
print("  " + ", ".join(f"{s}: {fmt(r.end_dowy)} ({r.censored:.0%})" for s, r in e_.iterrows()))
f26 = (ROOT if not a.toy else Path("toyroot")) / "analysis/all_basins_memory/nb26_observed_lag_by_year.csv"
if f26.exists():
    O = pd.read_csv(f26, dtype={"basin": str}); O["basin"] = O.basin.str.zfill(8); O = O.dropna(subset=["snowfall_centre_doy"])
    sc = pd.to_datetime([f"{w - 1 if d >= 274 else w}" for w, d in zip(O.wy, O.snowfall_centre_doy)], format="%Y") + pd.to_timedelta(O.snowfall_centre_doy - 1, "D")
    w0 = pd.to_datetime((O.wy - 1).astype(str) + "-10-01")
    for k in ("obs", "sim"):
        if f"{k}_lag_d" in O: O[f"{k}_melt_centre_dowy"] = (sc + pd.to_timedelta(O[f"{k}_lag_d"], "D") - w0).dt.days
    O[["basin", "wy", "obs_melt_centre_dowy", "sim_melt_centre_dowy"]].to_csv(D / "nb32_melt_centre_dates.csv", index=False)
    M = r1[r1.start.isin(WIN) & ~r1.memory_censored].merge(O, on=["basin", "wy"])
    if len(M):
        print(f"observed melt-flow centre in these basin-years: median {fmt(M.drop_duplicates(['basin', 'wy']).obs_melt_centre_dowy.median())}")
        for s, g in M.groupby("start"):
            dd = g.end_dowy - g.obs_melt_centre_dowy
            print(f"  start {s}: memory ends {np.median(dd):+.0f} d from the observed melt-flow centre (middle half {dd.quantile(.25):+.0f} to {dd.quantile(.75):+.0f}); "
                  f"rank correlation across basin-years {spearmanr(g.end_dowy, g.obs_melt_centre_dowy)[0]:+.2f}, n = {len(g)}")
        g = M[M.start == "12-01"]
        if len(g): print(f"ANSWER end date (start 1 Dec): the store's memory ends {np.median(g.end_dowy - g.obs_melt_centre_dowy):+.0f} d from the observed melt-flow centre (median), rank correlation {spearmanr(g.end_dowy, g.obs_melt_centre_dowy)[0]:+.2f} across {len(g)} basin-years")
else: print(f"(no {f26.name}: skip the comparison with observed melt timing)")

# ---- robustness: the same runs with a shorter horizon (longer spin-up before the nudge), files tagged _h<days> -----------------------
for hz in sorted({int(f.stem.split("_h")[-1]) for f in D.glob("nb32_memory_*_seed?_h*.csv")}):
    Bh = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in sorted(D.glob(f"nb32_memory_*_seed?_h{hz}.csv"))], ignore_index=True)
    key = ["basin", "seed", "wy", "start", "item"]; M = A[key + ["memory_d", "kind"]].merge(Bh[key + ["memory_d", "memory_censored"]], on=key, suffixes=("_240", f"_{hz}"))
    M = M[(M.memory_d_240 < hz) & ~M.memory_censored]                                    # only memories the shorter horizon can measure
    print(f"\nrobustness, horizon {hz} d ({365 - hz} d of spin-up before the nudge) against 240 d ({365 - 240} d), {Bh.groupby(['basin', 'seed']).ngroups} runs, memories shorter than {hz} d:")
    for it, lab in (("read1", "direction flow reads most"), ("probe", "snow probe"), ("snow_group", "snow cells together")):
        g = M[M.item == it]
        if len(g): dd = g[f"memory_d_{hz}"] - g.memory_d_240; print(f"  {lab}: difference median {dd.median():+.0f} d (middle half {dd.quantile(.25):+.0f} to {dd.quantile(.75):+.0f}), rank correlation {spearmanr(g.memory_d_240, g[f'memory_d_{hz}'])[0]:.2f}, n = {len(g)}")
    g = M[M.item == "read1"].groupby("start")[["memory_d_240", f"memory_d_{hz}"]].median()
    print(f"  direction flow reads most, median memory by start day (it is found again from the flow inside each horizon, so compare starts whose {hz} days reach the melt season: February to April):"); print(g.reindex([o for o in ORDER if o in g.index]).round(0).to_string())
    g = M[M.kind.isin(["snow cell", "other cell"])]
    if len(g): print(f"  single cells: rank correlation between the two horizons {spearmanr(g.memory_d_240, g[f'memory_d_{hz}'])[0]:.2f}, median difference {(g[f'memory_d_{hz}'] - g.memory_d_240).median():+.0f} d")
