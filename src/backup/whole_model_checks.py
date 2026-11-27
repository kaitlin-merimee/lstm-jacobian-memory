"""whole_model_checks: the remaining checks before the talk, and a side-by-side table of the per-basin / per-seed snow runs.

  --collect   table of every nb19_summary_*.csv (one row per basin and seed): does each run build a snow counter with the same behaviour?
  --check1    is lag_profiles's rain -> discharge lag (the 97 d / 42 d row) the quantity the synthetic test validated?
              Recomputes it on the synthetic S2 model for --n basins and compares with synthetic_3_recovery's own in_mean_lag / in_M50 and the truth.
  --check3    snow-unit timing in the four water years NOT used to pick the unit (seed 1, Encampment)
  --check4    is the forget-gate timescale the model's FAST response in the two real basins? Uses the rain-pulse responses saved by snow_unit_questions E.
With no flags, everything is run.   PYTHONPATH=src python src/backup/whole_model_checks.py
"""
import argparse, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from jtalk_core import Net, ROOT, SNOW, ARID, L_WIN
ap = argparse.ArgumentParser()
for f in ("collect", "check1", "check3", "check4", "toy"): ap.add_argument(f"--{f}", action="store_true")
ap.add_argument("--n", type=int, default=30); ap.add_argument("--scen", default="S2"); a = ap.parse_args()
ALL = not (a.collect or a.check1 or a.check3 or a.check4); R = ROOT if not a.toy else Path("toyroot"); OQ = R / "analysis/open_questions"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

if a.collect or ALL:
    files = sorted(OQ.glob("nb19_summary_*.csv"))
    print(f"\n== snow counter across basins and seeds ({len(files)} runs) ==")
    if files:
        S = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in files], ignore_index=True)
        S.to_csv(OQ / "nb22_snow_runs.csv", index=False)
        show = {"basin": "basin", "seed": "seed", "unit": "unit", "snow_reference": "snow ref", "unit_abs_r": "|r| snow", "cells_r_gt_0p8": "cells>0.8", "years_with_flip": "yrs flip",
                "turn_minus_snow_peak_d": "turn-peak snow", "turn_minus_sim_rise_d": "turn-rise", "flip_minus_sim_peak_d": "flip-peak Q", "sim_flow_done_at_flip": "flow done",
                "increment_r2_precip": "R2 precip", "increment_r2_temperature": "R2 temp", "swe_r2_unit_cell": "R2 cell", "swe_r2_unit_hidden": "R2 hidden",
                "precip_m30_melt_flow": "P-30% flow", "precip_p30_melt_flow": "P+30% flow", "unit_m30_melt_flow": "unit-30%", "snowcells_m30_melt_flow": "cells-30%",
                "snowcells_p30_melt_flow": "cells+30%", "random_m30_melt_flow": "random-30%"}
        print(S[[k for k in show if k in S]].rename(columns=show).round(2).to_string(index=False))
        print("How to read: a snow counter is there when |r| snow is high, the count turns near peak snow and before the rise, and the flip follows peak flow in most years.\n"
              "  'R2 cell' high with 'R2 hidden' near 0 = the amount is stored but hidden. 'cells-30%' far from 'random-30%' = the snow cells together carry volume.")

if a.check3 or ALL:
    print("\n== check 3: timing in the years not used to pick the unit ==")
    f = [p for p in (OQ / f"nb19_A_flip_timing_{SNOW}_seed1.csv", OQ / f"nb19_A_flip_timing_{SNOW}.csv") if p.exists()]
    if not f: print("  snow_unit_questions section A output not found")
    else:
        A = pd.read_csv(f[0]); ind = A[~A.wy.isin([2011, 2012])]
        for k in ("flip_minus_sim_peak_d", "flip_minus_obs_peak_d", "turn_minus_sim_rise_d", "sim_flow_done_at_flip", "turn_minus_snow_peak_d"):
            if k in A: print(f"  {k:26s} all six years: median {A[k].median():+.2f} (range {A[k].min():+.2f} to {A[k].max():+.2f}) | 2009, 2010, 2013, 2014 only: median {ind[k].median():+.2f} (range {ind[k].min():+.2f} to {ind[k].max():+.2f})")

if a.check4 or ALL:
    print("\n== check 4: forget-gate timescale against the model's fast response to rain ==")
    gates = {SNOW: 7.8, ARID: 0.74}
    g = OQ / "gate_timescale_by_season.csv"
    if g.exists():
        G = pd.read_csv(g, dtype={"basin": str}); gates = {b: float(G[(G.basin == b) & (G.days == "all days")].A_median.iloc[0]) for b in G.basin.unique()}
    months = pd.date_range("2008-10-01", "2013-09-01", freq="MS").month
    for basin in (SNOW, ARID):
        f = OQ / f"nb19_E_pulse_response_{basin}.npy"
        if not f.exists(): print(f"  {basin}: pulse responses not found (run snow_unit_questions --only E)"); continue
        Rr = np.load(f).astype(float); warm = np.isin(months[:len(Rr)], [5, 6, 7, 8, 9]) if basin == SNOW else np.ones(len(Rr), bool)
        r = Rr[warm].mean(0); kp = int(np.argmax(r[:10]))
        if r[kp] <= 0: print(f"  {basin}: no positive quick response to the pulse; not interpretable"); continue
        dummy = 0; below = np.where(r[kp:] < r[kp] / np.e)[0]; half = np.where(r[kp:] < r[kp] / 2)[0]
        tot = r.sum(); c10 = r[:10].sum() / tot if tot > 0 else np.nan
        print(f"  {basin}: after a rain pulse{' (May-Sep pulses)' if basin == SNOW else ''} the response peaks on day {kp} and falls to 1/e of its peak after {below[0] if len(below) else np.nan} d "
              f"(to half after {half[0] if len(half) else np.nan} d); {c10:.0%} of the year's extra flow arrives in the first 10 days. Forget-gate timescale: {gates.get(basin, np.nan):.1f} d.")
    print("  If the 1/e time of the quick response is close to the gate timescale, the gates are describing the model's fast response in the real basins too.")

if a.check1 or ALL:
    print(f"\n== check 1: lag_profiles's lag definition on the synthetic {a.scen} model, against synthetic_3_recovery's own numbers ==")
    try:
        if a.toy:
            basins = ["t1", "t2", "t3"]; T = pd.DataFrame(dict(basin=basins, in_mean_lag=[20., 30, 40], in_M50=[5., 9, 12], true_mean_lag=[21., 33, 39], true_M50=[5., 8, 13])); run = None
        else:
            run = Path(sorted(glob.glob(str(ROOT / f"runs/camels531_syn_{a.scen}_n00_h256_seed1_*")))[-1]).name
            T = pd.read_csv(ROOT / f"analysis/jacobians/followup/nb08b_basins_{a.scen}.csv", dtype={"basin": str}); T["basin"] = T.basin.str.zfill(8)
            T = T.sort_values("true_mean_lag").iloc[np.linspace(0, len(T) - 1, a.n).astype(int)]; basins = list(T.basin)
        lags = np.arange(L_WIN); rows = []
        for k, b in enumerate(basins):
            net = Net(k + 1 if a.toy else 1, b, a.toy, run=run); e = net.ends(every=10); _, _, S = net.run_windows(e, keep=True); g = net.adjoint(S)[0][:, :, net.p_idx]
            row = dict(basin=b)
            for nm, w in (("abs", np.abs(g).mean(1)), ("signed", np.clip(g.mean(1), 0, None)), ("per_window", (np.abs(g) / np.abs(g).sum(0)).mean(1))):
                w = w / w.sum(); row[f"mean_lag_{nm}"] = float((lags * w).sum()); row[f"M50_{nm}"] = int(np.searchsorted(np.cumsum(w), .5))
            rows.append(row)
        C = pd.DataFrame(rows).merge(T[["basin", "in_mean_lag", "in_M50", "true_mean_lag", "true_M50"]], on="basin"); C.to_csv(OQ / f"nb22_check1_{a.scen}.csv", index=False)
        out = []
        for nm in ("abs", "signed", "per_window"):
            for q, ref in (("mean_lag", "in_mean_lag"), ("M50", "in_M50"), ("mean_lag", "true_mean_lag"), ("M50", "true_M50")):
                x, y = C[f"{q}_{nm}"], C[ref]; out.append(dict(nb17_weighting=nm, quantity=q, compared_with=ref, rho=spearmanr(x, y)[0], median_ratio=float(np.median(x / np.maximum(y, 1e-9)))))
        O = pd.DataFrame(out); print(O.round(2).to_string(index=False))
        best = O[O.compared_with == "in_mean_lag"].assign(err=lambda d: (np.log(d.median_ratio)).abs()).sort_values("err").iloc[0]
        print(f"ANSWER check 1: lag_profiles with '{best.nb17_weighting}' weighting gives mean lags {best.median_ratio:.2f}x synthetic_3_recovery's (rank correlation {best.rho:.2f}) on {len(C)} synthetic basins. "
              f"A ratio near 1 and a high correlation mean the 97 d / 42 d row uses the validated quantity; otherwise rerun lag_profiles with --weight {best.nb17_weighting.replace('_', '-')} or drop the row. "
              f"lag_profiles uses windows ending in WY2009-2014; if synthetic_3_recovery used other years the ratio can differ a little for that reason alone.")
    except Exception as ex:
        print(f"  could not run on the synthetic model ({type(ex).__name__}: {ex}). The loader assumes the synthetic run loads like the real ones over WY2009-2014.")
