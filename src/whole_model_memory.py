"""whole_model_memory: memory timescale and rain-pulse response for ALL basins, compared with CAMELS attributes.

Per basin (seed 1, cached, safe to resubmit):
  lag     rain -> discharge lag from the chained Jacobians (lag_profiles definition): M50, mean lag, M90, all days,
          and separately for prediction days in the basin's 3 highest-flow and 3 lowest-flow months
  pulse   add `--pulse` mm of rain on the first of each month (WY2010 onward, --pulse-years years) and follow the extra flow:
          share returned within a year, days until half has arrived, same-day share; cold-day and warm-day pulses apart
  ratio   runoff ratio of simulated and observed flow (mean Q / mean P, WY2009-2014)

  python src/whole_model_memory.py --task 0 --n-tasks 54     # one slice of the basin list (Slurm array)
  python src/whole_model_memory.py --summarize               # merge, join CAMELS attributes, correlations, figure
  python src/whole_model_memory.py --toy --basins a b c      # code test
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import Net, ROOT, RUNS, L_WIN

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1)
ap.add_argument("--basins", nargs="*", default=None); ap.add_argument("--basin-file", default=str(ROOT / "basin_lists/camels_531_basins.txt"))
ap.add_argument("--pulse", type=float, default=10.0); ap.add_argument("--pulse-years", type=int, default=2); ap.add_argument("--every", type=int, default=10)
ap.add_argument("--summarize", action="store_true"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
R = ROOT if not a.toy else Path("toyroot"); CACHE = R / "analysis/all_basins_memory/cache"; CACHE.mkdir(parents=True, exist_ok=True); OUT = CACHE.parent
lags = np.arange(L_WIN)


def half(w):
    """Days until half (and 90%) of a non-negative profile has accumulated, and its mean lag."""
    w = np.clip(w, 0, None); tot = w.sum()
    if tot <= 0: return np.nan, np.nan, np.nan
    c = np.cumsum(w) / tot; return int(np.searchsorted(c, .5)), int(np.searchsorted(c, .9)), float((lags * w).sum() / tot)


def one_basin(b):
    net = Net(1, b, a.toy) if not a.toy else Net(abs(hash(b)) % 1000 + 1, b, True)
    if a.toy: net.y_center = 3.0
    net.gate_check(); row = dict(basin=b)
    # ---- lag profile (lag_profiles definition: mean |dQ/dP| over windows) ----
    e = net.ends(every=a.every); d = net.dates[e]; h, c, S = net.run_windows(e, keep=True); q = net.q(h)
    g = np.abs(net.adjoint(S)[0][:, :, net.p_idx]); del S
    row["M50_d"], row["M90_d"], row["mean_lag_d"] = half(g.mean(1))
    qm = pd.Series(q).groupby(d.month.values).mean().reindex(range(1, 13)); roll = np.array([qm.values[[(m + k) % 12 for k in range(3)]].mean() for m in range(12)])
    for name, m0 in (("highflow", int(np.nanargmax(roll))), ("lowflow", int(np.nanargmin(roll)))):
        months = [(m0 + k) % 12 + 1 for k in range(3)]; m = np.isin(d.month.values, months)
        row[f"{name}_first_month"] = months[0]
        if m.sum() >= 5: row[f"M50_{name}_d"] = half(g[:, m].mean(1))[0]; row[f"sens_{name}"] = float(g[:, m].sum(0).mean() * net.y_scale / net.p_scale)
    # ---- rain pulses ----
    P, T = net.precip(), net.temp(); starts = pd.date_range("2009-10-01", periods=12 * a.pulse_years, freq="MS"); Rr, cold = [], []
    for t in starts:
        t0 = int(np.where(net.dates == t)[0][0]); ee = t0 + np.arange(L_WIN)
        if ee[-1] >= len(net.X): break
        q0 = net.q(net.run_windows(ee)[0]); X0 = net.X; X1 = X0.copy(); X1[t0, net.p_idx] += a.pulse / net.p_scale; net.X = X1
        try: q1 = net.q(net.run_windows(ee)[0])
        finally: net.X = X0
        Rr.append((q1 - q0) / a.pulse); cold.append(T[t0] < 0)
    Rr, cold = np.array(Rr), np.array(cold); np.save(CACHE / f"pulse_{b}.npy", Rr.astype(np.float32))
    for name, m in (("", np.ones(len(Rr), bool)), ("_cold", cold), ("_warm", ~cold)):
        if m.sum() < 2: continue
        r = Rr[m].mean(0); row[f"pulse_returned{name}"] = float(r.sum()); row[f"pulse_M50{name}_d"] = half(r)[0]
        row[f"pulse_day0_share{name}"] = float(r[0] / r.sum()) if r.sum() > 0 else np.nan; row[f"pulse_first10d_share{name}"] = float(r[:10].sum() / r.sum()) if r.sum() > 0 else np.nan
    row["n_cold_pulses"] = int(cold.sum())
    # ---- runoff ratio ----
    ed = net.ends(); qs = net.q(net.run_windows(ed[::3])[0]); row["q_sim_mean"] = float(qs.mean()); row["p_mean"] = float(P[ed].mean()); row["runoff_ratio_sim"] = row["q_sim_mean"] / row["p_mean"]
    if net.q_obs is not None and np.isfinite(net.q_obs[ed]).mean() > .8: row["runoff_ratio_obs"] = float(np.nanmean(net.q_obs[ed]) / row["p_mean"])
    row["cold_precip_share"] = float(P[ed][T[ed] < 0].sum() / P[ed].sum())
    return row


if not a.summarize:
    basins = a.basins or [l.strip().zfill(8) for l in open(a.basin_file) if l.strip()]
    mine = basins[a.task::a.n_tasks] if not a.basins else basins
    for b in mine:
        f = CACHE / f"basin_{b}.csv"
        if f.exists(): print(b, "cached"); continue
        try: pd.DataFrame([one_basin(b)]).to_csv(f, index=False); print(b, "done")
        except Exception as ex: print(b, "FAILED", type(ex).__name__, ex)
else:
    files = sorted(CACHE.glob("basin_*.csv")); D = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in files], ignore_index=True)
    D["lowflow_minus_highflow_M50_d"] = D.get("M50_lowflow_d") - D.get("M50_highflow_d")
    print(f"{len(D)} basins"); pd.set_option("display.width", 220)
    att = None
    try:
        import yaml
        from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes
        cfg = yaml.safe_load(open(ROOT / "runs" / RUNS[1] / "config.yml")); att = load_camels_us_attributes(Path(cfg["data_dir"]), basins=list(D.basin))
        att.index = att.index.astype(str).str.zfill(8); D = D.merge(att, left_on="basin", right_index=True, how="left", suffixes=("", "_camels"))
    except Exception as ex: print("CAMELS attributes not joined:", type(ex).__name__, ex)
    D.to_csv(OUT / "whole_model_memory.csv", index=False)
    print("\nmedians:"); print(D[[c for c in ("M50_d", "mean_lag_d", "M90_d", "M50_highflow_d", "M50_lowflow_d", "pulse_M50_d", "pulse_returned", "runoff_ratio_sim", "runoff_ratio_obs") if c in D]].median().round(2).to_string())
    if att is not None:
        from scipy.stats import spearmanr
        tests = [("M50_d", ["frac_snow", "aridity", "baseflow_index", "runoff_ratio", "p_seasonality", "area_gages2", "soil_depth_pelletier"]),
                 ("pulse_M50_d", ["frac_snow", "aridity", "baseflow_index"]), ("pulse_returned", ["runoff_ratio", "aridity"]),
                 ("runoff_ratio_sim", ["runoff_ratio"]), ("lowflow_minus_highflow_M50_d", ["baseflow_index", "frac_snow"]), ("M50_lowflow_d", ["baseflow_index"])]
        print("\nSpearman rank correlations across basins:")
        for y, xs in tests:
            for x in xs:
                if y in D and x in D:
                    m = D[[x, y]].dropna(); print(f"  {y:30s} vs {x:22s} rho = {spearmanr(m[x], m[y])[0]:+.2f}  (n = {len(m)})")
        lowsnow = D[D.frac_snow < .1] if "frac_snow" in D else D
        if "baseflow_index" in D and len(lowsnow) > 20:
            m = lowsnow[["baseflow_index", "M50_lowflow_d"]].dropna(); print(f"  rain-fed basins only (snow fraction < 0.1): low-flow-season M50 vs baseflow_index rho = {spearmanr(m.baseflow_index, m.M50_lowflow_d)[0]:+.2f} (n = {len(m)})")
        print("ANSWER: memory tracks snow if M50 vs frac_snow is strongly positive; the model's water balance is physical if pulse_returned vs runoff_ratio is strongly positive;\n"
              "  a groundwater reading of the low-flow season is supported if low-flow M50 rises with baseflow_index in rain-fed basins.")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        for x_, (xc, yc, xl, yl) in zip(ax, [("frac_snow", "M50_d", "fraction of precipitation falling as snow", "half of rain's influence lies within (days)"),
                                              ("runoff_ratio", "pulse_returned", "observed runoff ratio", "share of a rain pulse that becomes flow"),
                                              ("baseflow_index", "M50_lowflow_d", "baseflow index", "same, low-flow season (days)")]):
            if xc in D and yc in D:
                m = D[[xc, yc, "aridity"]].dropna() if "aridity" in D else D[[xc, yc]].dropna()
                sc = x_.scatter(m[xc], m[yc], s=14, alpha=.6, c=m["aridity"] if "aridity" in m else "#3b6fb6", cmap="viridis_r", vmax=3)
                x_.set(xlabel=xl, ylabel=yl); x_.set_title(f"ρ = {spearmanr(m[xc], m[yc])[0]:+.2f}", loc="left", fontweight="bold")
                if xc == "runoff_ratio": x_.plot([0, 1], [0, 1], "k--", lw=1)
        plt.tight_layout(); plt.savefig(OUT / "nb23_memory_vs_attributes.png", dpi=220); print("figure:", OUT / "nb23_memory_vs_attributes.png")
    print("wrote", OUT / "whole_model_memory.csv")
