"""nb26: does the model's memory, in days, match a lag measured from observations? All snow basins.

The physical comparison (same definition as nb24 section O, which did one basin):
  observed lag = days from the centre of mass of freezing-day precipitation (October to June)
                 to the centre of mass of April-September flow above the cold-season base flow, per water year.
  It uses observed precipitation, temperature and flow only. The same calculation on the model's flow gives the simulated lag.
Then, across basins, the observed lag is set against what the memory tools read from the model (already in the nb23 table):
  M50_d              chained Jacobians: half of rain's influence on flow lies within this many days
  pulse_M50_cold_d   direct test: days until half of the response to a rain pulse on a freezing day has arrived

Basins: those with cold_precip_share above --cold (default 0.2) in the nb23 table.
  sbatch slurm/nb26_observed_lag.slurm                           # about 10 basins per task, CPU
  PYTHONPATH=src python src/nb26_observed_lag_snow_basins.py --summarize
  python src/nb26_observed_lag_snow_basins.py --toy --basins a b c d e f
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import Net, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0); ap.add_argument("--n-tasks", type=int, default=1); ap.add_argument("--basins", nargs="*", default=None)
ap.add_argument("--cold", type=float, default=.2); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
R = ROOT if not a.toy else Path("toyroot"); OUT = R / "analysis/all_basins_memory"; CACHE = OUT / "cache_obs_lag"; CACHE.mkdir(parents=True, exist_ok=True)
TABLE = OUT / "nb23_all_basins_memory.csv"


def lags_one_year(d, P, T, flows):
    """Snowfall centre and, for each flow series, melt-flow centre minus snowfall centre (days). nb24 section O definition."""
    dd = np.arange(len(d)); cold = (T < 0) & d.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6]); melt = d.month.isin([4, 5, 6, 7, 8, 9]); out = {}
    if P[cold].sum() <= 0: return out
    cp = (dd[cold] * P[cold]).sum() / P[cold].sum(); out["snowfall_centre_doy"] = int(d[int(cp)].dayofyear)
    for name, qq in flows.items():
        if qq is None or np.isnan(qq[melt]).mean() > .1: continue
        qq = np.nan_to_num(qq); base = np.percentile(qq[~melt], 25); ex = np.clip(qq - base, 0, None) * melt
        if ex.sum() > 0: out[f"{name}_lag_d"] = float((dd * ex).sum() / ex.sum() - cp)
    return out


def one_basin(b):
    net = Net(1, b, a.toy) if not a.toy else Net(abs(hash(b)) % 1000 + 1, b, True)
    if a.toy: net.y_center = 3.0
    e = net.ends(); d = net.dates[e]; wy = net.wy[e]; q = net.q(net.run_windows(e)[0]); P, T = net.precip()[e], net.temp()[e]
    qo = net.q_obs[e] if net.q_obs is not None else (np.roll(q, 6) * 1.05 if a.toy else None)
    rows = [dict(basin=b, wy=int(y), **lags_one_year(d[wy == y], P[wy == y], T[wy == y], dict(sim=q[wy == y], obs=None if qo is None else qo[wy == y]))) for y in np.unique(wy)]
    return pd.DataFrame(rows)


if not a.summarize:
    if a.basins: basins = a.basins
    else:
        t = pd.read_csv(TABLE, dtype={"basin": str}); basins = sorted(t.loc[t.cold_precip_share > a.cold, "basin"].str.zfill(8)); print(len(basins), f"basins with freezing-day precipitation share above {a.cold}")
    for b in (basins if a.basins else basins[a.task::a.n_tasks]):
        f = CACHE / f"lag_{b}.csv"
        if f.exists(): print(b, "cached"); continue
        try: one_basin(b).to_csv(f, index=False); print(b, "done")
        except Exception as ex: print(b, "FAILED", type(ex).__name__, ex)
else:
    from scipy.stats import spearmanr
    Y = pd.concat([pd.read_csv(f, dtype={"basin": str}) for f in sorted(CACHE.glob("lag_*.csv"))], ignore_index=True); Y.to_csv(OUT / "nb26_observed_lag_by_year.csv", index=False)
    cols = [c for c in ("obs_lag_d", "sim_lag_d") if c in Y]; B = Y.groupby("basin")[cols].median(); B["n_years"] = Y.groupby("basin").obs_lag_d.count() if "obs_lag_d" in Y else 0
    B = B[B.n_years >= 3].reset_index(); pd.set_option("display.width", 200)
    if a.toy: B["M50_d"] = B.obs_lag_d * .7 + np.random.default_rng(0).normal(0, 5, len(B)); B["pulse_M50_cold_d"] = B.obs_lag_d * .9
    elif TABLE.exists():
        t = pd.read_csv(TABLE, dtype={"basin": str}); t["basin"] = t.basin.str.zfill(8); B = B.merge(t[[c for c in ("basin", "M50_d", "pulse_M50_d", "pulse_M50_cold_d", "n_cold_pulses", "frac_snow", "aridity", "cold_precip_share") if c in t]], on="basin", how="left")
    B.to_csv(OUT / "nb26_observed_lag_by_basin.csv", index=False)
    print(f"\n{len(B)} snow basins with at least 3 water years of observed lag")
    print(f"observed lag, snowfall centre to melt-flow centre: median {B.obs_lag_d.median():.0f} d, quartiles {B.obs_lag_d.quantile(.25):.0f} to {B.obs_lag_d.quantile(.75):.0f}, range {B.obs_lag_d.min():.0f} to {B.obs_lag_d.max():.0f}")

    def line(y, label):
        m = B[["obs_lag_d", y]].dropna()
        if len(m) < 5: print(f"  {label}: too few basins"); return None
        diff = m[y] - m.obs_lag_d; rho = spearmanr(m.obs_lag_d, m[y])[0]
        print(f"  {label:<58s} rho = {rho:+.2f} | median {m[y].median():.0f} d | median difference from observed {diff.median():+.0f} d | within 14 d in {100 * (diff.abs() <= 14).mean():.0f}% of basins (n = {len(m)})"); return rho

    print("\nagainst the observed lag, across basins:")
    r_sim = line("sim_lag_d", "A. same calculation on the model's flow") if "sim_lag_d" in B else None
    r_jac = line("M50_d", "B. chained Jacobians (half of rain's influence)") if "M50_d" in B else None
    r_pul = line("pulse_M50_cold_d", "C. rain pulse on a freezing day (half arrived)") if "pulse_M50_cold_d" in B else None
    if "sim_lag_d" in Y and "obs_lag_d" in Y:
        m = Y[["basin", "obs_lag_d", "sim_lag_d"]].dropna(); an = m[["obs_lag_d", "sim_lag_d"]] - m.groupby("basin")[["obs_lag_d", "sim_lag_d"]].transform("mean")
        print(f"  year to year within a basin (departures from each basin's mean): observed against simulated rho = {spearmanr(an.obs_lag_d, an.sim_lag_d)[0]:+.2f} (n = {len(m)} basin-years)")
    print("\nANSWER: A says whether the model's flow has the observed snow lag. B and C say whether the memory read by the tools, in days, is that same physical lag:\n"
          "  a strong rho with a small median difference means the days can be read as the snowfall-to-melt delay; a strong rho with a large difference means the tools rank basins correctly but the days are not the lag itself.")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    P_ = [(c, l) for c, l in (("sim_lag_d", "same lag in the model's flow (days)"), ("M50_d", "model memory, chained Jacobians (days)"), ("pulse_M50_cold_d", "model: half of a cold-day rain pulse\nhas arrived (days)")) if c in B and B[c].notna().sum() >= 5]
    fig, ax = plt.subplots(1, len(P_), figsize=(5.4 * len(P_), 5.2)); ax = np.atleast_1d(ax)
    for x_, (c, l) in zip(ax, P_):
        m = B[["obs_lag_d", c]].dropna(); lim = [0, float(max(m.max().max(), 1)) * 1.05]
        x_.scatter(m.obs_lag_d, m[c], s=28, alpha=.7, color="#3b6fb6", edgecolor="none"); x_.plot(lim, lim, "k--", lw=1); x_.set_xlim(lim); x_.set_ylim(lim); x_.set_aspect("equal")
        x_.set_xlabel("observed: snowfall to melt flow (days)"); x_.set_ylabel(l); x_.set_title(f"rank correlation {spearmanr(m.obs_lag_d, m[c])[0]:.2f}, {len(m)} snow basins", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "nb26_observed_lag.png", dpi=200); print("wrote", OUT / "nb26_observed_lag.png")
