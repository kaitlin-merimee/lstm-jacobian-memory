"""nb25: is the model's memory longer in arid basins than in snow basins?

Reads the per-basin table written by `nb23_all_basins_memory.py --summarize` (no model runs, seconds on a login node).

  regime   snow  = frac_snow >= --snow            (default 0.3)
           arid  = not snow and aridity >= --arid (default 1.0, PET/P)
           humid = everything else
  A  medians and quartiles of each memory measure by regime, with a rank test between regimes
  B  snow fraction x aridity grid of median M50, so the two attributes are not confused with each other
  C  partial rank correlations: aridity with snow fraction held fixed, and the reverse; then also holding mean flow fixed
  D  boxplot figure

  python src/backup/nb25_memory_by_regime.py
  python src/backup/nb25_memory_by_regime.py --snow 0.4 --arid 1.5
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu, rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--csv", default=None); ap.add_argument("--snow", type=float, default=.3); ap.add_argument("--arid", type=float, default=1.0)
ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = Path("/tmp/nb25_toy") if a.toy else Path(a.root) / "analysis" / "all_basins_memory"; OUT.mkdir(parents=True, exist_ok=True)

if a.toy:
    r = np.random.default_rng(0); n = 531; fs = r.beta(.6, 2.5, n); ar = np.exp(r.normal(0, .5, n))
    D = pd.DataFrame(dict(basin=[f"{i:08d}" for i in range(n)], frac_snow=fs, aridity=ar, baseflow_index=r.uniform(.1, .9, n), q_sim_mean=3 / ar))
    D["M50_d"] = np.exp(3 + .8 * np.log(ar) + .5 * fs + r.normal(0, .5, n)); D["pulse_M50_d"] = D.M50_d * .6; D["pulse_returned"] = np.clip(.6 - .3 * np.log(ar), 0, 1)
    D["M50_lowflow_d"] = D.M50_d * 1.4; D["M50_highflow_d"] = D.M50_d * .9; D["mean_lag_d"] = D.M50_d * 1.5
else:
    D = pd.read_csv(a.csv or OUT / "nb23_all_basins_memory.csv", dtype={"basin": str})
need = [c for c in ("frac_snow", "aridity") if c not in D]
if need: raise SystemExit(f"missing columns {need}: rerun `nb23_all_basins_memory.py --summarize` so the CAMELS attributes are joined")

D["regime"] = np.where(D.frac_snow >= a.snow, "snow", np.where(D.aridity >= a.arid, "arid", "humid"))
M = [c for c in ("M50_d", "mean_lag_d", "M50_lowflow_d", "M50_highflow_d", "pulse_M50_d", "pulse_returned") if c in D]
order = ["humid", "arid", "snow"]; pd.set_option("display.width", 200)

print(f"\nA. regimes: snow = frac_snow >= {a.snow}; arid = other basins with aridity >= {a.arid}; humid = the rest")
print(D.groupby("regime")[["frac_snow", "aridity"] + (["baseflow_index"] if "baseflow_index" in D else [])].median().reindex(order).assign(n=D.regime.value_counts()).round(2).to_string())
for c in M:
    g = D.groupby("regime")[c].quantile([.25, .5, .75]).unstack().reindex(order)
    groups = [D.loc[D.regime == k, c].dropna() for k in order]
    p_all = kruskal(*[x for x in groups if len(x) > 2])[1] if sum(len(x) > 2 for x in groups) > 1 else np.nan
    p_as = mannwhitneyu(groups[1], groups[2])[1] if min(len(groups[1]), len(groups[2])) > 2 else np.nan
    print(f"\n  {c}: median [25%, 75%]   (any difference between regimes p = {p_all:.2g}; arid against snow p = {p_as:.2g})")
    for k in order: print(f"    {k:<6s} {g.loc[k, .5]:7.2f}  [{g.loc[k, .25]:.2f}, {g.loc[k, .75]:.2f}]")

print("\nB. median M50 (days) by snow fraction (rows) and aridity (columns); basin counts in brackets")
D["snow_bin"] = pd.cut(D.frac_snow, [-.01, .1, .3, 1.01], labels=["snow < 0.1", "0.1 to 0.3", "snow > 0.3"])
D["arid_bin"] = pd.cut(D.aridity, [0, .75, 1.25, 99], labels=["wet (< 0.75)", "middle", "dry (> 1.25)"])
t = D.pivot_table(index="snow_bin", columns="arid_bin", values="M50_d", aggfunc="median", observed=False); nn = D.pivot_table(index="snow_bin", columns="arid_bin", values="M50_d", aggfunc="count", observed=False)
print((t.round(0).astype("Int64").astype(str) + " (" + nn.astype(int).astype(str) + ")").to_string())


def partial(y, x, ctrl):
    """Spearman correlation of y and x after removing what the control columns explain (all in ranks)."""
    m = D[[y, x] + ctrl].dropna(); R = np.column_stack([rankdata(m[c]) for c in [y, x] + ctrl]); Z = np.column_stack([np.ones(len(m)), R[:, 2:]])
    ry = R[:, 0] - Z @ np.linalg.lstsq(Z, R[:, 0], rcond=None)[0]; rx = R[:, 1] - Z @ np.linalg.lstsq(Z, R[:, 1], rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1]), len(m)


print("\nC. rank correlations with M50, alone and with the other attributes held fixed")
print(f"  frac_snow against aridity across basins: rho = {spearmanr(D.frac_snow, D.aridity)[0]:+.2f}")
flow = [c for c in ("q_sim_mean",) if c in D]
for y in [c for c in ("M50_d", "pulse_M50_d") if c in D]:
    for x, other in (("aridity", "frac_snow"), ("frac_snow", "aridity")):
        m = D[[y, x]].dropna(); line = f"  {y:<12s} vs {x:<10s} alone {spearmanr(m[y], m[x])[0]:+.2f} | {other} fixed {partial(y, x, [other])[0]:+.2f}"
        if "baseflow_index" in D: line += f" | {other} and baseflow index fixed {partial(y, x, [other, 'baseflow_index'])[0]:+.2f}"
        if flow: line += f" | {other} and mean flow fixed {partial(y, x, [other] + flow)[0]:+.2f}"
        print(line)

med = D.groupby("regime").M50_d.median()
print(f"\nANSWER: median M50 is {med.get('arid', np.nan):.0f} d in arid basins, {med.get('snow', np.nan):.0f} d in snow basins and {med.get('humid', np.nan):.0f} d in humid basins. "
      "'Arid longer than snow' holds only if the arid median is clearly above the snow median and the arid-against-snow p value in A is small. "
      "If the aridity correlation in C falls towards zero once mean flow is held fixed, the long memory of dry basins is mostly the low-flow effect.")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = [(c, l) for c, l in (("M50_d", "half of rain's influence\nlies within (days)"), ("pulse_M50_d", "half of a rain pulse's\nresponse has arrived (days)"), ("pulse_returned", "share of a rain pulse\nthat becomes flow")) if c in D]
fig, ax = plt.subplots(1, len(P), figsize=(3.6 * len(P), 3.6)); ax = np.atleast_1d(ax); col = dict(humid="#5a9e6f", arid="#c98a3b", snow="#3b6fb6")
for x_, (c, l) in zip(ax, P):
    bp = x_.boxplot([D.loc[D.regime == k, c].dropna() for k in order], patch_artist=True, showfliers=False, widths=.6, medianprops=dict(color="k"))
    for b_, k in zip(bp["boxes"], order): b_.set_facecolor(col[k]); b_.set_alpha(.8)
    x_.set_xticks([1, 2, 3], [f"{k}\n(n = {(D.regime == k).sum()})" for k in order]); x_.set_ylabel(l); x_.spines[["top", "right"]].set_visible(False)
fig.suptitle("Model memory by regime, 531 basins, one model version", fontsize=10); fig.tight_layout(); fig.savefig(OUT / "nb25_memory_by_regime.png", dpi=200)
D[["basin", "regime", "frac_snow", "aridity"] + M].to_csv(OUT / "nb25_memory_by_regime.csv", index=False); print("wrote", OUT / "nb25_memory_by_regime.png")
