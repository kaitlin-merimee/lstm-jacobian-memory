"""lag_profiles: Jacobian rain -> discharge lag profiles for the two walkthrough basins (replaces 8 d / 0.7 d).

For every window (one every `--every` days, ending in WY2009-2014) the transposed daily Jacobians are
chained back from the prediction day. That gives, per lag k = 0..364:
  dQ_T / dP_{T-k}   (mm/d of discharge per mm/d of precipitation, k days earlier)
  reach-back        ||dQ_T / d state_{T-k}|| relative to lag 0 (= ||w_out||)
Summaries: mean lag and M50 (lag by which half of the summed |sensitivity| has arrived), overall and by season.

CHECK BEFORE QUOTING: synthetic_3_recovery's `in_mean_lag` / `in_M50` were validated against planted timescales. This
script weights lags by mean |dQ/dP| over windows. If synthetic_3_recovery used a different weighting (signed, or per
window then averaged), set --weight to match so the real-basin numbers are comparable to the synthetic test.

  python src/backup/lag_profiles.py --seeds 1 2 3
  python src/backup/lag_profiles.py --toy
"""
import argparse, numpy as np, pandas as pd
from jtalk_core import Net, ROOT, SNOW, ARID, L_WIN, season_of

ap = argparse.ArgumentParser()
ap.add_argument("--basins", nargs="*", default=[SNOW, ARID]); ap.add_argument("--seeds", type=int, nargs="*", default=[1])
ap.add_argument("--every", type=int, default=10); ap.add_argument("--weight", choices=["abs", "signed", "per-window"], default="abs")
ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else __import__("pathlib").Path("toyroot")) / "analysis/lagprofiles"; OUT.mkdir(parents=True, exist_ok=True)
lags = np.arange(L_WIN)


def summarize(g):
    """g: [L, n] sensitivities. Returns the weight profile over lags (sums to 1), mean lag, M50, M90, negative share."""
    if a.weight == "per-window":
        w = (np.abs(g) / np.abs(g).sum(0)).mean(1)
    elif a.weight == "signed":
        w = np.clip(g.mean(1), 0, None); w = w / w.sum()
    else:
        w = np.abs(g).mean(1); w = w / w.sum()
    cum = np.cumsum(w)
    return w, float((lags * w).sum()), int(np.searchsorted(cum, .5)), int(np.searchsorted(cum, .9)), float(np.clip(-g, 0, None).sum() / np.abs(g).sum())


rows, prof = [], {}
for seed in a.seeds:
    for basin in a.basins:
        net = Net(seed, basin, a.toy); net.gate_check()
        e = net.ends(every=a.every); d = net.dates[e]; seas = season_of(d.month.values)
        h, c, S = net.run_windows(e, keep=True); q = net.q(h)
        dQdx, reach = net.adjoint(S)
        g = dQdx[:, :, net.p_idx] * net.y_scale / net.p_scale           # mm/d per mm/d
        rb = reach / reach[0]
        hi = q >= np.quantile(q, .75)
        for name, m in [("all", np.ones(len(e), bool)), ("high flow (top 25%)", hi), ("rest", ~hi)] + [(s, seas == s) for s in pd.unique(seas)]:
            if m.sum() < 3: continue
            w, ml, m50, m90, neg = summarize(g[:, m])
            rows.append(dict(seed=seed, basin=basin, subset=name, n_windows=int(m.sum()), mean_lag_d=ml, M50_d=m50, M90_d=m90,
                             negative_share=neg, total_gain=float(g[:, m].sum(0).mean()),
                             reach30=float(rb[30, m].mean()), reach90=float(rb[90, m].mean()), reach180=float(rb[180, m].mean()),
                             q_mean=float(q[m].mean())))
            if name == "all": prof[(seed, basin)] = w
        np.savez_compressed(OUT / f"nb17_{basin}_seed{seed}.npz", g=g.astype(np.float32), reach=rb.astype(np.float32),
                            q=q, dates=d.values.astype("datetime64[D]"))
df = pd.DataFrame(rows); df.to_csv(OUT / "nb17_lag_summary.csv", index=False)
pd.set_option("display.width", 220); print(df.round(3).to_string(index=False))
print("\nSLIDE NUMBERS (subset = all):")
for (basin), gdf in df[df.subset == "all"].groupby("basin"):
    print(f"  {basin}: mean lag {gdf.mean_lag_d.mean():.1f} d (seeds {gdf.mean_lag_d.min():.1f}-{gdf.mean_lag_d.max():.1f}), "
          f"M50 {gdf.M50_d.median():.0f} d, M90 {gdf.M90_d.median():.0f} d, negative share {gdf.negative_share.mean():.2f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 5.2)); col = {a.basins[0]: "#3b6fb6", a.basins[-1]: "#c8782a"}
for (seed, basin), w in prof.items():
    ax.plot(lags + 1, w, color=col.get(basin, "k"), lw=2.5 if seed == a.seeds[0] else 1, alpha=1 if seed == a.seeds[0] else .5,
            label=basin if seed == a.seeds[0] else None)
    if seed == a.seeds[0]:
        m50 = df[(df.seed == seed) & (df.basin == basin) & (df.subset == "all")].M50_d.iloc[0]
        ax.axvline(m50 + 1, color=col.get(basin, "k"), ls="--", lw=1.2)
ax.set(xscale="log", yscale="log", xlabel="days before the predicted day (lag + 1)", ylabel="share of discharge's\nsensitivity to rain")
ax.legend(frameon=False, title="dashed = half of sensitivity (M50)"); plt.tight_layout()
plt.savefig(OUT / "lag_profiles.png", dpi=250); print("figure:", OUT / "lag_profiles.png")
