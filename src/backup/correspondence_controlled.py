"""correspondence_controlled.py

Confound-controlled correspondence: does learned tau carry storage information beyond climate/wetness?
Outputs analysis/jacobians/correspondence_controlled.csv
Recovered from jacobian_followup_experiments.md (Oct 2026). Items marked VERIFY must be
checked against the NeuralHydrology 1.13 install before trusting outputs.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from jlens_utils import OUT_DIR, PROJECT, latest_run, Model, regime

import sys; sys.path.insert(0, "src")
from jlens_utils import OUT_DIR, PROJECT
FIG_DIR = PROJECT / "analysis" / "figures"

rng = np.random.default_rng(0)
base = pd.read_csv(OUT_DIR / "analysis_table.csv", dtype={"basin": str}).set_index("basin")

# ---- assemble per-seed learned tau (sweep files from the early notebooks naming) ----
seed_files = {1: "sweep_summary.csv", 2: "sweep_summary_seed2.csv", 3: "sweep_summary_seed3.csv"}
tau = pd.DataFrame({s: pd.read_csv(OUT_DIR / f, dtype={"basin": str}).set_index("basin")["tau_readout_log10"]
                    for s, f in seed_files.items()})
tau["mean"] = tau[[1, 2, 3]].mean(axis=1)

# ---- static attributes exactly as the model saw them (raw, pre-standardization) ----
m1 = Model(latest_run("camels531_daymet_h256_seed1*"))
attr_dir = m1.cfg.data_dir / "camels_attributes_v2.0"
attrs = pd.concat([pd.read_csv(p, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
                   for p in sorted(attr_dir.glob("camels_*.txt"))], axis=1)
attrs = attrs.loc[:, ~attrs.columns.duplicated()]
STATICS = m1.statics
huc = attrs["huc_02"].astype(int)          # for spatially blocked CV (from camels_name.txt)

df = base.join(tau, rsuffix="_seed").join(attrs[STATICS + ["huc_02"]], rsuffix="_attr")
df["log_tau_rec_p90"] = np.log10(df["tau_rec_p90"])
TARGETS = {"pt_median": "log_tau_rec_pt", "p90": "log_tau_rec_p90",
           "event_fit": "log_tau_rec_ev", "BFI": "baseflow_index"}
COVSETS = {"none": [],
           "climate": ["aridity", "p_mean", "frac_snow"],     # model inputs
           "wetness": ["runoff_ratio"],                      # observed signature: conservative
           "climate+wetness": ["aridity", "p_mean", "frac_snow", "runoff_ratio"]}


def partial_spearman(d, x, y, covs):
    """Spearman partial correlation via rank residualization."""
    r = d[[x, y] + covs].dropna().rank()
    if not covs:
        return stats.spearmanr(r[x], r[y]).statistic, len(r)
    Z = np.column_stack([np.ones(len(r))] + [r[c] for c in covs])
    res = lambda v: r[v] - Z @ np.linalg.lstsq(Z, r[v], rcond=None)[0]
    return np.corrcoef(res(x), res(y))[0, 1], len(r)


def boot_ci(d, x, y, covs, n=2000):
    d = d[[x, y] + covs].dropna()
    vals = [partial_spearman(d.iloc[rng.integers(0, len(d), len(d))], x, y, covs)[0] for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])

# ---- the full pre-specified table: estimator x covariate set x regime x seed ----
rows = []
for seed in [1, 2, 3, "mean"]:
    for tname, tcol in TARGETS.items():
        for cname, covs in COVSETS.items():
            for reg in ["all", "snow", "mixed", "humid-rain", "arid"]:
                d = df if reg == "all" else df[df.regime == reg]
                rho, n = partial_spearman(d, seed, tcol, covs)
                lo, hi = boot_ci(d, seed, tcol, covs, n=500 if reg != "all" else 2000)
                rows.append(dict(seed=seed, estimator=tname, covariates=cname, regime=reg,
                                 rho=rho, ci_lo=lo, ci_hi=hi, n=n))
tab = pd.DataFrame(rows)

# Holm correction over the seed-1 family actually reported (16 estimator x covset cells, pooled)
fam = tab[(tab.seed == 1) & (tab.regime == "all")].copy()
fam["p"] = [stats.t.sf(abs(r) * np.sqrt((n - 2) / (1 - r**2)), n - 2) * 2 for r, n in zip(fam.rho, fam.n)]
fam = fam.sort_values("p")
m = len(fam)
fam["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * fam["p"].values))
tab.to_csv(OUT_DIR / "correspondence_controlled.csv", index=False)
print(fam.round(3))


# ---------------- incremental-information test ----------------
y = df["log_tau_rec_p90"]
ok = y.notna()
X_stat = df.loc[ok, STATICS].values
X_tau = df.loc[ok, ["mean"]].values
groups = huc[ok.index[ok]].values
gkf = GroupKFold(n_splits=6)      # leave-HUC-regions-out: spatial blocking

def oof(make, X):
    pred = np.empty(ok.sum())
    for tr, te in gkf.split(X, y[ok], groups):
        mdl = make(); mdl.fit(X[tr], y[ok].values[tr]); pred[te] = mdl.predict(X[te])
    return pred

rf = lambda: RandomForestRegressor(500, min_samples_leaf=5, random_state=0, n_jobs=-1)
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
preds = {
    "statics_RF": oof(rf, X_stat),
    "statics+tau_RF": oof(rf, np.hstack([X_stat, X_tau])),
    "statics_ridge": oof(ridge, X_stat),
    "statics+tau_ridge": oof(ridge, np.hstack([X_stat, X_tau])),
    "tau_only": X_tau.ravel(),
}
yy = y[ok].values
for k, p in preds.items():
    print(f"{k:20s} OOF Spearman = {stats.spearmanr(p, yy).statistic:+.2f}")

# bootstrap the gain in OOF Spearman from adding tau
gain = []
for _ in range(2000):
    i = rng.integers(0, len(yy), len(yy))
    gain.append(stats.spearmanr(preds["statics+tau_RF"][i], yy[i]).statistic
                - stats.spearmanr(preds["statics_RF"][i], yy[i]).statistic)
print("Δρ from adding learned tau (RF):", np.percentile(gain, [2.5, 50, 97.5]).round(3))

# ===================== Correspondence by regime (replacement for the 531-dot scatter) =====================
# Does learned memory track observed recession, before and after removing climate + wetness?
# One dot per seed, bar = median over seeds, thin line = that seed's 95% CI.
import numpy as np, pandas as pd, matplotlib.pyplot as plt

EST = "p90"            # observed recession estimator: "p90", "pt_median", "event_fit" or "BFI"
corr = pd.read_csv(OUT_DIR / "correspondence_controlled.csv")
REG_ORDER = ["humid-rain", "mixed", "snow", "arid"]
REG_LABEL = {"humid-rain": "Humid, rain", "mixed": "Mixed", "snow": "Snow", "arid": "Arid"}
SETS = [("none", "as measured", "#9E9E9E"), ("climate+wetness", "after removing climate\n+ wetness", "#2171B5")]

fig, ax = plt.subplots(figsize=(9, 5.5))
d = corr[(corr.estimator == EST) & corr.regime.isin(REG_ORDER)]
for j, (cov, lab, col) in enumerate(SETS):
    for i, reg in enumerate(REG_ORDER):
        r = d[(d.regime == reg) & (d.covariates == cov)].sort_values("seed")
        x = i + (j - 0.5) * 0.32
        ax.bar(x, r.rho.median(), width=0.28, color=col, alpha=0.85, label=lab if i == 0 else None)
        for k, (_, row) in enumerate(r.iterrows()):
            xs = x + (k - 1) * 0.07
            ax.plot([xs, xs], [row.ci_lo, row.ci_hi], color="0.25", lw=1)
            ax.plot(xs, row.rho, "o", color="0.15", ms=4)
ax.axhline(0, color="k", lw=0.8)
n_by = d[d.covariates == "none"].groupby("regime").n.first()
ax.set_xticks(range(len(REG_ORDER)))
ax.set_xticklabels([f"{REG_LABEL[r]}\n(n = {n_by.get(r, '?')})" for r in REG_ORDER])
ax.set_ylabel("Rank correlation: model memory\nvs observed recession")
ax.legend(frameon=False, loc="upper right", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / f"nb10_correspondence_by_regime_{EST}.pdf", bbox_inches="tight")
