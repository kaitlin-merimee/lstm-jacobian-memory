"""nb09_sim_decomposition.py

Simulated-hydrograph decomposition (I->S->O path) and the low-flow version of the frozen hypothesis.
Outputs analysis/jacobians/sim_decomposition_seed{n}.csv, sim_decomposition_paths.csv
Recovered from jacobian_followup_experiments.md (Oct 2026). Items marked VERIFY must be
checked against the NeuralHydrology 1.13 install before trusting outputs.
"""
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from jlens_utils import OUT_DIR, latest_run, Model, recession_taus

base = pd.read_csv(OUT_DIR / "analysis_table.csv", dtype={"basin": str}).set_index("basin")


def lyne_hollick_bfi(q, alpha=0.925, passes=3):
    """Baseflow index via the Lyne–Hollick filter (forward/backward passes).
    Used identically on obs and sim, so the comparison is internally consistent even
    if it differs slightly from the CAMELS-provided baseflow_index (Ladson et al. variant)."""
    q = np.asarray(q, float)
    b = q.copy()
    for p in range(passes):
        src = b
        qf = np.zeros_like(src)
        rng_ = range(1, len(src)) if p % 2 == 0 else range(len(src) - 2, -1, -1)
        prev = 0 if p % 2 == 0 else len(src) - 1
        for t in rng_:
            qf[t] = alpha * qf[prev] + 0.5 * (1 + alpha) * (src[t] - src[prev])
            qf[t] = min(max(qf[t], 0.0), src[t])
            prev = t
        b = src - qf
    return np.nansum(b) / np.nansum(q)


def log_nse(o, s, eps_frac=0.01):
    eps = eps_frac * np.nanmean(o)
    lo, ls = np.log(o + eps), np.log(np.clip(s, 0, None) + eps)
    return 1 - np.nansum((lo - ls) ** 2) / np.nansum((lo - np.nanmean(lo)) ** 2)


def fdc_flv(o, s, l=0.3):
    """Yilmaz et al. (2008) low-flow volume bias (%), lowest 30% of the FDC, log space.
    VERIFY sign convention against neuralhydrology.evaluation.metrics.fdc_flv."""
    o = np.sort(o[~np.isnan(o)])[::-1]; s = np.sort(np.clip(s[~np.isnan(s)], 1e-6, None))[::-1]
    o = np.clip(o, 1e-6, None)
    no, ns = int(np.round((1 - l) * len(o))), int(np.round((1 - l) * len(s)))
    lo, ls = np.log(o[no:]), np.log(s[ns:])
    qo = np.sum(lo - lo.min()); qs = np.sum(ls - ls.min())
    return -1 * (qs - qo) / (qo + 1e-6) * 100


def nse(o, s):
    return 1 - np.nansum((o - s) ** 2) / np.nansum((o - np.nanmean(o)) ** 2)


def one_basin(model, b):
    o, s = model.obs_sim(b)
    m = o.notna() & s.notna()
    o, s = o[m].values, np.clip(s[m].values, 0, None)
    both_decl = (np.diff(o) < 0) & (np.diff(s) < 0)        # common-day mask variant
    r_o = recession_taus(o)
    r_s = recession_taus(s)
    r_sc = recession_taus(s, mask=both_decl)
    r_oc = recession_taus(o, mask=both_decl)
    return dict(basin=b,
                **{f"obs_{k}": v for k, v in r_o.items()},
                **{f"sim_{k}": v for k, v in r_s.items()},
                **{f"simC_{k}": v for k, v in r_sc.items()},
                **{f"obsC_{k}": v for k, v in r_oc.items()},
                bfi_obs=lyne_hollick_bfi(o), bfi_sim=lyne_hollick_bfi(s),
                nse=nse(o, s), log_nse=log_nse(o, s), flv=fdc_flv(o, s))

results = {}
for seed, pat, sweep in [(1, "camels531_daymet_h256_seed1*", "sweep_summary.csv"),
                         (2, "camels531_daymet_h256_seed2*", "sweep_summary_seed2.csv"),
                         (3, "camels531_daymet_h256_seed3*", "sweep_summary_seed3.csv")]:
    model = Model(latest_run(pat))
    rows = [one_basin(model, b) for b in base.index]     # fast; no unroll needed
    d = pd.DataFrame(rows).set_index("basin")
    d["tau_internal"] = pd.read_csv(OUT_DIR / sweep, dtype={"basin": str}).set_index("basin")["tau_readout_log10"]
    for k in ["tau_rec_pt", "tau_rec_p90", "tau_rec_ev"]:
        for pre in ["obs", "sim", "obsC", "simC"]:
            d[f"log_{pre}_{k}"] = np.log10(d[f"{pre}_{k}"])
    d = d.join(base[["regime", "karst_flag", "aridity", "frac_snow"]])
    d.to_csv(OUT_DIR / f"sim_decomposition_seed{seed}.csv")
    results[seed] = d


# ---------------- path decomposition ----------------
def sp(a, b): 
    m = a.notna() & b.notna(); return stats.spearmanr(a[m], b[m]).statistic

def partial(d, x, y, z):
    r = d[[x, y, z]].dropna().rank()
    Z = np.column_stack([np.ones(len(r)), r[z]])
    res = lambda v: r[v] - Z @ np.linalg.lstsq(Z, r[v], rcond=None)[0]
    return np.corrcoef(res(x), res(y))[0, 1]

out = []
for seed, d in results.items():
    for est in ["tau_rec_p90", "tau_rec_pt", "tau_rec_ev"]:
        for reg, dd in [("all", d)] + list(d.groupby("regime")):
            I, S, O = dd["tau_internal"], dd[f"log_sim_{est}"], dd[f"log_obs_{est}"]
            out.append(dict(seed=seed, est=est, regime=reg,
                            rho_IS=sp(I, S),            # is internal tau the model's output mechanism?
                            rho_SO=sp(S, O),            # does the model's output recession match reality?
                            rho_IO=sp(I, O),            # existing headline
                            rho_IO_given_S=partial(dd, "tau_internal", f"log_obs_{est}", f"log_sim_{est}"),
                            bfi_rho_SO=sp(dd.bfi_sim, dd.bfi_obs)))
paths = pd.DataFrame(out)
paths.to_csv(OUT_DIR / "sim_decomposition_paths.csv", index=False)
print(paths[paths.regime == "all"].round(2))


# ---------------- low-flow test (state the prediction BEFORE running):
# |delta| correlates negatively with log-NSE and positively with |FLV|, but not with NSE.
ht = []
for seed, d in results.items():
    d = d.copy()
    d["delta"] = d["log_sim_tau_rec_p90"] - d["log_obs_tau_rec_p90"]
    d["abs_delta"] = d["delta"].abs()
    d["abs_resid_q1"] = base["tau_readout_log10"].sub(   # old nb03 residual, for comparison
        np.polyval(np.polyfit(base["log_tau_rec_pt"], base["tau_readout_log10"], 1), base["log_tau_rec_pt"])).abs()
    for reg, dd in [("all", d)] + list(d.groupby("regime")):
        for fid in ["abs_delta", "abs_resid_q1"]:
            for skill in ["nse", "log_nse", "flv"]:
                y = dd[skill].abs() if skill == "flv" else dd[skill]   # |FLV| = size of low-flow bias
                ht.append(dict(seed=seed, regime=reg, fidelity=fid, skill=skill, rho=sp(dd[fid], y), n=len(dd)))
        # signed: does too-flashy (delta<0) produce negative low-flow bias?
        ht.append(dict(seed=seed, regime=reg, fidelity="delta_signed", skill="flv_signed",
                       rho=sp(dd["delta"], dd["flv"]), n=len(dd)))
ht = pd.DataFrame(ht)
print(ht.pivot_table(index=["fidelity", "skill"], columns=["regime", "seed"], values="rho").round(2))
