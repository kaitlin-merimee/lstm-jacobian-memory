"""nb08b_recovery.py — synthetic ground-truth recovery (nb08).

Does the Jacobian lens recover timescales we planted? For each scenario (S1, S2, S3, G) and basin:
  truth      planted tau_s / tau_f / phi_s (truth_params_{scen}.csv) and the generator's own lag
             sensitivity |dQ_T/dP_{T-k}| (autograd through the reservoir model, same windows as the LSTM)
  LSTM lens  forget-gate readout tau and band fractions (portfolio), input-gradient lag profile,
             state-propagator lag profile, recession tau of the LSTM's simulated flow
  baseline   recession tau of the synthetic "observed" flow itself

Self-contained: it does NOT import nb08a (importing nb08a reruns the whole generator and rewrites
data/synthetic/). The reservoir model is re-implemented here and checked against the saved
synthetic flows for every basin before anything is computed (gate: max |dQ| < 1e-8 mm/d).

Usage (from the project root, neuralhydrology env active):
  python src/nb08b_recovery.py --scen S1 --basins 40 --n-jobs 8      # quick test, ~5 min
  sbatch -A <your-account> -p <your-partition> -n 1 -c 16 -t 6:00:00 -J nb08b -o logs/nb08b_%j.out \
         --wrap "python src/nb08b_recovery.py --n-jobs 16"            # all scenarios, all basins
  python src/nb08b_recovery.py --summary-only                          # redo table + figures from CSVs

Outputs
  analysis/jacobians/followup/nb08b_basins_{scen}.csv   one row per basin
  analysis/jacobians/followup/nb08b_recovery.csv        one row per (scenario, estimator, truth)
  analysis/jacobians/followup/nb08b_summary.txt         the numbers for the slide
  analysis/figures/followup/nb08b_recovery_{scen}.png   planted vs recovered, one figure per scenario
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens_utils as J                                                     # noqa: E402

SYN_DIR = J.PROJECT / "data" / "synthetic"
OUT = J.OUT_DIR / "followup"
FIG = J.FIG_DIR / "followup"
SCENARIOS = ["S1", "S2", "S3", "G"]
RUN_PATTERN = "camels531_syn_{scen}_n00_h256_seed1_*"
TRUE_EVERY = 2          # true lag profile on every 2nd LSTM window (the generator is cheap; 2 is plenty)
NSE_GATE = 0.95         # below this the LSTM has not learned the synthetic flows; recovery is not meaningful


# ============================================================ model without test_results
class SynModel(J.Model):
    """jlens_utils.Model, but test_results.p is optional (training jobs don't run nh-run evaluate)."""

    def __init__(self, run_dir):
        import torch
        from neuralhydrology.utils.config import Config
        from neuralhydrology.datautils.utils import load_scaler
        self.run_dir = Path(run_dir)
        self.cfg = Config(self.run_dir / "config.yml")
        self.seq = self.cfg.seq_length
        self.dyn = list(self.cfg.dynamic_inputs)
        self.statics = list(self.cfg.static_attributes)
        self.n_dyn = len(self.dyn)
        self.weights = sorted(self.run_dir.glob("model_epoch*.pt"))[-1]
        sd = torch.load(self.weights, map_location="cpu")
        self.W_ih = sd["lstm.weight_ih_l0"].double().numpy()
        self.W_hh = sd["lstm.weight_hh_l0"].double().numpy()
        self.b = (sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]).double().numpy()
        hk = [k for k in sd if "head" in k and "weight" in k][0]
        self.w_out = sd[hk].double().numpy().squeeze()
        self.b_out = float(sd[hk.replace("weight", "bias")].double())
        self.w_abs = np.abs(self.w_out)
        self.H, self.D = self.W_hh.shape[1], self.W_ih.shape[1]
        self.scaler = load_scaler(self.run_dir)
        self.t = {k: torch.tensor(getattr(self, k)) for k in ("W_ih", "W_hh", "b", "w_out")}
        tgt = self.cfg.target_variables[0]
        self.target = tgt
        self.y_center = float(self.scaler["xarray_feature_center"][tgt].values)
        self.y_scale = float(self.scaler["xarray_feature_scale"][tgt].values)
        prcp = [i for i, n in enumerate(self.dyn) if "prcp" in n.lower()]
        assert len(prcp) == 1, f"expected exactly one precipitation input, got {self.dyn}"
        self.j_prcp = prcp[0]
        tr = sorted((self.run_dir / "test").glob("model_epoch*/test_results.p"))
        self.test_results = pickle.load(open(tr[-1], "rb")) if tr else None

    def basins(self):
        return [l.strip() for l in open(self.cfg.test_basin_file) if l.strip()]


# ============================================================ reservoir model (copy of nb08a.generate)
def generate(P, TM, p, x0=None, keep_states=False):
    """Identical equations to nb08a.generate. P, TM: torch [N, T]. p: dict of [N] arrays + 'snow'.
    keep_states=True also returns the (swe, qf, qs) trajectory after each day, each [N, T]."""
    import torch
    N, T = P.shape
    tf, ts = torch.as_tensor(p["tau_f"], dtype=P.dtype), torch.as_tensor(p["tau_s"], dtype=P.dtype)
    af, as_ = torch.exp(-1 / tf), torch.exp(-1 / ts)
    phi, c, ddf = (torch.as_tensor(p[k], dtype=P.dtype) for k in ("phi_s", "c", "ddf"))
    swe, qf, qs = (torch.zeros(N, dtype=P.dtype) for _ in range(3)) if x0 is None else x0
    Q, S = [], []
    for t in range(T):
        sf = torch.sigmoid(-TM[:, t]) if p["snow"] else torch.zeros_like(TM[:, t])
        swe = swe + P[:, t] * sf
        melt = torch.minimum(swe, ddf * torch.relu(TM[:, t])) if p["snow"] else torch.zeros_like(swe)
        swe = swe - melt
        u = c * (P[:, t] * (1 - sf) + melt)
        qf = af * qf + (1 - af) * (1 - phi) * u
        qs = as_ * qs + (1 - as_) * phi * u
        Q.append(qf + qs)
        if keep_states:
            S.append(torch.stack([swe, qf, qs]))
    Q = torch.stack(Q, 1)
    if keep_states:
        return Q, torch.stack(S, 2)          # [3, N, T]
    return Q, (swe, qf, qs)


def true_lag_profile(P1, TM1, p1, states, end_pos, seq):
    """Mean |dQ_T/dP_{T-k}| over windows ending at end_pos, all windows batched in one backward pass.
    P1, TM1: torch [T] one basin; states: [3, T] trajectory from the full run; lag-indexed (0 = same day)."""
    import torch
    W = len(end_pos)
    Pw = torch.stack([P1[e - seq + 1:e + 1] for e in end_pos]).clone().requires_grad_(True)
    Tw = torch.stack([TM1[e - seq + 1:e + 1] for e in end_pos])
    x0 = tuple(torch.stack([states[j, e - seq] for e in end_pos]).detach() for j in range(3))
    pw = {k: (np.repeat(v, W) if isinstance(v, np.ndarray) else v) for k, v in p1.items()}
    Q, _ = generate(Pw, Tw, pw, x0=x0)
    Q[:, -1].sum().backward()
    return np.abs(Pw.grad.numpy())[:, ::-1].mean(0)


def read_basin_forcing(data_dir, basin, idx):
    """P and mean temperature on the generator's index, exactly as nb08a built them."""
    from neuralhydrology.datasetzoo.camelsus import load_camels_us_forcings
    df, _ = load_camels_us_forcings(data_dir, basin, "daymet")
    df = df.reindex(idx)
    assert df[["prcp(mm/day)", "tmax(C)", "tmin(C)"]].notna().all().all(), f"{basin}: forcing gaps on idx"
    return df["prcp(mm/day)"].values, ((df["tmax(C)"] + df["tmin(C)"]) / 2).values


# ============================================================ LSTM side
def state_propagator_profile(mdl, X):
    """jlens_utils.state_propagator_profile with one fix: the unroll must track gradients
    (input_grad=True), otherwise c.retain_grad() fails because nothing in the graph requires grad."""
    import torch
    y, _, cs = mdl.unroll_torch(X, input_grad=True, keep_c=True)
    y.sum().backward()
    G = torch.stack([c.grad for c in cs], dim=1).numpy()
    C = torch.stack(cs, dim=1).detach().numpy()
    return np.abs(G * C.std(axis=1, keepdims=True)).sum(-1).mean(0)[::-1]


def continuous_X(mdl, ds):
    """[T, D] inputs for the whole test period: first window in full, then the last day of each window."""
    first = mdl.build_X(ds, [0])[0]
    rows = [first]
    for k in range(1, len(ds)):
        rows.append(mdl.build_X(ds, [k])[0][-1:])
    X = np.concatenate(rows, 0)
    for k in (1, len(ds) // 2, len(ds) - 1):                   # windows must overlap by seq-1 days
        w = mdl.build_X(ds, [k])[0]
        assert np.allclose(w, X[k:k + mdl.seq]), f"window {k} does not line up with the stitched series"
    return X


def window_end_dates(ds, mdl, n):
    """End date of each sample. Uses the sample's own 'date' when NeuralHydrology provides it."""
    s = ds[0]
    if "date" in s:
        return pd.DatetimeIndex([pd.Timestamp(np.asarray(ds[k]["date"])[-1]) for k in range(n)])
    start = pd.Timestamp(mdl.cfg.test_start_date)
    return pd.date_range(start, periods=n, freq="D")


def unroll_continuous(mdl, X):
    """Zero-initialised run through X (365 warm-up days included). Returns y in mm/d, [T]."""
    h = np.zeros(mdl.H); c = np.zeros(mdl.H); XW = X @ mdl.W_ih.T + mdl.b; y = np.empty(len(X))
    for t in range(len(X)):
        pre = XW[t] + mdl.W_hh @ h
        ai, af, ag, ao = np.split(pre, 4)
        c = 1 / (1 + np.exp(-af)) * c + 1 / (1 + np.exp(-ai)) * np.tanh(ag)
        h = 1 / (1 + np.exp(-ao)) * np.tanh(c)
        y[t] = h @ mdl.w_out + mdl.b_out
    return y * mdl.y_scale + mdl.y_center


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    return 1 - np.sum((s[m] - o[m]) ** 2) / np.sum((o[m] - o[m].mean()) ** 2)


# ============================================================ one basin
def analyze(scen, run_dir, b, q_syn, p1):
    import torch
    torch.set_num_threads(1)
    if run_dir not in MODELS:
        MODELS[run_dir] = SynModel(run_dir)
    mdl = MODELS[run_dir]
    t0 = time.time()

    # ---- generator gate: our copy of the reservoir model must reproduce the saved flows ----
    idx = q_syn.index
    P_np, TM_np = read_basin_forcing(mdl.cfg.data_dir, b, idx)
    P1, TM1 = torch.tensor(P_np), torch.tensor(TM_np)
    with torch.no_grad():
        Qg, states = generate(P1[None], TM1[None], p1, keep_states=True)
    gen_err = float(np.abs(Qg[0].numpy() - q_syn.values).max())
    states = states[:, 0]

    # ---- LSTM lens on stride windows (as nb02/nb10) ----
    ds = mdl.dataset(b)
    idxs = mdl.window_idxs(ds)
    fw, prof_in, prof_st, y_win = [], [], [], []
    for c0 in range(0, len(idxs), 16):
        X = mdl.build_X(ds, idxs[c0:c0 + 16])
        g, yw = mdl.unroll_np(X); fw.append(g["f"]); y_win.append(yw)
        prof_in.append(J.lag_profile_input(mdl, X, j=mdl.j_prcp) * X.shape[0])
        prof_st.append(state_propagator_profile(mdl, X) * X.shape[0])
    prof_in = np.sum(prof_in, 0) / len(idxs); prof_st = np.sum(prof_st, 0) / len(idxs)
    port, _ = J.portfolio(mdl, np.concatenate(fw, 0))

    # ---- continuous simulation over the test period (for NSE and recession of the LSTM's flow) ----
    ends = window_end_dates(ds, mdl, len(ds))
    Xc = continuous_X(mdl, ds)
    yc = unroll_continuous(mdl, Xc)[mdl.seq - 1:]
    sim = pd.Series(yc, index=ends)
    win_gap = float(np.abs(sim.values[idxs] - (np.concatenate(y_win) * mdl.y_scale + mdl.y_center)).max())
    obs = q_syn.reindex(ends)

    # ---- truth: generator lag sensitivity on the same window ends ----
    end_pos = idx.get_indexer(ends[idxs])
    assert (end_pos >= mdl.seq).all(), "window starts before the generator spin-up"
    tprof = true_lag_profile(P1, TM1, p1, states, end_pos[::TRUE_EVERY], mdl.seq)

    row = dict(basin=b, scen=scen, gen_err=gen_err, nse_syn=nse(obs.values, sim.values),
               win_vs_cont_mm=win_gap, n_windows=len(idxs), secs=time.time() - t0, **port)
    row.update({f"in_{k}": v for k, v in J.profile_stats(prof_in).items()})
    row.update({f"state_{k}": v for k, v in J.profile_stats(prof_st).items()})
    row.update({f"true_{k}": v for k, v in J.profile_stats(tprof).items()})
    row.update({f"rec_{k}": v for k, v in J.recession_taus(obs.values).items()})
    row.update({f"recsim_{k}": v for k, v in J.recession_taus(np.clip(sim.values, 0, None)).items()})
    return row


MODELS = {}


# ============================================================ recovery table and figures
def recover(df, est, truth, log=True):
    from scipy import stats
    a, t = df[est].astype(float), df[truth].astype(float)
    if log:
        a, t = np.log10(a.clip(lower=1e-3)), np.log10(t.clip(lower=1e-3))
    m = a.notna() & t.notna() & np.isfinite(a) & np.isfinite(t)
    if m.sum() < 10 or t[m].nunique() < 3:
        return dict(n=int(m.sum()), rho=np.nan, rho_lo=np.nan, rho_hi=np.nan, slope=np.nan, bias=np.nan)
    if a[m].nunique() < 3:
        return dict(n=int(m.sum()), rho=np.nan, rho_lo=np.nan, rho_hi=np.nan, slope=np.nan, bias=float(np.median(a[m] - t[m])))
    warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
    rho = stats.spearmanr(a[m], t[m]).statistic
    rng = np.random.default_rng(0); av, tv = a[m].values, t[m].values
    boot = [stats.spearmanr(av[i], tv[i]).statistic for i in (rng.integers(0, len(av), len(av)) for _ in range(500))]
    slope = stats.linregress(tv, av).slope
    return dict(n=int(m.sum()), rho=rho, rho_lo=np.percentile(boot, 2.5), rho_hi=np.percentile(boot, 97.5),
                slope=slope, bias=float(np.median(av - tv)))


def pairs_for(scen):
    p = [("tau_readout_days", "tau_s", True, "forget-gate readout τ (d)", "planted slow τ (d)"),
         ("state_mean_lag", "true_mean_lag", False, "state-propagator mean lag (d)", "true mean lag (d)"),
         ("in_mean_lag", "true_mean_lag", False, "input-gradient mean lag (d)", "true mean lag (d)"),
         ("in_M50", "true_M50", False, "input-gradient M50 (d)", "true M50 (d)"),
         ("rec_tau_rec_p90", "tau_s", True, "p90 recession τ of synthetic flow (d)", "planted slow τ (d)"),
         ("recsim_tau_rec_p90", "tau_s", True, "p90 recession τ of LSTM flow (d)", "planted slow τ (d)")]
    if scen != "S1":
        p.append(("readout_frac_quick", "fast_frac_true", False, "readout share on <3-d units", "planted fast fraction"))
        p.append(("state_mean_lag", "tau_s", True, "state-propagator mean lag (d)", "planted slow τ (d)"))
    return p


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, lines = [], ["nb08b synthetic recovery  (" + time.strftime("%Y-%m-%d %H:%M") + ")", ""]
    for scen in SCENARIOS:
        f = OUT / f"nb08b_basins_{scen}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"basin": str})
        d["tau_readout_days"] = 10 ** d["tau_readout_log10"]
        d["fast_frac_true"] = 1 - d["phi_s"]
        ok = d.nse_syn >= NSE_GATE
        lines.append(f"{scen}: {len(d)} basins | median synthetic NSE {d.nse_syn.median():.3f} "
                     f"({ok.mean():.0%} ≥ {NSE_GATE}) | generator check max {d.gen_err.max():.1e} mm/d | "
                     f"window vs continuous sim max {d.win_vs_cont_mm.max():.1e} mm/d")
        if d.nse_syn.median() < NSE_GATE:
            lines.append(f"   !! median NSE below {NSE_GATE}: the LSTM did not learn these flows; read the rest with care")
        for est, tr, lg, *_ in pairs_for(scen):
            for subset, dd in (("all", d), ("NSE≥gate", d[ok])):
                r = recover(dd, est, tr, log=lg)
                rows.append(dict(scen=scen, estimator=est, truth=tr, subset=subset, log=lg, **r))
                if subset == "all":
                    lines.append(f"   {est:22s} vs {tr:14s} ρ = {r['rho']:+.2f} [{r['rho_lo']:+.2f}, {r['rho_hi']:+.2f}]"
                                 f"  slope {r['slope']:+.2f}  bias {r['bias']:+.2f}{' (log10)' if lg else ''}  n={r['n']}")
        lines.append("")

        prs = pairs_for(scen)
        fig, ax = plt.subplots(2, (len(prs) + 1) // 2, figsize=(4 * ((len(prs) + 1) // 2), 7.5))
        for a, (est, tr, lg, xl, tl) in zip(ax.flat, prs):
            x, y = d[tr].astype(float), d[est].astype(float)
            a.scatter(x, y, s=8, alpha=.5, c=np.where(ok, "tab:blue", "tab:red"), lw=0)
            if lg:
                a.set_xscale("log"); a.set_yscale("log")
            fin = np.isfinite(x) & np.isfinite(y) & ((x > 0) & (y > 0) if lg else True)
            if fin.sum():
                lo, hi = min(x[fin].min(), y[fin].min()), max(x[fin].max(), y[fin].max())
                a.plot([lo, hi], [lo, hi], "k--", lw=.8)
            r = recover(d, est, tr, log=lg)
            a.set(xlabel=tl, ylabel=xl, title=f"ρ = {r['rho']:.2f}")
        for a in list(ax.flat)[len(prs):]:
            a.axis("off")
        fig.suptitle(f"{scen}: planted (x) vs recovered (y), {len(d)} basins; red = synthetic NSE < {NSE_GATE}; dashed = 1:1")
        fig.tight_layout(); fig.savefig(FIG / f"nb08b_recovery_{scen}.png", dpi=200); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "nb08b_recovery.csv", index=False)
    (OUT / "nb08b_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scen", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--basins", type=int, default=0, help="random subset of N basins (quick test)")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        return summarize()
    from joblib import Parallel, delayed

    for scen in args.scen:
        runs = sorted((J.PROJECT / "runs").glob(RUN_PATTERN.format(scen=scen)))
        if not runs:
            print(f"[skip] no run for {scen}"); continue
        run_dir = runs[-1]
        mdl = SynModel(run_dir)
        print(f"[{scen}] run {run_dir.name} | weights {mdl.weights.name} | target {mdl.target} | "
              f"prcp input #{mdl.j_prcp} = {mdl.dyn[mdl.j_prcp]}", flush=True)
        truth = pd.read_csv(SYN_DIR / f"truth_params_{scen}.csv", index_col=0)
        truth.index = truth.index.astype(str).str.zfill(8)
        with open(SYN_DIR / f"syn_{scen}_n00.p", "rb") as f:
            syn = pickle.load(f)
        basins = [b for b in mdl.basins() if b in truth.index and b in syn]
        if args.basins:
            basins = sorted(np.random.default_rng(0).choice(basins, args.basins, replace=False))
        snow = scen in ("S3", "G")
        jobs = []
        for b in basins:
            p1 = {k: truth.loc[b, [k]].values.astype(float) for k in ("c", "tau_f", "tau_s", "phi_s", "ddf")}
            p1["snow"] = snow
            jobs.append(delayed(analyze)(scen, run_dir, b, syn[b]["q_syn"], p1))
        t0 = time.time()
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)
        d = pd.DataFrame(rows).set_index("basin").join(truth)
        bad = d.gen_err > 1e-8
        if bad.any():
            print(f"[{scen}] !! generator copy disagrees with saved flows in {bad.sum()} basins "
                  f"(max {d.gen_err.max():.1e}); truth profiles for those are not trustworthy")
        d.to_csv(OUT / f"nb08b_basins_{scen}.csv")
        print(f"[{scen}] {len(d)} basins in {(time.time() - t0) / 60:.1f} min; "
              f"median NSE {d.nse_syn.median():.3f}", flush=True)
    summarize()


if __name__ == "__main__":
    main()
