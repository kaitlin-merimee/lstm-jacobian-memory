"""nb15_all_basins.py — windowed state memory and discharge reach-back for every CAMELS basin.

Same analysis as nb15 Cell 5 (and nb14 Cells 11–12), one basin at a time, written to the SAME cache
(analysis/lyapunov/nb15_cache/{gauge}.p, same settings record), so nb15 Cell 5b loads everything.
Basins already cached with identical settings are skipped, so a re-submitted or timed-out task resumes.

  compute:    sbatch slurm/backup/nb15_all.slurm                       (array job; one chunk of basins per task)
  one task:   python src/backup/nb15_all_basins.py --task 0 --ntasks 54
  summarize:  python src/backup/nb15_all_basins.py --summarize          (after all tasks finish; CPU, ~1 min)

Summary outputs (analysis/lyapunov/):
  nb15_all_basins.csv        one row per basin: attributes, regime, reach-back by season, state memory, visibility,
                             and the flow-level check (below)
  nb15_all_month_regime.csv  median reach-back at 90 d by month of the predicted day and regime
  nb15_all_reach90.png       reach-back by month per regime (raw, and after removing each basin's flow-level effect)

Flow-level check: within each basin, log(reach-back at 90 d) is regressed on log(simulated Q) across windows.
The residual's seasonal amplitude (max − min of monthly medians, in log10 units) is what is left of the seasonal
pattern once flow level is accounted for. If snow basins keep a large residual amplitude and rain basins do not,
the melt-season reach-back is not just "high flow makes everything more sensitive".
"""
import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT    = Path(".")
RUN_DIR = ROOT / "runs/camels531_daymet_h256_seed1_1207_194330"
OUT     = ROOT / "analysis/lyapunov"
CACHE   = OUT / "nb15_cache"
PERIOD  = ("01/10/2008", "30/09/2014")
STRIDE_W, K_W, KEEP = 10, 64, 180
LAGS = (30, 90, 180, 300)
SEASONS = [("Dec–Mar", [12, 1, 2, 3]), ("Apr–Jun", [4, 5, 6]), ("Jul–Sep", [7, 8, 9]), ("Oct–Nov", [10, 11])]
META = dict(run=RUN_DIR.name, period=PERIOD, stride=STRIDE_W, k=K_W, keep=KEEP, lags=LAGS)   # must match nb15

# globals filled by load_model()
H = D = None
W_ih = W_hh = b = w_head = None
b_head = y_scale = y_center = 0.0
CTX = {}

# ---------------------------------------------------------------- core functions (identical to nb14/nb15)
def sig(z): return 1/(1+np.exp(-z))

def unroll(X, h0=None, c0=None):
    '''Continuous run. Returns per-day gates and states; hp/cp = state going INTO day t.'''
    T = len(X); h = np.zeros(H) if h0 is None else h0.copy(); c = np.zeros(H) if c0 is None else c0.copy()
    S = {k: np.zeros((T, H)) for k in ["h", "c", "i", "f", "g", "o", "hp", "cp"]}
    XW = X @ W_ih.T + b
    for t in range(T):
        z = XW[t] + W_hh @ h
        i, f, g, o = sig(z[:H]), sig(z[H:2*H]), np.tanh(z[2*H:3*H]), sig(z[3*H:])
        S["hp"][t], S["cp"][t] = h, c
        c = f*c + i*g; h = o*np.tanh(c)
        S["i"][t], S["f"][t], S["g"][t], S["o"][t], S["h"][t], S["c"][t] = i, f, g, o, h, c
    return S

def unroll_windows(X, ends, L):
    '''Zero-initialised run over each window X[e-L+1 : e+1], all windows at once (what NeuralHydrology does).'''
    n = len(ends); h = np.zeros((n, H)); c = np.zeros((n, H))
    for s in range(L):
        z = X[ends - L + 1 + s] @ W_ih.T + b + h @ W_hh.T
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2*H]), np.tanh(z[:, 2*H:3*H]), sig(z[:, 3*H:])
        c = f*c + i*g; h = o*np.tanh(c)
    return h, c

def step_jacobian(S, t):
    '''Closed form d[h_t; c_t] / d[h_{t-1}; c_{t-1}]  (2H x 2H).'''
    Wi, Wf, Wg, Wo = W_hh[:H], W_hh[H:2*H], W_hh[2*H:3*H], W_hh[3*H:]
    i, f, g, o, c, cp = (S[k][t] for k in ["i", "f", "g", "o", "c", "cp"])
    tc = np.tanh(c); a = o*(1 - tc**2)
    dc_dh = (cp*f*(1-f))[:, None]*Wf + (g*i*(1-i))[:, None]*Wi + (i*(1-g**2))[:, None]*Wg
    dh_dh = (tc*o*(1-o))[:, None]*Wo + a[:, None]*dc_dh
    return np.block([[dh_dh, np.diag(a*f)], [dc_dh, np.diag(f)]])

def qr_step(J, Q):
    '''One Benettin step: returns new Q and the per-direction log stretch ln R_kk (signs fixed so R_kk > 0).'''
    Q, R = np.linalg.qr(J @ Q)
    s = np.sign(np.diag(R)); s[s == 0] = 1
    return Q*s, np.log(np.maximum(np.abs(np.diag(R)), 1e-300))

def to_tau(lam):
    '''Effective memory in days; inf where lambda >= 0 (no decay).'''
    lam = np.asarray(lam, float)
    return np.where(lam < 0, -1/np.minimum(lam, -1e-12), np.inf)

# ---------------------------------------------------------------- model and inputs
def load_model(toy=False):
    global H, D, W_ih, W_hh, b, w_head, b_head, y_scale, y_center
    if toy:
        rng = np.random.default_rng(1); H, D = 16, 6
        W_ih = rng.normal(0, .5, (4*H, D)); W_hh = rng.normal(0, .25, (4*H, H)); b = rng.normal(0, .5, 4*H); b[H:2*H] += 3
        w_head, b_head, y_scale, y_center = rng.normal(0, .3, H), 0.0, 1.0, 0.0
        g = [f"0{i}000000" for i in range(1, 9)]
        CTX["basins"] = g
        CTX["attrs"] = pd.DataFrame({"frac_snow": np.linspace(0, .8, 8), "aridity": np.linspace(2.5, .4, 8),
                                     "runoff_ratio": .3, "p_mean": 3.0}, index=g)
        return
    import torch
    from neuralhydrology.utils.config import Config
    from neuralhydrology.modelzoo import get_model
    from neuralhydrology.datautils.utils import load_scaler
    torch.set_num_threads(1)
    cfg = Config(RUN_DIR / "config.yml")
    cfg.update_config({"test_start_date": PERIOD[0], "test_end_date": PERIOD[1], "device": "cpu"})
    model = get_model(cfg).to("cpu")
    model.load_state_dict(torch.load(sorted(RUN_DIR.glob("model_epoch*.pt"))[-1], map_location="cpu")); model.eval()
    lstm = model.lstm
    W_ih = lstm.weight_ih_l0.detach().double().numpy(); W_hh = lstm.weight_hh_l0.detach().double().numpy()
    b = (lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().double().numpy()
    lin = [m for m in model.head.modules() if isinstance(m, torch.nn.Linear)]; assert len(lin) == 1
    w_head = lin[0].weight.detach().double().numpy()[0]; b_head = float(lin[0].bias.detach())
    H, D = W_hh.shape[1], W_ih.shape[1]
    scaler = load_scaler(RUN_DIR); tgt = cfg.target_variables[0]
    y_scale = float(scaler["xarray_feature_scale"][tgt].values); y_center = float(scaler["xarray_feature_center"][tgt].values)
    CTX.update(cfg=cfg, model=model, scaler=scaler, torch=torch,
               basins=[l.strip() for l in open(cfg.test_basin_file) if l.strip()])


def basin_inputs(gauge):
    if "model" not in CTX:   # toy
        rng = np.random.default_rng(int(gauge) % 997)
        dates = pd.date_range(pd.to_datetime(PERIOD[0], dayfirst=True) - pd.Timedelta(days=364), pd.to_datetime(PERIOD[1], dayfirst=True))
        doy = dates.dayofyear.values; snow = CTX["attrs"].loc[gauge, "frac_snow"]
        X = np.c_[np.sin(2*np.pi*doy/365.25), rng.gamma(.3, 3, len(dates)) * (1 + snow*np.cos(2*np.pi*doy/365.25)),
                  rng.normal(0, 1, (len(dates), D-2))]
        return X, dates
    from neuralhydrology.datasetzoo import get_dataset
    torch, cfg, model = CTX["torch"], CTX["cfg"], CTX["model"]
    ds = get_dataset(cfg, is_train=False, basin=gauge, period="test", scaler=CTX["scaler"])
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)
    rows, first = [], True
    with torch.no_grad():
        for batch in loader:
            xb = model.embedding_net(batch).double().numpy()
            if first:
                rows.append(xb[:-1, 0, :])
                yh = model(batch)["y_hat"][:, -1, 0].double().numpy()
                for j in (0, min(100, xb.shape[1]-1)):
                    S1 = unroll(xb[:, j, :]); err = abs(S1["h"][-1] @ w_head + b_head - yh[j])
                    assert err < 1e-4, f"{gauge}: numpy vs model {err:.1e}"
                assert np.allclose(xb[1:, 0, :], xb[:-1, 1, :]), f"{gauge}: windows do not overlap"
                first = False
            rows.append(xb[-1, :, :])
    X = np.concatenate(rows, 0)
    start = pd.to_datetime(PERIOD[0], dayfirst=True) - pd.Timedelta(days=cfg.seq_length - 1)
    dates = pd.date_range(start, pd.to_datetime(PERIOD[1], dayfirst=True), freq="D")
    assert len(dates) == len(X) and np.isfinite(X).all(), f"{gauge}: {len(X)} rows vs {len(dates)} days, or NaNs"
    return X, dates


def analyze_basin(gauge):
    # identical to nb15 Cell 5
    X, dates = basin_inputs(gauge)
    ends = np.arange(364, len(X))[::STRIDE_W]
    lam, gn, vis, qsim, cmax = [], [], {k: [] for k in LAGS}, [], 0.0
    for e in ends:
        Sw = unroll(X[e-364:e+1]); T_w = len(Sw["f"]); cmax = max(cmax, np.abs(Sw["c"]).max())
        qsim.append((Sw["h"][-1] @ w_head + b_head) * y_scale + y_center)
        Js = [step_jacobian(Sw, t) for t in range(T_w)]
        kw = min(K_W, 2*H)
        Q = np.linalg.qr(np.random.default_rng(0).normal(size=(2*H, kw)))[0]
        L = np.zeros((T_w, kw)); Qs = {}
        for t in range(T_w):
            Q, L[t] = qr_step(Js[t], Q)
            if T_w - 1 - t in LAGS: Qs[T_w - 1 - t] = Q[:, :10].copy()
        lam.append(L[-KEEP:].mean(0))
        v = np.r_[w_head, np.zeros(H)]; g = np.zeros(T_w)
        for k in range(T_w):
            g[k] = np.linalg.norm(v)
            if k in Qs: vis[k].append(np.linalg.norm(v @ Qs[k])**2 / max(g[k]**2, 1e-300))
            v = v @ Js[T_w - 1 - k]
        gn.append(g)
    gn = np.array(gn)
    return dict(end_dates=dates[ends], tau=to_tau(np.median(lam, 0)), rel=gn / gn[:, :1],
                vis={k: float(np.median(v_)) for k, v_ in vis.items()}, qsim=np.array(qsim), cmax=cmax)


def is_cached(gauge):
    f = CACHE / f"{gauge}.p"
    if not f.exists():
        return False
    try:
        return pickle.load(open(f, "rb"))["meta"] == META
    except Exception:
        return False


def compute(task, ntasks, toy):
    load_model(toy)
    mine = CTX["basins"][task::ntasks]
    print(f"task {task}/{ntasks}: {len(mine)} basins", flush=True)
    for g in mine:
        if is_cached(g):
            print(f"  {g}: cached, skip", flush=True); continue
        t0 = time.time()
        try:
            res = analyze_basin(g)
        except Exception as ex:                       # one bad basin must not kill the chunk
            print(f"  {g}: FAILED {type(ex).__name__}: {ex}", flush=True); continue
        f = CACHE / f"{g}.p"; tmp = f.with_suffix(f".tmp{task}")
        pickle.dump({"meta": META, "res": res}, open(tmp, "wb")); tmp.replace(f)
        print(f"  {g}: {len(res['end_dates'])} windows in {time.time()-t0:.0f} s", flush=True)


# ---------------------------------------------------------------- summary across basins
def regime(row):
    # same rule as jlens_utils.regime
    if row.frac_snow >= 0.3: return "snow"
    if row.aridity >= 1.5 and row.frac_snow < 0.15: return "arid"
    if row.frac_snow < 0.15: return "humid-rain"
    return "mixed"


def summarize(toy):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if toy:
        load_model(True); attrs = CTX["attrs"]
    else:
        from neuralhydrology.utils.config import Config
        from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes
        cfg = Config(RUN_DIR / "config.yml")
        basins = [l.strip() for l in open(cfg.test_basin_file) if l.strip()]
        attrs = load_camels_us_attributes(cfg.data_dir, basins=basins)
    rows, monthly, monthly_res = [], [], []
    files = sorted(CACHE.glob("*.p"))
    for f in files:
        d = pickle.load(open(f, "rb"))
        if d["meta"] != META: continue
        g, r = f.stem, d["res"]
        if g not in attrs.index: continue
        mon = pd.DatetimeIndex(r["end_dates"]).month
        r90 = np.log10(np.clip(r["rel"][:, 90], 1e-12, None))
        lq = np.log10(np.clip(r["qsim"], 0, None) + 0.01)
        slope, icpt = np.polyfit(lq, r90, 1) if np.ptp(lq) > 0 else (0.0, r90.mean())
        resid = r90 - (icpt + slope * lq)
        m_raw = pd.Series(r90, index=mon).groupby(level=0).median().reindex(range(1, 13))
        m_res = pd.Series(resid, index=mon).groupby(level=0).median().reindex(range(1, 13))
        a = attrs.loc[g]
        row = {"gauge": g, "regime": regime(a), "frac_snow": a.frac_snow, "aridity": a.aridity,
               "n_windows": len(mon), "dirs_gt_1yr": int((r["tau"] > 365).sum()), "dirs_gt_30d": int((r["tau"] > 30).sum()),
               "visibility_slow10_90d_pct": 100 * r["vis"][90],
               "rho_reach90_vs_logQ": pd.Series(r90).corr(pd.Series(lq), method="spearman"),
               "slope_reach90_on_logQ": slope,
               "season_amp_raw_log10": float(m_raw.max() - m_raw.min()),
               "season_amp_after_flow_log10": float(m_res.max() - m_res.min()),
               "peak_month_raw": int(m_raw.idxmax()), "peak_month_after_flow": int(m_res.idxmax())}
        for s, ms in SEASONS:
            m = np.isin(mon, ms)
            for lag in (30, 90, 180):
                row[f"{s}_{lag}d_pct"] = 100 * float(np.median(r["rel"][m][:, lag])) if m.sum() else np.nan
        rows.append(row)
        monthly.append(pd.Series(10 ** m_raw.values, index=range(1, 13), name=g))
        monthly_res.append(pd.Series(m_res.values, index=range(1, 13), name=g))
    if not rows:
        print(f"no cached basins with current settings in {CACHE}"); return
    T = pd.DataFrame(rows).set_index("gauge")
    T.to_csv(OUT / "nb15_all_basins.csv")
    M = pd.concat(monthly, axis=1).T; M["regime"] = T.regime
    MR = pd.concat(monthly_res, axis=1).T; MR["regime"] = T.regime
    (100 * M.groupby("regime").median()).round(3).to_csv(OUT / "nb15_all_month_regime.csv")
    print(f"{len(T)} basins summarized ({len(files)} cache files)")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(T.groupby("regime")[["dirs_gt_1yr", "visibility_slow10_90d_pct", "rho_reach90_vs_logQ",
                               "season_amp_raw_log10", "season_amp_after_flow_log10"]].median().round(3).assign(n=T.regime.value_counts()))

    order = [10, 11, 12] + list(range(1, 10)); cols = {"snow": "tab:blue", "mixed": "tab:purple", "humid-rain": "tab:green", "arid": "tab:orange"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for reg, col in cols.items():
        sel = M.regime == reg
        if not sel.any(): continue
        med = M[sel].drop(columns="regime").median().loc[order]
        lo, hi = M[sel].drop(columns="regime").quantile([.25, .75]).loc[:, order].values
        ax[0].fill_between(range(12), 100*lo, 100*hi, color=col, alpha=.15)
        ax[0].plot(range(12), 100*med.values, "o-", color=col, label=f"{reg} (n={sel.sum()})")
        rm = MR[sel].drop(columns="regime").median().loc[order]
        ax[1].plot(range(12), rm.values, "o-", color=col, label=reg)
    ax[0].set_yscale("log"); ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax[0].set(ylabel="sensitivity to the state 90 d earlier\n(% of same day; median, IQR shaded)", title="Reach-back by month, all basins")
    ax[1].axhline(0, color="0.6", lw=.8)
    ax[1].set(ylabel="log10 residual after each basin's\nflow-level fit (median)", title="Same, after removing the flow-level effect")
    for a in ax: a.set(xticks=range(12), xticklabels=list("ONDJFMAMJJAS"), xlabel="month of the predicted day"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "nb15_all_reach90.png", dpi=200)
    print("wrote", OUT / "nb15_all_basins.csv", "and", OUT / "nb15_all_reach90.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--toy", action="store_true", help="random small model, fake basins, writes under ./toy_out (testing)")
    a = ap.parse_args()
    if a.toy:
        OUT = Path("toy_out"); CACHE = OUT / "nb15_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    summarize(a.toy) if a.summarize else compute(a.task, a.ntasks, a.toy)
