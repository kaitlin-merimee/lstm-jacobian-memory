"""nb08a_generate_synthetic.py

Synthetic ground-truth generator: S1/S2/S3/G scenarios from real Daymet forcings with planted timescales.
Writes data/synthetic/syn_{scen}_n{noise}.p and truth_params_{scen}.csv
Recovered from jacobian_followup_experiments.md (Oct 2026). Items marked VERIFY must be
checked against the NeuralHydrology 1.13 install before trusting outputs.
"""
import numpy as np, pandas as pd, torch, pickle
from neuralhydrology.datasetzoo.camelsus import load_camels_us_forcings, load_camels_us_attributes  # VERIFY names in 1.13
from jlens_utils import PROJECT, latest_run, Model

m1 = Model(latest_run("camels531_daymet_h256_seed1*"))
DATA = m1.cfg.data_dir
BASINS = sorted(m1.test_results.keys())
SYN_DIR = PROJECT / "data" / "synthetic"; SYN_DIR.mkdir(parents=True, exist_ok=True)

# ---- forcings for all basins on a common daily index ----
forc = {}
for b in BASINS:
    df, _area = load_camels_us_forcings(DATA, b, "daymet")
    forc[b] = df[["prcp(mm/day)", "tmax(C)", "tmin(C)"]]
idx = forc[BASINS[0]].index
for b in BASINS:
    idx = idx.intersection(forc[b].index)
P = torch.tensor(np.stack([forc[b].loc[idx, "prcp(mm/day)"].values for b in BASINS]))           # [N, T]
TM = torch.tensor(np.stack([((forc[b].loc[idx, "tmax(C)"] + forc[b].loc[idx, "tmin(C)"]) / 2).values for b in BASINS]))

attrs = load_camels_us_attributes(DATA, basins=BASINS).loc[BASINS].select_dtypes("number")
z = (attrs - attrs.mean()) / attrs.std()
USED = ["aridity", "slope_mean", "soil_porosity", "soil_depth_pelletier", "frac_forest", "sand_frac", "carbonate_rocks_frac", "geol_permeability", "elev_mean"]
assert not z[USED].isna().any().any(), z[USED].isna().sum()[lambda s: s > 0]


def params_for(scen):
    clip = lambda v, lo, hi: np.clip(v, lo, hi)
    c = clip(0.9 - 0.35 * attrs["aridity"].values, 0.05, 0.9)
    p = {"c": c, "snow": scen in ("S3", "G")}
    if scen == "S1":
        p["tau_f"] = np.full(len(BASINS), 1.0)          # unused (phi_s = 1)
        p["tau_s"] = 10 ** clip(1.0 + 0.35 * z.soil_depth_pelletier + 0.25 * z.frac_forest, 0.0, 2.3)
        p["phi_s"] = np.ones(len(BASINS))
    else:
        p["tau_f"] = 10 ** clip(0.3 - 0.2 * z.slope_mean, -0.3, 0.7)              # ~0.5–5 d
        p["phi_s"] = 1 / (1 + np.exp(-0.8 * z.soil_porosity.values))               # slow fraction
        if scen == "G":
            p["tau_s"] = 10 ** clip(1.3 + 0.6 * z.carbonate_rocks_frac + 0.25 * z.geol_permeability, 1.0, 2.5)
        else:
            p["tau_s"] = 10 ** clip(1.6 + 0.3 * z.soil_depth_pelletier + 0.2 * z.sand_frac, 1.0, 2.5)
    p["ddf"] = clip(2.5 + 1.0 * z.elev_mean.values, 1.0, 5.0)                       # mm/°C/d
    return {k: (np.asarray(v, float) if k != "snow" else v) for k, v in p.items()}


def generate(P, TM, p, x0=None):
    """Batched generator. P, TM: [N, T] tensors. Returns Q [N, T] and final state.
    Smooth rain/snow split (sigmoid around 0 °C) keeps it differentiable."""
    N, T = P.shape
    tf = torch.tensor(p["tau_f"]); ts = torch.tensor(p["tau_s"])
    af, as_ = torch.exp(-1 / tf), torch.exp(-1 / ts)
    phi = torch.tensor(p["phi_s"]); c = torch.tensor(p["c"]); ddf = torch.tensor(p["ddf"])
    swe, qf, qs = (torch.zeros(N, dtype=P.dtype) for _ in range(3)) if x0 is None else x0
    Q = []
    for t in range(T):
        sf = torch.sigmoid(-TM[:, t]) if p["snow"] else torch.zeros_like(TM[:, t])
        swe = swe + P[:, t] * sf
        melt = torch.minimum(swe, ddf * torch.relu(TM[:, t])) if p["snow"] else torch.zeros_like(swe)
        swe = swe - melt
        u = c * (P[:, t] * (1 - sf) + melt)
        qf = af * qf + (1 - af) * (1 - phi) * u
        qs = as_ * qs + (1 - as_) * phi * u
        Q.append(qf + qs)
    return torch.stack(Q, 1), (swe, qf, qs)


rng = np.random.default_rng(0)
truth = {}
for scen in ["S1", "S2", "S3", "G"]:
    p = params_for(scen)
    with torch.no_grad():
        Q, _ = generate(P, TM, p)
    for noise in [0.0, 0.1]:
        Qn = Q.numpy() * (np.exp(rng.normal(0, noise, Q.shape)) if noise else 1.0)
        feats = {b: pd.DataFrame({"q_syn": Qn[i]}, index=idx) for i, b in enumerate(BASINS)}
        with open(SYN_DIR / f"syn_{scen}_n{int(noise*100):02d}.p", "wb") as f:
            pickle.dump(feats, f)
    truth[scen] = pd.DataFrame({k: v for k, v in p.items() if k != "snow"}, index=BASINS)
    truth[scen].to_csv(SYN_DIR / f"truth_params_{scen}.csv")

# sanity: synthetic flows should look like plausible hydrographs (plot 4 basins per scenario)


# ---------------- true lag sensitivity (planted M50 / mean lag) ----------------
def true_lag_profile(scen, b_i, end_positions, seq=365):
    """Mean |dQ_T/dP_{T-k}| over windows ending at `end_positions` (indices into idx). Lag-indexed."""
    p = params_for(scen)
    p1 = {k: (v[b_i:b_i + 1] if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    profs = []
    for e in end_positions:
        s0 = e - seq + 1
        with torch.no_grad():                                     # spin-up to window start
            _, x0 = generate(P[b_i:b_i + 1, :s0], TM[b_i:b_i + 1, :s0], p1)
        Pw = P[b_i:b_i + 1, s0:e + 1].clone().requires_grad_(True)
        Q, _ = generate(Pw, TM[b_i:b_i + 1, s0:e + 1], p1, x0=tuple(v.detach() for v in x0))
        Q[0, -1].backward()
        profs.append(np.abs(Pw.grad.numpy()[0])[::-1])
    return np.mean(profs, 0)
