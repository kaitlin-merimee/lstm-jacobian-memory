"""jlens_utils.py

Shared weight-loading, unroll, windowing, timescale and recession code for nb08-nb11.
Recovered from jacobian_followup_experiments.md (Oct 2026). Items marked VERIFY must be
checked against the NeuralHydrology 1.13 install before trusting outputs.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler

PROJECT = Path(".")
OUT_DIR = PROJECT / "analysis" / "jacobians"
FIG_DIR = PROJECT / "analysis" / "figures"
STRIDE = 45
TAU_CAP = 1e4
BANDS = {"quick": (0, 3), "event": (3, 30), "seasonal": (30, 365), "integrator": (365, np.inf)}


def latest_run(pattern: str) -> Path:
    return sorted((PROJECT / "runs").glob(pattern))[-1]


class Model:
    """Weights + config + scaler for one trained run, float64 numpy and torch views."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.cfg = Config(self.run_dir / "config.yml")
        self.seq = self.cfg.seq_length
        self.dyn = list(self.cfg.dynamic_inputs)
        self.statics = list(self.cfg.static_attributes)
        self.n_dyn = len(self.dyn)
        sd = torch.load(sorted(self.run_dir.glob("model_epoch*.pt"))[-1], map_location="cpu")
        self.W_ih = sd["lstm.weight_ih_l0"].double().numpy()
        self.W_hh = sd["lstm.weight_hh_l0"].double().numpy()
        self.b = (sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]).double().numpy()
        hk = [k for k in sd if "head" in k and "weight" in k][0]
        self.w_out = sd[hk].double().numpy().squeeze()          # [H]
        self.b_out = float(sd[hk.replace("weight", "bias")].double())
        self.w_abs = np.abs(self.w_out)
        self.H = self.W_hh.shape[1]
        self.D = self.W_ih.shape[1]
        self.scaler = load_scaler(self.run_dir)
        self.t = {k: torch.tensor(getattr(self, k)) for k in ("W_ih", "W_hh", "b", "w_out")}
        with open(sorted((self.run_dir / "test").glob("model_epoch*"))[-1] / "test_results.p", "rb") as f:
            self.test_results = pickle.load(f)

    # ---------- data ----------
    def dataset(self, basin):
        return get_dataset(self.cfg, is_train=False, period="test", basin=basin, scaler=self.scaler)

    def window_idxs(self, ds, stride=STRIDE):
        return np.arange(0, len(ds), stride)

    def build_X(self, ds, idxs):
        """[B, T, D] float64: [dynamics | statics] per day (NH 1.13: x_d is a tensor)."""
        rows = []
        for k in idxs:
            s = ds[k]
            xd = s["x_d"]
            if isinstance(xd, dict):
                xd = torch.cat([xd[n].reshape(self.seq, -1) for n in self.dyn], dim=-1)
            xs = s["x_s"].reshape(1, -1).expand(self.seq, -1)
            rows.append(torch.cat([xd, xs], dim=-1).double().numpy())
        return np.stack(rows)

    def obs_sim(self, basin):
        xr_ = self.test_results[basin]["1D"]["xr"]
        obs = [v for v in xr_.data_vars if v.endswith("_obs")][0]
        sim = [v for v in xr_.data_vars if v.endswith("_sim")][0]
        dates = pd.to_datetime(xr_["date"].values)
        return (pd.Series(xr_[obs].values.squeeze(), index=dates),
                pd.Series(xr_[sim].values.squeeze(), index=dates))

    # ---------- forward passes ----------
    def unroll_np(self, X):
        """Batched numpy forward. Returns gates dict [B,T,H] and y [B] (normalized units)."""
        B, T, _ = X.shape
        h = np.zeros((B, self.H)); c = np.zeros((B, self.H))
        f_all = np.empty((B, T, self.H))
        for t in range(T):
            pre = X[:, t] @ self.W_ih.T + h @ self.W_hh.T + self.b
            ai, af, ag, ao = np.split(pre, 4, axis=-1)
            f = 1 / (1 + np.exp(-af))
            c = f * c + (1 / (1 + np.exp(-ai))) * np.tanh(ag)
            h = (1 / (1 + np.exp(-ao))) * np.tanh(c)
            f_all[:, t] = f
        return {"f": f_all}, h @ self.w_out + self.b_out

    def unroll_torch(self, X, input_grad=False, keep_c=False):
        """Differentiable unroll. Returns y [B], X tensor, list of c_t (grad-retaining if keep_c)."""
        Xt = torch.tensor(X, dtype=torch.float64, requires_grad=input_grad)
        B, T, _ = Xt.shape
        h = torch.zeros(B, self.H, dtype=torch.float64)
        c = torch.zeros(B, self.H, dtype=torch.float64)
        cs = []
        for t in range(T):
            pre = Xt[:, t] @ self.t["W_ih"].T + h @ self.t["W_hh"].T + self.t["b"]
            ai, af, ag, ao = pre.chunk(4, -1)
            c = torch.sigmoid(af) * c + torch.sigmoid(ai) * torch.tanh(ag)
            if keep_c:
                c.retain_grad()
                cs.append(c)
            h = torch.sigmoid(ao) * torch.tanh(c)
        y = h @ self.t["w_out"] + self.b_out
        return y, Xt, cs


# ---------- timescale statistics ----------
def forget_tau(f):
    return -1.0 / np.log(np.clip(f, 1e-12, 1 - 1e-12))


def portfolio(model, f_windows):
    """f_windows: [B,T,H] forget gates over all windows of one basin. Matches nb02 definitions."""
    tau_u = np.median(np.median(forget_tau(f_windows), axis=1), axis=0)     # [H]
    ltau = np.log10(np.clip(tau_u, 1e-2, TAU_CAP))
    w = model.w_abs
    out = {"tau_readout_log10": float((w * ltau).sum() / w.sum())}
    for name, (lo, hi) in BANDS.items():
        m = (tau_u >= lo) & (tau_u < hi)
        out[f"readout_frac_{name}"] = float(w[m].sum() / w.sum())
    return out, tau_u


def lag_profile_input(model, X, j=0):
    """Window-mean |dy_T / dx_{T-k, j}|, index 0 = lag 0. j=0 is prcp."""
    y, Xt, _ = model.unroll_torch(X, input_grad=True)
    y.sum().backward()
    g = np.abs(Xt.grad.numpy()[:, :, j]).mean(axis=0)
    return g[::-1]


def state_propagator_profile(model, X):
    """Propagator-based memory: window-mean of sum_i |dy_T/dc_{T-k,i}| * sd(c_i).

    This measures how much a typical perturbation of the cell state k days ago still
    moves today's prediction. It includes cross-unit coupling through h, which the
    per-unit forget-gate tau ignores. Returns lag-indexed profile (index 0 = lag 0).
    """
    y, _, cs = model.unroll_torch(X, keep_c=True)
    y.sum().backward()
    G = torch.stack([c.grad for c in cs], dim=1).numpy()        # [B,T,H]  dy_T/dc_t
    C = torch.stack(cs, dim=1).detach().numpy()                 # [B,T,H]
    sd = C.std(axis=1, keepdims=True)                           # typical state variability per unit
    prof = np.abs(G * sd).sum(-1).mean(0)                       # [T]
    return prof[::-1]


def profile_stats(prof):
    """M50 and mean lag of a lag-indexed nonnegative profile (truncated at the window)."""
    p = np.asarray(prof, float)
    cdf = np.cumsum(p) / p.sum()
    lags = np.arange(len(p))
    return {"M50": int(np.searchsorted(cdf, 0.5)),
            "mean_lag": float((lags * p).sum() / p.sum())}


# ---------- recession and signatures (copied from nb03 so all notebooks agree) ----------
def recession_taus(q, min_flow=1e-3, skip_after_peak=2, min_event_len=7, mask=None):
    """Point-wise and event-fit recession tau. `mask` (bool array, len(q)-1) optionally
    restricts which day-pairs count as recession (e.g., days declining in BOTH obs and sim)."""
    qv = np.asarray(q, float)
    declining = np.diff(qv) < 0
    if mask is not None:
        declining &= mask
    pt, ev = [], []
    i = 0
    while i < len(declining):
        if declining[i]:
            j = i
            while j < len(declining) and declining[j]:
                j += 1
            seg = np.arange(i, j + 1)[skip_after_peak:]
            if len(seg) >= 2:
                qs = qv[seg]
                ok = (qs[:-1] > min_flow) & (qs[1:] > min_flow)
                r = qs[1:][ok] / qs[:-1][ok]
                r = r[(r > 0) & (r < 1)]
                pt.extend((-1.0 / np.log(r)).tolist())
                if len(qs) >= min_event_len and (qs > min_flow).all():
                    slope = np.polyfit(np.arange(len(qs)), np.log(qs), 1)[0]
                    if slope < 0:
                        ev.append(-1.0 / slope)
            i = j
        else:
            i += 1
    pt, ev = np.array(pt), np.array(ev)
    return {"tau_rec_pt": np.median(pt) if len(pt) > 20 else np.nan,
            "tau_rec_p90": np.percentile(pt, 90) if len(pt) > 20 else np.nan,
            "tau_rec_ev": np.median(ev) if len(ev) >= 5 else np.nan}


def regime(row):
    if row.frac_snow >= 0.3: return "snow"
    if row.aridity >= 1.5 and row.frac_snow < 0.15: return "arid"
    if row.frac_snow < 0.15: return "humid-rain"
    return "mixed"
