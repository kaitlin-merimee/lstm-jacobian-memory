"""Shared functions for the AGU follow-up analyses (nudge_through_melt nudge, lag_profiles lag profiles, lyapunov_seeds Lyapunov seeds).

Same conventions as the early notebooks: numpy LSTM in PyTorch gate order (i, f, g, o), 365-day windows that start
from h = c = 0, windows ending in the unseen years WY2009-2014. The NeuralHydrology loading is the an early notebook
Cell 2 pattern. `--toy` in every script swaps in a small random LSTM so the code can be tested anywhere.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("JTALK_ROOT", Path(__file__).resolve().parents[1]))   # project folder with runs/, data/, analysis/; default: this repo
RUNS = {1: "camels531_daymet_h256_seed1_1207_194330",
        2: "camels531_daymet_h256_seed2_1207_234531",
        3: "camels531_daymet_h256_seed3_1207_234546"}
SNOW, ARID = "06623800", "08165300"
PERIOD = ("01/10/2008", "30/09/2014")          # prediction days; NeuralHydrology adds the lookback in front
L_WIN = 365


def sig(z):
    return 1 / (1 + np.exp(-z))


class Net:
    """Weights, stitched inputs X (T x D, as nn.LSTM sees them) and scalers for one run and basin."""

    def __init__(self, seed=1, basin=SNOW, toy=False, root=ROOT, run=None):
        self.seed, self.basin, self.toy = seed, basin, toy
        if toy:
            self._toy(seed)
        else:
            self._load(Path(root) / "runs" / (run or RUNS[seed]), basin)       # run = folder name under runs/ (e.g. a synthetic run)
        self.H, self.D = self.W_hh.shape[1], self.W_ih.shape[1]
        H = self.H
        self.Wi, self.Wf, self.Wg, self.Wo = (self.W_hh[k * H:(k + 1) * H] for k in range(4))
        self.wy = (self.dates.year + (self.dates.month >= 10)).values

    def _toy(self, seed):
        rng = np.random.default_rng(seed); H, D = 24, 6
        self.W_ih = rng.normal(0, .5, (4 * H, D)); self.W_hh = rng.normal(0, .25, (4 * H, H))
        self.b = rng.normal(0, .5, 4 * H); self.b[H:2 * H] += 3
        self.w_head, self.b_head = rng.normal(0, .3, H), 0.0
        self.dates = pd.date_range("2007-10-02", "2014-09-30", freq="D")
        doy = self.dates.dayofyear.values
        self.X = np.c_[rng.gamma(.3, 3, len(doy)), np.sin(2 * np.pi * doy / 365.25),
                       rng.normal(0, 1, (len(doy), D - 2))]
        self.y_scale, self.y_center, self.p_idx, self.p_scale = 1.0, 0.0, 0, 1.0
        self.dyn = ['prcp', 'tmean'] + [f'x{k}' for k in range(D - 2)]
        self.x_center, self.x_scale = np.zeros(D), np.ones(D); self.x_scale[1] = 10.0
        self.p_center, self.t_cols, self.q_obs = 0.0, [1], None
        self.check_windows = []

    def _load(self, run_dir, basin):
        import torch
        from neuralhydrology.utils.config import Config
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.datasetzoo import get_dataset
        from neuralhydrology.datautils.utils import load_scaler

        cfg = Config(run_dir / "config.yml")
        cfg.update_config({"test_start_date": PERIOD[0], "test_end_date": PERIOD[1], "device": "cpu"})
        model = get_model(cfg).to("cpu")
        wfile = sorted(run_dir.glob("model_epoch*.pt"))[-1]
        model.load_state_dict(torch.load(wfile, map_location="cpu")); model.eval()
        lstm = model.lstm
        self.W_ih = lstm.weight_ih_l0.detach().double().numpy()
        self.W_hh = lstm.weight_hh_l0.detach().double().numpy()
        self.b = (lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().double().numpy()
        lin = [m for m in model.head.modules() if isinstance(m, torch.nn.Linear)]
        assert len(lin) == 1, "expected a single Linear readout"
        self.w_head = lin[0].weight.detach().double().numpy()[0]; self.b_head = float(lin[0].bias.detach())

        scaler = load_scaler(run_dir)
        ds = get_dataset(cfg, is_train=False, basin=basin, period="test", scaler=scaler)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)
        rows, ys, self.check_windows, first = [], [], [], True
        with torch.no_grad():
            for batch in loader:
                xb = model.embedding_net(batch).double().numpy()       # [seq, batch, D]
                if first:
                    rows.append(xb[:-1, 0, :])
                    yh = model(batch)["y_hat"][:, -1, 0].double().numpy()
                    self.check_windows = [(xb[:, j, :], yh[j]) for j in (0, 1, 100, 255) if j < xb.shape[1]]
                    assert np.allclose(xb[1:, 0, :], xb[:-1, 1, :]), "windows do not overlap: check for gaps"
                    first = False
                rows.append(xb[-1, :, :])
                try: ys.append(batch['y'][:, -1, 0].double().numpy())
                except Exception: ys = None
        self.X = np.concatenate(rows, 0)
        start = pd.to_datetime(PERIOD[0], dayfirst=True) - pd.Timedelta(days=cfg.seq_length - 1)
        self.dates = pd.date_range(start, pd.to_datetime(PERIOD[1], dayfirst=True), freq="D")
        assert len(self.dates) == len(self.X), f"{len(self.X)} input rows vs {len(self.dates)} days: samples were dropped"
        assert np.isfinite(self.X).all(), "NaNs in the inputs"
        assert cfg.seq_length == L_WIN, f"seq_length {cfg.seq_length} != {L_WIN}"

        tgt = cfg.target_variables[0]
        self.y_scale = float(scaler["xarray_feature_scale"][tgt].values)
        self.y_center = float(scaler["xarray_feature_center"][tgt].values)
        dyn = list(cfg.dynamic_inputs)
        pname = [n for n in dyn if "prcp" in n.lower()][0]
        self.p_idx = dyn.index(pname)                                   # dynamic inputs come first in the embedding
        self.p_scale = float(scaler["xarray_feature_scale"][pname].values)
        self.dyn = dyn
        self.x_center = np.array([float(scaler["xarray_feature_center"][n].values) for n in dyn])
        self.x_scale = np.array([float(scaler["xarray_feature_scale"][n].values) for n in dyn])
        self.p_center = self.x_center[self.p_idx]
        self.t_cols = [i for i, n in enumerate(dyn) if n.lower().startswith(('tmax', 'tmin', 'tmean', 'temp'))]
        self.q_obs = None                                               # observed Q (mm/d) aligned with X; NaN before the first prediction day
        if ys:
            qo = np.concatenate(ys) * self.y_scale + self.y_center
            if len(qo) == len(self.X) - (L_WIN - 1): self.q_obs = np.r_[np.full(L_WIN - 1, np.nan), qo]
        if self.q_obs is None: print('  note: observed discharge not read from the dataset; obs comparisons are skipped')
        print(f"[seed {self.seed}] {run_dir.name} | {wfile.name} | basin {basin} | prcp input #{self.p_idx} = {pname}")

    def precip(self):
        """Precipitation in mm/d for every row of X."""
        return self.X[:, self.p_idx] * self.p_scale + self.p_center

    def temp(self):
        """Air temperature in deg C (mean of the temperature inputs) for every row of X."""
        return np.mean([self.X[:, i] * self.x_scale[i] + self.x_center[i] for i in self.t_cols], 0)

    # ---- window bookkeeping ---------------------------------------------------------------
    def ends(self, start="2008-10-01", stop="2014-09-30", every=1):
        """Indices into X of prediction days in [start, stop] that have a full 365-day window."""
        e = np.where((self.dates >= start) & (self.dates <= stop))[0]
        return e[e >= L_WIN - 1][::every]

    def q(self, h):
        """Discharge (mm/d) from hidden state(s)."""
        return (h @ self.w_head + self.b_head) * self.y_scale + self.y_center

    # ---- forward ------------------------------------------------------------------------------
    def run_windows(self, ends, keep=False, nudge=None):
        """Zero-start 365-day run for every window ending at `ends` (all windows at once).

        keep=True stores gates and states for every step: arrays [L, n, H] (needed for Jacobians).
        nudge=(unit, step_per_window, factor): multiply c[unit] by `factor` at the END of window step
        step_per_window[w] (-1 = leave that window alone), then carry on. Returns h, c at the last day
        (and the stored dict if keep)."""
        H, n = self.H, len(ends); h = np.zeros((n, H)); c = np.zeros((n, H))
        S = {k: np.zeros((L_WIN, n, H)) for k in "ifgo"} if keep else None
        if keep: S["c"], S["cp"] = np.zeros((L_WIN, n, H)), np.zeros((L_WIN, n, H))
        for s in range(L_WIN):
            z = self.X[ends - L_WIN + 1 + s] @ self.W_ih.T + self.b + h @ self.W_hh.T
            i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
            if keep: S["cp"][s] = c
            c = f * c + i * g
            if nudge is not None:
                u, steps, fac = nudge; w = np.where(steps == s)[0]
                c[np.ix_(w, np.atleast_1d(u))] *= fac                    # u may be one unit or a set of units
            h = o * np.tanh(c)
            if keep:
                S["i"][s], S["f"][s], S["g"][s], S["o"][s], S["c"][s] = i, f, g, o, c
        return (h, c, S) if keep else (h, c)

    # ---- backward: sensitivity of the last day's discharge -----------------------------------------
    def adjoint(self, S):
        """Chain the transposed daily Jacobians back from the prediction day.

        Returns dQ/dx [L, n, D] with index k = lag in days (k = 0 is the prediction day), in normalised
        units, and reach [L, n] = ||dQ_T / d state_{T-k}||, with reach[0] = ||w_head|| by construction."""
        n = S["c"].shape[1]
        a_h = np.tile(self.w_head, (n, 1)); a_c = np.zeros((n, self.H))
        dQdx = np.zeros((L_WIN, n, self.D)); reach = np.zeros((L_WIN, n))
        for k in range(L_WIN):
            s = L_WIN - 1 - k
            reach[k] = np.sqrt((a_h ** 2).sum(1) + (a_c ** 2).sum(1))
            i, f, g, o, c, cp = (S[x][s] for x in ("i", "f", "g", "o", "c", "cp"))
            tc = np.tanh(c)
            gc = a_c + a_h * o * (1 - tc ** 2)
            gz = np.concatenate([gc * g * i * (1 - i), gc * cp * f * (1 - f), gc * i * (1 - g ** 2),
                                 a_h * tc * o * (1 - o)], 1)
            dQdx[k] = gz @ self.W_ih
            a_h, a_c = gz @ self.W_hh, gc * f
        return dQdx, reach

    # ---- one-day Jacobian applied to a set of directions (for QR) -----------------------------------
    def apply_J(self, S, s, w, Qh, Qc):
        """J(s) @ [Qh; Qc] for window w at step s, without building the 2H x 2H matrix."""
        i, f, g, o, c, cp = (S[x][s, w] for x in ("i", "f", "g", "o", "c", "cp"))
        tc = np.tanh(c); beta = o * (1 - tc ** 2)
        C = (cp * f * (1 - f))[:, None] * self.Wf + (g * i * (1 - i))[:, None] * self.Wi + (i * (1 - g ** 2))[:, None] * self.Wg
        dc = C @ Qh + f[:, None] * Qc
        dh = (tc * o * (1 - o))[:, None] * (self.Wo @ Qh) + beta[:, None] * dc
        return dh, dc

    def step_jacobian(self, S, s, w):
        """Full 2H x 2H Jacobian (the matrix on the anatomy slide). Used for checks only."""
        I = np.eye(self.H); Z = np.zeros((self.H, self.H))
        dh1, dc1 = self.apply_J(S, s, w, I, Z); dh2, dc2 = self.apply_J(S, s, w, Z, I)
        return np.block([[dh1, dh2], [dc1, dc2]])

    def gate_check(self):
        """Stop unless the numpy run reproduces NeuralHydrology's own prediction (normalised units)."""
        for xw, yh in self.check_windows:
            keepX, self.X = self.X, xw
            h, _ = self.run_windows(np.array([L_WIN - 1])); self.X = keepX
            y = h[0] @ self.w_head + self.b_head
            assert abs(y - yh) < 1e-4, f"numpy {y:+.6f} vs model {yh:+.6f}"
        print(f"gate: numpy run = model prediction on {len(self.check_windows)} windows" if self.check_windows
              else "gate: skipped (toy)")


def season_of(months):
    lab = np.empty(len(months), dtype=object)
    for name, ms in {"Apr-Jun": (4, 5, 6), "Jul-Sep": (7, 8, 9), "Oct-Nov": (10, 11), "Dec-Mar": (12, 1, 2, 3)}.items():
        lab[np.isin(months, ms)] = name
    return lab
