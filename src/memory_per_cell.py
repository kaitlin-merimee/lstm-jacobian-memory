"""memory_per_cell: how long does each part of the state remember, and how much does flow feel it? Through the year. One basin, one model version.

Earlier tests asked how far back FLOW depends on the weather (memory days) or on the state. This one asks it per part of the state:
for each of the 256 cells, and for a few named directions, nudge the cell state on a start day and follow the nudge forward.

  A  Memory in the state. Nudge the cell state by one standard deviation along a cell or direction on the start day and carry the
     nudge forward with the chained daily Jacobians (forward, not backward). Memory = days until the nudge's own size along that
     cell or direction has fallen to 1/e (37%). "Spread memory" = the same for the nudge anywhere in the state (it can move into
     other cells). Censored at the horizon: a value equal to the horizon means "at least this long".
     The nudge is carried inside the one 365-day window that ends `--horizon` days after the start day, so the start day has at
     least 365 - horizon days of spin-up in front of it. The spin-up check printed below compares that state with the state the
     model has on the start day in its own window.
  B  Effect on flow. For every day T from the start day to the end of the horizon, the gradient of T's flow with respect to the
     start day's cell state, each T in its own 365-day window, as NeuralHydrology runs it (the same gradient as where_snow_patching, all days, not
     only April-September). Per cell or direction, per one standard deviation: total extra flow over the horizon (mm), the same
     without signs, the April-September part, and the days until half of it (without signs) has arrived.
  C  Unit swings in the melt season. For the named unit: is the day-to-day change of its hidden output in April-July set by the
     output gate (when to report) or by the cell (how much is stored)? And how big are those swings in flow (mm/d) through the
     direct readout, against peak flow?

Directions (all of length one in standardised cell units, so a nudge of one standard deviation along each):
  cell_j           each of the 256 cells
  read1..readK     the top directions flow reads, found again for each start day from B's gradients (all days in the horizon)
  snow_group       all snow-tracking cells moved together, each with the sign of its correlation with snow (where_snow_patching rule)
  probe            linear snow probe direction (ridge, fitted on all years; direction only, so in-sample is fine here)
  pc1..pc3         directions of most variance (PCA of the daily cell states)
  rand1..randN     random directions

  sbatch slurm/memory_per_cell.slurm                                  # 5 snow basins x 3 model versions, CPU
  PYTHONPATH=src python src/memory_per_cell.py --basin 06623800 --seed 1 --unit 77
  python src/memory_per_cell.py --toy                                  # code test on a random LSTM
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from jtalk_core import Net, ROOT, SNOW, L_WIN, sig

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--unit", default="auto")
ap.add_argument("--starts", nargs="*", default=[f"{m:02d}-01" for m in (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)], help="start days, MM-DD, inside each water year")
ap.add_argument("--horizon", type=int, default=240, help="days followed after the start day (at most 364)")
ap.add_argument("--k-read", type=int, default=3); ap.add_argument("--n-random", type=int, default=10); ap.add_argument("--group-thresh", type=float, default=.8)
ap.add_argument("--ddf", type=float, default=3.0); ap.add_argument("--chunk", type=int, default=60, help="windows per forward pass (memory)"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
assert 30 <= a.horizon <= L_WIN - 1
OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/cell_memory"; OUT.mkdir(parents=True, exist_ok=True); TAG = f"{a.basin}_seed{a.seed}" + ("" if a.horizon == 240 else f"_h{a.horizon}")      # other horizons: robustness runs, kept apart
YEARS = range(2009, 2015); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); rng = np.random.default_rng(0)
NAMES = {"06623800": "Encampment R., WY", "12178100": "Newhalem Cr., WA", "13235000": "SF Payette R., ID", "09066300": "Middle Cr., CO", "09312600": "White R., UT"}
E1 = np.exp(-1)

net = Net(a.seed, a.basin, a.toy)
if a.toy: net.y_center = 3.0
else: net.gate_check()
H = net.H; e = net.ends(); d = net.dates[e]; wy = net.wy[e]; P, T = net.precip(), net.temp(); cold = T < 0
if cold.sum() < 30: cold = T < np.median(T)                                     # toy weather has no freezing days
hT, cT = net.run_windows(e); mu, sd = cT.mean(0), cT.std(0) + 1e-9; Z = (cT - mu) / sd


# ---- snow reference and snow-tracking cells (same rule as where_snow_patching / snow_cells_role) ------------------------------------------------------------
swe = np.zeros(len(P))
for t in range(1, len(P)): swe[t] = max(swe[t - 1] + P[t] * cold[t] - a.ddf * max(T[t], 0), 0)
REF, REFNAME = swe, "degree-day snow index"
try:
    if a.toy: raise FileNotFoundError
    def pick(df):
        dc = [k for k in df.columns if "date" in k.lower()][0]; sc = [k for k in df.columns if "swe" in k.lower()][0]; return sc, pd.Series(df[sc].values, index=pd.to_datetime(df[dc]))
    f_ = ROOT / f"analysis/spires/spires_{a.basin}.csv"
    if f_.exists(): sc, ser = pick(pd.read_csv(f_))
    else:
        parts = []
        for g_ in sorted((ROOT / "analysis/spires/all").glob("spires_WY*.csv")):
            df = pd.read_csv(g_, dtype=str); bc = [k for k in df.columns if any(w in k.lower() for w in ("basin", "gauge", "gage", "site"))][0]; df = df[df[bc].str.zfill(8) == a.basin]
            if len(df): sc, s1 = pick(df); parts.append(s1.astype(float))
        ser = pd.concat(parts).sort_index()
    ser = ser[~ser.index.duplicated()].astype(float).reindex(net.dates).values
    if np.isfinite(ser[e]).sum() > 500: REF, REFNAME = ser, f"SPIReS {sc}"
except Exception: pass
y_ref = REF[e]; ok = np.isfinite(y_ref)
def rho_cells(C, y):
    rc = rankdata(C, axis=0); ry = rankdata(y); rc = (rc - rc.mean(0)) / (rc.std(0) + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12); return (rc * ry[:, None]).mean(0)
pk = {y: np.nanmax(y_ref[wy == y]) for y in YEARS if ok[wy == y].sum() > 100}; big, small = max(pk, key=pk.get), min(pk, key=pk.get)
r2 = np.array([rho_cells(cT[(wy == y) & ok], y_ref[(wy == y) & ok]) for y in (big, small)]); strength = np.where(np.sign(r2[0]) == np.sign(r2[1]), np.abs(r2).min(0), 0)
GROUP = np.where(strength > a.group_thresh)[0]
if len(GROUP) < 2: GROUP = np.argsort(-strength)[:max(2, H // 12)]
U = int(np.argmax(strength)) if a.unit == "auto" else int(a.unit)
print(f"{NAMES.get(a.basin, '')} {a.basin}, model version {a.seed} | snow reference: {REFNAME} | {len(GROUP)} snow-tracking cells | named unit {U} | horizon {a.horizon} d")

# ---- fixed directions, in standardised cell units, each of length one ----------------------------------------------------------
unit = lambda v: v / (np.linalg.norm(v) + 1e-30)
Zc = Z[ok] - Z[ok].mean(0); wP = np.linalg.solve(Zc.T @ Zc + 1e-2 * ok.sum() * np.eye(H), Zc.T @ (y_ref[ok] - y_ref[ok].mean()))
_, _, Vp = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
vg = np.zeros(H); vg[GROUP] = np.sign(r2[0][GROUP]); FIXED = [("snow_group", "snow group", unit(vg)), ("probe", "probe", unit(wP))]
FIXED += [(f"pc{k + 1}", "PCA", Vp[k]) for k in range(3)] + [(f"rand{k + 1}", "random", unit(rng.normal(size=H))) for k in range(a.n_random)]


def grad_at(S, steps):
    """d(last-day flow, mm/d) / d(cell state at step steps[w]) for every window (as where_snow_patching)."""
    n = S["c"].shape[1]; a_h = np.tile(net.w_head, (n, 1)); a_c = np.zeros((n, H)); G = np.zeros((n, H))
    for k in range(L_WIN):
        s = L_WIN - 1 - k; i, f, g, o, c, cp = (S[x][s] for x in ("i", "f", "g", "o", "c", "cp")); tc = np.tanh(c)
        gc = a_c + a_h * o * (1 - tc ** 2); w = steps == s; G[w] = gc[w]
        if s <= steps.min(): break
        gz = np.concatenate([gc * g * i * (1 - i), gc * cp * f * (1 - f), gc * i * (1 - g ** 2), a_h * tc * o * (1 - o)], 1)
        a_h, a_c = gz @ net.W_hh, gc * f
    return G * net.y_scale


def first_below(curve, thr):
    """First day the curve (curve[0] = 1) falls below thr; len-1 (censored) if never."""
    k = np.where(np.abs(curve) < thr)[0]; return (int(k[0]), False) if len(k) else (len(curve) - 1, True)


# ---- A and B, per water year and start day ----------------------------------------------------------------------------------------
rows, curves, spin = [], {}, []
for y in YEARS:
    for md in a.starts:
        t0 = pd.Timestamp(f"{y - 1 if int(md[:2]) >= 10 else y}-{md}"); w_ = np.where(net.dates == t0)[0]
        if not len(w_) or w_[0] < L_WIN - 1: continue
        tn = int(w_[0]); hz = min(a.horizon, len(net.dates) - 1 - tn)
        if hz < 30: continue
        # B: flow gradients, every day T in [t0, t0 + hz], each in its own window
        ee = np.arange(tn, tn + hz + 1); steps = L_WIN - 1 - (ee - tn); G = np.zeros((len(ee), H))
        for k in range(0, len(ee), a.chunk):
            _, _, S = net.run_windows(ee[k:k + a.chunk], keep=True); G[k:k + a.chunk] = grad_at(S, steps[k:k + a.chunk]); del S
        Gs = G * sd                                                                           # mm/d of flow per one standard deviation of each cell
        _, sr, Vr = np.linalg.svd(Gs, full_matrices=False)
        read = [(f"read{k + 1}", "read", Vr[k] * np.sign((Gs @ Vr[k]).sum() or 1)) for k in range(a.k_read)]
        items = [(f"cell_{j}", "snow cell" if j in GROUP else "other cell", np.eye(H)[j]) for j in range(H)] + read + FIXED
        V = np.array([v for _, _, v in items])                                                # [K, H]
        R = Gs @ V.T                                                                          # [hz+1, K] flow response, mm/d per 1 sd
        # A: the same nudges carried forward inside the window that ends hz days after t0
        _, _, S1 = net.run_windows(np.array([tn + hz]), keep=True); s0 = L_WIN - 1 - hz
        c0 = S1["c"][s0, 0]; o0 = S1["o"][s0, 0]; dc = sd[:, None] * V.T; dh = (o0 * (1 - np.tanh(c0) ** 2))[:, None] * dc
        own = np.zeros((hz + 1, len(V))); spread = np.zeros((hz + 1, len(V))); own[0], spread[0] = 1, 1
        for k in range(1, hz + 1):
            dh, dc = net.apply_J(S1, s0 + k, 0, dh, dc); dz = dc / sd[:, None]
            own[k] = (V * dz.T).sum(1); spread[k] = np.linalg.norm(dz, axis=0)
        if not spin:                                                                          # once: forward Jacobian against a small finite nudge
            jr = len(items) - len(FIXED) - a.k_read; eps = 1e-4; X_ = net.X; hh = np.zeros((2, H)); cc = np.zeros((2, H))
            for s in range(L_WIN):
                z = X_[np.full(2, tn + hz) - L_WIN + 1 + s] @ net.W_ih.T + net.b + hh @ net.W_hh.T
                i_, f_, g_, o_ = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:]); cc = f_ * cc + i_ * g_
                if s == s0: cc[1] += eps * sd * V[jr]
                hh = o_ * np.tanh(cc)
            fd = (cc[1] - cc[0]) / eps / sd; ad = dz[:, jr]
            print(f"  forward-Jacobian check (read1, {hz} d on): correlation with a finite nudge {np.corrcoef(fd, ad)[0, 1]:.4f}, size ratio {np.linalg.norm(fd) / (np.linalg.norm(ad) + 1e-30):.3f} (both should be about 1)")
        c_own = net.run_windows(np.array([tn]))[1][0]; spin.append(dict(start=md, wy=y, r=np.corrcoef(c0, c_own)[0, 1], rel=np.abs(c0 - c_own).sum() / (np.abs(c_own).sum() + 1e-12)))
        lnf = np.log(np.clip(S1['f'][s0 + 1:, 0], 1e-12, 1)).mean(0); gate_tau = -1 / np.minimum(lnf, -1e-12); del S1      # each cell's forget gate alone, over the horizon
        melt = np.isin(net.dates[ee].month, range(4, 10))
        for j, (name, kind, _) in enumerate(items):
            r = R[:, j]; ab = np.abs(r); cab = np.cumsum(ab); m_own, cen = first_below(own[:, j], E1); m_spr, _ = first_below(spread[:, j], E1)
            rows.append(dict(basin=a.basin, seed=a.seed, wy=y, start=md, horizon=hz, item=name, kind=kind, cell=int(name[5:]) if name.startswith("cell_") else -1,
                             r_with_snow=float(strength[int(name[5:])]) if name.startswith("cell_") else np.nan,
                             gate_tau_d=float(gate_tau[int(name[5:])]) if name.startswith("cell_") else np.nan,
                             memory_d=m_own, memory_censored=cen, spread_memory_d=m_spr,
                             kept_30d=own[min(30, hz), j], kept_90d=own[min(90, hz), j], kept_180d=own[min(180, hz), j] if hz >= 180 else np.nan,
                             flow_total_mm=r.sum(), flow_abs_mm=ab.sum(), flow_melt_mm=r[melt].sum(), flow_same_day_mm=r[0],
                             flow_half_d=int(np.searchsorted(cab, cab[-1] / 2)) if cab[-1] > 0 else np.nan))
        keep_named = [j for j, (n_, k_, _) in enumerate(items) if not n_.startswith("cell_") or int(n_[5:]) in set(GROUP) | {U}]
        curves[f"{y}_{md}"] = dict(names=[items[j][0] for j in keep_named], own=own[:, keep_named].astype(np.float32), spread=spread[:, keep_named].astype(np.float32),
                                   flow=R[:, keep_named].astype(np.float32), own_all=own.astype(np.float16), read_share=(sr[:a.k_read] ** 2 / (sr ** 2).sum()).astype(np.float32))
        print(f"  WY{y} start {md}: horizon {hz} d, top {a.k_read} read directions carry {(sr[:a.k_read] ** 2).sum() / (sr ** 2).sum():.0%} of flow sensitivity", flush=True)

A = pd.DataFrame(rows)
A["month"] = A.start.str[:2].astype(int); A.to_csv(OUT / f"nb32_memory_{TAG}.csv", index=False); pd.to_pickle(curves, OUT / f"nb32_curves_{TAG}.pkl")
SP = pd.DataFrame(spin); print(f"\nspin-up check: start-day state in the long window against its own window, correlation median {SP.r.median():.3f} (min {SP.r.min():.3f}), "
                                f"relative difference median {SP.rel.median():.1%} (max {SP.rel.max():.1%}). Close to 1 and 0% means the forward nudge starts from the model's own state.")

# ---- summaries -----------------------------------------------------------------------------------------------------------------
cells = A[A.cell >= 0].copy(); cells["flow_share"] = cells.flow_abs_mm / cells.groupby(["wy", "start"]).flow_abs_mm.transform("sum")
grp = lambda df: df.groupby(["start", "kind"], sort=False).agg(memory_d=("memory_d", "median"), censored=("memory_censored", "mean"), spread_memory_d=("spread_memory_d", "median"),
                                                                  flow_abs_mm=("flow_abs_mm", "median"), flow_melt_mm=("flow_melt_mm", "median"), flow_half_d=("flow_half_d", "median"))
named = A[A.cell < 0].copy(); named.loc[named.kind == "read", "kind"] = named.item              # keep read1, read2, read3 apart
Tb = pd.concat([grp(cells), grp(named)]).reset_index(); Tb.to_csv(OUT / f"nb32_summary_{TAG}.csv", index=False)
print("\n== A and B: medians over the six water years (cells: also over cells). memory_d = days until a 1-sd nudge falls to 37% along itself ==")
for md in a.starts:
    s = Tb[Tb.start == md]
    if len(s): print(f"\nstart {md}"); print(s.drop(columns="start").round(2).to_string(index=False))

print("\n== across the 256 cells: does a long memory go with a large effect on flow? (rank correlation, per start day, median over years) ==")
for md in a.starts:
    c_ = cells[cells.start == md]
    if len(c_): print(f"  start {md}: memory vs flow effect rho = {np.median([spearmanr(g.memory_d, g.flow_abs_mm)[0] for _, g in c_.groupby('wy')]):+.2f}; "
                      f"share of flow effect on the 10% longest-memory cells {np.median([g.loc[g.memory_d >= g.memory_d.quantile(.9), 'flow_share'].sum() for _, g in c_.groupby('wy')]):.0%}; "
                      f"on the snow-tracking cells {np.median([g.loc[g.kind == 'snow cell', 'flow_share'].sum() for _, g in c_.groupby('wy')]):.0%} ({len(GROUP)} of {H} cells)")
u_ = cells[cells.cell == U].groupby("start", sort=False)[["memory_d", "memory_censored", "flow_abs_mm"]].median()
print(f"\nunit {U}, by start day (median over years):"); print(u_.round(3).to_string())

for md in ("12-01", "03-01"):
    s = Tb[Tb.start == md].set_index("kind")
    if not len(s): continue
    g_ = lambda k, c: s.loc[k, c] if k in s.index else np.nan
    print(f"ANSWER A/B ({md}): memory in the state, median days: snow cells {g_('snow cell', 'memory_d'):.0f}, other cells {g_('other cell', 'memory_d'):.0f}, read1 {g_('read1', 'memory_d'):.0f}, "
          f"snow group {g_('snow group', 'memory_d'):.0f}, probe {g_('probe', 'memory_d'):.0f}, PCA {g_('PCA', 'memory_d'):.0f}, random {g_('random', 'memory_d'):.0f} (horizon {a.horizon}). "
          f"Flow effect, mm per 1 sd: snow cells {g_('snow cell', 'flow_abs_mm'):.2f}, other cells {g_('other cell', 'flow_abs_mm'):.2f}, read1 {g_('read1', 'flow_abs_mm'):.2f}, "
          f"snow group {g_('snow group', 'flow_abs_mm'):.2f}, probe {g_('probe', 'flow_abs_mm'):.2f}, random {g_('random', 'flow_abs_mm'):.2f}.")

# ---- C: the named unit's swings in April-July ---------------------------------------------------------------------------------------
em = e[np.isin(d.month, (4, 5, 6, 7))]; Hc, Oc, Cc = [], [], []
for k in range(0, len(em), 32):
    _, _, S = net.run_windows(em[k:k + 32], keep=True); Oc.append(S["o"][-1][:, U]); Cc.append(S["c"][-1][:, U]); del S
Oc, Cc = np.concatenate(Oc), np.concatenate(Cc); Hc = Oc * np.tanh(Cc); dm = net.dates[em]; Tm = T[em]
C = pd.DataFrame(dict(date=dm, wy=net.wy[em], h=Hc, o=Oc, tanh_c=np.tanh(Cc), c=Cc, temp=Tm, q_direct_mm=net.w_head[U] * Hc * net.y_scale, q_model=net.q(hT[np.isin(e, em)])))
C.to_csv(OUT / f"nb32_C_unit{U}_melt_{TAG}.csv", index=False)
same_wy = np.diff(C.wy) == 0; dH = np.diff(C.h)[same_wy]; dGate = (np.diff(C.o) * C.tanh_c.values[1:])[same_wy]
var_o = np.corrcoef(dH, dGate)[0, 1] ** 2                                      # share of day-to-day change that the gate term alone explains
print(f"\n== C: unit {U}, April-July, {len(C)} days ==")
print(f"  |tanh c| median {np.median(np.abs(C.tanh_c)):.3f} (1 = saturated); hidden output range {C.h.min():+.2f} to {C.h.max():+.2f}")
print(f"  day-to-day changes in hidden output: share explained by the output gate alone (change in gate x current tanh c) {var_o:.0%}; "
      f"rank correlation of hidden output with air temperature {spearmanr(C.h, C.temp)[0]:+.2f}, with the output gate {spearmanr(C.h, C.o)[0]:+.2f}")
print(f"ANSWER C: unit {U}'s swings move flow directly by at most {np.abs(C.q_direct_mm - C.q_direct_mm.median()).max():.2f} mm/d through the readout (peak modelled flow {C.q_model.max():.1f} mm/d). "
      f"Its effect through other cells is in A/B: flow effect of a 1-sd nudge, by start day, above.")
