"""nb27: where in the state does the model keep the snow it uses? One snow basin, one model version per run.

Lees et al. (2022) show snow can be READ from the cell state with a linear probe. Their reviewer asked how many signals the state
holds and how much they matter; the authors answered with PCA and warned that variance is not importance. This script ranks
directions by what melt-season flow listens to, and tests use directly.

Everything is measured on the cell state on 1 March, against flow from 1 April to 30 September of the same water year.

  A  Read directions. For every melt-season day, the gradient of that day's flow with respect to the 1 March cell state (chained
     Jacobians). Decompose the set of gradients the way PCA decomposes states. How many directions carry most of the sensitivity,
     next to how many PCA components carry most of the variance, and how much of the sensitivity the top PCA components hold.
  B  Probe. A linear probe from the cell state to snow water equivalent, scored on held-out years (ridge; Lees used elastic net).
     How much of the probe direction lies inside the read directions, against chance.
  C  Use (activation patching). Rerun with freezing-day precipitation before 1 March scaled by 0.7 and 1.3. That changes melt flow and
     changes the 1 March cell state. Then put only PART of that state change into the unchanged run and see how much of the flow
     change comes back: the read directions, the top PCA components, the snow-tracking cells, the probe direction, random directions.

  PYTHONPATH=src python src/nb27_read_directions.py                       # Encampment, seed 1; a few minutes on a compute node, CPU
  PYTHONPATH=src python src/nb27_read_directions.py --basin 13235000 --seed 2
  python src/nb27_read_directions.py --toy
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from jtalk_core import Net, ROOT, SNOW, L_WIN, sig

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--unit", default="auto", help="single cell to patch alone: a number, or auto = the cell that tracks snow best")
ap.add_argument("--group-thresh", type=float, default=.8); ap.add_argument("--ddf", type=float, default=3.0); ap.add_argument("--ks", type=int, nargs="*", default=[1, 3, 10, 30])
ap.add_argument("--date", default="03-01", help="month-day of the state that is examined; 04-01 is the water-management convention"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/read_directions"; OUT.mkdir(parents=True, exist_ok=True); TAG = f"{a.basin}_seed{a.seed}" + ("" if a.date == "03-01" else f"_d{a.date}")
DL = f"{int(a.date[3:])} {pd.Timestamp('2001-' + a.date).month_name()}"; _print = print
def print(*x, **k): _print(*[str(v).replace("1 March", DL) for v in x], **k)      # every message names the date that was used
YEARS = range(2009, 2015); pd.set_option("display.width", 220); rng = np.random.default_rng(0)

net = Net(a.seed, a.basin, a.toy); net.gate_check()
if a.toy: net.y_center = 3.0
H = net.H; e = net.ends(); d = net.dates[e]; wy = net.wy[e]; P, T = net.precip(), net.temp(); cold = T < 0
if cold.sum() < 30: cold = T < np.median(T)                                     # toy weather has no freezing days
hT, cT = net.run_windows(e); q_all = net.q(hT); mu, sd = cT.mean(0), cT.std(0) + 1e-9; Z = (cT - mu) / sd      # daily cell state, standardised


def fwd(ee, steps, add=None, X=None):
    """Zero-start windows ending at ee. Returns flow on the last day and the cell state at step steps[w]; `add` [n, H] is added there."""
    X = net.X if X is None else X; n = len(ee); h = np.zeros((n, H)); c = np.zeros((n, H)); cap = np.zeros((n, H)); ar = np.arange(n)
    for s in range(L_WIN):
        z = X[ee - L_WIN + 1 + s] @ net.W_ih.T + net.b + h @ net.W_hh.T
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
        c = f * c + i * g; w = ar[steps == s]
        if len(w):
            cap[w] = c[w]
            if add is not None: c[w] = c[w] + add[w]
        h = o * np.tanh(c)
    return net.q(h), cap


def grad_at(S, steps):
    """d(last-day flow, mm/d) / d(cell state at step steps[w]) for every window: the adjoint of jtalk_core, stopped at that step."""
    n = S["c"].shape[1]; a_h = np.tile(net.w_head, (n, 1)); a_c = np.zeros((n, H)); G = np.zeros((n, H))
    for k in range(L_WIN):
        s = L_WIN - 1 - k; i, f, g, o, c, cp = (S[x][s] for x in ("i", "f", "g", "o", "c", "cp")); tc = np.tanh(c)
        gc = a_c + a_h * o * (1 - tc ** 2); w = steps == s; G[w] = gc[w]
        if s <= steps.min(): break
        gz = np.concatenate([gc * g * i * (1 - i), gc * cp * f * (1 - f), gc * i * (1 - g ** 2), a_h * tc * o * (1 - o)], 1)
        a_h, a_c = gz @ net.W_hh, gc * f
    return G * net.y_scale


# ---- snow reference: SPIReS if there is a file for this basin, else a degree-day index (same as nb19) --------------------------
swe = np.zeros(len(P))
for t in range(1, len(P)): swe[t] = max(swe[t - 1] + P[t] * cold[t] - a.ddf * max(T[t], 0), 0)
REF, REFNAME = swe, f"degree-day snow index (ddf {a.ddf:g})"
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
except Exception as ex: print("  (no SPIReS series for this basin:", type(ex).__name__, "- using the degree-day snow index)")
y_ref = REF[e]; ok = np.isfinite(y_ref)

# snow-tracking cells: |rank correlation| above the threshold in both the biggest and the smallest snow year, same sign
def rho_cells(C, y):
    rc = rankdata(C, axis=0); ry = rankdata(y); rc = (rc - rc.mean(0)) / (rc.std(0) + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12); return (rc * ry[:, None]).mean(0)
pk = {y: np.nanmax(y_ref[wy == y]) for y in YEARS if ok[wy == y].sum() > 100}; big, small = max(pk, key=pk.get), min(pk, key=pk.get)
R2y = np.array([rho_cells(cT[(wy == y) & ok], y_ref[(wy == y) & ok]) for y in (big, small)]); strength = np.where(np.sign(R2y[0]) == np.sign(R2y[1]), np.abs(R2y).min(0), 0)
GROUP = np.where(strength > a.group_thresh)[0]; U = int(np.argmax(strength)) if a.unit == "auto" else int(a.unit)
print(f"snow reference: {REFNAME} | snow years WY{big} (big), WY{small} (small) | {len(GROUP)} cells with |r| > {a.group_thresh:g} in both | single cell: unit {U} (|r| {strength[U]:.2f})")

# ---- pass 1: per year, the unchanged run, the two snowfall runs, and the gradients ------------------------------------------------
Y, Gs = [], []
for y in YEARS:
    tn = int(np.where(net.dates == pd.Timestamp(f"{y}-{a.date}"))[0][0]); t0 = int(np.where(net.dates == pd.Timestamp(f"{y - 1}-10-01"))[0][0])
    m = (wy == y) & np.isin(d.month, range(4, 10)) & (e > tn); ee = e[m]; steps = L_WIN - 1 - (ee - tn)
    q0, c0 = fwd(ee, steps); rec = dict(wy=y, ee=ee, steps=steps, q0=q0, c0=c0)
    for fac in (.7, 1.3):
        X1 = net.X.copy(); rows = np.arange(t0, tn + 1); rows = rows[cold[rows]]; X1[rows, net.p_idx] = (P[rows] * fac - net.p_center) / net.p_scale
        rec[fac] = fwd(ee, steps, X=X1)                                          # (flow, 1 March cell state) with scaled snowfall
    _, _, S = net.run_windows(ee, keep=True); G = grad_at(S, steps); del S; Gs.append(G); rec["G"] = G; Y.append(rec)
G = np.concatenate(Gs) * sd                                                        # mm/d of flow per standard deviation of each cell
r0 = Y[0]; eps = 1e-3 * sd * rng.normal(size=(len(r0["ee"]), H)); lin = (r0["G"] * eps).sum(1); act = fwd(r0["ee"], r0["steps"], add=eps)[0] - r0["q0"]
print(f"gradient check: predicted against actual flow change for a tiny state change, correlation {np.corrcoef(lin, act)[0, 1]:.4f}, ratio {np.abs(act).sum() / (np.abs(lin).sum() + 1e-30):.3f} (both should be 1)")

# ---- A: read directions against variance directions -------------------------------------------------------------------------------
Ur, sr, Vr = np.linalg.svd(G, full_matrices=False); er = sr ** 2 / (sr ** 2).sum()            # rows of Vr: read directions
Up, sp, Vp = np.linalg.svd(Z - Z.mean(0), full_matrices=False); ep = sp ** 2 / (sp ** 2).sum()  # rows of Vp: PCA components
nfor = lambda en, t: int(np.searchsorted(np.cumsum(en), t) + 1); pr = lambda en: float(1 / (en ** 2).sum())
inside = lambda V, k, M: float(((M @ V[:k].T) ** 2).sum() / (M ** 2).sum())
print(f"\n== A: how many directions of the 1 March cell state does melt-season flow read? ({len(G)} melt-season days, {H} cells) ==")
A = pd.DataFrame({"directions": [1, 2, 3, 5, 10, 20, 30, 50]}); A["share_of_flow_sensitivity"] = [er[:k].sum() for k in A.directions]; A["share_of_state_variance_PCA"] = [ep[:k].sum() for k in A.directions]
A["flow_sensitivity_inside_top_PCA"] = [inside(Vp, k, G) for k in A.directions]; A["state_variance_inside_top_read"] = [inside(Vr, k, Z - Z.mean(0)) for k in A.directions]
print(A.round(3).to_string(index=False)); A.to_csv(OUT / f"nb27_A_spectrum_{TAG}.csv", index=False)
print(f"ANSWER A: {nfor(er, .8)} directions carry 80% of melt flow's sensitivity to the 1 March state and {nfor(er, .95)} carry 95% (effective number {pr(er):.1f}). "
      f"PCA needs {nfor(ep, .8)} components for 80% of the state's variance. The top {nfor(ep, .8)} PCA components hold {inside(Vp, nfor(ep, .8), G):.0%} of the flow sensitivity, "
      f"and the top {nfor(er, .8)} read directions hold {inside(Vr, nfor(er, .8), Z - Z.mean(0)):.0%} of the state's variance.")

# ---- B: linear probe for snow, and where its direction sits ---------------------------------------------------------------------------
def ridge(Zt, yt, lam): Zc = Zt - Zt.mean(0); w = np.linalg.solve(Zc.T @ Zc + lam * len(Zt) * np.eye(H), Zc.T @ (yt - yt.mean())); return w, yt.mean() - Zt.mean(0) @ w
yrs = [y for y in YEARS if ok[wy == y].sum() > 100]; best = None
for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    pred = np.full(len(e), np.nan)
    for y in yrs: tr = ok & (wy != y); w_, b_ = ridge(Z[tr], y_ref[tr], lam); pred[(wy == y) & ok] = Z[(wy == y) & ok] @ w_ + b_
    mm = np.isfinite(pred); r2 = 1 - ((y_ref[mm] - pred[mm]) ** 2).sum() / ((y_ref[mm] - y_ref[mm].mean()) ** 2).sum()
    if best is None or r2 > best[0]: best = (r2, lam)
wP, bP = ridge(Z[ok], y_ref[ok], best[1]); wn = wP / np.linalg.norm(wP); probe = lambda c_: ((c_ - mu) / sd) @ wP + bP
gm = np.array([r["G"].sum(0) for r in Y]).mean(0) * sd; gm /= np.linalg.norm(gm)                 # gradient of total melt-season flow
print(f"\n== B: linear probe from the cell state to {REFNAME} ==")
B = pd.DataFrame({"read_directions": a.ks}); B["share_of_probe_direction_inside"] = [float(((Vr[:k] @ wn) ** 2).sum()) for k in a.ks]; B["chance"] = [min(k / H, 1) for k in a.ks]
B["same_for_top_PCA"] = [float(((Vp[:k] @ wn) ** 2).sum()) for k in a.ks]; print(B.round(3).to_string(index=False)); B.to_csv(OUT / f"nb27_B_probe_{TAG}.csv", index=False)
snow_seen = np.median([np.median(probe(r[1.3][1]) - probe(r["c0"])) / max(np.median(probe(r["c0"])), 1e-9) for r in Y])
print(f"ANSWER B: the probe explains {best[0]:.0%} of snow in held-out years. {B.share_of_probe_direction_inside.iloc[-1]:.0%} of its direction lies inside the top {a.ks[-1]} read directions "
      f"(chance {min(a.ks[-1] / H, 1):.0%}); its alignment with the gradient of total melt flow is {abs(float(wn @ gm)):.2f} (1 = same direction, chance about {1 / np.sqrt(H):.2f}). "
      f"With 30% more snowfall the probe reads {snow_seen:+.0%} more snow on 1 March.")

# ---- C: put part of the snow-driven state change into the unchanged run -----------------------------------------------------------
proj = lambda V, k: (lambda dz: dz @ V[:k].T @ V[:k]); mask = lambda idx: (lambda dz: dz * np.isin(np.arange(H), idx))
parts = {"whole state change": lambda dz: dz}
for k in a.ks: parts[f"top {k} read directions"] = proj(Vr, k)
parts[f"everything outside the top {a.ks[-1]} read directions"] = lambda dz: dz - proj(Vr, a.ks[-1])(dz)
for k in a.ks: parts[f"top {k} PCA components"] = proj(Vp, k)
parts["probe direction"] = lambda dz: np.outer(dz @ wn, wn)
if len(GROUP): parts[f"the {len(GROUP)} snow-tracking cells"] = mask(GROUP); parts["all other cells"] = mask(np.setdiff1d(np.arange(H), GROUP))
parts[f"unit {U} alone"] = mask([U])
for j in range(3): Qr = np.linalg.qr(rng.normal(size=(H, 10)))[0].T; parts[f"10 random directions #{j + 1}"] = proj(Qr, 10)
rows = []
for r in Y:
    for fac in (.7, 1.3):
        q1, c1 = r[fac]; dz = (c1 - r["c0"]) / sd; full = q1.sum() / r["q0"].sum() - 1
        for name, fn in parts.items():
            dzp = fn(dz); qp = fwd(r["ee"], r["steps"], add=dzp * sd)[0]
            rows.append(dict(wy=r["wy"], factor=fac, part=name.split(" #")[0], melt_flow_change=qp.sum() / r["q0"].sum() - 1, snowfall_run_change=full,
                             recovered=(qp.sum() - r["q0"].sum()) / (q1.sum() - r["q0"].sum()), share_of_state_change=float((dzp ** 2).sum() / (dz ** 2).sum()),
                             first_order_prediction=float((r["G"] * dzp * sd).sum() / (q1.sum() - r["q0"].sum()))))
C = pd.DataFrame(rows); C.to_csv(OUT / f"nb27_C_patching_{TAG}.csv", index=False)
gC = C.groupby(["part", "factor"], sort=False)[["melt_flow_change", "recovered", "first_order_prediction", "share_of_state_change"]].median().unstack("factor")
print(f"\n== C: scale freezing-day precipitation before 1 March, then put part of the 1 March state change into the unchanged run ==")
print(f"  snowfall x0.7 / x1.3 changes Apr-Sep flow by {C[C.factor == .7].snowfall_run_change.median():+.1%} / {C[C.factor == 1.3].snowfall_run_change.median():+.1%} (median of {len(Y)} years)")
print("  recovered = share of that flow change that comes back; first_order_prediction = the same from the gradients alone; share_of_state_change = size of the part")
print(gC.round(3).to_string()); rec = C.groupby("part", sort=False).recovered.median()
kk = next((k for k in a.ks if rec[f"top {k} read directions"] > .8), None)
print(f"ANSWER C: putting back the whole 1 March state change recovers {rec['whole state change']:.0%} of the snowfall effect on melt flow. "
      + (f"The top {kk} read directions alone recover {rec[f'top {kk} read directions']:.0%}" if kk else f"Even the top {a.ks[-1]} read directions recover only {rec[f'top {a.ks[-1]} read directions']:.0%}")
      + f"; the top {a.ks[-1]} PCA components {rec[f'top {a.ks[-1]} PCA components']:.0%}; the probe direction {rec['probe direction']:.0%}; "
      + (f"the {len(GROUP)} snow-tracking cells {rec[f'the {len(GROUP)} snow-tracking cells']:.0%} and all other cells {rec['all other cells']:.0%}; " if len(GROUP) else "")
      + f"unit {U} alone {rec[f'unit {U} alone']:.0%}; 10 random directions {rec['10 random directions']:.0%}.\n"
      "  If the whole change recovers well below 100%, the snow also travels in the hidden state on 1 March. If first_order_prediction is far from recovered, the response is not linear in the state.")
pd.DataFrame([dict(basin=a.basin, seed=a.seed, snow_ref=REFNAME, n_snow_cells=len(GROUP), unit=U, n80_read=nfor(er, .8), n95_read=nfor(er, .95), eff_read=pr(er), n80_pca=nfor(ep, .8),
                   probe_r2=best[0], probe_inside_top_read=B.share_of_probe_direction_inside.iloc[-1], probe_cos_melt_grad=abs(float(wn @ gm)), **{"rec_" + ("snow-tracking cells" if "snow-tracking" in k else "single snow cell" if k.startswith("unit ") else k): v for k, v in rec.items()})]).to_csv(OUT / f"nb27_summary_{TAG}.csv", index=False)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 2, figsize=(13, 5.2)); kx = np.arange(1, min(60, H) + 1)
ax[0].plot(kx, np.cumsum(er)[:len(kx)], color="#3b6fb6", lw=2.5, label="flow sensitivity, in read directions"); ax[0].plot(kx, np.cumsum(ep)[:len(kx)], color="#c8782a", lw=2.5, label="state variance, in PCA components")
ax[0].plot(kx, [inside(Vp, k, G) for k in kx], color="#c8782a", lw=2, ls="--", label="flow sensitivity, in PCA components"); ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("number of directions"); ax[0].set_ylabel("share captured"); ax[0].legend(frameon=False, fontsize=12); ax[0].set_title("Few directions are read", fontsize=14)
ax[1].plot(a.ks, [rec[f"top {k} read directions"] for k in a.ks], "o-", color="#3b6fb6", lw=2.5, label="read directions"); ax[1].plot(a.ks, [rec[f"top {k} PCA components"] for k in a.ks], "o-", color="#c8782a", lw=2.5, label="PCA components")
for name, col in ([(f"the {len(GROUP)} snow-tracking cells", "#5a9e6f")] if len(GROUP) else []) + [("probe direction", "#8a5fb0"), ("10 random directions", "#888888")]: ax[1].axhline(rec[name], color=col, ls=":", lw=2, label=name)
ax[1].axhline(rec["whole state change"], color="k", lw=1, label="whole state change"); ax[1].set_xscale("log"); ax[1].set_xlabel("number of directions put back"); ax[1].set_ylabel("share of the snowfall effect\non melt flow recovered")
ax[1].legend(frameon=False, fontsize=11); ax[1].set_title("Where the snow the model uses is kept", fontsize=14)
fig.suptitle(f"Basin {a.basin}, model version {a.seed}, cell state on {DL}", fontsize=12); fig.tight_layout(); fig.savefig(OUT / f"nb27_read_directions_{TAG}.png", dpi=200); print("wrote", OUT / f"nb27_read_directions_{TAG}.png")
