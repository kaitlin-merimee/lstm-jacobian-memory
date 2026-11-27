"""nb28: follow-up to nb27. What are the few directions melt flow reads, and are they shared between basins? One model version per run.

nb27 found that about three directions of the 1 March cell state carry the snowfall effect on melt flow, and that the snow-tracking
cells and a linear snow probe do not. Two questions remain.

  D  What do the read directions represent?
     The gradient of total April-September flow with respect to the 1 March cell state gives ONE coordinate of the state, in mm of
     melt-season flow per unit of state. Call it the stored melt flow. Three tests of whether it is the snowpack amount:
       D1  through the year: does it follow snow water equivalent day by day?
       D2  between years: on 1 March, does it rank the six years by winter snowfall, by snow on the ground and by the melt flow
           that follows? The snow probe and the best single snow cell are scored the same way.
       D3  dose: scale winter snowfall from x0.5 to x1.5. How many mm of melt flow per mm of snowfall does the model deliver, how
           much of that is already written in the stored melt flow on 1 March, and is it proportional?
  E  Are the directions shared? Take the read directions found in basin A and use them to patch basin B (nb27 section C).
     Also one pooled set of directions from all basins together.

  sbatch slurm/nb28_followup.slurm                       # three model versions, CPU
  PYTHONPATH=src python src/nb28_read_directions_followup.py --seed 1
  python src/nb28_read_directions_followup.py --toy
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from jtalk_core import Net, ROOT, L_WIN, sig

ap = argparse.ArgumentParser()
ap.add_argument("--basins", nargs="*", default=["06623800", "12178100", "13235000", "09066300", "09312600"]); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--ks", type=int, nargs="*", default=[3, 10]); ap.add_argument("--ddf", type=float, default=3.0); ap.add_argument("--date", default="03-01", help="month-day of the state that is examined; 04-01 is the water-management convention"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/read_directions"; OUT.mkdir(parents=True, exist_ok=True); TAG = f"seed{a.seed}" + ("" if a.date == "03-01" else f"_d{a.date}")
DL = f"{int(a.date[3:])} {pd.Timestamp('2001-' + a.date).month_name()}"; _print = print
def print(*x, **k): _print(*[str(v).replace("1 March", DL) for v in x], **k)      # every message names the date that was used
YEARS = range(2009, 2015); FACS = (.5, .7, .85, 1.15, 1.3, 1.5); KMAX = 30; pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
if a.toy: a.basins = a.basins[:3]


def fwd(net, ee, steps, add=None, X=None):
    """Zero-start windows ending at ee: flow on the last day and the cell state at step steps[w]; `add` [n, H] is added there."""
    H = net.H; X = net.X if X is None else X; n = len(ee); h = np.zeros((n, H)); c = np.zeros((n, H)); cap = np.zeros((n, H)); ar = np.arange(n)
    for s in range(L_WIN):
        z = X[ee - L_WIN + 1 + s] @ net.W_ih.T + net.b + h @ net.W_hh.T
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
        c = f * c + i * g; w = ar[steps == s]
        if len(w):
            cap[w] = c[w]
            if add is not None: c[w] = c[w] + add[w]
        h = o * np.tanh(c)
    return net.q(h), cap


def grad_at(net, S, steps):
    """d(last-day flow, mm/d) / d(cell state at step steps[w]) for every window."""
    H = net.H; n = S["c"].shape[1]; a_h = np.tile(net.w_head, (n, 1)); a_c = np.zeros((n, H)); G = np.zeros((n, H))
    for k in range(L_WIN):
        s = L_WIN - 1 - k; i, f, g, o, c, cp = (S[x][s] for x in ("i", "f", "g", "o", "c", "cp")); tc = np.tanh(c)
        gc = a_c + a_h * o * (1 - tc ** 2); w = steps == s; G[w] = gc[w]
        if s <= steps.min(): break
        gz = np.concatenate([gc * g * i * (1 - i), gc * cp * f * (1 - f), gc * i * (1 - g ** 2), a_h * tc * o * (1 - o)], 1)
        a_h, a_c = gz @ net.W_hh, gc * f
    return G * net.y_scale


def snow_reference(net, basin, P, T, cold, e):
    swe = np.zeros(len(P))
    for t in range(1, len(P)): swe[t] = max(swe[t - 1] + P[t] * cold[t] - a.ddf * max(T[t], 0), 0)
    try:
        if a.toy: raise FileNotFoundError
        def pick(df):
            dc = [k for k in df.columns if "date" in k.lower()][0]; sc = [k for k in df.columns if "swe" in k.lower()][0]; return sc, pd.Series(df[sc].values, index=pd.to_datetime(df[dc]))
        f_ = ROOT / f"analysis/spires/spires_{basin}.csv"
        if f_.exists(): sc, ser = pick(pd.read_csv(f_))
        else:
            parts = []
            for g_ in sorted((ROOT / "analysis/spires/all").glob("spires_WY*.csv")):
                df = pd.read_csv(g_, dtype=str); bc = [k for k in df.columns if any(w in k.lower() for w in ("basin", "gauge", "gage", "site"))][0]; df = df[df[bc].str.zfill(8) == basin]
                if len(df): sc, s1 = pick(df); parts.append(s1.astype(float))
            ser = pd.concat(parts).sort_index()
        ser = ser[~ser.index.duplicated()].astype(float).reindex(net.dates).values
        if np.isfinite(ser[e]).sum() > 500: return ser, f"SPIReS {sc}"
    except Exception: pass
    return swe, "degree-day snow index"


def rho_cells(C, y):
    rc = rankdata(C, axis=0); ry = rankdata(y); rc = (rc - rc.mean(0)) / (rc.std(0) + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12); return (rc * ry[:, None]).mean(0)


# ---- pass 1: every basin -------------------------------------------------------------------------------------------------------------
R = {}
for bi, b in enumerate(a.basins):
    net = Net(a.seed, b, a.toy) if not a.toy else Net(a.seed + bi, b, True)
    if a.toy: net.y_center = 3.0
    else: net.gate_check()
    H = net.H; e = net.ends(); d = net.dates[e]; wy = net.wy[e]; P, T = net.precip(), net.temp(); cold = T < 0
    if cold.sum() < 30: cold = T < np.median(T)
    hT, cT = net.run_windows(e); q_all = net.q(hT); mu, sd = cT.mean(0), cT.std(0) + 1e-9; Z = (cT - mu) / sd
    REF, REFNAME = snow_reference(net, b, P, T, cold, e); y_ref = REF[e]; ok = np.isfinite(y_ref)
    pk = {y: np.nanmax(y_ref[wy == y]) for y in YEARS if ok[wy == y].sum() > 100}; big, small = max(pk, key=pk.get), min(pk, key=pk.get)
    r2 = np.array([rho_cells(cT[(wy == y) & ok], y_ref[(wy == y) & ok]) for y in (big, small)]); strength = np.where(np.sign(r2[0]) == np.sign(r2[1]), np.abs(r2).min(0), 0); U = int(np.argmax(strength))
    def ridge(tr): Zc = Z[tr] - Z[tr].mean(0); w = np.linalg.solve(Zc.T @ Zc + 1e-2 * tr.sum() * np.eye(H), Zc.T @ (y_ref[tr] - y_ref[tr].mean())); return w, y_ref[tr].mean() - Z[tr].mean(0) @ w
    wP = ridge(ok)[0]; probe_out = np.full(len(e), np.nan)                           # ridge snow probe; each year is read by a probe fitted WITHOUT that year
    for y in YEARS:
        tr = ok & (wy != y)
        if tr.sum() > 100: w_, b_ = ridge(tr); probe_out[wy == y] = Z[wy == y] @ w_ + b_
    yrs, Gs = [], []
    for y in YEARS:
        tn = int(np.where(net.dates == pd.Timestamp(f"{y}-{a.date}"))[0][0]); t0 = int(np.where(net.dates == pd.Timestamp(f"{y - 1}-10-01"))[0][0])
        m = (wy == y) & np.isin(d.month, range(4, 10)) & (e > tn); ee = e[m]; steps = L_WIN - 1 - (ee - tn); rows = np.arange(t0, tn + 1); rows = rows[cold[rows]]
        q0, c0 = fwd(net, ee, steps); rec = dict(wy=y, ee=ee, steps=steps, q0=q0, c0=c0, coldP=float(P[rows].sum()), i_mar=int(np.where(e == tn)[0][0]), swe_mar=float(REF[tn]),
                                                  q_sim=float(q0.sum()), q_obs=float(np.nansum(net.q_obs[ee])) if net.q_obs is not None and np.isfinite(net.q_obs[ee]).mean() > .9 else np.nan)
        for fac in FACS:
            X1 = net.X.copy(); X1[rows, net.p_idx] = (P[rows] * fac - net.p_center) / net.p_scale; rec[fac] = fwd(net, ee, steps, X=X1)
        _, _, S = net.run_windows(ee, keep=True); rec["G"] = grad_at(net, S, steps); del S; Gs.append(rec["G"]); yrs.append(rec)
    G = np.concatenate(Gs); _, sr, Vr = np.linalg.svd(G * sd, full_matrices=False); KM = min(KMAX, H)
    gm = np.mean([r["G"].sum(0) for r in yrs], 0)                                    # mm of Apr-Sep flow per unit of each cell, on 1 March
    R[b] = dict(net=net, H=H, e=e, d=d, wy=wy, cT=cT, q_all=q_all, sd=sd, mu=mu, ref=y_ref, refname=REFNAME, U=U, usign=float(np.sign(r2[0][U]) or 1), wP=wP, probe_out=probe_out, years=yrs,
                Vraw=Vr[:KM] / sd, sraw=sr[:KM], gm=gm)                                     # read directions as gradients in raw cell units
    print(f"[{b}] snow reference: {REFNAME}; best snow cell {U}; {len(G)} melt-season gradients")
pd.to_pickle({b: {k: v for k, v in r.items() if k in ("sd", "mu", "Vraw", "sraw", "gm")} for b, r in R.items()}, OUT / f"nb28_directions_{TAG}.pkl")

# ---- D1: through the year ---------------------------------------------------------------------------------------------------------------
print("\n== D1: stored melt flow (state on each day, read with the 1 March gradient of total Apr-Sep flow) against snow water equivalent, day by day ==")
rows = []
for b, r in R.items():
    stored = r["cT"] @ r["gm"]; stored = stored - np.percentile(stored, 2); r["stored"] = stored; ok = np.isfinite(r["ref"]) & np.isfinite(r["probe_out"]); probe = r["probe_out"]; cell = r["cT"][:, r["U"]] * r["usign"]
    rows.append(dict(basin=b, snow_reference=r["refname"], **{f"{n}_vs_swe": spearmanr(v[ok], r["ref"][ok])[0] for n, v in (("stored_melt_flow", stored), ("probe", probe), ("best_cell", cell))},
                     stored_peak_month=int(pd.Series(stored, index=r["d"]).groupby(r["d"].month).mean().idxmax()), swe_peak_month=int(pd.Series(r["ref"], index=r["d"]).groupby(r["d"].month).mean().idxmax())))
    pd.DataFrame(dict(date=r["d"], stored_melt_flow_mm=stored, snow_reference=r["ref"], probe=probe, best_cell=cell, q_sim=r["q_all"])).to_csv(OUT / f"nb28_D1_daily_{b}_{TAG}.csv", index=False)
D1 = pd.DataFrame(rows); print(D1.round(2).to_string(index=False))
print(f"ANSWER D1: rank correlation with snow water equivalent, median over basins: stored melt flow {D1.stored_melt_flow_vs_swe.median():+.2f}, probe {D1.probe_vs_swe.median():+.2f}, best snow cell {D1.best_cell_vs_swe.median():+.2f}. "
      "The 1 March gradient is used on every day, so away from late winter this is a rough reading. The probe reads each year with weights fitted on the other years.")

# ---- D2: between years -------------------------------------------------------------------------------------------------------------------
rows = []
for b, r in R.items():
    for y in r["years"]:
        c = r["cT"][y["i_mar"]]; rows.append(dict(basin=b, wy=y["wy"], stored_melt_flow=float(c @ r["gm"]), probe=float(r["probe_out"][y["i_mar"]]), best_cell=float(c[r["U"]] * r["usign"]),
                                                  winter_snowfall_mm=y["coldP"], snow_on_1_march=y["swe_mar"], melt_flow_model_mm=y["q_sim"], melt_flow_observed_mm=y["q_obs"]))
D2 = pd.DataFrame(rows); D2.to_csv(OUT / f"nb28_D2_between_years_{TAG}.csv", index=False)
Zs = D2.groupby("basin").transform(lambda v: (v - v.mean()) / (v.std() + 1e-12)) if len(D2) else D2; readers = ["stored_melt_flow", "probe", "best_cell"]; targets = ["winter_snowfall_mm", "snow_on_1_march", "melt_flow_model_mm", "melt_flow_observed_mm"]
tab = pd.DataFrame({t: [spearmanr(Zs[rd], Zs[t], nan_policy="omit")[0] for rd in readers] for t in targets}, index=readers)
print(f"\n== D2: on 1 March, which internal reading ranks the years correctly? Rank correlation of year-to-year departures, {D2.basin.nunique()} basins x {D2.wy.nunique()} years pooled ==")
print(tab.round(2).to_string())
per = D2.groupby("basin").apply(lambda g: pd.Series({rd: spearmanr(g[rd], g.melt_flow_model_mm)[0] for rd in readers})); print("\n  per basin, against the model's melt flow (6 years each):"); print(per.round(2).to_string())
print(f"ANSWER D2: against winter snowfall the stored melt flow scores {tab.loc['stored_melt_flow', 'winter_snowfall_mm']:+.2f}, the probe {tab.loc['probe', 'winter_snowfall_mm']:+.2f}, the best snow cell {tab.loc['best_cell', 'winter_snowfall_mm']:+.2f}. "
      f"Against the melt flow that follows (model): {tab.loc['stored_melt_flow', 'melt_flow_model_mm']:+.2f}, {tab.loc['probe', 'melt_flow_model_mm']:+.2f}, {tab.loc['best_cell', 'melt_flow_model_mm']:+.2f}; "
      f"observed: {tab.loc['stored_melt_flow', 'melt_flow_observed_mm']:+.2f}, {tab.loc['probe', 'melt_flow_observed_mm']:+.2f}, {tab.loc['best_cell', 'melt_flow_observed_mm']:+.2f}. Six years per basin, so read these as rankings, not precise numbers.")

# ---- D3: dose ----------------------------------------------------------------------------------------------------------------------------
rows = []
for b, r in R.items():
    for y in r["years"]:
        for fac in FACS:
            q1, c1 = y[fac]; rows.append(dict(basin=b, wy=y["wy"], factor=fac, snowfall_change_mm=(fac - 1) * y["coldP"], melt_flow_change_mm=float(q1.sum() - y["q0"].sum()),
                                              stored_change_mm=float((y["G"] * (c1 - y["c0"])).sum()), melt_flow_change=float(q1.sum() / y["q0"].sum() - 1)))
D3 = pd.DataFrame(rows); D3["flow_per_mm_snowfall"] = D3.melt_flow_change_mm / D3.snowfall_change_mm; D3["stored_per_mm_snowfall"] = D3.stored_change_mm / D3.snowfall_change_mm; D3.to_csv(OUT / f"nb28_D3_dose_{TAG}.csv", index=False)
g3 = D3.groupby(["basin", "factor"])[["melt_flow_change", "flow_per_mm_snowfall", "stored_per_mm_snowfall"]].median().unstack("factor")
print("\n== D3: scale freezing-day precipitation before 1 March; mm of Apr-Sep flow per mm of snowfall, and how much of it the 1 March stored melt flow already shows ==")
print(g3["flow_per_mm_snowfall"].round(2).rename_axis("flow per mm snowfall, by factor", axis=1).to_string()); print(g3["stored_per_mm_snowfall"].round(2).rename_axis("stored per mm snowfall, by factor", axis=1).to_string())
eff = D3.groupby("basin").flow_per_mm_snowfall.median(); st = D3.groupby("basin").stored_per_mm_snowfall.median(); lo, hi = D3[D3.factor == .5].flow_per_mm_snowfall.median(), D3[D3.factor == 1.5].flow_per_mm_snowfall.median()
print(f"ANSWER D3: each mm of winter snowfall becomes {eff.median():.2f} mm of Apr-Sep flow (basins {eff.min():.2f} to {eff.max():.2f}); the 1 March stored melt flow shows {st.median():.2f} mm ({(st / eff).median():.0%} of it). "
      f"Proportional if the number is the same for less and more snow: {lo:.2f} at x0.5, {hi:.2f} at x1.5.")

# ---- E: are the directions shared between basins? ----------------------------------------------------------------------------------------
def basis(Vraw, sd_target, k):                                                   # source gradients, expressed for the target basin, orthonormal rows
    return np.linalg.qr((Vraw[:k] * sd_target).T)[0].T
pool = np.concatenate([r["sraw"][:, None] * r["Vraw"] / np.linalg.norm(r["sraw"]) for r in R.values()]); Vpool = np.linalg.svd(pool, full_matrices=False)[2]
rows = []
for bt, rt in R.items():
    for src, Vsrc in [(bs, rs["Vraw"]) for bs, rs in R.items()] + [("pooled", Vpool)]:
        for k in a.ks:
            B = basis(Vsrc, rt["sd"], min(k, len(Vsrc))); recs = []
            for y in rt["years"]:
                for fac in (.7, 1.3):
                    q1, c1 = y[fac]; dz = (c1 - y["c0"]) / rt["sd"]; qp = fwd(rt["net"], y["ee"], y["steps"], add=(dz @ B.T @ B) * rt["sd"])[0]; recs.append((qp.sum() - y["q0"].sum()) / (q1.sum() - y["q0"].sum()))
            rows.append(dict(target=bt, source=src, k=k, recovered=float(np.median(recs))))
E = pd.DataFrame(rows); E.to_csv(OUT / f"nb28_E_transfer_{TAG}.csv", index=False)
print("\n== E: share of the snowfall effect recovered in the TARGET basin (rows) using read directions from the SOURCE basin (columns) ==")
for k in a.ks: print(f"\n  top {k} directions"); print(E[E.k == k].pivot(index="target", columns="source", values="recovered")[list(R) + ["pooled"]].round(2).to_string())
k0 = a.ks[0]; Ek = E[E.k == k0]; own = Ek[Ek.target == Ek.source].recovered; oth = Ek[(Ek.target != Ek.source) & (Ek.source != "pooled")].recovered; po = Ek[Ek.source == "pooled"].recovered
print(f"ANSWER E: with the top {k0} directions, a basin's own directions recover {own.median():.0%}; another basin's directions recover {oth.median():.0%} (range {oth.min():.0%} to {oth.max():.0%}); "
      f"one pooled set recovers {po.median():.0%} (range {po.min():.0%} to {po.max():.0%}). Close to the own-basin number means one shared snow store across basins; close to zero means each basin uses its own.")

# ---- figures -----------------------------------------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}); nb = len(R)
fig, ax = plt.subplots(nb, 1, figsize=(12, 2.4 * nb), sharex=True); ax = np.atleast_1d(ax)
for x_, (b, r) in zip(ax, R.items()):
    x_.plot(r["d"], r["stored"], color="#3b6fb6", lw=1.8); x_.set_ylabel("stored melt\nflow (mm)", color="#3b6fb6"); x2 = x_.twinx(); x2.plot(r["d"], r["ref"], color="#888888", lw=1.3); x2.set_ylabel("snow", color="#666666"); x2.spines["top"].set_visible(False)
    x_.set_title(f"basin {b}  (grey: {r['refname']})", fontsize=12, loc="left")
fig.tight_layout(); fig.savefig(OUT / f"nb28_D1_stored_melt_flow_{TAG}.png", dpi=200)
fig, ax = plt.subplots(1, 3, figsize=(15, 5)); cols = plt.cm.tab10(np.arange(nb))
for x_, rd, lab in zip(ax, readers, ("stored melt flow (read directions)", "linear snow probe", "best single snow cell")):
    for cc, b in zip(cols, R): mm = D2.basin == b; x_.scatter(Zs.loc[mm, rd], Zs.loc[mm, "melt_flow_model_mm"], color=cc, s=45, label=b)
    x_.axhline(0, color="k", lw=.5); x_.axvline(0, color="k", lw=.5); x_.set_xlabel(f"{lab}\non {DL} (departure from basin mean)"); x_.set_title(f"rank correlation {tab.loc[rd, 'melt_flow_model_mm']:+.2f}", fontsize=13)
ax[0].set_ylabel("Apr-Sep flow that follows\n(departure from basin mean)"); ax[0].legend(frameon=False, fontsize=10, title="basin"); fig.tight_layout(); fig.savefig(OUT / f"nb28_D2_between_years_{TAG}.png", dpi=200)
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5)); M = E[E.k == k0].pivot(index="target", columns="source", values="recovered")[list(R) + ["pooled"]]; im = ax.imshow(M.values, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(M.shape[1]), M.columns, rotation=45, ha="right"); ax.set_yticks(range(M.shape[0]), M.index); ax.set_xlabel("directions taken from"); ax.set_ylabel("used in basin")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M.values[i, j]:.0%}", ha="center", va="center", color="w" if M.values[i, j] > .6 else "k", fontsize=11)
ax.set_title(f"Snowfall effect recovered, top {k0} directions", fontsize=13); fig.tight_layout(); fig.savefig(OUT / f"nb28_E_transfer_{TAG}.png", dpi=200)
print("wrote figures and tables to", OUT)
