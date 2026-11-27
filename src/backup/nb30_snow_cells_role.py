"""nb30: what do the snow-tracking cells do? One snow basin, one model version per run.

nb27 showed that on 1 March the cells that track snow hold almost none of the snow the model uses. Open questions:

  F  Are they all switches, like unit 77? For every snow-tracking cell: how open is its hidden state (beta = o x (1 - tanh^2 c),
     the share of a change in the cell that shows in what it passes on) in winter and in late summer.
  G  Do they hold the AMOUNT of snow?
       G1  With winter snowfall x1.3, by how much does each cell's 1 March value change? A snowfall counter would change by 30%.
       G2  Remove the usual seasonal cycle from snow water equivalent and from each reading, then correlate what is left over all
           days. This asks whether a reading follows snow that is above or below normal for the date, on every day, not on one date.
  H  Do they pass the amount on before 1 March? Rerun the snowfall runs with the snow cells HELD at their normal-run values:
       their hidden output held up to 1 March / their cell and hidden values held all year.
     If the snowfall effect on melt flow survives, the amount never travels through them. The reverse test puts their snowfall-run
     values into the normal run all year.
  I  Does the model use them for TIMING? Warm (and cool) 1 March to 30 June by 2 degrees C. Melt flow shifts earlier (later).
     Hold the snow cells at normal-run values: how much of the shift in timing is lost? Put their warm-run values into the normal
     run: how much of the shift appears? The same number of random cells is the yardstick.
  J  Which cells make up the directions flow reads (needs the nb28 file for this model version): how many cells, and do they
     overlap the snow-tracking cells?

  sbatch slurm/backup/nb30_snow_cells.slurm
  PYTHONPATH=src python src/backup/nb30_snow_cells_role.py --basin 06623800 --seed 1 --unit 77
  python src/backup/nb30_snow_cells_role.py --toy
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from jtalk_core import Net, ROOT, SNOW, L_WIN, sig

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--unit", default="auto"); ap.add_argument("--group-thresh", type=float, default=.8)
ap.add_argument("--ddf", type=float, default=3.0); ap.add_argument("--dT", type=float, default=2.0); ap.add_argument("--date", default="03-01"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/read_directions"; OUT.mkdir(parents=True, exist_ok=True); SUF = "" if a.date == "03-01" else f"_d{a.date}"; TAG = f"{a.basin}_seed{a.seed}{SUF}"
YEARS = range(2009, 2015); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); rng = np.random.default_rng(0)
NAMES = {"06623800": "Encampment R., WY", "12178100": "Newhalem Cr., WA", "13235000": "SF Payette R., ID", "09066300": "Middle Cr., CO", "09312600": "White R., UT"}

net = Net(a.seed, a.basin, a.toy)
if a.toy: net.y_center = 3.0
else: net.gate_check()
H = net.H; e = net.ends(); d = net.dates[e]; wy = net.wy[e]; P, T = net.precip(), net.temp(); cold = T < 0
if cold.sum() < 30: cold = T < np.median(T)
hT, cT = net.run_windows(e)


def fwd(ee, X=None, idx=None, set_c=None, set_h=None, upto=None, rec=False):
    """Zero-start windows ending at ee. For the cells idx: optionally overwrite their cell and/or hidden values at every step
    (set_c, set_h: [L, n, len(idx)]), only up to step upto[w] if given. rec=True also returns those cells' values at every step."""
    X = net.X if X is None else X; n = len(ee); h = np.zeros((n, H)); c = np.zeros((n, H)); Hr = Cr = None
    if rec: Hr = np.zeros((L_WIN, n, len(idx))); Cr = np.zeros((L_WIN, n, len(idx)))
    for s in range(L_WIN):
        z = X[ee - L_WIN + 1 + s] @ net.W_ih.T + net.b + h @ net.W_hh.T
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
        c = f * c + i * g; m = np.ones(n, bool) if upto is None else s <= upto
        if set_c is not None: c[np.ix_(m, idx)] = set_c[s][m]
        h = o * np.tanh(c)
        if set_h is not None: h[np.ix_(m, idx)] = set_h[s][m]
        if rec: Hr[s] = h[:, idx]; Cr[s] = c[:, idx]
    return (net.q(h), Hr, Cr) if rec else net.q(h)


# ---- snow reference and snow-tracking cells (same rule as nb27) -------------------------------------------------------------------
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
U = int(np.argmax(strength)) if a.unit == "auto" else int(a.unit); NG = len(GROUP); OTHERS = np.setdiff1d(np.arange(H), GROUP)
RAND = [np.sort(rng.choice(OTHERS, NG, replace=False)) for _ in range(3)]
print(f"{NAMES.get(a.basin, '')} {a.basin}, model version {a.seed} | snow reference: {REFNAME} | {NG} snow-tracking cells (|r| > {a.group_thresh:g} in WY{big} and WY{small}) | named unit {U}")

# ---- F: are the snow cells switches? -----------------------------------------------------------------------------------------------
# beta on each day = o x (1 - tanh^2 c) at the end of that day's window, from one more pass that keeps the output gate
def beta_daily():
    n = len(e); h = np.zeros((n, H)); c = np.zeros((n, H))
    for s in range(L_WIN):
        z = net.X[e - L_WIN + 1 + s] @ net.W_ih.T + net.b + h @ net.W_hh.T
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:]); c = f * c + i * g; h = o * np.tanh(c)
    return o * (1 - np.tanh(c) ** 2)
BETA = beta_daily(); win = np.isin(d.month, [12, 1, 2, 3]); summ = np.isin(d.month, [8, 9]); SHUT = .05
F = pd.DataFrame(dict(cell=np.arange(H), snow_cell=np.isin(np.arange(H), GROUP), r_with_snow=strength, beta_winter=np.median(BETA[win], 0), winter_days_shut=(BETA[win] < SHUT).mean(0),
                      beta_aug_sep=np.median(BETA[summ], 0), aug_sep_days_shut=(BETA[summ] < SHUT).mean(0)))
F["switch"] = (F.winter_days_shut > .9) & (F.aug_sep_days_shut < .5); F["always_shut"] = (F.winter_days_shut > .9) & (F.aug_sep_days_shut >= .5); F.to_csv(OUT / f"nb30_F_cells_{TAG}.csv", index=False)
fs, fo = F[F.snow_cell], F[~F.snow_cell]
print(f"\n== F: hidden state of each snow-tracking cell (shut = beta below {SHUT}; switch = shut on over 90% of Dec-Mar days and open on at least half of Aug-Sep days) ==")
print(fs.sort_values("r_with_snow", ascending=False).drop(columns="snow_cell").round(3).to_string(index=False))
print(f"ANSWER F: of {NG} snow-tracking cells, {int((fs.winter_days_shut > .9).sum())} are shut through winter ({int(fs.switch.sum())} switches that open by late summer, {int(fs.always_shut.sum())} shut all year). "
      f"Among the other {len(fo)} cells, {(fo.winter_days_shut > .9).mean():.0%} are shut through winter and {fo.switch.mean():.0%} are switches. Unit {U}: winter days shut {F.winter_days_shut[U]:.0%}, Aug-Sep days shut {F.aug_sep_days_shut[U]:.0%}.")

# ---- runs: normal, snowfall x0.7 / x1.3, spring warmer / cooler ------------------------------------------------------------------
ALL = np.unique(np.concatenate([GROUP] + RAND + [[U]])); pos = {c_: k for k, c_ in enumerate(ALL)}; sub = lambda A, idx: A[:, :, [pos[c_] for c_ in idx]]
Y = []
for y in YEARS:
    tn = int(np.where(net.dates == pd.Timestamp(f"{y}-{a.date}"))[0][0]); t0 = int(np.where(net.dates == pd.Timestamp(f"{y - 1}-10-01"))[0][0]); tj = int(np.where(net.dates == pd.Timestamp(f"{y}-06-30"))[0][0])
    m = (wy == y) & np.isin(d.month, range(4, 10)) & (e > tn); ee = e[m]; steps = L_WIN - 1 - (ee - tn); rec = dict(wy=y, ee=ee, steps=steps, i_date=int(np.where(e == tn)[0][0]))
    rec["norm"] = fwd(ee, idx=ALL, rec=True); Xs = {}
    for fac in (.7, 1.3):
        X1 = net.X.copy(); rows = np.arange(t0, tn + 1); rows = rows[cold[rows]]; X1[rows, net.p_idx] = (P[rows] * fac - net.p_center) / net.p_scale; Xs[f"snow x{fac}"] = X1
    for sgn in (1, -1):
        X1 = net.X.copy()
        for ic in net.t_cols: X1[tn:tj + 1, ic] += sgn * a.dT / net.x_scale[ic]
        Xs[f"spring {'+' if sgn > 0 else '-'}{a.dT:g}C"] = X1
    rec["X"] = Xs; rec["pert"] = {k: fwd(ee, X=X1, idx=ALL, rec=True) for k, X1 in Xs.items()}; Y.append(rec)
centre = lambda q: float((np.arange(len(q)) * np.clip(q, 0, None)).sum() / np.clip(q, 0, None).sum())      # day within 1 Apr - 30 Sep at the centre of the flow
SNOWK = [k for k in Y[0]["pert"] if k.startswith("snow")]; TEMPK = [k for k in Y[0]["pert"] if k.startswith("spring")]

# ---- G: do the snow cells hold the amount? ---------------------------------------------------------------------------------------
print("\n== G1: winter snowfall x1.3. Change in each reading on the date, as a share of its normal value (a snowfall counter would give +30%) ==")
g1 = []
for r in Y:
    c0 = r["norm"][2][r["steps"], np.arange(len(r["ee"]))]; c1 = r["pert"]["snow x1.3"][2][r["steps"], np.arange(len(r["ee"]))]      # [n, len(ALL)] on the date
    g1.append(np.median((c1 - c0) / np.where(np.abs(c0) > 1e-6, np.abs(c0), np.nan), 0))
g1 = np.nanmedian(np.array(g1), 0); G1 = pd.DataFrame(dict(cell=ALL, change=g1 * np.sign(np.array([r2[0][c_] for c_ in ALL])))).set_index("cell")
gs = G1.loc[GROUP, "change"]; print(gs.round(3).to_frame("change_towards_more_snow").T.to_string())
print(f"ANSWER G1: with 30% more winter snowfall, the snow-tracking cells move by a median of {gs.median():+.1%} towards 'more snow' (range {gs.min():+.1%} to {gs.max():+.1%}); "
      f"{int((gs > .15).sum())} of {NG} move by more than half of the 30%. Unit {U}: {G1.change.get(U, np.nan):+.1%}.")

doy = d.dayofyear.values
def anomaly(v):
    s = pd.Series(v); return (s - s.groupby(doy).transform("mean")).values
ya = anomaly(np.where(ok, y_ref, np.nan)); snowy = ok & (pd.Series(y_ref).groupby(doy).transform("mean").values > .05 * np.nanmax(y_ref))
cell_anom = np.array([spearmanr(anomaly(cT[:, j])[snowy], ya[snowy])[0] for j in range(H)]) * np.sign(r2[0] + 1e-12); readings = {}
Zc = (cT - cT.mean(0)) / (cT.std(0) + 1e-9); pr_out = np.full(len(e), np.nan)
for y in YEARS:                                                                   # each year is read by a probe fitted without that year
    tr = ok & (wy != y)
    if tr.sum() > 100: Zo = Zc[tr] - Zc[tr].mean(0); w_ = np.linalg.solve(Zo.T @ Zo + 1e-2 * tr.sum() * np.eye(H), Zo.T @ (y_ref[tr] - y_ref[tr].mean())); pr_out[wy == y] = (Zc[wy == y] - Zc[tr].mean(0)) @ w_ + y_ref[tr].mean()
readings["linear snow probe (held-out years)"] = pr_out
pk_file = OUT / f"nb28_directions_seed{a.seed}{SUF}.pkl"; DIRS = pd.read_pickle(pk_file).get(a.basin) if pk_file.exists() else None
if DIRS is not None: readings["directions flow reads (stored melt flow)"] = cT @ DIRS["gm"]
print("\n== G2: seasonal cycle removed (each day compared with the six-year mean for that day of year), days with snow normally on the ground ==")
G2 = pd.DataFrame([dict(reading=k, through_the_year=spearmanr(v[ok], y_ref[ok], nan_policy="omit")[0], above_or_below_normal=spearmanr(anomaly(v)[snowy], ya[snowy], nan_policy="omit")[0]) for k, v in readings.items()]
                  + [dict(reading=f"unit {U}", through_the_year=strength[U], above_or_below_normal=cell_anom[U]), dict(reading=f"snow-tracking cells, median of {NG}", through_the_year=np.median(strength[GROUP]), above_or_below_normal=np.median(cell_anom[GROUP])),
                     dict(reading=f"snow-tracking cells, best of {NG}", through_the_year=np.max(strength[GROUP]), above_or_below_normal=np.max(cell_anom[GROUP])), dict(reading="all other cells, best", through_the_year=np.max(strength[OTHERS]), above_or_below_normal=np.max(np.abs(cell_anom[OTHERS])))])
print(G2.round(2).to_string(index=False)); G2.to_csv(OUT / f"nb30_G2_anomaly_{TAG}.csv", index=False)
print("ANSWER G2: 'through_the_year' is the score a probe study reports. 'above_or_below_normal' is the same reading with the seasonal cycle taken out: it says whether the reading knows a big snowpack from a small one on the same date. "
      "Six years of data, and neighbouring days are not independent, so compare the rows with each other more than with zero.")

# ---- H and I: hold the snow cells at normal-run values, or put the other run's values into the normal run ---------------------------
def path_tests(keys, metric, label):
    rows = []
    for r in Y:
        q0 = r["norm"][0]
        for k in keys:
            q1, H1, C1 = r["pert"][k]; full = metric(q1) - metric(q0)
            for gname, idx in [("snow-tracking cells", GROUP), (f"unit {U}", np.array([U]))] + [(f"random cells", rr) for rr in RAND]:
                Hn, Cn = sub(r["norm"][1], idx), sub(r["norm"][2], idx); Hp, Cp = sub(H1, idx), sub(C1, idx)
                tests = {"held (cell and hidden) all year": fwd(r["ee"], X=r["X"][k], idx=idx, set_c=Cn, set_h=Hn), "other run's values put into the normal run": fwd(r["ee"], idx=idx, set_c=Cp, set_h=Hp)}
                if k in SNOWK: tests["hidden output held up to the date"] = fwd(r["ee"], X=r["X"][k], idx=idx, set_h=Hn, upto=r["steps"])
                for tname, qq in tests.items(): rows.append(dict(wy=r["wy"], run=k, cells=gname, test=tname, full_effect=full, effect=metric(qq) - metric(q0)))
    D_ = pd.DataFrame(rows); D_["share"] = D_.effect / D_.full_effect; D_["share_lost"] = np.where(D_.test.str.startswith("other"), np.nan, 1 - D_.share); D_["share_produced"] = np.where(D_.test.str.startswith("other"), D_.share, np.nan)
    return D_
vol = lambda q: float(q.sum())
Hd = path_tests(SNOWK, vol, "amount"); Hd.to_csv(OUT / f"nb30_H_amount_{TAG}.csv", index=False)
print(f"\n== H: amount. Snowfall x0.7 / x1.3 changes Apr-Sep flow by {np.median([vol(r['pert']['snow x0.7'][0]) / vol(r['norm'][0]) - 1 for r in Y]):+.1%} / {np.median([vol(r['pert']['snow x1.3'][0]) / vol(r['norm'][0]) - 1 for r in Y]):+.1%} ==")
th = Hd.groupby(["cells", "test"], sort=False)[["share_lost", "share_produced"]].median(); print(th.round(3).to_string())
gl = lambda t_, c_="snow-tracking cells", col="share_lost", D_=Hd: float(D_[(D_.cells == c_) & (D_.test == t_)][col].median())
print(f"ANSWER H: holding the snow cells' hidden output at normal values up to the date loses {gl('hidden output held up to the date'):.0%} of the snowfall effect on melt flow; holding their cell and hidden values all year loses {gl('held (cell and hidden) all year'):.0%} "
      f"(the same number of random cells: {gl('held (cell and hidden) all year', 'random cells'):.0%}). Putting their snowfall-run values into the normal run produces {gl(chr(111) + 'ther run' + chr(39) + 's values put into the normal run', col='share_produced'):.0%} of it. "
      "Near zero on all three means the amount does not pass through these cells at any time of year.")

Id = path_tests(TEMPK, centre, "timing"); Id.to_csv(OUT / f"nb30_I_timing_{TAG}.csv", index=False)
shift = {k: np.median([centre(r["pert"][k][0]) - centre(r["norm"][0]) for r in Y]) for k in TEMPK}; dvol = {k: np.median([vol(r["pert"][k][0]) / vol(r["norm"][0]) - 1 for r in Y]) for k in TEMPK}
print(f"\n== I: timing. Temperature changed by {a.dT:g} C from the date to 30 June. Centre of Apr-Sep flow moves by " + ", ".join(f"{v:+.1f} d ({k})" for k, v in shift.items()) + "; flow volume changes by " + ", ".join(f"{v:+.1%}" for v in dvol.values()) + " ==")
ti = Id.groupby(["cells", "test"], sort=False)[["share_lost", "share_produced"]].median(); print(ti.round(3).to_string())
ot = "other run's values put into the normal run"
print(f"ANSWER I: holding the snow cells at normal values loses {gl('held (cell and hidden) all year', D_=Id):.0%} of the shift in melt timing (random cells: {gl('held (cell and hidden) all year', 'random cells', D_=Id):.0%}; unit {U} alone: {gl('held (cell and hidden) all year', f'unit {U}', D_=Id):.0%}). "
      f"Putting their warm-run or cool-run values into the normal run produces {gl(ot, col='share_produced', D_=Id):.0%} of the shift (random cells: {gl(ot, 'random cells', col='share_produced', D_=Id):.0%}). "
      "Well above the random cells means the model does use these cells for timing; near the random cells means it does not.")

# ---- J: which cells make up the directions flow reads -------------------------------------------------------------------------------
if DIRS is not None:
    V = DIRS["Vraw"][:3] * DIRS["sd"]; V = V / np.linalg.norm(V, axis=1, keepdims=True); w2 = (V ** 2).sum(0) / 3; order = np.argsort(-w2); n80 = int(np.searchsorted(np.cumsum(w2[order]), .8) + 1)
    top = order[:15]; J = pd.DataFrame(dict(cell=top, share_of_weight=w2[top], snow_cell=np.isin(top, GROUP), r_with_snow=strength[top], winter_days_shut=F.winter_days_shut.values[top]))
    print("\n== J: the three directions flow reads, cell by cell (each direction is a list of 256 weights, one per cell) =="); print(J.round(3).to_string(index=False)); J.to_csv(OUT / f"nb30_J_direction_cells_{TAG}.csv", index=False)
    print(f"ANSWER J: {n80} cells carry 80% of the weight of the three directions (256 would be an even spread; 1 would be a single cell). The snow-tracking cells carry {w2[GROUP].sum():.0%} of the weight, against {NG / H:.0%} if weight were spread evenly; "
          f"{int(np.isin(order[:n80], GROUP).sum())} of those {n80} cells are snow-tracking cells.")
else: print(f"\n(J skipped: {pk_file.name} not found; run nb28 for this model version first)")

pd.DataFrame([dict(basin=a.basin, seed=a.seed, n_snow_cells=NG, unit=U, snow_cells_shut_in_winter=int((fs.winter_days_shut > .9).sum()), snow_cells_switch=int(fs.switch.sum()), other_cells_switch_share=float(fo.switch.mean()),
                   G1_median_change=float(gs.median()), amount_lost_hidden_to_date=gl("hidden output held up to the date"), amount_lost_held_all_year=gl("held (cell and hidden) all year"), amount_lost_random=gl("held (cell and hidden) all year", "random cells"),
                   amount_produced=gl(ot, col="share_produced"), timing_shift_days=float(np.mean(np.abs(list(shift.values())))), timing_lost_held=gl("held (cell and hidden) all year", D_=Id), timing_lost_random=gl("held (cell and hidden) all year", "random cells", D_=Id),
                   timing_produced=gl(ot, col="share_produced", D_=Id), timing_produced_random=gl(ot, "random cells", col="share_produced", D_=Id))]).to_csv(OUT / f"nb30_summary_{TAG}.csv", index=False)
print("\nwrote tables to", OUT)
