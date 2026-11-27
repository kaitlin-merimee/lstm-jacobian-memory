"""nb19: open questions from the 16 Nov slide review. One section per question; each prints an ANSWER block.

  A  Is the snow unit's sign flip tied to the melt hydrograph, in every year?  (sim and observed Q)
  B  What does the counter count: days, cold days, or snowfall amount?
  C  Is it the only snow unit? How many units track snow, and how much better is the whole state?
  D  Does the WHOLE MODEL know snow volume? Scale cold-day precipitation and follow melt-season flow
     (and whether the counter and its flip respond).
  E  Rain -> discharge lag of the model by a finite rain pulse (signed; cross-checks nb17's 97 d / 42 d).
  F  Recession timescale of the observed and the simulated hydrograph.
  G  How interconnected are the units? (shares of the one-day Jacobian)
  H  nb18 restated as "fraction of a state direction still present after 365 days".
  I  Degree days: what drives the counter's decline, and do any units track positive degree-days?
  J  Where is the snow volume? Change ALL snow-tracking cells together (and the unit alone, and random cells) and follow melt flow.

Other basins and seeds:  --basin 12178100 --seed 2 --unit auto --only ABCDJ
  --unit auto picks the snow unit by the nb12 rule (rank correlation of each cell with snow in the biggest and the smallest
  snow year, same sign, |rho| > 0.8, weighted by readout weight). The snow reference is SPIReS SWE when a file is found for
  the basin, otherwise a degree-day snow index computed from the model's own precipitation and temperature inputs.
  Each run writes a one-row summary; `python src/backup/nb22_checks.py --collect` prints them side by side.

  PYTHONPATH=src python src/backup/nb19_open_questions.py            # about 10 min on 8 cores (compute node)
  python src/backup/nb19_open_questions.py --toy                     # code test on a random LSTM
Not answered here: how unit 77 was first picked (nb12), and the NSE-loss question (needs a retraining).
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import Net, ROOT, SNOW, ARID, L_WIN

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--other", default=ARID); ap.add_argument("--unit", default="77", help="unit number, or auto")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--group-thresh", type=float, default=0.8, help="|r| with snow for a cell to count as snow-tracking (J)")
ap.add_argument("--ddf", type=float, default=3.0, help="degree-day factor (mm/degC/d) for the snow index in C")
ap.add_argument("--pulse", type=float, default=10.0, help="rain pulse in mm for E")
ap.add_argument("--only", default="ABCDEFGHIJ"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/open_questions"; OUT.mkdir(parents=True, exist_ok=True)
YEARS = range(2009, 2015); NaT = pd.NaT; TAG = f"{a.basin}_seed{a.seed}"; SUM = dict(basin=a.basin, seed=a.seed)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})

net = Net(a.seed, a.basin, a.toy); net.gate_check()
if a.toy: net.y_center = 3.0                       # keeps toy discharge positive
e = net.ends(); d = net.dates[e]; wy = net.wy[e]
h, c = net.run_windows(e); q = net.q(h)
P, T = net.precip(), net.temp(); cold = T < 0
qobs = net.q_obs[e] if net.q_obs is not None else (q * (1 + .1 * np.random.default_rng(0).normal(size=len(q))) if a.toy else None)


def flip_dates(cu, dd):
    p = int(np.argmax(np.abs(cu))); s = np.sign(cu[p]); after = np.arange(p + 1, len(cu))
    fl = after[np.sign(cu[after]) != s]; un = after[np.abs(cu[after]) < 2]
    return (dd[fl[0]] if len(fl) else NaT), (dd[un[0]] if len(un) else NaT), dd[p], cu[p]


def with_X(X1, ends):
    X0, net.X = net.X, X1
    try: return net.run_windows(ends)
    finally: net.X = X0


# one full Oct-Sep window per water year, with gates: used by the unit selection and by B, C, G, I
e30 = np.array([np.where(net.dates == pd.Timestamp(f"{y}-09-30"))[0][0] for y in YEARS])
_, _, S = net.run_windows(e30, keep=True)
idx = e30[None, :] - L_WIN + 1 + np.arange(L_WIN)[:, None]                   # row of X for every step [365, 6]

# ---- snow reference (SPIReS if there is a file for this basin, else a degree-day index) and the snow unit -------
swe = np.zeros(len(P))
for t in range(1, len(P)): swe[t] = max(swe[t - 1] + P[t] * cold[t] - a.ddf * max(T[t], 0), 0)
SNOWREF = {f"degree-day snow index (ddf {a.ddf:g})": swe}                       # daily series aligned with net.dates
def read_spires(basin):
    """Per-basin file first, then the all-basin yearly files. Returns a daily series on net.dates, or raises."""
    def pick(df):
        dc = [k for k in df.columns if "date" in k.lower()][0]; sc = [k for k in df.columns if "swe" in k.lower()][0]
        return sc, pd.Series(df[sc].values, index=pd.to_datetime(df[dc]))
    f = ROOT / f"analysis/spires/spires_{basin}.csv"
    if f.exists(): sc, ser = pick(pd.read_csv(f))
    else:
        parts = []
        for g in sorted((ROOT / "analysis/spires/all").glob("spires_WY*.csv")):
            df = pd.read_csv(g, dtype=str); bc = [k for k in df.columns if any(w in k.lower() for w in ("basin", "gauge", "gage", "site"))][0]
            df = df[df[bc].str.zfill(8) == basin]
            if len(df): sc, ser = pick(df); parts.append(ser.astype(float))
        ser = pd.concat(parts).sort_index()
    return sc, ser[~ser.index.duplicated()].reindex(net.dates).values.astype(float)
try:
    if a.toy: raise FileNotFoundError
    sc, ser = read_spires(a.basin)
    if np.isfinite(ser[idx]).sum() > 500: SNOWREF[f"SPIReS {sc}"] = ser
except Exception as ex: print("  (no SPIReS series for this basin:", type(ex).__name__, "- using the degree-day snow index)")
REFNAME = list(SNOWREF)[-1]; REF = SNOWREF[REFNAME]; targets = {k: v[idx] for k, v in SNOWREF.items()}
from scipy.stats import rankdata
def spearman_cells(C, y):
    ok = np.isfinite(y)
    if ok.sum() < 100: return np.zeros(C.shape[1])
    rc = rankdata(C[ok], axis=0); ry = rankdata(y[ok]); rc = (rc - rc.mean(0)) / (rc.std(0) + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return (rc * ry[:, None]).mean(0)
RHO = np.array([spearman_cells(S["c"][:, w], REF[idx][:, w]) for w in range(len(e30))])       # [year, unit]
peaks = np.array([np.nanmax(REF[idx][:, w]) if np.isfinite(REF[idx][:, w]).sum() > 100 else np.nan for w in range(len(e30))])
big, small = int(np.nanargmax(peaks)), int(np.nanargmin(peaks)); wabs = np.abs(net.w_head) / np.abs(net.w_head).max()
min_abs = np.minimum(np.abs(RHO[big]), np.abs(RHO[small])); same = np.sign(RHO[big]) == np.sign(RHO[small])
score = min_abs * wabs * (same & (min_abs > 0.8)); AUTO = int(np.argmax(score)) if score.max() > 0 else int(np.argmax(min_abs * same))
okp = np.isfinite(REF[idx]).ravel()
RPOOL = np.array([abs(np.corrcoef(S["c"][:, :, j].ravel()[okp], REF[idx].ravel()[okp])[0, 1]) if S["c"][:, :, j].std() > 0 else 0 for j in range(net.H)])
GROUP = np.where(RPOOL > a.group_thresh)[0]
U = AUTO if str(a.unit).lower() == "auto" else (3 if a.toy else int(a.unit))
print(f"\nsnow reference: {REFNAME} | selection years WY{list(YEARS)[big]} (big) and WY{list(YEARS)[small]} (small) | cells passing the nb12 rule: {int((score > 0).sum())}")
print("  top by the nb12 rule: " + ", ".join(f"{j} (min|rho| {min_abs[j]:.2f}, weight {wabs[j]:.2f})" for j in np.argsort(score)[::-1][:5]) + f" | using unit {U}"
      + (" (auto)" if str(a.unit).lower() == "auto" else f" (given; auto would pick {AUTO})"))
SUM.update(unit=U, auto_unit=AUTO, snow_reference=REFNAME.split(" (")[0], cells_passing_rule=int((score > 0).sum()), snow_cells=len(GROUP), unit_min_abs_rho=float(min_abs[U]))
cU = S["c"][:, :, U]; ipk = np.abs(cU).argmax(0); sgn = np.sign(cU[ipk, np.arange(len(e30))])

# ---- A: flip timing vs the melt hydrograph -------------------------------------------------------
if "A" in a.only:
    rows = []; fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    for y, x in zip(YEARS, ax.ravel()):
        m = wy == y; dd = d[m]; fl, un, pk, cpk = flip_dates(c[m, U], dd); r = dict(wy=y, peak_c=cpk, turn_date=pk, unsat_date=un, flip_date=fl)
        tg = REF[e][m]
        if np.isfinite(tg).sum() > 100 and np.nanmax(tg) > 0:
            ip = int(np.nanargmax(tg)); off = np.where((np.arange(len(tg)) > ip) & (tg < .05 * tg[ip]))[0]
            r.update(snow_peak_date=dd[ip], turn_minus_snow_peak_d=(pk - dd[ip]).days)
            if len(off) and fl is not NaT: r.update(snow_off_date=dd[off[0]], flip_minus_snow_off_d=(fl - dd[off[0]]).days)
        for name, qq in (("sim", q[m]), ("obs", None if qobs is None else qobs[m])):
            if qq is None or np.isnan(qq).mean() > .1: continue
            qq = pd.Series(qq).interpolate(limit_direction="both").values; s = np.isin(dd.month, range(4, 10))
            iq = int(np.argmax(np.where(s, qq, -np.inf))); cum = np.cumsum(np.where(s, qq, 0)); ih = int(np.searchsorted(cum, cum[-1] / 2))
            qq = np.clip(qq, 0, None); base = qq[s].min(); rise = np.where(s & (qq - base >= .2 * (qq[iq] - base)))[0][0]
            r.update({f"{name}_rise_date": dd[rise], f"{name}_peak_date": dd[iq], f"{name}_half_flow_date": dd[ih],
                      f"turn_minus_{name}_rise_d": (pk - dd[rise]).days})
            if fl is not NaT:
                r.update({f"flip_minus_{name}_peak_d": (fl - dd[iq]).days, f"flip_minus_{name}_half_d": (fl - dd[ih]).days,
                          f"{name}_q_at_flip_over_peak": qq[dd == fl][0] / qq[iq], f"{name}_flow_done_at_flip": cum[dd == fl][0] / cum[-1]})
            x.plot(dd, qq, color="k" if name == "sim" else "0.6", lw=1.5 if name == "sim" else 1, label=name)
        for t, col, lab in ((pk, "#3b6fb6", "count turns"), (un, "#8a4fa0", "un-saturates"), (fl, "#e08a00", "sign flip")):
            if t is not NaT: x.axvline(t, color=col, lw=1.5, label=lab)
        x.set_title(f"WY{y}", loc="left"); x.set_xlim(pd.Timestamp(f"{y}-03-01"), pd.Timestamp(f"{y}-09-30")); x.tick_params(axis="x", rotation=30)
        rows.append(r)
    ax[0, 0].legend(frameon=False, fontsize=10); ax[0, 0].set_ylabel("Q (mm/d)"); ax[1, 0].set_ylabel("Q (mm/d)")
    plt.tight_layout(); plt.savefig(OUT / f"nb19_A_flip_vs_hydrograph_{TAG}.png", dpi=220); plt.close()
    A = pd.DataFrame(rows); A.to_csv(OUT / f"nb19_A_flip_timing_{TAG}.csv", index=False)
    print("\n== A: flip timing vs hydrograph =="); print(A.filter(regex="wy|minus|over_peak|done_at").round(2).to_string(index=False))
    if "turn_minus_snow_peak_d" in A:
        indep = A[~A.wy.isin([list(YEARS)[big], list(YEARS)[small]])]
        print(f"ANSWER A (snow, {REFNAME}): the count turns {A.turn_minus_snow_peak_d.median():+.0f} d from peak snow (range {A.turn_minus_snow_peak_d.min():+.0f} to "
              f"{A.turn_minus_snow_peak_d.max():+.0f})" + (f"; the sign flip is {A.flip_minus_snow_off_d.median():+.0f} d from snow-off (snow below 5% of its peak)." if "flip_minus_snow_off_d" in A else "."))
        print(f"  the four years not used to pick the unit: turn {indep.turn_minus_snow_peak_d.median():+.0f} d from peak snow"
              + (f", flip {indep.flip_minus_sim_peak_d.median():+.0f} d from peak flow." if "flip_minus_sim_peak_d" in indep else "."))
        SUM.update(turn_minus_snow_peak_d=A.turn_minus_snow_peak_d.median(), flip_minus_snow_off_d=A.flip_minus_snow_off_d.median() if "flip_minus_snow_off_d" in A else np.nan)
    for k_ in ("flip_minus_sim_peak_d", "turn_minus_sim_rise_d", "sim_flow_done_at_flip", "flip_minus_obs_peak_d"):
        if k_ in A: SUM[k_] = A[k_].median()
    SUM["years_with_flip"] = int(A.flip_date.notna().sum()); SUM["peak_count"] = float(A.peak_c.abs().median())
    for name in ("sim", "obs"):
        if f"flip_minus_{name}_peak_d" in A:
            print(f"ANSWER A ({name}): sign flip is {A[f'flip_minus_{name}_peak_d'].median():+.0f} d from peak flow (range {A[f'flip_minus_{name}_peak_d'].min():+.0f} to "
                  f"{A[f'flip_minus_{name}_peak_d'].max():+.0f}); by the flip {A[f'{name}_flow_done_at_flip'].median():.0%} of Apr-Sep flow has passed and Q is "
                  f"{A[f'{name}_q_at_flip_over_peak'].median():.0%} of peak. The count turns {A[f'turn_minus_{name}_rise_d'].median():+.0f} d from the start of the rise.")


# ---- B: what does the counter count? -----------------------------------------------------------------
if "B" in a.only:
    inc = (S["c"][:, :, U] - S["cp"][:, :, U]) * sgn                           # daily change, positive = counting up
    acc = np.arange(L_WIN)[:, None] <= ipk[None, :]                            # accumulation phase: 1 Oct to the turn
    y_ = inc[acc]; cand = {"every day the same (constant)": np.ones(acc.sum()), "cold day (T < 0)": cold[idx][acc].astype(float),
                           "air temperature": T[idx][acc], "precipitation": P[idx][acc], "snowfall (P on cold days)": (P * cold)[idx][acc]}
    r2 = {k: (0.0 if v.std() == 0 else np.corrcoef(v, y_)[0, 1] ** 2) for k, v in cand.items()}
    Xb = np.c_[np.ones(acc.sum()), cand["cold day (T < 0)"], cand["snowfall (P on cold days)"]]; beta = np.linalg.lstsq(Xb, y_, rcond=None)[0]
    r2both = 1 - ((y_ - Xb @ beta) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    cd = cand["cold day (T < 0)"] > 0; snow = cand["snowfall (P on cold days)"]
    print("\n== B: what the counter counts (accumulation phase, six water years) ==")
    print(f"  mean daily increment {y_.mean():.3f} (cold days {y_[cd].mean():.3f}, other days {y_[~cd].mean():.3f}); mean forget gate {S['f'][:, :, U][acc].mean():.4f}")
    mean_ = lambda m: y_[m].mean() if m.any() else np.nan
    print(f"  on cold days: increment with snowfall > 5 mm {mean_(cd & (snow > 5)):.3f} vs dry cold days {mean_(cd & (snow < 0.1)):.3f}")
    for k, v in r2.items(): print(f"  R2 of daily increment vs {k:32s} {v:.2f}")
    SUM.update(increment_r2_precip=r2["precipitation"], increment_r2_cold_day=r2["cold day (T < 0)"], increment_r2_temperature=r2["air temperature"])
    print(f"ANSWER B: cold-day indicator explains {r2['cold day (T < 0)']:.0%} of the daily increment, snowfall amount {r2['snowfall (P on cold days)']:.0%}, both {r2both:.0%}. "
          f"A unit that counts snow WATER needs the snowfall term to dominate; a season timer does not.")
    pd.Series(r2).to_csv(OUT / f"nb19_B_counter_increment_r2_{TAG}.csv")

# ---- C: is it the only snow unit? -----------------------------------------------------------------------
if "C" in a.only:
    hh = S["o"] * np.tanh(S["c"])

    def loyo_r2(F, y, alpha=10.0):
        pred = np.full(y.shape, np.nan)
        for w in range(F.shape[1]):
            tr = [k for k in range(F.shape[1]) if k != w]; A_ = F[:, tr].reshape(-1, F.shape[2]); b_ = y[:, tr].ravel(); ok = np.isfinite(b_)
            A_, b_ = A_[ok], b_[ok]; mu, sd = A_.mean(0), A_.std(0) + 1e-9; A_ = (A_ - mu) / sd
            W = np.linalg.solve(A_.T @ A_ + alpha * np.eye(A_.shape[1]), A_.T @ (b_ - b_.mean())); pred[:, w] = ((F[:, w] - mu) / sd) @ W + b_.mean()
        ok = np.isfinite(y); return 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()

    print("\n== C: which units track snow? ==")
    for name, y_ in targets.items():
        ok = np.isfinite(y_).ravel()
        if ok.sum() < 200: print(f"  {name}: too few values"); continue
        rc = np.array([abs(np.corrcoef(S["c"][:, :, j].ravel()[ok], y_.ravel()[ok])[0, 1]) for j in range(net.H)])
        rank = int((rc > rc[U]).sum()) + 1; top = np.argsort(rc)[::-1][:8]
        res = dict(all_h=loyo_r2(hh, y_), all_c=loyo_r2(S["c"], y_), unit_c=loyo_r2(S["c"][:, :, [U]], y_), unit_h=loyo_r2(hh[:, :, [U]], y_),
                   all_h_without_unit=loyo_r2(np.delete(hh, U, 2), y_))
        print(f"  target: {name}\n    |r| of unit {U}'s cell: {rc[U]:.2f} (rank {rank} of {net.H}); units with |r| > 0.8: {(rc > .8).sum()}, > 0.6: {(rc > .6).sum()}; top: "
              + ", ".join(f"{j} ({rc[j]:.2f})" for j in top))
        print(f"ANSWER C ({name}): held-out-year R2 from unit {U}'s cell {res['unit_c']:.2f}, its hidden value {res['unit_h']:.2f}, all hidden values {res['all_h']:.2f}, "
              f"all hidden values without unit {U} {res['all_h_without_unit']:.2f}, all cells {res['all_c']:.2f}.")
        if name == REFNAME: SUM.update(unit_rank=rank, unit_abs_r=float(rc[U]), cells_r_gt_0p8=int((rc > .8).sum()), swe_r2_unit_cell=res["unit_c"],
                                       swe_r2_unit_hidden=res["unit_h"], swe_r2_all_hidden=res["all_h"], swe_r2_all_hidden_without_unit=res["all_h_without_unit"])
        pd.DataFrame({"abs_r_cell": rc}).to_csv(OUT / f"nb19_C_unit_snow_corr_{TAG}_{name.split()[0]}.csv")

# ---- D: does the whole model know snow volume? ------------------------------------------------------------
if "D" in a.only:
    rows = []
    for fac in (0.5, 0.7, 1.3, 1.5):
        X1 = net.X.copy(); X1[cold, net.p_idx] += (fac - 1) * P[cold] / net.p_scale
        h1, c1 = with_X(X1, e); q1 = net.q(h1)
        for y in YEARS:
            m = wy == y; dd = d[m]; s = np.isin(dd.month, range(4, 10)); f0, _, _, c0 = flip_dates(c[m, U], dd); f1, _, _, cn = flip_dates(c1[m, U], dd)
            win = (net.wy == y)
            rows.append(dict(wy=y, factor=fac, snowfall_share_of_precip=P[win & cold].sum() / P[win].sum(), melt_flow_change=q1[m][s].sum() / q[m][s].sum() - 1,
                             peak_flow_change=q1[m][s].max() / q[m][s].max() - 1, peak_date_shift_d=(dd[s][q1[m][s].argmax()] - dd[s][q[m][s].argmax()]).days,
                             counter_peak_change=cn / c0 - 1, flip_shift_d=(f1 - f0).days if (f0 is not NaT and f1 is not NaT) else np.nan))
    D = pd.DataFrame(rows); D["elasticity"] = D.melt_flow_change / (D.factor - 1); D.to_csv(OUT / f"nb19_D_precip_scaling_{TAG}.csv", index=False)
    g = D.groupby("factor").agg(melt_flow_change=("melt_flow_change", "median"), peak_flow_change=("peak_flow_change", "median"), peak_date_shift_d=("peak_date_shift_d", "median"),
                                counter_peak_change=("counter_peak_change", "median"), flip_shift_d=("flip_shift_d", "median"), elasticity=("elasticity", "median"))
    print("\n== D: scale precipitation on cold days (median over six water years) =="); print(g.round(3))
    print(f"  cold-day precipitation is {D.snowfall_share_of_precip.median():.0%} of annual precipitation")
    lo, hi = g.loc[0.7], g.loc[1.3]
    SUM.update(precip_m30_melt_flow=lo.melt_flow_change, precip_p30_melt_flow=hi.melt_flow_change, precip_m30_unit_peak=lo.counter_peak_change, precip_p30_unit_peak=hi.counter_peak_change)
    print(f"ANSWER D: -30% / +30% snowfall changes Apr-Sep flow by {lo.melt_flow_change:+.0%} / {hi.melt_flow_change:+.0%} (whole model), "
          f"the counter's peak by {lo.counter_peak_change:+.0%} / {hi.counter_peak_change:+.0%} and its flip date by {lo.flip_shift_d:+.0f} / {hi.flip_shift_d:+.0f} d. "
          f"How to read it: flow changes but the counter does not -> the model uses snow volume, held outside this unit; neither changes -> the model does not use "
          f"winter precipitation volume for melt flow; both change -> the unit does track amount. Static attributes (e.g. mean precipitation) were not changed.")

# ---- E, F: both basins ---------------------------------------------------------------------------------------
def rec_tau(qq):
    qq = np.asarray(qq, float); dq = np.diff(qq); run = np.zeros(len(dq), int)
    for i in range(len(dq)): run[i] = (run[i - 1] + 1 if i else 1) if dq[i] < 0 else 0
    ok = (run >= 5) & np.isfinite(dq) & (qq[1:] > 0); tau = -(qq[1:][ok] + qq[:-1][ok]) / 2 / dq[ok]
    return (np.median(tau), np.percentile(tau, 25), np.percentile(tau, 75), int(ok.sum())) if ok.sum() > 20 else (np.nan,) * 3 + (int(ok.sum()),)

if "E" in a.only or "F" in a.only:
    rowsE, rowsF = [], []; fig, ax = plt.subplots(figsize=(9, 4.8))
    for basin, col in ((a.basin, "#3b6fb6"), (a.other, "#c8782a")):
        nb = net if basin == a.basin else Net(a.seed, basin, a.toy); net = nb; nb.y_center = 3.0 if a.toy else nb.y_center                    # with_X uses `net`
        eb = nb.ends(); qb = nb.q(nb.run_windows(eb)[0]); Tb = nb.temp()
        if "F" in a.only:
            qo = nb.q_obs[eb] if nb.q_obs is not None else (qb * (1 + .1 * np.random.default_rng(1).normal(size=len(qb))) if a.toy else None)
            for name, qq in (("simulated", qb), ("observed", qo)):
                if qq is None: continue
                md, p25, p75, n = rec_tau(pd.Series(qq).interpolate(limit_direction="both").values)
                rowsF.append(dict(basin=basin, hydrograph=name, recession_tau_median_d=md, p25=p25, p75=p75, n_days=n))
        if "E" in a.only:
            t0s = np.array([np.where(nb.dates == t)[0][0] for t in pd.date_range("2008-10-01", "2013-09-01", freq="MS")])
            R = []
            for t0 in t0s:
                ee = t0 + np.arange(L_WIN); h0, _ = nb.run_windows(ee); X1 = nb.X.copy(); X1[t0, nb.p_idx] += a.pulse / nb.p_scale
                R.append((nb.q(with_X(X1, ee)[0]) - nb.q(h0)) / a.pulse)
            R = np.array(R); np.save(OUT / f"nb19_E_pulse_response_{basin}.npy", R.astype(np.float32)); pc = Tb[t0s] < 0
            for name, m in (("all pulses", np.ones(len(t0s), bool)), ("pulse on a cold day", pc), ("pulse on a warm day", ~pc)):
                if m.sum() < 3: continue
                r = R[m].mean(0); tot = r.sum(); cum = np.cumsum(r)
                rowsE.append(dict(basin=basin, pulses=name, n=int(m.sum()), returned_within_365d=tot, M50_d=int(np.argmax(cum >= tot / 2)) if tot > 0 else np.nan,
                                  M90_d=int(np.argmax(cum >= .9 * tot)) if tot > 0 else np.nan, mean_lag_d=(np.arange(L_WIN) * r).sum() / tot if tot > 0 else np.nan,
                                  negative_share=np.clip(-r, 0, None).sum() / np.abs(r).sum(), day0_share=r[0] / tot if tot > 0 else np.nan))
            ax.plot(np.arange(L_WIN) + 1, np.cumsum(R.mean(0)) / R.mean(0).sum(), color=col, lw=2.2, label=basin)
    if net.basin != a.basin: net = Net(a.seed, a.basin, a.toy); net.y_center = 3.0 if a.toy else net.y_center
    if rowsE:
        E = pd.DataFrame(rowsE); E.to_csv(OUT / "nb19_E_pulse_lags.csv", index=False)
        ax.axhline(.5, color="0.6", lw=.8); ax.set(xscale="log", xlabel="days after the rain pulse", ylabel="share of the extra flow that has arrived"); ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(OUT / "nb19_E_pulse_response.png", dpi=220)
        print(f"\n== E: response of simulated Q to a {a.pulse:g} mm rain pulse (first of each month, WY2009-2013) =="); print(E.round(3).to_string(index=False))
        for basin, g in E[E.pulses == "all pulses"].groupby("basin", sort=False):
            print(f"ANSWER E ({basin}): half of the extra flow arrives within {g.M50_d.iloc[0]:.0f} d (90% within {g.M90_d.iloc[0]:.0f} d); {g.returned_within_365d.iloc[0]:.0%} of the pulse "
                  f"returns as flow within a year. Compare with nb17's M50 (97 d snow, 42 d arid), which weights by absolute sensitivity.")
    plt.close()
    if rowsF:
        F = pd.DataFrame(rowsF); F.to_csv(OUT / "nb19_F_recession.csv", index=False)
        print("\n== F: recession timescale, -Q / (dQ/dt) on runs of 5+ falling days =="); print(F.round(1).to_string(index=False))
        for basin, g in F.groupby("basin", sort=False):
            print(f"ANSWER F ({basin}): " + "; ".join(f"{r.hydrograph} {r.recession_tau_median_d:.0f} d" for r in g.itertuples()))

# ---- G: how interconnected are the units? ----------------------------------------------------------------------
if "G" in a.only:
    H = net.H; rows = []
    for w in range(len(e30)):
        for s in range(30, L_WIN, 15):
            J = net.step_jacobian(S, s, w); J2 = J ** 2; tot = J2.sum(); same = np.eye(H, dtype=bool)
            Cb = np.abs(J[H:, :H]); srt = np.sort(Cb, 1)[:, ::-1]; n90 = (np.cumsum(srt, 1) < .9 * srt.sum(1, keepdims=True)).sum(1) + 1
            rows.append(dict(month=net.dates[idx[s, w]].month, h_from_h=J2[:H, :H].sum() / tot, h_from_c=J2[:H, H:].sum() / tot, c_from_h=J2[H:, :H].sum() / tot,
                             c_from_c=J2[H:, H:].sum() / tot, cross_unit=1 - sum(J2[i:i + H, j:j + H][same].sum() for i in (0, H) for j in (0, H)) / tot,
                             units_feeding_a_cell_90pct=np.median(n90)))
    G = pd.DataFrame(rows); G.to_csv(OUT / f"nb19_G_coupling_{TAG}.csv", index=False); gm = G.drop(columns="month").median()
    print("\n== G: shares of the squared one-day Jacobian (median over sampled days) =="); print(gm.round(3).to_string())
    print(f"ANSWER G: {gm.cross_unit:.0%} of the one-day Jacobian (squared) links DIFFERENT units; a cell's daily update draws on a median of "
          f"{gm.units_feeding_a_cell_90pct:.0f} of {H} units for 90% of its incoming sensitivity. (h is bounded and c is not, so block shares compare unlike scales.)")

# ---- H: nb18 as fraction still present after a year -----------------------------------------------------------------
if "H" in a.only:
    print("\n== H: state directions, fraction still present after 365 days (from nb18) ==")
    for seed in (1, 2, 3):
        f = (ROOT if not a.toy else Path("toyroot")) / f"analysis/lyapunov/nb18_windowed_{a.basin}_seed{seed}.npz"
        if not f.exists(): print(f"  seed {seed}: {f.name} not found (run nb18 first)"); continue
        kept = np.exp(365 * np.minimum(np.load(f)["lam"], 0))
        print(f"ANSWER H (seed {seed}): of the {len(kept)} slowest directions, {(kept > .5).sum()} keep more than half after a year, {(kept > 1 / np.e).sum()} more than 37%, {(kept > .1).sum()} more than 10%.")
# ---- I: positive degree-days ---------------------------------------------------------------------------------------
if "I" in a.only:
    nW = len(e30); step = np.arange(L_WIN)[:, None]; pdd = np.clip(T, 0, None)
    flip = np.array([next((s_ for s_ in range(ipk[w] + 1, L_WIN) if np.sign(cU[s_, w]) != sgn[w]), L_WIN - 1) for w in range(nW)])
    dep = (step > ipk[None, :]) & (step <= flip[None, :])                        # depletion phase: the turn to the sign flip
    y_ = -((S["c"][:, :, U] - S["cp"][:, :, U]) * sgn)[dep]                      # daily decline, positive = counting down
    cand = {"positive degree-days (max(T, 0))": pdd[idx][dep], "air temperature": T[idx][dep], "precipitation": P[idx][dep]}
    for k_, nm in enumerate(net.dyn[:len(net.x_scale)]):
        if "srad" in nm.lower(): cand["solar radiation"] = net.X[:, k_][idx][dep]
    print("\n== I: degree days ==")
    print(f"  depletion phase: {dep.sum()} days over {nW} water years, mean daily decline {y_.mean():.3f}")
    r2 = {k: (0.0 if v.std() == 0 else np.corrcoef(v, y_)[0, 1] ** 2) for k, v in cand.items()}
    for k, v in r2.items(): print(f"  R2 of the counter's daily decline vs {k:34s} {v:.2f}")
    print(f"ANSWER I (decline): positive degree-days explain {r2['positive degree-days (max(T, 0))']:.0%} of the unit's day-to-day decline (accumulation was driven by precipitation, section B).")
    cum = np.cumsum(pdd[idx], 0); day = np.broadcast_to(step, cum.shape).astype(float); hh = S["o"] * np.tanh(S["c"])
    def absr(F, y): return np.array([abs(np.corrcoef(F[:, :, j].ravel(), y.ravel())[0, 1]) if F[:, :, j].std() > 0 else 0 for j in range(net.H)])
    rc_cum, rc_day, rh_pdd = absr(S["c"], cum), absr(S["c"], day), absr(hh, pdd[idx])
    better = np.where((rc_cum > .8) & (rc_cum > rc_day + .05))[0]; top = np.argsort(rc_cum)[::-1][:8]; toph = np.argsort(rh_pdd)[::-1][:8]
    print(f"  cells vs cumulative degree-days since 1 Oct: {(rc_cum > .8).sum()} units with |r| > 0.8; {len(better)} of them track it clearly better than the day count itself; top: "
          + ", ".join(f"{j} ({rc_cum[j]:.2f}; day count {rc_day[j]:.2f})" for j in top))
    print(f"  hidden values vs today's degree-days: {(rh_pdd > .8).sum()} units with |r| > 0.8; top: " + ", ".join(f"{j} ({rh_pdd[j]:.2f})" for j in toph))
    print(f"ANSWER I (units): unit {U} cell vs cumulative degree-days |r| = {rc_cum[U]:.2f} (vs day count {rc_day[U]:.2f}). Any cumulative quantity rises through the year, "
          f"so only units that beat the plain day count ({len(better)}) are evidence of a degree-day sum.")
    pd.DataFrame({"abs_r_cell_cumPDD": rc_cum, "abs_r_cell_daycount": rc_day, "abs_r_hidden_dailyPDD": rh_pdd}).to_csv(OUT / f"nb19_I_degree_days_{TAG}.csv")
# ---- J: where is the snow volume? change the snow-tracking cells together ------------------------------------------------
if "J" in a.only:
    print(f"\n== J: change stored values on 1 March and follow flow to 30 September ({len(GROUP)} cells with |r| > {a.group_thresh:g} against {REFNAME}) ==")
    if len(GROUP) < 2: print("  fewer than 2 snow-tracking cells: skipped")
    else:
        rng = np.random.default_rng(0); rest = np.setdiff1d(np.arange(net.H), GROUP)
        sets = {"snow cells": GROUP, f"unit {U} alone": np.array([U]), "snow cells without the unit": np.setdiff1d(GROUP, [U])}
        for k_ in range(3): sets[f"random cells #{k_ + 1}"] = rng.choice(rest, min(len(GROUP), len(rest)), replace=False)
        rows = []
        for y in YEARS:
            m = wy == y; ee = e[m]; dd = d[m]; tn = int(np.where(net.dates == pd.Timestamp(f"{y}-03-01"))[0][0]); q0 = q[m]
            steps = np.where(ee >= tn, L_WIN - 1 - (ee - tn), -1); sm = np.isin(dd.month, range(4, 10))
            for name, units in sets.items():
                for fac in (0.5, 0.7, 1.3, 1.5):
                    q1 = net.q(net.run_windows(ee, nudge=(units, steps, fac))[0])
                    rows.append(dict(wy=y, cells=name.split(" #")[0], n=len(units), factor=fac, melt_flow_change=q1[sm].sum() / q0[sm].sum() - 1,
                                     max_abs_dq=np.abs(q1 - q0).max(), peak_date_shift_d=(dd[sm][q1[sm].argmax()] - dd[sm][q0[sm].argmax()]).days))
        J = pd.DataFrame(rows); J.to_csv(OUT / f"nb19_J_group_nudge_{TAG}.csv", index=False)
        gJ = J.groupby(["cells", "factor"], sort=False).agg(n=("n", "first"), melt_flow_change=("melt_flow_change", "median"), max_abs_dq=("max_abs_dq", "median"), peak_date_shift_d=("peak_date_shift_d", "median"))
        print(gJ.round(3)); g_ = lambda nm, f_: gJ.loc[(nm, f_), "melt_flow_change"]
        print(f"ANSWER J: scaling the {len(GROUP)} snow cells by 0.7 / 1.3 changes Apr-Sep flow by {g_('snow cells', .7):+.1%} / {g_('snow cells', 1.3):+.1%}; "
              f"the same number of random cells {g_('random cells', .7):+.1%} / {g_('random cells', 1.3):+.1%}; unit {U} alone {g_(f'unit {U} alone', .7):+.2%} / {g_(f'unit {U} alone', 1.3):+.2%}. "
              f"Compare with scaling cold-day precipitation in D. Changing many cells at once pushes the state outside what the model saw in training.")
        SUM.update(snowcells_m30_melt_flow=g_("snow cells", .7), snowcells_p30_melt_flow=g_("snow cells", 1.3), random_m30_melt_flow=g_("random cells", .7), random_p30_melt_flow=g_("random cells", 1.3),
                   unit_m30_melt_flow=g_(f"unit {U} alone", .7), unit_p30_melt_flow=g_(f"unit {U} alone", 1.3), unit_nudge_max_abs_dq=float(J[(J.n == 1) & J.factor.isin([.7, 1.3])].max_abs_dq.max()))
pd.DataFrame([SUM]).to_csv(OUT / f"nb19_summary_{TAG}.csv", index=False)
print("\noutputs in", OUT, "| one-row summary:", f"nb19_summary_{TAG}.csv")
