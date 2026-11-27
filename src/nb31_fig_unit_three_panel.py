"""nb31: slide figure for one snow-tracking unit, one water year, three panels (replaces the four-panel nb16 figure).

  1  simulated flow: as trained, and after the unit's cell state is nudged by x0.7 and x1.3 on 1 March, on the SAME axis,
     so the size of the change is seen against the flow itself
  2  the unit's cell state c (what it stores), with snow on the ground (SPIReS SWE, or a degree-day index if there is no file)
     drawn behind it as a grey area, scaled to the cell's range and on the cell's side of zero, so the two shapes can be compared
  3  the unit's hidden state h (what it passes on), shaded where it is shut (beta = o x (1 - tanh^2 c) below 0.05)
  Lines on all panels: the nudge day, the day the hidden state STARTS to change, and the days the cell changes sign (after melt,
  and in autumn when the count begins). The nudged runs are drawn on all three panels; --hide-nudged-states keeps them on the flow panel only.

"Starts to change" = first day after 1 March on which h has left its December-February value by more than --start-frac
(default 2%) of its range over the water year, and stays there for 5 days. The table printed for all six years gives the
date with 2%, 5% and 10%, days from peak flow, and the share of April-September flow that has passed.

  PYTHONPATH=src python src/nb31_fig_unit_three_panel.py                       # unit 77, Encampment, seed 1, WY2011
  PYTHONPATH=src python src/nb31_fig_unit_three_panel.py --wy 2012 --obs
  python src/nb31_fig_unit_three_panel.py --toy
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import Net, ROOT, SNOW, L_WIN

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--unit", type=int, default=77); ap.add_argument("--wy", type=int, default=2011)
ap.add_argument("--nudge-day", default="03-01"); ap.add_argument("--factors", type=float, nargs="*", default=[.7, 1.3]); ap.add_argument("--start-frac", type=float, default=.02)
ap.add_argument("--obs", action="store_true", help="also draw observed flow"); ap.add_argument("--no-swe", action="store_true"); ap.add_argument("--hide-nudged-states", action="store_true", help="draw the nudged runs on the flow panel only"); ap.add_argument("--shade", action="store_true", help="grey where the hidden state is shut (a change in the cell cannot reach it)"); ap.add_argument("--no-count-start", action="store_true"); ap.add_argument("--ddf", type=float, default=3.0); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
NAMES = {"06623800": "Encampment R., WY", "12178100": "Newhalem Cr., WA", "13235000": "SF Payette R., ID", "09066300": "Middle Cr., CO", "09312600": "White R., UT"}
net = Net(a.seed, a.basin, a.toy); OUT = (ROOT if not a.toy else Path("toyroot")) / "analysis/nudge"; OUT.mkdir(parents=True, exist_ok=True)
if a.toy: a.unit = 3; net.y_center = 3.0
else: net.gate_check()
U = a.unit; SHUT = .05


def snow_reference():
    """Snow on the ground for every row of X: SPIReS SWE if there is a file for this basin, else a degree-day index (as nb27)."""
    P, T = net.precip(), net.temp(); cold = T < 0
    if cold.sum() < 30: cold = T < np.median(T)
    swe = np.zeros(len(P))
    for t in range(1, len(P)): swe[t] = max(swe[t - 1] + P[t] * cold[t] - a.ddf * max(T[t], 0), 0)
    try:
        if a.toy: raise FileNotFoundError
        def pick(df):
            dc = [k for k in df.columns if "date" in k.lower()][0]; sc = [k for k in df.columns if "swe" in k.lower()][0]; return pd.Series(df[sc].values, index=pd.to_datetime(df[dc]))
        f_ = ROOT / f"analysis/spires/spires_{a.basin}.csv"
        if f_.exists(): ser = pick(pd.read_csv(f_))
        else:
            parts = []
            for g_ in sorted((ROOT / "analysis/spires/all").glob("spires_WY*.csv")):
                df = pd.read_csv(g_, dtype=str); bc = [k for k in df.columns if any(w in k.lower() for w in ("basin", "gauge", "gage", "site"))][0]; df = df[df[bc].str.zfill(8) == a.basin]
                if len(df): parts.append(pick(df).astype(float))
            ser = pd.concat(parts).sort_index()
        ser = ser[~ser.index.duplicated()].astype(float).reindex(net.dates).values
        if np.isfinite(ser).sum() > 500: return ser, "snow on the ground (satellite-based SWE)"
    except Exception: pass
    return swe, "snow on the ground (degree-day index)"


def start_of_change(h, d, frac):
    """First day after 1 March on which h is more than frac of its yearly range away from its Dec-Feb value, for 5 days running."""
    hw = np.median(h[np.isin(d.month, [12, 1, 2])]); R = h.max() - h.min(); away = np.abs(h - hw) > frac * R; i0 = int(np.argmax((d.month >= 3) & (d.month < 10)))
    run = np.convolve(away.astype(int), np.ones(5, int), "valid") == 5; idx = np.where(run)[0]; idx = idx[idx >= i0]
    return (int(idx[0]) if len(idx) else None), hw


def one_year(wy):
    e = net.ends(f"{wy - 1}-10-01", f"{wy}-09-30"); d = net.dates[e]; h0, c0 = net.run_windows(e); q0 = net.q(h0); c, h = c0[:, U], h0[:, U]
    tc = np.tanh(c); beta = np.where(np.abs(tc) > 1e-9, h / np.where(np.abs(tc) > 1e-9, tc, 1), 0) * (1 - tc ** 2)
    p = int(np.argmax(np.abs(c))); after = np.arange(p + 1, len(c)); fl = after[np.sign(c[after]) != np.sign(c[p])]; i_flip = int(fl[0]) if len(fl) else None
    before = np.arange(1, p); fa = before[(np.sign(c[before]) == np.sign(c[p])) & (np.sign(c[before - 1]) != np.sign(c[p]))]; i_aut = int(fa[-1]) if len(fa) else None   # last day the count starts, before its peak
    melt = np.isin(d.month, range(4, 10)); cum = np.cumsum(q0 * melt) / (q0 * melt).sum(); i_pk = int(np.argmax(q0))
    return dict(e=e, d=d, q0=q0, c=c, h=h, beta=beta, i_flip=i_flip, i_aut=i_aut, i_pk=i_pk, cum=cum, starts={f: start_of_change(h, d, f)[0] for f in sorted({a.start_frac, .02, .05, .10})})


rows = []
for wy in range(2009, 2015):
    r = one_year(wy); d = r["d"]; row = dict(wy=wy, peak_flow=d[r["i_pk"]].date(), sign_flip=d[r["i_flip"]].date() if r["i_flip"] is not None else None, count_starts=d[r["i_aut"]].date() if r["i_aut"] is not None else None)
    if r["i_flip"] is not None: row["flip_days_after_peak"] = r["i_flip"] - r["i_pk"]; row["melt_flow_passed_at_flip"] = r["cum"][r["i_flip"]]
    for f, i in r["starts"].items():
        row[f"h_starts_{f:.0%}"] = d[i].date() if i is not None else None
        if f == a.start_frac and i is not None: row["start_days_after_peak"] = i - r["i_pk"]; row["melt_flow_passed_at_start"] = r["cum"][i]; row["start_to_flip_days"] = (r["i_flip"] - i) if r["i_flip"] is not None else np.nan
    rows.append(row)
T = pd.DataFrame(rows); pd.set_option("display.width", 220); print(f"\n{NAMES.get(a.basin, '')} {a.basin}, model version {a.seed}, unit {U}"); print(T.round(2).to_string(index=False)); T.to_csv(OUT / f"nb31_unit{U}_dates_{a.basin}_seed{a.seed}.csv", index=False)
if "start_days_after_peak" in T:
    print(f"\nSLIDE NUMBERS: the hidden state starts to change {T.start_days_after_peak.min():.0f} to {T.start_days_after_peak.max():.0f} days after peak flow (negative = before), "
          f"when {T.melt_flow_passed_at_start.min():.0%} to {T.melt_flow_passed_at_start.max():.0%} of Apr-Sep flow has passed; "
          + (f"the cell changes sign {T.start_to_flip_days.min():.0f} to {T.start_to_flip_days.max():.0f} days later, when {T.melt_flow_passed_at_flip.min():.0%} to {T.melt_flow_passed_at_flip.max():.0%} has passed." if "start_to_flip_days" in T else ""))

# ---- figure -------------------------------------------------------------------------------------------------------------------------
r = one_year(a.wy); d, e = r["d"], r["e"]; nd = pd.Timestamp(f"{a.wy}-{a.nudge_day}"); tn = int(np.where(net.dates == nd)[0][0]); steps = np.where(e >= tn, L_WIN - 1 - (e - tn), -1); down = e >= tn
N = {}
for fac in a.factors: h1, c1 = net.run_windows(e, nudge=(U, steps, fac)); N[fac] = (net.q(h1), c1[:, U], h1[:, U])
dqmax = max(np.abs(N[f][0] - r["q0"])[down].max() for f in a.factors); print(f"\nWY{a.wy}: largest change in flow on any day after the nudge: {dqmax:.4f} mm/d (peak flow {r['q0'].max():.1f} mm/d)")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False}); BLUE, GREEN, ORANGE, RED = "#3b6fb6", "#3f8f5a", "#e08a00", "#c0392b"; LS = {a.factors[0]: "--", a.factors[-1]: ":"}
fig, ax = plt.subplots(3, 1, figsize=(11, 9.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, 1]})
if a.obs and net.q_obs is not None: ax[0].plot(d, net.q_obs[e], color="0.75", lw=1.5, label="observed")
ax[0].plot(d, r["q0"], color="k", lw=2.4, label="as trained")
for fac in a.factors: ax[0].plot(d[down], N[fac][0][down], LS[fac], color=RED, lw=2, label=f"cell ×{fac:g} on {nd.day} {nd.strftime('%B')}" if fac == a.factors[0] else f"cell ×{fac:g}")
ax[0].set_ylabel("simulated flow\n(mm/d)"); ax[0].set_ylim(0, None)
ax[0].legend(frameon=False, fontsize=13, loc="upper left", title=f"largest change: {dqmax:.2f} mm/d" if dqmax >= .005 else f"largest change: {dqmax:.3f} mm/d", title_fontsize=13, alignment="left")
ax[1].plot(d, r["c"], color=BLUE, lw=2.4)
if not a.hide_nudged_states:
    for fac in a.factors: ax[1].plot(d[down], N[fac][1][down], LS[fac], color=BLUE, lw=1.8)
if not a.no_swe:
    REF, REFNAME = snow_reference(); sw = REF[e]
    if np.isfinite(sw).sum() > 100 and np.nanmax(sw) > 0:
        pk = r["c"][int(np.argmax(np.abs(r["c"])))]; ax[1].fill_between(d, 0, np.nan_to_num(sw) / np.nanmax(sw) * pk, color="#bbbbbb", alpha=.7, lw=0, zorder=0, label="snow on the ground (SWE, scaled)")
        ax[1].plot([], [], color=BLUE, lw=2.4, label=f"cell state"); ax[1].legend(frameon=False, fontsize=12, loc="lower left")
        ok_ = np.isfinite(sw); print(f"WY{a.wy}: rank correlation of |cell state| with {REFNAME}: {pd.Series(np.abs(r['c'][ok_])).corr(pd.Series(sw[ok_]), method='spearman'):.2f}")
ax[1].axhline(0, color="0.6", lw=.8); ax[1].set_ylabel(f"cell state\n(what unit {U} stores)")
ax[2].plot(d, r["h"], color=GREEN, lw=2.4)
if not a.hide_nudged_states:
    for fac in a.factors: ax[2].plot(d[down], N[fac][2][down], LS[fac], color=GREEN, lw=1.8)
if a.shade:
    shut = np.abs(r["beta"]) < SHUT; ax[2].fill_between(d, 0, 1, where=shut, transform=ax[2].get_xaxis_transform(), color="0.92", lw=0, zorder=0)
    ax[2].text(.01, .06, "grey: a change in the cell does not show here", transform=ax[2].transAxes, fontsize=12, color="0.35")
ax[2].axhline(0, color="0.6", lw=.8); ax[2].set_ylabel(f"hidden state\n(what it passes on)")
i_s, i_f = r["starts"][a.start_frac], r["i_flip"]; marks = [(nd, "0.4", "nudge", 1.2)] + ([(d[i_s], GREEN, "hidden state\nstarts to change", 2)] if i_s is not None else []) + ([(d[i_f], ORANGE, "cell changes\nsign", 2)] if i_f is not None else []) + ([(d[r["i_aut"]], ORANGE, "count\nstarts", 2)] if r["i_aut"] is not None and not a.no_count_start else [])
for x_ in ax:
    for when, col, _, lw in marks: x_.axvline(when, color=col, lw=lw * .7, alpha=.8)
top = ax[0].get_ylim()[1]
marks_s = sorted(marks, key=lambda m: m[0]); span = (d[-1] - d[0]).days
for k, (when, col, lab, _) in enumerate(marks_s):
    close = k > 0 and (when - marks_s[k - 1][0]).days < .12 * span                 # two labels close together: lift the later one
    ax[0].text(when, top * (1.17 if close else 1.02), lab, ha="center", va="bottom", fontsize=12, color=col)
import matplotlib.dates as mdates
ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10, 12, 2, 4, 6, 8])); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.align_ylabels(); fig.tight_layout(rect=(0, 0, 1, .93)); f_ = OUT / f"nb31_unit{U}_{a.basin}_WY{a.wy}.png"; fig.savefig(f_, dpi=220); print("wrote", f_)
