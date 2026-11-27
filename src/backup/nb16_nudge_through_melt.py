"""nb16: does the snow counter's stored AMOUNT matter once melt arrives?

The earlier nudge changed the counter by 30% and saw < 0.005 mm/d. While the unit is saturated
(|c| large, beta = o(1 - tanh^2 c) = 0) that is true by construction. The real test is downstream:
nudge the counter on a winter day, keep running, and follow discharge through melt.

For every water year 2009-2014 and every nudge day, each prediction day T after the nudge gets its
own 365-day zero-start window (as NeuralHydrology runs it); c[unit] is multiplied by `factor` on the
nudge day inside that window. Reported per nudge: largest and summed discharge change over the rest
of the water year, and how far the unit's sign flip and un-saturation dates move.

  python src/backup/nb16_nudge_through_melt.py                 # counter 77, basin 06623800, seed 1
  python src/backup/nb16_nudge_through_melt.py --toy           # code test on a random LSTM
"""
import argparse, numpy as np, pandas as pd
from jtalk_core import Net, ROOT, SNOW, L_WIN

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--unit", type=int, default=77)
ap.add_argument("--controls", type=int, nargs="*", default=[109], help="other units given the same nudges (109 = switch)")
ap.add_argument("--n-random", type=int, default=4, help="extra random control units")
ap.add_argument("--factors", type=float, nargs="*", default=[0.0, 0.5, 0.7, 1.3, 1.5])
ap.add_argument("--nudge-days", nargs="*", default=["01-01", "03-01", "04-15"], help="MM-DD inside each water year")
ap.add_argument("--rel-to-flip", type=int, nargs="*", default=[-14, -3], help="also nudge this many days from the flip")
ap.add_argument("--fig-wy", type=int, default=2011); ap.add_argument("--toy", action="store_true")
a = ap.parse_args()

net = Net(1, a.basin, a.toy); net.gate_check()
OUT = (ROOT if not a.toy else __import__("pathlib").Path("toyroot")) / "analysis/nudge"; OUT.mkdir(parents=True, exist_ok=True)
if a.toy: a.unit, a.controls = 3, [5]
rng = np.random.default_rng(0)
controls = list(a.controls) + [int(u) for u in rng.choice([u for u in range(net.H) if u not in (a.unit, *a.controls)], a.n_random, replace=False)]


def flip_dates(c, dates):
    """Sign-flip and un-saturation (|c| < 2, where beta is no longer ~0) dates after the peak |c|."""
    p = int(np.argmax(np.abs(c))); s = np.sign(c[p]); after = np.arange(p + 1, len(c))
    fl = after[np.sign(c[after]) != s]; un = after[np.abs(c[after]) < 2]
    return (dates[fl[0]] if len(fl) else pd.NaT), (dates[un[0]] if len(un) else pd.NaT), dates[p], c[p]


rows, keep_fig = [], {}
for wy in range(2009, 2015):
    e = net.ends(f"{wy - 1}-10-01", f"{wy}-09-30"); d = net.dates[e]
    h0, c0 = net.run_windows(e); q0 = net.q(h0)
    f0, u0, pk, cpk = flip_dates(c0[:, a.unit], d)
    o_last = None
    days = [pd.Timestamp(f"{wy if int(md[:2]) < 10 else wy - 1}-{md}") for md in a.nudge_days]
    days += [f0 + pd.Timedelta(days=k) for k in a.rel_to_flip if f0 is not pd.NaT]
    for unit in [a.unit] + controls:
        for nd in days:
            if nd < d[0] or nd > d[-1]: continue
            tn = int(np.where(net.dates == nd)[0][0])
            steps = np.where(e >= tn, L_WIN - 1 - (e - tn), -1)       # window step that holds the nudge day
            down = e >= tn
            for fac in a.factors:
                h1, c1 = net.run_windows(e, nudge=(unit, steps, fac)); dq = net.q(h1) - q0
                f1, u1, _, _ = flip_dates(c1[:, unit], d) if unit == a.unit else (pd.NaT,) * 4
                j = int(np.argmax(np.abs(dq) * down))
                rows.append(dict(wy=wy, unit=unit, role="counter" if unit == a.unit else "control",
                                 nudge_date=nd.date(), factor=fac, c_on_nudge_day=c0[e == tn, unit][0] if (e == tn).any() else np.nan,
                                 max_abs_dq=np.abs(dq[down]).max(), date_of_max=d[j].date(),
                                 sum_dq_mm=dq[down].sum(), sum_abs_dq_mm=np.abs(dq[down]).sum(),
                                 q_peak=q0.max(), q_total_mm=q0[down].sum(),
                                 flip_base=f0, flip_nudged=f1, flip_shift_d=(f1 - f0).days if unit == a.unit and f1 is not pd.NaT and f0 is not pd.NaT else np.nan,
                                 unsat_base=u0, unsat_nudged=u1))
                if wy == a.fig_wy and unit == a.unit and fac in (0.7, 1.3) and nd == days[1]:
                    keep_fig[fac] = (dq, c1[:, unit])
    if wy == a.fig_wy:
        _, _, S = net.run_windows(e[-1:], keep=True)                   # one full window for the gate traces
        keep_fig["base"] = (d, q0, c0[:, a.unit], h0[:, a.unit], f0, u0, days[1])
    print(f"WY{wy}: peak c = {cpk:+.1f} on {pk.date()}, sign flip {f0}, un-saturates {u0}")

df = pd.DataFrame(rows); df.to_csv(OUT / f"nb16_nudge_{a.basin}_u{a.unit}.csv", index=False)
pd.set_option("display.width", 200)
print("\nLargest |dQ| after the nudge, mm/d (median over water years; peak flow for scale):")
print(df.assign(nudge=df.nudge_date.astype(str).str[5:]).groupby(["role", "factor"])
        .agg(max_abs_dq=("max_abs_dq", "median"), worst=("max_abs_dq", "max"), sum_abs_dq_mm=("sum_abs_dq_mm", "median"),
             flip_shift_d=("flip_shift_d", "median"), q_peak=("q_peak", "median")).round(4))
cn = df[(df.role == "counter") & df.factor.isin([0.7, 1.3])]
print(f"\nSLIDE NUMBERS  counter unit {a.unit}, +/-30% nudge, followed to 30 Sep:")
print(f"  largest |dQ| any day, any year: {cn.max_abs_dq.max():.4f} mm/d  (seasonal peak flow median {cn.q_peak.median():.2f} mm/d)")
print(f"  flip-date shift: median {cn.flip_shift_d.median():.0f} d, range {cn.flip_shift_d.min():.0f} to {cn.flip_shift_d.max():.0f} d")
er = df[(df.role == "counter") & (df.factor == 0.0)]
if len(er): print(f"  erasing the count (factor 0): largest |dQ| {er.max_abs_dq.max():.4f} mm/d, flip shift median {er.flip_shift_d.median():.0f} d")

# ---- slide figure: one water year, four stacked panels ---------------------------------------
if "base" in keep_fig:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    d, q0, c, h, f0, u0, nd = keep_fig["base"]
    tc = np.tanh(c); o = np.where(np.abs(tc) > 1e-12, h / np.where(tc == 0, 1, tc), np.nan); beta = o * (1 - tc ** 2)
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1]})
    ax[0].plot(d, q0, color="k"); ax[0].set_ylabel("simulated Q\n(mm/d)")
    ax[1].plot(d, c, color="#3b6fb6", label="as trained")
    for fac, ls in ((0.7, "--"), (1.3, ":")):
        if fac in keep_fig: ax[1].plot(d, keep_fig[fac][1], ls, color="#3b6fb6", label=f"nudged x{fac}")
    ax[1].axhline(0, color="0.6", lw=.8); ax[1].set_ylabel(f"counter cell\nc (unit {a.unit})"); ax[1].legend(frameon=False, fontsize=12, loc="lower right", ncol=3)
    ax[2].semilogy(d, np.maximum(np.abs(beta), 1e-20), color="#8a4fa0"); ax[2].set_ylabel("β: how much\nh hears c")
    for fac, ls in ((0.7, "--"), (1.3, ":")):
        if fac in keep_fig: ax[3].plot(d, keep_fig[fac][0], ls, color="#c0392b", label=f"x{fac}")
    ax[3].set_ylabel("change in Q\n(mm/d)"); ax[3].legend(frameon=False, fontsize=12)
    for x in ax:
        x.axvline(nd, color="0.4", lw=1)
        if f0 is not pd.NaT: x.axvline(f0, color="#e08a00", lw=1.5)
    ax[0].text(nd, ax[0].get_ylim()[1], " nudge", va="top", fontsize=12)
    if f0 is not pd.NaT: ax[0].text(f0, ax[0].get_ylim()[1], " sign flip", va="top", color="#e08a00", fontsize=12)
    fig.align_ylabels(); plt.tight_layout(); plt.savefig(OUT / f"nb16_nudge_{a.basin}_WY{a.fig_wy}.png", dpi=250)
    print("figure:", OUT / f"nb16_nudge_{a.basin}_WY{a.fig_wy}.png")
