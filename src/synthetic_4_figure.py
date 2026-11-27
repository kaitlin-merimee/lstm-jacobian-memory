"""Slide figure for the synthetic test: what forget gates and chained Jacobians read, against the planted truth (scenario S2).

Reads the per-basin table synthetic_3_recovery already wrote; nothing is recomputed.
  python src/synthetic_4_figure.py --three        # gates vs fast bucket | gates vs slow bucket | Jacobians vs slow bucket
  python src/synthetic_4_figure.py --same-axis    # two panels, both against the slow bucket
  python src/synthetic_4_figure.py                # two panels, Jacobian lag against the true mean lag
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--scen", default="S2")
ap.add_argument("--same-axis", action="store_true"); ap.add_argument("--three", action="store_true"); ap.add_argument("--four", action="store_true", help="2 x 2: gates and Jacobians, each against the fast and the slow bucket")
ap.add_argument("--csv", default=None, help="path to an nb08b_basins_*.csv (default: found under analysis/jacobians/followup)")
a = ap.parse_args()
src = [Path(a.csv)] if a.csv else sorted((ROOT / "analysis/jacobians/followup").glob(f"nb08b_basins_*{a.scen}*.csv"))
assert src, "no nb08b_basins_*.csv found for this scenario"
df = pd.read_csv(src[0]); print("read", src[0].name)
if "tau_readout_days" not in df: df["tau_readout_days"] = 10 ** df["tau_readout_log10"]   # synthetic_3_recovery stores log10 days
ORANGE, BLUE = "#c8782a", "#3b6fb6"
gate_slow = ("tau_readout_days", "tau_s", "Forget gates vs slow bucket", "planted slow bucket (days)", "gate memory (days)", ORANGE)
gate_fast = ("tau_readout_days", "tau_f", "Forget gates vs fast bucket", "planted fast bucket (days)", "gate memory (days)", ORANGE)
jac_slow = ("state_mean_lag", "tau_s", "Chained Jacobians vs slow bucket", "planted slow bucket (days)", "Jacobian memory lag (days)", BLUE)
jac_fast = ("in_M50", "tau_f", "Chained Jacobians vs fast bucket", "planted fast bucket (days)", "Jacobian: half the response by (days)", BLUE)
jac_lag = ("in_mean_lag", "true_mean_lag", "Chained Jacobians", "true mean lag (days)", "Jacobian mean lag (days)", BLUE)
pairs, tag = ([gate_fast, gate_slow, jac_fast, jac_slow], "_four") if a.four else ([gate_fast, gate_slow, jac_slow], "_three") if a.three else ([gate_slow, jac_slow], "_same_axis") if a.same_axis else ([gate_slow, jac_lag], "")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 10.4)) if a.four else plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5.2)); ax = np.ravel(ax)
for x, (yc, xc, title, xl, yl, col) in zip(ax, pairs):
    d = df[[xc, yc]].replace([np.inf, -np.inf], np.nan).dropna(); d = d[(d > 0).all(axis=1)]
    rho = spearmanr(d[xc], d[yc])[0]
    x.loglog(d[xc], d[yc], "o", ms=4, alpha=.5, color=col)
    lim = [min(d.min()), max(d.max())]
    if yc != "in_M50": x.plot(lim, lim, "k--", lw=1)             # M50 is not on the same scale as a bucket timescale: no 1:1 line
    if lim[1] < 10:                                               # narrow range: plain tick labels instead of crowded log ones
        from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
        tk = [t for t in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16) if lim[0] * .9 <= t <= lim[1] * 1.1]
        for axis in (x.xaxis, x.yaxis):
            axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); axis.set_minor_formatter(NullFormatter())
    x.set(xlabel=xl, ylabel=yl); x.set_title(f"{title}\nρ = {rho:+.2f}", loc="left", fontweight="bold", fontsize=16)
    print(f"  {title}: rho = {rho:+.2f} (n = {len(d)})")
plt.tight_layout(); out = (src[0].parent if a.csv else ROOT / "analysis/figures/followup") / f"fig_synthetic_two_panel_{a.scen}{tag}.png"
plt.savefig(out, dpi=250); print("figure:", out)
