"""nb20: how is the per-unit forget-gate timescale (the 8 d / 0.7 d on the slide) summarized, and does it change by season?

For each basin, one 365-day window per `--every` days ending in WY2009-2014. On each prediction day every unit
has f_j, so tau_j = -1 / ln f_j. Several summaries are printed so you can see which one reproduces nb02's numbers:
  A  per day: |w_out|-weighted mean of log10(tau) over units, back to days   -> then median / mean over days, and by season
  B  per day: |w_out|-weighted median of tau over units                      -> same
  C  per unit: tau from the time-mean of ln f over all days, then |w_out|-weighted mean of log10(tau)   (one number)
  D  as C but unweighted median over units

  PYTHONPATH=src python src/backup/nb20_gate_tau_by_season.py          # about 2 min
"""
import argparse, numpy as np, pandas as pd
from jtalk_core import Net, ROOT, SNOW, ARID, L_WIN, season_of
ap = argparse.ArgumentParser(); ap.add_argument("--basins", nargs="*", default=[SNOW, ARID]); ap.add_argument("--every", type=int, default=5)
ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else __import__("pathlib").Path("toyroot")) / "analysis/open_questions"; OUT.mkdir(parents=True, exist_ok=True)
rows = []
for basin in a.basins:
    net = Net(1, basin, a.toy); net.gate_check(); e = net.ends(every=a.every); d = net.dates[e]; w = np.abs(net.w_head) / np.abs(net.w_head).sum()
    lnf = np.concatenate([np.log(net.run_windows(e[k:k + 32], keep=True)[2]["f"][-1]) for k in range(0, len(e), 32)])    # [days, H], prediction-day gates
    tau = -1 / np.minimum(lnf, -1e-12); order = np.argsort(tau, 1); cw = np.cumsum(np.take_along_axis(np.broadcast_to(w, tau.shape), order, 1), 1)
    A = 10 ** (np.log10(tau) @ w); B = np.take_along_axis(tau, order, 1)[np.arange(len(tau)), (cw >= .5).argmax(1)]
    tau_u = -1 / np.minimum(lnf.mean(0), -1e-12); C = 10 ** (np.log10(tau_u) @ w); D = np.median(tau_u)
    seas = season_of(d.month.values)
    for name, m in [("all days", np.ones(len(d), bool))] + [(s, seas == s) for s in ("Oct-Nov", "Dec-Mar", "Apr-Jun", "Jul-Sep")]:
        rows.append(dict(basin=basin, days=name, n=int(m.sum()), A_median=np.median(A[m]), A_mean=A[m].mean(), A_p10=np.percentile(A[m], 10), A_p90=np.percentile(A[m], 90),
                         B_median=np.median(B[m]), B_mean=B[m].mean(), units_tau_gt_30d=float((tau[m] > 30).sum(1).mean())))
    print(f"\n{basin}: C (time-mean ln f per unit, weighted log-mean) = {C:.2f} d | D (same, unweighted median over units) = {D:.2f} d")
df = pd.DataFrame(rows); df.to_csv(OUT / "nb20_gate_tau_by_season.csv", index=False)
pd.set_option("display.width", 200); print(df.round(2).to_string(index=False))
for basin, g in df.groupby("basin", sort=False):
    g = g.set_index("days"); print(f"ANSWER ({basin}): weighted gate timescale, median over days {g.A_median['all days']:.1f} d (10th-90th percentile of days {g.A_p10['all days']:.1f}-{g.A_p90['all days']:.1f}); "
                                   f"Apr-Jun {g.A_median['Apr-Jun']:.1f} d vs Dec-Mar {g.A_median['Dec-Mar']:.1f} d.")
