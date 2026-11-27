"""rain_pulse_by_month: two follow-ups for the headline slide (two example basins).

  M  Where does each month's rain go? Uses the rain-pulse responses snow_unit_questions section E already saved (no model needed):
     for a pulse on the first of each month: share that becomes flow, same-day share, days until half has arrived,
     and the calendar month in which the extra flow arrives.
  O  Observed check (loads the model, run on a compute node): per water year, the lag between the centre of mass of
     freezing-day precipitation and the centre of mass of April-September flow, for observed and simulated flow;
     and the runoff ratio of observed and simulated flow next to the share of a rain pulse that becomes flow.

  PYTHONPATH=src python src/backup/rain_pulse_by_month.py            # both; --only M runs on a login node
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import ROOT, SNOW, ARID, L_WIN
ap = argparse.ArgumentParser(); ap.add_argument("--only", default="MO"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OQ = (ROOT if not a.toy else Path("toyroot")) / "analysis/open_questions"; pd.set_option("display.width", 220)
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]; ret = {}

if "M" in a.only:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14}); fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    for x_, (basin, label) in zip(ax, ((SNOW, "snow basin"), (ARID, "arid basin"))):
        f = [p for p in (OQ / f"nb19_E_pulse_response_{basin}.npy", OQ / f"nb19_E_pulse_response_{basin}_seed1.npy") if p.exists()]
        if not f: print(f"{basin}: pulse responses not found (run snow_unit_questions --only E first)"); continue
        Rr = np.load(f[0]).astype(float); starts = pd.date_range("2008-10-01", periods=len(Rr), freq="MS"); ret[basin] = float(Rr.mean(0).sum())
        rows = []; H = np.zeros((12, 12))
        for mo in [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            sel = np.where(starts.month == mo)[0]; r = Rr[sel].mean(0); tot = r.sum(); c = np.cumsum(np.clip(r, 0, None))
            for i in sel:
                arr = (starts[i] + pd.to_timedelta(np.arange(L_WIN), "D")).month.values
                for k in range(12): H[mo - 1, k] += Rr[i][arr == k + 1].sum() / len(sel)          # signed: gains and losses within a month cancel
            peak = int(np.argmax(H[mo - 1])); rows.append(dict(pulse_month=MN[mo - 1], n=len(sel), becomes_flow=tot, same_day_share=r[0] / tot if tot > 0 else np.nan,
                first_10d_share=r[:10].sum() / tot if tot > 0 else np.nan, half_arrived_d=int(np.searchsorted(c, c[-1] / 2)) if c[-1] > 0 else np.nan,
                most_arrives_in=MN[peak], negative_share=np.clip(-r, 0, None).sum() / np.abs(r).sum()))
        M = pd.DataFrame(rows); M.to_csv(OQ / f"nb24_pulse_by_month_{basin}.csv", index=False)
        print(f"\n== M: {label} {basin}: a 10 mm rain pulse on the first of each month =="); print(M.round(2).to_string(index=False))
        order = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]; Hn = np.clip(H, 0, None)[order][:, order]; im = x_.imshow(Hn / max(Hn.max(), 1e-12), cmap="Blues", aspect="auto")
        x_.set_xticks(range(12)); x_.set_xticklabels([MN[i] for i in order], rotation=45); x_.set_yticks(range(12)); x_.set_yticklabels([MN[i] for i in order])
        x_.set(xlabel="month the extra flow arrives", ylabel="month of the rain pulse"); x_.set_title(label, loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig(OQ / "nb24_where_each_months_rain_goes.png", dpi=220); print("figure:", OQ / "nb24_where_each_months_rain_goes.png")
    print("How to read: a physical snow response has winter pulses arriving in May-July with nothing the same day, and summer pulses arriving quickly.\n"
          "  Summer or autumn pulses that mostly arrive the NEXT spring would not be physical for rain.")

if "O" in a.only:
    from jtalk_core import Net
    rows = []
    for basin, label in ((SNOW, "snow basin"), (ARID, "arid basin")):
        net = Net(1, basin, a.toy); net.gate_check()
        if a.toy: net.y_center = 3.0
        e = net.ends(); d = net.dates[e]; wy = net.wy[e]; q = net.q(net.run_windows(e)[0]); P, T = net.precip()[e], net.temp()[e]
        qo = net.q_obs[e] if net.q_obs is not None else (q * 1.05 if a.toy else None)
        rr = dict(basin=basin, runoff_ratio_sim=q.mean() / P.mean(), runoff_ratio_obs=np.nanmean(qo) / P.mean() if qo is not None else np.nan, pulse_becomes_flow=ret.get(basin, np.nan),
                  freezing_day_precip_share=P[T < 0].sum() / P.sum())
        print(f"\n== O: {label} {basin} ==\n  runoff ratio: simulated {rr['runoff_ratio_sim']:.2f}, observed {rr['runoff_ratio_obs']:.2f}; share of a rain pulse that becomes flow {rr['pulse_becomes_flow']:.2f}")
        if rr["freezing_day_precip_share"] > .2:
            lagrows = []
            for y in np.unique(wy):
                m = wy == y; dd = np.arange(m.sum()); cold = (T[m] < 0) & (d[m].month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])); melt = d[m].month.isin([4, 5, 6, 7, 8, 9])
                if P[m][cold].sum() <= 0: continue
                cp = (dd[cold] * P[m][cold]).sum() / P[m][cold].sum(); r_ = dict(wy=int(y), snowfall_centre=d[m][int(cp)].date())
                for name, qq in (("sim", q[m]), ("obs", None if qo is None else qo[m])):
                    if qq is None or np.isnan(qq[melt]).mean() > .1: continue
                    qq = np.nan_to_num(qq); base = np.percentile(qq[~melt], 25); ex = np.clip(qq - base, 0, None) * melt
                    if ex.sum() > 0: cq = (dd * ex).sum() / ex.sum(); r_[f"{name}_melt_flow_centre"] = d[m][int(cq)].date(); r_[f"{name}_lag_d"] = cq - cp
                lagrows.append(r_)
            Lg = pd.DataFrame(lagrows); Lg.to_csv(OQ / f"nb24_snowfall_to_flow_lag_{basin}.csv", index=False); print(Lg.round(0).to_string(index=False))
            for name in ("obs", "sim"):
                if f"{name}_lag_d" in Lg: print(f"ANSWER O ({name}): centre of freezing-day precipitation to centre of melt-season flow: median {Lg[f'{name}_lag_d'].median():.0f} d (range {Lg[f'{name}_lag_d'].min():.0f} to {Lg[f'{name}_lag_d'].max():.0f}).")
            print("  Compare with the model's response to a rain pulse on a freezing day (half arrived by about 124 d) and the whole-Jacobian lag (97 d).")
        rows.append(rr)
    pd.DataFrame(rows).to_csv(OQ / "nb24_runoff_ratio.csv", index=False)
