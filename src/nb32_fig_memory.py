"""nb32_fig_memory: talk figures for the per-cell memory test. Run nb32_collect.py first.

  Panel 1  Memory through the year: how long a nudge lasts in the state, by start day, for the directions flow reads, the
           snow-tracking cells, the snow probe, the directions of most variance and all other cells (median of the runs).
  Panel 2  One start day, one run: every cell as a dot, memory in the state against effect on flow; named directions marked.

  PYTHONPATH=src python src/nb32_fig_memory.py                       # both panels
  PYTHONPATH=src python src/nb32_fig_memory.py --panels 1            # memory through the year only (slide version)
  PYTHONPATH=src python src/nb32_fig_memory.py --panels 2 --start 12-01 --basin 06623800 --seed 1 --unit 77
  PYTHONPATH=src python src/nb32_fig_memory.py --panels 3            # when the store's memory ends, against observed melt timing
  Panel 4  Memory in days, by start day (1 November to 1 April), next to the days left until the observed centre of melt-season flow:
           the direction flow reads most, and the cells that track snow. Simplest slide version of panel 3.
  Panel 3  For starts 1 November to 1 April (--full: October to May, with every run as a thin line): the day of the nudge and the day the nudge along the direction flow reads most has faded,
           next to the observed centre of melt-season flow (nb26, same basin-years).
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--panels", type=int, nargs="*", default=[1, 2]); ap.add_argument("--start", default="12-01")
ap.add_argument("--basin", default="06623800"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--unit", type=int, default=77); ap.add_argument("--full", action="store_true", help="panel 3: also show the 1 October and 1 May starts and every run as a thin line"); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
D = (ROOT if not a.toy else Path("toyroot")) / "analysis/cell_memory"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, PURPLE, GREY = "#3b6fb6", "#c8782a", "#5a9e6f", "#8a5fb0", "#999999"          # same meanings as nb29
NAMES = {"06623800": "Encampment R., WY", "12178100": "Newhalem Cr., WA", "13235000": "SF Payette R., ID", "09066300": "Middle Cr., CO", "09312600": "White R., UT"}
ORDER = ["10-01", "11-01", "12-01", "01-01", "02-01", "03-01", "04-01", "05-01", "06-01", "07-01", "08-01", "09-01"]
MON = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
LINES = [("read1", "direction flow reads most", BLUE, "-"), ("snow cell", "cells that track snow", GREEN, "-"), ("probe", "linear snow probe", PURPLE, "-"),
         ("PCA", "directions of most variance", ORANGE, "-"), ("other cell", "all other cells", GREY, "-")]
A = pd.read_csv(D / "nb32_all_runs.csv", dtype={"basin": str}); S = pd.read_csv(D / "nb32_by_start.csv"); HZ = int(A.horizon.max())

if 1 in a.panels:
    fig, ax = plt.subplots(figsize=(11, 5.6)); x = np.arange(12)
    for k, lab, col, ls in LINES:
        s = S[S.kind2 == k].set_index("start").reindex(ORDER)
        if s.memory_d.isna().all(): continue
        ax.plot(x, s.memory_d, ls, color=col, lw=2.5, marker="o", ms=8, label=lab)
        if k in ("read1", "snow cell"): ax.fill_between(x, s.mem_lo, s.mem_hi, color=col, alpha=.15, lw=0)
    ax.axhline(HZ, color="k", lw=1, ls="--"); ax.text(-.2, HZ - 4, f"longest we can measure ({HZ} d)", fontsize=12, va="top")
    ax.set_xticks(x, MON); ax.set_xlim(-.3, 11.3); ax.legend(frameon=False, ncol=3, fontsize=12, loc="upper right", bbox_to_anchor=(1, .97)); ax.set_ylim(0, HZ * 1.3); ax.set_xlabel("day the state is nudged (1st of the month)")
    ax.set_ylabel("memory: days until a nudge\nhas faded to 37%")
    runs = A.groupby(["basin", "seed"]).ngroups; ax.set_title(f"Median of {runs} runs ({A.basin.nunique()} snow basins x {A.seed.nunique()} model versions); bands: middle half of runs", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(D / "nb32_memory_through_year.png", dpi=220); print("wrote", D / "nb32_memory_through_year.png")

if 2 in a.panels:
    c = A[(A.basin == a.basin) & (A.seed == a.seed) & (A.start == a.start)]
    g = c.groupby(["item", "kind", "cell"]).agg(memory_d=("memory_d", "median"), flow_abs_mm=("flow_abs_mm", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 7.2)); floor = max(g.flow_abs_mm[g.flow_abs_mm > 0].min(), 1e-4)
    m = g[g.kind == "other cell"]; ax.scatter(m.memory_d, np.maximum(m.flow_abs_mm, floor), s=20, color="#c4c4c4", edgecolor="none", label="other cells", zorder=2)
    m = g[g.kind == "snow cell"]; ax.scatter(m.memory_d, np.maximum(m.flow_abs_mm, floor), s=75, color="#2e8b4e", edgecolor="k", linewidth=.9, label="cells that track snow", zorder=3)   # solid, outlined: must stand out from the grey
    for its, col, mk, lab in ((("read1", "read2", "read3"), BLUE, "D", "3 directions flow reads"), (("probe",), PURPLE, "D", "linear snow probe"), (("snow_group",), GREEN, "s", "all snow cells moved together")):
        m = g[g.item.isin(its)]
        if len(m): ax.scatter(m.memory_d, m.flow_abs_mm, s=150, marker=mk, color=col, edgecolor="k", linewidth=.9, zorder=4, label=lab)
    u = g[g.cell == a.unit]
    if len(u): ax.annotate(f"unit {a.unit}", (u.memory_d.iloc[0], max(u.flow_abs_mm.iloc[0], floor)), xytext=(14, 14), textcoords="offset points", fontsize=13, fontweight="bold", arrowprops=dict(arrowstyle="-", color="k", lw=.9), zorder=5)
    ax.axvline(HZ, color="k", lw=1, ls="--"); ax.set_yscale("log"); ax.set_xlim(0, HZ * 1.05)
    ax.set_xlabel(f"memory in the state (days until a nudge on {int(a.start[3:])} {pd.Timestamp('2001-' + a.start).month_name()} fades to 37%)")
    ax.set_ylabel(f"effect on flow over the next {HZ} days\n(mm per one-standard-deviation nudge)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, fontsize=12, markerscale=.9, columnspacing=1.2, handletextpad=.3); ax.set_title(f"{NAMES.get(a.basin, a.basin)} {a.basin}, model version {a.seed}; median of six water years", fontsize=12, loc="left")
    fig.tight_layout(); f = D / f"nb32_memory_vs_flow_{a.basin}_seed{a.seed}_{a.start}.png"; fig.savefig(f, dpi=220); print("wrote", f)

if 3 in a.panels:
    E = pd.read_csv(D / "nb32_read1_end_dates.csv", dtype={"basin": str}); WIN = ORDER[:8] if a.full else ORDER[1:7]; x = np.arange(len(WIN))   # 1 Oct is half censored, 1 May is after melt begins
    E = E[E.start.isin(WIN)]; start_dowy = [(pd.Timestamp(f"{2000 if int(s_[:2]) >= 10 else 2001}-{s_}") - pd.Timestamp("2000-10-01")).days for s_ in WIN]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fm = D / "nb32_melt_centre_dates.csv"
    if fm.exists():
        Mc = pd.read_csv(fm, dtype={"basin": str}); Mc = Mc[Mc.basin.isin(E.basin.unique())].obs_melt_centre_dowy.dropna()
        ax.axhspan(Mc.quantile(.25), Mc.quantile(.75), color="#bbbbbb", alpha=.5, lw=0, label="observed centre of melt-season flow (middle half)")
    run = E.groupby(["basin", "seed", "start"]).end_dowy.median().unstack("start").reindex(columns=WIN); med = run.median()
    if a.full:
        for _, m in run.iterrows(): ax.plot(x, m, color=BLUE, lw=1, alpha=.3)
    else: ax.fill_between(x, run.quantile(.25), run.quantile(.75), color=BLUE, alpha=.2, lw=0, label="middle half of runs")
    ax.plot(x, med, color=BLUE, lw=3, marker="o", ms=9, label="day the store has forgotten the nudge (37% left)")
    ax.plot(x, start_dowy, color="k", lw=1.5, ls=":", marker="o", ms=5, label="day of the nudge")
    for j in range(len(WIN)): ax.annotate("", (x[j], med.values[j]), (x[j], start_dowy[j]), arrowprops=dict(arrowstyle="-", color=BLUE, lw=.8, alpha=.5))
    mt = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]; ax.set_yticks(mt, MON); ax.set_ylim(0, 400 if a.full else 365)
    ax.set_xticks(x, [f"1 {MON[ORDER.index(s_)]}" for s_ in WIN]); ax.set_xlabel("day the state is nudged along the direction flow reads most"); ax.set_ylabel("date")
    n = E.groupby(["basin", "seed"]).ngroups; ax.set_title(f"Thick line: median of {n} runs ({E.basin.nunique()} snow basins x {E.seed.nunique()} model versions)" + ("; thin lines: each run" if a.full else ""), fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=12); fig.tight_layout(); fig.savefig(D / ("nb32_memory_end_date_full.png" if a.full else "nb32_memory_end_date.png"), dpi=220); print("wrote", D / ("nb32_memory_end_date_full.png" if a.full else "nb32_memory_end_date.png"))

if 4 in a.panels:
    E = pd.read_csv(D / "nb32_read1_end_dates.csv", dtype={"basin": str}); WIN = ORDER[1:7]; x = np.arange(len(WIN))
    sd0 = {s_: (pd.Timestamp(f"{2000 if int(s_[:2]) >= 10 else 2001}-{s_}") - pd.Timestamp("2000-10-01")).days for s_ in WIN}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fm = D / "nb32_melt_centre_dates.csv"
    if fm.exists():
        Mc = pd.read_csv(fm, dtype={"basin": str}); Mc = Mc[Mc.basin.isin(E.basin.unique())].obs_melt_centre_dowy.dropna()
        left = np.array([Mc.median() - sd0[s_] for s_ in WIN]); ax.plot(x, left, "--", color="k", lw=2, label="days left until the observed centre of melt flow", zorder=4)
    for k, lab, col in (("read1", "direction flow reads most", BLUE), ("snow cell", "cells that track snow (one at a time)", GREEN)):
        s = S[S.kind2 == k].set_index("start").reindex(WIN)
        ax.fill_between(x, s.mem_lo, s.mem_hi, color=col, alpha=.18, lw=0); ax.plot(x, s.memory_d, color=col, lw=3, marker="o", ms=9, label=lab, zorder=3)
    ax.set_xticks(x, [f"1 {MON[ORDER.index(s_)]}" for s_ in WIN]); ax.set_ylim(0, 250); ax.set_xlim(-.3, len(WIN) - .7)
    ax.set_xlabel("day the state is nudged"); ax.set_ylabel("memory: days until a nudge\nhas faded to 37%")
    n = E.groupby(["basin", "seed"]).ngroups; ax.set_title(f"Median of {n} runs ({E.basin.nunique()} snow basins x {E.seed.nunique()} model versions); bands: middle half of runs", fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=13); fig.tight_layout(); fig.savefig(D / "nb32_memory_days_left.png", dpi=220); print("wrote", D / "nb32_memory_days_left.png")
