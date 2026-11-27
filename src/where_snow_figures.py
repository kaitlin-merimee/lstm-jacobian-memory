"""where_snow_figures: two talk figures for the snow-store result, drawn from the where_snow_patching and where_snow_followup tables (no model runs).

  1  nb29_recovered.png      share of the snowfall effect on melt flow that comes back when one part of the 1 March state
                             change is put into the unchanged run; bars are medians, dots are the runs (basin x model version)
  2  nb29_when_vs_how_much.png   two questions for three internal readings (best snow cell, linear snow probe, read directions):
                             does it follow snow through the year, and does it rank the years by how much snow fell?

  python src/where_snow_collect.py          # first, if nb27_all_runs.csv is not there yet
  python src/where_snow_figures.py
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from jtalk_core import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "analysis/read_directions")); ap.add_argument("--pca-k", type=int, default=3, help="PCA components in figure 1: 3 (like for like) or 10")
ap.add_argument("--panels", type=int, default=3, help="2 drops the observed-melt-flow panel of figure 2"); ap.add_argument("--date", default="03-01", help="use the runs made with this --date"); a = ap.parse_args(); D = Path(a.dir)
SUF = "" if a.date == "03-01" else f"_d{a.date}"; DL = f"{int(a.date[3:])} {pd.Timestamp('2001-' + a.date).month_name()}"
NAMES = {"06623800": "Encampment R., WY", "12178100": "Newhalem Cr., WA", "13235000": "SF Payette R., ID", "09066300": "Middle Cr., CO", "09312600": "White R., UT"}   # check against camels_name.txt
nm = lambda b: f"{NAMES[b]}\n{b}" if b in NAMES else b
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False}); BLUE, ORANGE, GREEN, PURPLE, GREY = "#3b6fb6", "#c8782a", "#5a9e6f", "#8a5fb0", "#999999"; rng = np.random.default_rng(0)

# ---- figure 1 ------------------------------------------------------------------------------------------------------------------------
T = pd.read_csv(D / f"nb27_all_runs{SUF}.csv", dtype={"basin": str})
bars = [("recovered: top 3 read directions", "3 directions\nflow reads", BLUE), (f"recovered: top {a.pca_k} PCA components", f"{a.pca_k} directions of\nmost variance (PCA)", ORANGE), ("recovered: probe direction", "linear snow\nprobe", PURPLE),
        ("recovered: snow-tracking cells", "cells that\ntrack snow", GREEN), ("recovered: 10 random directions", "10 random\ndirections", GREY)]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for j, (c, lab, col) in enumerate(bars):
    v = T[c].values; ax.bar(j, np.median(v), color=col, alpha=.85, width=.68); ax.scatter(j + rng.uniform(-.2, .2, len(v)), v, s=22, color="k", alpha=.55, zorder=3)
    ax.text(j, 1.0, f"{np.median(v):.0%}", ha="center", va="top", fontsize=20, fontweight="bold", color=col, transform=ax.get_xaxis_transform())
ax.axhline(1, color="k", lw=1, ls="--"); ax.text(len(bars) - .55, 1.02, "whole state change", ha="right", va="bottom", fontsize=12); ax.set_xlim(-.6, len(bars) - .4); ax.axhline(0, color="k", lw=.8)
ax.set_xticks(range(len(bars)), [b[1] for b in bars]); ax.set_ylabel("share of the snowfall effect\non melt flow recovered"); ax.set_ylim(min(-.2, T[[b[0] for b in bars]].min().min() - .05), max(1.4, T[[b[0] for b in bars]].max().max() + .3)); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_xlabel(f"Bars and numbers: median. Dots: {len(T)} runs ({T.basin.nunique()} snow basins x {T.seed.nunique()} model versions).", fontsize=12, labelpad=12)
fig.tight_layout(); fig.savefig(D / f"nb29_recovered{SUF}.png", dpi=220); print("wrote", D / f"nb29_recovered{SUF}.png"); print(T[[b[0] for b in bars]].describe().loc[["min", "50%", "max"]].round(2).to_string())

# ---- figure 2 ------------------------------------------------------------------------------------------------------------------------
readers = [("best_cell", "best snow cell", GREEN), ("probe", "linear snow probe", PURPLE), ("stored_melt_flow", "directions flow reads", BLUE)]; rows = []
for f in sorted(D.glob(f"nb28_D2_between_years_seed?{SUF}.csv")):
    seed = int(f.stem.split("seed")[-1][0]); D2 = pd.read_csv(f, dtype={"basin": str}); Z = D2.drop(columns=["basin", "wy"]).groupby(D2.basin).transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    for rd, _, _ in readers:
        day = []
        for b in D2.basin.unique():
            g = pd.read_csv(D / f"nb28_D1_daily_{b}_seed{seed}{SUF}.csv"); col = "stored_melt_flow_mm" if rd == "stored_melt_flow" else rd; ok = g.snow_reference.notna(); day.append(spearmanr(g.loc[ok, col], g.loc[ok, "snow_reference"])[0])
        rows.append(dict(seed=seed, reader=rd, through_the_year=float(np.median(day)), **{t: spearmanr(Z[rd], Z[t], nan_policy="omit")[0] for t in ("winter_snowfall_mm", "snow_on_1_march", "melt_flow_model_mm", "melt_flow_observed_mm")}))
W = pd.DataFrame(rows); W.to_csv(D / f"nb29_when_vs_how_much{SUF}.csv", index=False); print(f"\nrank correlations, median of model versions (snow_on_1_march = the snow product on {DL}, the quantity the probe was fitted to):"); print(W.groupby("reader", sort=False).median(numeric_only=True).drop(columns="seed").round(2).to_string())
panels = [("through_the_year", "When: follows snow\nthrough the year"), ("winter_snowfall_mm", "How much: ranks the years\nby winter snowfall"), ("melt_flow_observed_mm", "How much: ranks the years\nby observed melt flow")][:a.panels]
fig, ax = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 5.2), sharey=True)
for x_, (t, title) in zip(ax, panels):
    for j, (rd, lab, col) in enumerate(readers):
        v = W.loc[W.reader == rd, t].values; x_.bar(j, np.median(v), color=col, alpha=.85, width=.68); x_.scatter(j + np.linspace(-.15, .15, len(v)), v, s=30, color="k", alpha=.6, zorder=3)
        x_.text(j, max(np.nanmax(v), 0) + .04, f"{np.median(v):.2f}", ha="center", va="bottom", fontsize=16, fontweight="bold", color=col)
    x_.set_xticks(range(3), [r[1].replace(" ", "\n", 1) for r in readers], fontsize=13); x_.set_title(title, fontsize=15); x_.set_ylim(min(0, np.nanmin(W[[p_[0] for p_ in panels]].values) - .05), 1.15); x_.axhline(0, color="k", lw=.8)
ax[0].set_ylabel("rank correlation"); fig.suptitle(f"Dots: {W.seed.nunique()} model versions. Five snow basins. \"How much\" uses the reading on {DL} in each of six years.", fontsize=12, y=.02, va="bottom")
fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(D / f"nb29_when_vs_how_much{SUF}.png", dpi=220); print("wrote", D / f"nb29_when_vs_how_much{SUF}.png")

# ---- figure 3: are the directions shared? (same basin order on both axes, median over model versions) ----------------------------------
E = pd.concat([pd.read_csv(f, dtype={"target": str, "source": str}) for f in sorted(D.glob(f"nb28_E_transfer_seed?{SUF}.csv"))]); E = E[E.k == E.k.min()]
M = E.groupby(["target", "source"]).recovered.median().unstack("source"); order = sorted(M.index, key=lambda b: (b == "12178100", b)); M = M.loc[order, order + ["pooled"]]
fig, ax = plt.subplots(figsize=(8.6, 6.6)); ax.imshow(np.clip(M.values, 0, 1), vmin=0, vmax=1, cmap="Blues")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M.values[i, j]:.0%}", ha="center", va="center", color="w" if M.values[i, j] > .6 else "k", fontsize=13)
ax.axvline(len(order) - .5, color="w", lw=4); ax.set_xticks(range(M.shape[1]), [nm(b) for b in M.columns[:-1]] + ["all basins\npooled"], rotation=45, ha="right", fontsize=11); ax.set_yticks(range(M.shape[0]), [nm(b) for b in M.index], fontsize=11)
ax.set_xlabel("three directions taken from basin"); ax.set_ylabel("used in basin"); ax.set_title(f"Share of the snowfall effect recovered (median of {E.groupby(['target', 'source']).size().max()} model versions)", fontsize=12, loc="left")
fig.tight_layout(); fig.savefig(D / f"nb29_shared{SUF}.png", dpi=220); print("wrote", D / f"nb29_shared{SUF}.png"); print(M.round(2).to_string())

# ---- figure 4: one basin, the store against snow water equivalent, each scaled to its own six-year range ----------------------------------
for f in sorted(D.glob(f"nb28_D1_daily_06623800_seed1{SUF}.csv")) or sorted(D.glob(f"nb28_D1_daily_*_seed1{SUF}.csv"))[:1]:
    g = pd.read_csv(f, parse_dates=["date"]); sc = lambda v: (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)); fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(g.date, sc(g.snow_reference), color="#bbbbbb", alpha=.7, lw=0, label="snow on the ground (satellite-based SWE)"); ax.plot(g.date, sc(g.stored_melt_flow_mm), color=BLUE, lw=2.5, label="what flow reads from the state")
    ax.set_yticks([0, 1], ["low", "high"]); ax.set_ylim(0, 1.25); ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=13); b_ = f.stem.split("_")[3]; ax.set_title(f"{NAMES.get(b_, 'Basin')} ({b_}), one model version; each line scaled to its own range", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(D / f"nb29_store_vs_snow{SUF}.png", dpi=220); print("wrote", D / f"nb29_store_vs_snow{SUF}.png")
