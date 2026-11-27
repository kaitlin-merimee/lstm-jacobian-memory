# ===================== CELL 4b — Figure A, slide version (4 panels) =====================
# Uses what nb11 already has in memory: BASINS, DAILY[role]["tau"|"h"|"dates"], OBS, SIM, FORC, model.w_out, model.b_out
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.ticker as mticker

PCOL, TCOL = "prcp(mm/day)", "tmean"          # <-- set to nb11's forcing column names
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
BAND_COLORS = {"quick": "#BDBDBD", "event": "#6BAED6", "seasonal": "#2171B5", "integrator": "#08306B"}
BAND_LABEL  = {"quick": "< 3 days", "event": "3–30 days", "seasonal": "1–12 months", "integrator": "> 1 year"}
REF = [(1, "1 day"), (7, "1 week"), (30, "1 month"), (365, "1 year")]

def band_of(t):
    return np.select([t < 3, t < 30, t < 365], ["quick", "event", "seasonal"], "integrator")

def figure_a_slide(role, title, mem_ylim=(0.3, 1000), outfile=None):
    D = DAILY[role]; dates = D["dates"]; w = np.asarray(model.w_out); wa = np.abs(w)
    o, s, f = OBS[role].loc[dates], SIM[role].loc[dates], FORC[role].loc[dates]

    # readout-weighted memory length per day, in days
    mem = 10 ** ((np.log10(D["tau"]) * wa).sum(1) / wa.sum())
    # exact split of the prediction by memory band (readout is linear), mapped to mm/d
    y_norm = D["h"] @ w + model.b_out
    a, c0 = np.polyfit(y_norm, s.values, 1)
    ub = band_of(np.median(D["tau"], axis=0))
    contrib = {k: a * (D["h"][:, ub == k] @ w[ub == k]) for k in BAND_COLORS}

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.3, 1, 1.3]})
    # 1 forcing: rain vs snow
    snow = f[TCOL] < 0
    ax[0].bar(dates[~snow], f[PCOL][~snow], color="#2171B5", width=1, label="rain")
    ax[0].bar(dates[snow],  f[PCOL][snow],  color="#9ECAE1", width=1, label="snow (T < 0 °C)")
    ax[0].set_ylabel("Precip\n(mm/d)"); ax[0].legend(loc="upper right", frameon=False, ncol=2)
    # 2 discharge
    ax[1].plot(dates, o, color="k", lw=1.6, label="observed")
    ax[1].plot(dates, s, color="#D95F02", lw=1.6, label="LSTM")
    ax[1].set_ylabel("Discharge\n(mm/d)"); ax[1].legend(loc="upper right", frameon=False, ncol=2)
    # 3 memory length in days
    ax[2].plot(dates, mem, color="#08306B", lw=1.8)
    ax[2].set_yscale("log"); ax[2].set_ylim(*mem_ylim)
    for v, lab in REF:
        ax[2].axhline(v, color="0.75", lw=0.8, ls=":")
        ax[2].text(dates[-1], v, " " + lab, va="center", fontsize=12, color="0.4")
    ax[2].set_yticks([v for v, _ in REF]); ax[2].set_yticklabels([str(v) for v, _ in REF])
    ax[2].set_ylabel("Memory\nlength (days)")
    # 4 where the prediction comes from
    pos = np.zeros(len(dates)); neg = np.zeros(len(dates))
    for k, col in BAND_COLORS.items():
        v = contrib[k]; base = np.where(v >= 0, pos, neg)
        ax[3].fill_between(dates, base, base + v, color=col, lw=0, step="mid", label=BAND_LABEL[k])
        pos = np.where(v >= 0, pos + v, pos); neg = np.where(v < 0, neg + v, neg)
    ax[3].axhline(0, color="k", lw=0.6)
    ax[3].set_ylabel("Contribution to\nprediction (mm/d)")
    ax[3].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=4,
                 title="prediction carried by units with memory of", title_fontsize=13, alignment="left")
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(title, loc="left", fontweight="bold")
    fig.align_ylabels(ax); fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig

# for role, b in BASINS.items():
#     figure_a_slide(role, f"{b['name']} ({b['id']})", outfile=f"figs/figA_slide_{role}.png")


# ===================== CELL 12 — recession plot (slide 12 replacement) =====================
# -dQ/dt vs Q on rain-free recession days, obs vs LSTM. Uses the full record, not one WY.
def recession_points(q, p, p_max=1.0, lag_dry=2):
    """Daily recession pairs (Q, -dQ/dt), Brutsaert–Nieber style: centred difference,
    only falling days with no rain today or in the previous `lag_dry` days."""
    q = q.astype(float)
    dq = (q.shift(-1) - q.shift(1)) / 2.0
    qm = q
    dry = (p.rolling(lag_dry + 1).max() <= p_max)
    keep = dry & (dq < 0) & (q.shift(-1) < q) & (q < q.shift(1)) & (qm > 0)
    return qm[keep].values, -dq[keep].values

def binned(qv, rv, nbins=15, min_n=10):
    edges = np.logspace(np.log10(np.percentile(qv, 1)), np.log10(np.percentile(qv, 99.5)), nbins + 1)
    idx = np.digitize(qv, edges)
    xs, ys = [], []
    for i in range(1, nbins + 1):
        m = idx == i
        if m.sum() >= min_n:
            xs.append(np.median(qv[m])); ys.append(np.median(rv[m]))
    return np.array(xs), np.array(ys)

def recession_figure(roles_titles, outfile=None):
    fig, axs = plt.subplots(1, len(roles_titles), figsize=(6.5 * len(roles_titles), 6))
    axs = np.atleast_1d(axs); out = {}
    for x, (role, title) in zip(axs, roles_titles):
        p = FORC[role][PCOL]
        for lab, q, col in [("observed", OBS[role], "k"), ("LSTM", SIM[role], "#D95F02")]:
            q = q.reindex(p.index)
            qv, rv = recession_points(q, p)
            x.scatter(qv, rv, s=5, color=col, alpha=0.12, lw=0)
            bx, by = binned(qv, rv)
            b, loga = np.polyfit(np.log10(bx), np.log10(by), 1)
            x.plot(bx, by, "o-", color=col, lw=2.2, ms=6, label=f"{lab}  (slope b = {b:.2f})")
            out[(role, lab)] = dict(n=len(qv), b=b, a=10**loga)
        x.set_xscale("log"); x.set_yscale("log")
        for axis in (x.xaxis, x.yaxis):
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        x.set_xlabel("Discharge Q (mm/d)"); x.set_ylabel("Recession rate  −dQ/dt (mm/d²)")
        x.set_title(title, loc="left", fontweight="bold"); x.legend(frameon=False, loc="upper left")
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig, out

# fig, rec = recession_figure([("snow", "Encampment River, WY"), ("arid", "N. Fk Guadalupe, TX")],
#                             outfile="figs/recession_two_basin.png")
# print(pd.DataFrame(rec).T.round(3))
