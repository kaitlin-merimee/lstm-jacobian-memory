"""nb18: coupled-state memory (Lyapunov, inside 365-day windows) vs per-unit forget gates, seeds 1-3.

Every Lyapunov number so far is seed 1. This repeats the windowed calculation for each seed: inside
each zero-start window the daily Jacobians are multiplied with a QR re-orthonormalisation each day,
lambda_k = mean daily log stretch of direction k (after `--burn` days), tau_k = -1/lambda_k.
The uncoupled reference is each unit's forget gate alone, tau_j = -1/<ln f_j>, over the same days.

Seed 1 is the consistency check: `dirs > 365 d` should be close to the 18 from nb14 Cells 11-12.
If it is not, the two scripts define the window average differently (burn-in, k, spacing); compare seeds
within this script only, and do not mix its counts with nb14's on a slide.

  python src/backup/nb18_lyap_seeds.py --seeds 1 2 3            # about 5 min per seed on 8 cores
  python src/backup/nb18_lyap_seeds.py --toy
"""
import argparse, time, numpy as np, pandas as pd
from jtalk_core import Net, ROOT, SNOW, L_WIN

ap = argparse.ArgumentParser()
ap.add_argument("--basin", default=SNOW); ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
ap.add_argument("--k", type=int, default=64); ap.add_argument("--every", type=int, default=10)
ap.add_argument("--burn", type=int, default=30); ap.add_argument("--toy", action="store_true"); a = ap.parse_args()
OUT = (ROOT if not a.toy else __import__("pathlib").Path("toyroot")) / "analysis/lyapunov"; OUT.mkdir(parents=True, exist_ok=True)


def to_tau(lam):
    lam = np.asarray(lam, float); return np.where(lam < 0, -1 / np.minimum(lam, -1e-12), np.inf)


rows, spec = [], {}
for seed in a.seeds:
    net = Net(seed, a.basin, a.toy); net.gate_check(); t0 = time.time()
    k = min(a.k, 2 * net.H); e = net.ends(every=a.every)
    lam_w, lnf_w = [], []
    for w0 in range(0, len(e), 16):                                    # 16 windows at a time keeps memory small
        ew = e[w0:w0 + 16]; _, _, S = net.run_windows(ew, keep=True)
        for w in range(len(ew)):
            Q = np.linalg.qr(np.random.default_rng(w0 + w).normal(size=(2 * net.H, k)))[0]; Lg = np.zeros((L_WIN, k))
            for s in range(L_WIN):
                dh, dc = net.apply_J(S, s, w, Q[:net.H], Q[net.H:])
                Q, R = np.linalg.qr(np.vstack([dh, dc])); Lg[s] = np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
            lam_w.append(Lg[a.burn:].mean(0)); lnf_w.append(np.log(S["f"][a.burn:, w]).mean(0))
    lam = np.mean(lam_w, 0); tau = to_tau(lam)
    tau_unit = np.sort(to_tau(np.mean(lnf_w, 0)))[::-1]
    r = slice(1, 20); ok = np.isfinite(tau[r]) & np.isfinite(tau_unit[r])
    rows.append(dict(seed=seed, basin=a.basin, n_windows=len(e), k=k, dirs_gt_365=int((tau > 365).sum()), dirs_gt_30=int((tau > 30).sum()),
                     units_gt_365=int((tau_unit > 365).sum()), units_gt_30=int((tau_unit > 30).sum()),
                     tau_rank1=tau[0], tau_rank10=tau[min(9, k - 1)], tau_rank20=tau[min(19, k - 1)],
                     gate_over_coupled_rank2_20_min=float((tau_unit[r][ok] / tau[r][ok]).min()) if ok.any() else np.nan,
                     gate_over_coupled_rank2_20_max=float((tau_unit[r][ok] / tau[r][ok]).max()) if ok.any() else np.nan,
                     minutes=(time.time() - t0) / 60))
    spec[seed] = (tau, tau_unit)
    np.savez_compressed(OUT / f"nb18_windowed_{a.basin}_seed{seed}.npz", lam=lam, tau_unit=tau_unit, lam_windows=np.array(lam_w, np.float32))
df = pd.DataFrame(rows); df.to_csv(OUT / f"nb18_seeds_{a.basin}.csv", index=False)
pd.set_option("display.width", 220); print(df.round(2).to_string(index=False))
print(f"\nSLIDE NUMBERS  directions lasting > 365 d (of the {k} slowest): " + ", ".join(f"seed {r.seed}: {r.dirs_gt_365}" for r in df.itertuples()))
print("  gate memory / coupled memory, ranks 2-20: " + ", ".join(f"seed {r.seed}: {r.gate_over_coupled_rank2_20_min:.0f}-{r.gate_over_coupled_rank2_20_max:.0f}x" for r in df.itertuples()))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 5.4)); cap = 1e5
for n, (seed, (tau, tu)) in enumerate(spec.items()):
    rk = np.arange(1, len(tau) + 1)
    ax.loglog(rk, np.minimum(tau, cap), "o-", ms=4, color="#3b6fb6", alpha=1 if n == 0 else .45, lw=2.2 if n == 0 else 1,
              label="whole coupled state (chained Jacobians)" if n == 0 else None)
    m = min(len(tau), len(tu))
    ax.loglog(rk[:m], np.minimum(tu[:m], cap), "s-", ms=4, color="#c8782a", alpha=1 if n == 0 else .45, lw=2.2 if n == 0 else 1,
              label="each unit's forget gate alone" if n == 0 else None)
ax.axhline(365, color="k", ls="--", lw=1); ax.text(1.05, 420, "365-day window: the model never sees further back", fontsize=12)
ax.set(xlabel="rank (longest memory first)", ylabel="memory (days to fade by 1/e)"); ax.legend(frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(OUT / f"nb18_spectrum_seeds_{a.basin}.png", dpi=250); print("figure:", OUT / f"nb18_spectrum_seeds_{a.basin}.png")
