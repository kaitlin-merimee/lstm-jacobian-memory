"""spires_extract_all.py — basin-average SPIReS SWE for many basins, one water year per run.

Designed as a Slurm array (one task per water year). Each day of the mosaic is read ONCE and
every basin is averaged from it, so cost grows with days, not with basins.
Basins whose outline falls outside the SPIReS grid are skipped; partly covered basins are kept
and flagged with the fraction of their area inside the grid (see coverage_WY{yr}.csv).

Geometry, projection and the MATLAB transpose are handled exactly as in spires_basin_extract.py
(validated against pyproj and an independent calculation).

usage:
  python src/spires_extract_all.py --h5dir <spires_swe> --basins data/basins \
      --basin-file basin_lists/camels_531_basins.txt --year 2011 --out analysis/spires/all
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spires_basin_extract import albers_from_attrs, xy_to_rc, grid_rc_to_xy, load_outline, NODATA
from matplotlib.path import Path as MPath

VARS = ["swe", "sweHybrid", "melt"]


def ring_area_km2(xy):
    x, y = xy[:, 0], xy[:, 1]
    return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2 / 1e6


def basin_cells(rings, fwd, R, nrow, ncol):
    """Cell (row, col) indices (0-based) whose centres are inside the outline, plus the outline area."""
    xy_rings = [np.column_stack(fwd(r[:, 0], r[:, 1])) for r in rings]
    area = sum(ring_area_km2(r) for r in xy_rings)
    allxy = np.vstack(xy_rings)
    rr, cc = xy_to_rc(R, allxy[:, 0], allxy[:, 1])
    r0, r1 = max(int(np.floor(rr.min())) - 2, 1), min(int(np.ceil(rr.max())) + 2, nrow)
    c0, c1 = max(int(np.floor(cc.min())) - 2, 1), min(int(np.ceil(cc.max())) + 2, ncol)
    if r0 > r1 or c0 > c1:
        return np.empty(0, int), np.empty(0, int), area
    Rg, Cg = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    X, Y = grid_rc_to_xy(R, Rg, Cg)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for ring in xy_rings:
        inside |= MPath(ring).contains_points(pts)
    return Rg.ravel()[inside] - 1, Cg.ravel()[inside] - 1, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5dir", required=True); ap.add_argument("--basins", required=True)
    ap.add_argument("--basin-file", required=True); ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--min-coverage", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gauges = [l.strip() for l in Path(a.basin_file).read_text().splitlines() if l.strip()]
    fpath = Path(a.h5dir) / f"mosaicreconstruction_{a.year}.h5"
    with h5py.File(fpath, "r") as f:
        G = f["Grid"]; R = np.asarray(G.attrs["ReferencingMatrix"], float); fwd = albers_from_attrs(G)
        dates = pd.to_datetime(np.ravel(f.attrs["MATLABdates"]).astype(float) - 719529, unit="D").normalize()
        nday, ncol, nrow = G["swe"].shape
        assert len(dates) == nday and dates[0] == pd.Timestamp(a.year - 1, 10, 1), f"bad dates in {fpath.name}"

        cov, cells = [], {}
        for g in gauges:
            gj = Path(a.basins) / f"{g}.geojson"
            if not gj.exists():
                cov.append({"gauge": g, "status": "no_outline"}); continue
            r, c, area = basin_cells(load_outline(gj), fwd, R, nrow, ncol)
            frac = len(r) * 0.25 / area if area > 0 else 0.0           # 500 m cells = 0.25 km2
            status = "outside_grid" if len(r) == 0 else ("partial" if frac < a.min_coverage else "ok")
            cov.append({"gauge": g, "status": status, "area_km2": round(area, 1), "n_cells": len(r),
                        "frac_in_grid": round(frac, 3)})
            if status == "ok":
                cells[g] = (r, c)
        pd.DataFrame(cov).to_csv(out / f"coverage_WY{a.year}.csv", index=False)
        print(f"WY{a.year}: {len(cells)} basins inside the grid, of {len(gauges)}", flush=True)
        if not cells:
            return

        res = {v: np.full((nday, len(cells)), np.nan) for v in VARS}
        valid = np.full((nday, len(cells)), np.nan)
        for d in range(nday):
            for v in VARS:
                day = G[v][d]                                           # (col, row): MATLAB order
                for k, (g, (r, c)) in enumerate(cells.items()):
                    vals = day[c, r].astype(float)                       # index as (col, row) = transpose
                    vals[vals == NODATA] = np.nan
                    res[v][d, k] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                    if v == "swe":
                        valid[d, k] = np.isfinite(vals).mean()
            if d % 30 == 0:
                print(f"  day {d + 1}/{nday}", flush=True)

    keys = list(cells)
    df = pd.concat([pd.DataFrame({"date": dates, "gauge": g,
                                  **{f"{v}_mm": res[v][:, k] for v in VARS},
                                  "frac_valid": valid[:, k]}) for k, g in enumerate(keys)])
    df.to_csv(out / f"spires_WY{a.year}.csv", index=False)
    print(f"WY{a.year}: wrote {len(df)} rows for {len(keys)} basins")


if __name__ == "__main__":
    main()
