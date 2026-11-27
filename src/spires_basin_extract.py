"""spires_basin_extract.py — basin-average SPIReS SWE for a few gauges.

Reads the SPIReS reconstruction mosaics (one HDF5 per water year, MATLAB-written), averages
the 500 m cells whose centres fall inside each basin outline, and writes one daily CSV per basin
plus a check figure per basin (outline over that year's peak-SWE map).

Georeferencing comes from the file itself (Grid attrs): Albers equal-area, NAD83, parallels
29.5/45.5, origin 37.5N 113W, MATLAB referencing matrix  [x y] = [row col 1] * R  (1-based).
MATLAB stores arrays column-major, so h5py returns each day as (col, row): transposed here.
The Albers forward projection is implemented directly (Snyder 1987, eqs 14-1 to 14-14) so the
script needs only numpy, h5py, matplotlib and pandas.

Basin outlines: GeoJSON from the USGS NLDI service, one file per gauge:
  curl -s "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-06623800/basin" -o basins/06623800.geojson

usage:
  python spires_basin_extract.py --h5dir <spires_swe> --basins <dir with GAUGE.geojson> \
      --gauges 06623800 [GAUGE ...] --years 2009 2010 ... --out <outdir>
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

NODATA = 65535


def albers_from_attrs(g):
    """Forward Albers (ellipsoidal) using the parameters stored in the file's Grid attrs."""
    a, e = (float(v) for v in np.ravel(g.attrs["geoid"]))
    p1, p2 = np.radians(np.ravel(g.attrs["mapparallels"]).astype(float))
    lat0, lon0 = np.radians(np.ravel(g.attrs["origin"])[:2].astype(float))
    fe, fn = float(np.ravel(g.attrs["falseeasting"])[0]), float(np.ravel(g.attrs["falsenorthing"])[0])
    e2 = e * e
    m = lambda p: np.cos(p) / np.sqrt(1 - e2 * np.sin(p) ** 2)
    q = lambda p: (1 - e2) * (np.sin(p) / (1 - e2 * np.sin(p) ** 2)
                              - np.log((1 - e * np.sin(p)) / (1 + e * np.sin(p))) / (2 * e))
    m1, m2, q1, q2, q0 = m(p1), m(p2), q(p1), q(p2), q(lat0)
    n = (m1 ** 2 - m2 ** 2) / (q2 - q1)
    C = m1 ** 2 + n * q1
    rho0 = a * np.sqrt(C - n * q0) / n

    def fwd(lon_deg, lat_deg):
        lam, phi = np.radians(lon_deg), np.radians(lat_deg)
        rho = a * np.sqrt(C - n * q(phi)) / n
        th = n * (lam - lon0)
        return fe + rho * np.sin(th), fn + rho0 - rho * np.cos(th)
    return fwd


def grid_rc_to_xy(R, row, col):
    """MATLAB referencing matrix (h5py shows it as 2x3): x = R[0]@[row,col,1], y = R[1]@[row,col,1]; 1-based."""
    return R[0, 0] * row + R[0, 1] * col + R[0, 2], R[1, 0] * row + R[1, 1] * col + R[1, 2]


def xy_to_rc(R, x, y):
    M = np.array([[R[0, 0], R[0, 1]], [R[1, 0], R[1, 1]]])
    rc = np.linalg.solve(M, np.vstack([x - R[0, 2], y - R[1, 2]]))
    return rc[0], rc[1]


def load_outline(path):
    gj = json.loads(Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    rings = []
    for ft in feats:
        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rings += [np.asarray(p[0], float) for p in polys]      # outer rings only
    return rings


def basin_mask(rings, fwd, R, shape_rc):
    """Boolean mask (rows, cols) of cells whose centres fall inside the outline, plus its bounding box."""
    xy_rings = [np.column_stack(fwd(r[:, 0], r[:, 1])) for r in rings]
    allxy = np.vstack(xy_rings)
    rr, cc = xy_to_rc(R, allxy[:, 0], allxy[:, 1])
    r0, r1 = int(np.floor(rr.min())) - 2, int(np.ceil(rr.max())) + 2
    c0, c1 = int(np.floor(cc.min())) - 2, int(np.ceil(cc.max())) + 2
    r0, c0 = max(r0, 1), max(c0, 1); r1, c1 = min(r1, shape_rc[0]), min(c1, shape_rc[1])
    R_, C_ = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    X, Y = grid_rc_to_xy(R, R_, C_)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for ring in xy_rings:
        inside |= MPath(ring).contains_points(pts)
    return inside.reshape(R_.shape), (r0, r1, c0, c1), xy_rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5dir", required=True); ap.add_argument("--basins", required=True)
    ap.add_argument("--gauges", nargs="+", required=True); ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = {g: [] for g in a.gauges}
    for yr in a.years:
        fpath = Path(a.h5dir) / f"mosaicreconstruction_{yr}.h5"
        with h5py.File(fpath, "r") as f:
            G = f["Grid"]; R = np.asarray(G.attrs["ReferencingMatrix"], float)
            fwd = albers_from_attrs(G)
            # MATLAB datenums (the ISOdates attr is one space-separated string with some separators missing)
            mdates = np.ravel(f.attrs["MATLABdates"]).astype(float)
            dates = pd.to_datetime(mdates - 719529, unit="D").normalize()     # 719529 = datenum(1970,1,1)
            nday, ncol, nrow = G["swe"].shape                          # h5py order: (day, col, row)
            assert len(dates) == nday, f"{fpath.name}: {len(dates)} dates vs {nday} days of data"
            assert dates[0] == pd.Timestamp(yr - 1, 10, 1), f"{fpath.name}: first date {dates[0].date()}, expected {yr - 1}-10-01"
            for g in a.gauges:
                mask, (r0, r1, c0, c1), xy_rings = basin_mask(load_outline(Path(a.basins) / f"{g}.geojson"),
                                                              fwd, R, (nrow, ncol))
                if mask.sum() == 0:
                    raise SystemExit(f"{g}: no SPIReS cells inside the outline in {yr} (outside the domain?)")
                print(f"{yr} {g}: {mask.sum()} cells ({mask.sum() * 0.25:.0f} km2), rows {r0}-{r1}, cols {c0}-{c1}", flush=True)
                sl = (slice(c0 - 1, c1), slice(r0 - 1, r1))             # 1-based inclusive -> python, (col, row)
                cube = {}
                for var in ["swe", "sweHybrid", "melt"]:
                    v = G[var][:, sl[0], sl[1]].transpose(0, 2, 1).astype(float)   # -> (day, row, col)
                    v[v == NODATA] = np.nan
                    cube[var] = v
                for d in range(nday):
                    rows[g].append({"date": dates[d],
                                    **{f"{var}_mm": np.nanmean(cube[var][d][mask]) for var in cube},
                                    "n_cells": int(mask.sum()),
                                    "frac_valid": float(np.isfinite(cube["swe"][d][mask]).mean())})
                # check figure: this year's peak SWE with the outline on top (catches a flipped grid)
                pk = np.nanmax(cube["swe"], axis=0)
                X0, Y0 = grid_rc_to_xy(R, r0 - 0.5, c0 - 0.5); X1, Y1 = grid_rc_to_xy(R, r1 + 0.5, c1 + 0.5)
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(pk, extent=[X0, X1, Y1, Y0], cmap="Blues")
                ax.set_ylim(min(Y0, Y1), max(Y0, Y1))
                for ring in xy_rings: ax.plot(ring[:, 0], ring[:, 1], color="#D95F02", lw=1.5)
                ax.set_title(f"USGS {g}, WY{yr}: peak SPIReS SWE (mm)")
                ax.set_xlabel("Albers x (m)"); ax.set_ylabel("Albers y (m)")
                fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
                fig.savefig(out / f"check_{g}_WY{yr}.png", dpi=120); plt.close(fig)
    for g, r in rows.items():
        df = pd.DataFrame(r).set_index("date").sort_index()
        df.to_csv(out / f"spires_{g}.csv")
        print(f"{g}: wrote {len(df)} days; peak basin-mean SWE {df.swe_mm.max():.0f} mm on {df.swe_mm.idxmax().date()}")


if __name__ == "__main__":
    main()
