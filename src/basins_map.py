"""Locator map: the 531 CAMELS basins (grey) and the five snow basins used for the snow results (labelled), on US state
outlines, in an equal-area projection (CONUS Albers, EPSG:5070), with a scale bar and a north arrow.

Needs geopandas. State outlines: Natural Earth 1:50m states (public domain), downloaded once to data/naturalearth/ if not
there yet (or pass --states with any US states file geopandas can read). Gauge locations: camels_topo.txt; official gauge
names: camels_name.txt, printed so the names on the slides can be checked.

  PYTHONPATH=src python src/basins_map.py
  PYTHONPATH=src python src/basins_map.py --attrs /path/to/camels_attributes_v2.0 --states /path/to/states.shp
"""
import argparse, urllib.request, numpy as np, pandas as pd
from pathlib import Path
from jtalk_core import ROOT

NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces_lakes.zip"
ap = argparse.ArgumentParser()
ap.add_argument("--attrs", default=None, help="folder holding camels_topo.txt and camels_name.txt (default: searched under data/)")
ap.add_argument("--states", default=None, help="US states file (default: Natural Earth, downloaded once to data/naturalearth/)")
ap.add_argument("--basin-list", default=None, help="file with the 531 basin ids (default: the file in basin_lists/)")
ap.add_argument("--basins", nargs="*", default=["06623800", "09066300", "09312600", "13235000", "12178100"])
ap.add_argument("--labels", nargs="*", default=["Encampment R., WY", "Middle Cr., CO", "White R., UT", "SF Payette R., ID", "Newhalem Cr., WA"])
ap.add_argument("--scale-km", type=float, default=500); ap.add_argument("--out", default=None); a = ap.parse_args()
import geopandas as gpd
from pyproj import Transformer
CRS = "EPSG:5070"

# ---- gauges -----------------------------------------------------------------------------------------------------------
if a.attrs: topo = Path(a.attrs) / "camels_topo.txt"
else:
    hits = sorted((ROOT / "data").rglob("camels_topo.txt")); assert hits, "camels_topo.txt not found under data/: pass --attrs"; topo = hits[0]
T = pd.read_csv(topo, sep=";", dtype={"gauge_id": str}); T["gauge_id"] = T.gauge_id.str.zfill(8)
nm = topo.parent / "camels_name.txt"
if nm.exists(): N = pd.read_csv(nm, sep=";", dtype={"gauge_id": str}); N["gauge_id"] = N.gauge_id.str.zfill(8); T = T.merge(N[["gauge_id", "gauge_name"]], on="gauge_id", how="left")
bl = Path(a.basin_list) if a.basin_list else next(iter(sorted((ROOT / "basin_lists").glob("*"))), None)
A = T[T.gauge_id.isin({l.strip().zfill(8) for l in bl.read_text().split() if l.strip()})] if bl is not None and bl.exists() else T
G = gpd.GeoDataFrame(A, geometry=gpd.points_from_xy(A.gauge_lon, A.gauge_lat), crs="EPSG:4326").to_crs(CRS)
S = G.set_index("gauge_id").loc[a.basins]
print(f"read {topo}; {len(G)} basins on the map")
for (gid, r), lab in zip(S.iterrows(), a.labels):
    print(f"  {gid}  slide label: {lab:<20}  official name: {r.get('gauge_name', '?')}  ({r.gauge_lat:.2f}, {r.gauge_lon:.2f})")

# ---- states -----------------------------------------------------------------------------------------------------------
if a.states: st = gpd.read_file(a.states)
else:
    f = ROOT / "data/naturalearth" / Path(NE_URL).name
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True); print("downloading Natural Earth states (once):", NE_URL); urllib.request.urlretrieve(NE_URL, f)
    st = gpd.read_file(f)
cols = {c.lower(): c for c in st.columns}
if "admin" in cols: st = st[st[cols["admin"]] == "United States of America"]
if "name" in cols: st = st[~st[cols["name"]].isin(["Alaska", "Hawaii", "Puerto Rico"])]
st = st.to_crs(CRS)

# ---- figure -----------------------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(11, 7))
st.plot(ax=ax, color="#EFEDE5", edgecolor="white", linewidth=.8)
st.dissolve().boundary.plot(ax=ax, color="#9AA5B1", linewidth=.9)
ax.scatter(G.geometry.x, G.geometry.y, s=9, color="#b5b5b5", edgecolor="none", zorder=2, label=f"{len(G)} CAMELS basins (gauge locations)")
ax.scatter(S.geometry.x, S.geometry.y, s=130, color="#2A62A0", edgecolor="k", linewidth=.9, zorder=3, label="the five snow basins")
off = {"06623800": (12, 8), "09066300": (12, -10), "09312600": (-12, -14), "13235000": (12, 4), "12178100": (12, 6)}
for (gid, r), lab in zip(S.iterrows(), a.labels):
    dx, dy = off.get(gid, (12, 4))
    ax.annotate(lab, (r.geometry.x, r.geometry.y), xytext=(dx, dy), textcoords="offset points", fontsize=14, fontweight="bold", color="#1B2A41",
                ha="right" if dx < 0 else "left", va="center", zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.75))
x0, y0, x1, y1 = st.total_bounds; ax.set_xlim(x0 - .02 * (x1 - x0), x1 + .02 * (x1 - x0)); ax.set_ylim(y0 - .12 * (y1 - y0), y1 + .03 * (y1 - y0))
ax.set_aspect("equal"); ax.set_axis_off()

# scale bar (Albers is equal-area: distances are close to true near the middle of the map)
L = a.scale_km * 1000; bx, by = x0 + .56 * (x1 - x0), y0 - .07 * (y1 - y0)
for k in range(4): ax.add_patch(plt.Rectangle((bx + k * L / 4, by), L / 4, .012 * (y1 - y0), facecolor="#1B2A41" if k % 2 == 0 else "white", edgecolor="#1B2A41", lw=1, zorder=5))
for v in (0, L / 2, L): ax.text(bx + v, by - .012 * (y1 - y0), f"{v / 1000:g}", ha="center", va="top", fontsize=12)
ax.text(bx + L + .015 * (x1 - x0), by + .006 * (y1 - y0), "km", ha="left", va="center", fontsize=12)

# north arrow, pointing along the true meridian where it is drawn
to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True); to_xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
nx, ny = x1 - .04 * (x1 - x0), y0 + .10 * (y1 - y0); lo, la = to_ll.transform(nx, ny); tx, ty = to_xy.transform(lo, la + 1.5)
d = np.array([tx - nx, ty - ny]); d = d / np.linalg.norm(d) * .09 * (y1 - y0)
ax.annotate("", xy=(nx + d[0], ny + d[1]), xytext=(nx, ny), arrowprops=dict(arrowstyle="-|>", color="#1B2A41", lw=2.2, mutation_scale=22), zorder=5)
ax.text(nx + d[0] * 1.25, ny + d[1] * 1.25, "N", ha="center", va="center", fontsize=16, fontweight="bold", color="#1B2A41")

ax.legend(frameon=False, loc="lower left", fontsize=13, bbox_to_anchor=(0, -.02))
ax.text(1, -.03, "Equal-area projection (CONUS Albers). State outlines: Natural Earth.", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#55637A")
out = Path(a.out) if a.out else ROOT / "analysis/figures/basins_map.png"; out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight"); print("wrote", out)
