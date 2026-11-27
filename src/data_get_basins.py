"""data_get_basins.py — download USGS NLDI drainage-basin outlines (GeoJSON) for a list of gauges.

Skips gauges whose file already exists, retries each request, and writes a log of failures.
Light network work (one small request per gauge), fine on a login node.

usage: python src/data_get_basins.py --basin-file basin_lists/camels_531_basins.txt --out data/basins
"""
import argparse, json, time, urllib.request
from pathlib import Path

URLS = ["https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{g}/basin",
        "https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{g}/basin"]

ap = argparse.ArgumentParser()
ap.add_argument("--basin-file", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
gauges = [l.strip() for l in Path(a.basin_file).read_text().splitlines() if l.strip()]
failed = []
for n, g in enumerate(gauges, 1):
    f = out / f"{g}.geojson"
    if f.exists() and f.stat().st_size > 200:
        continue
    ok = False
    for url in URLS:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url.format(g=g), timeout=60) as r:
                    gj = json.loads(r.read())
                if gj.get("features"):
                    f.write_text(json.dumps(gj)); ok = True
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if ok:
            break
    if not ok:
        failed.append(g)
    if n % 25 == 0:
        print(f"{n}/{len(gauges)} done, {len(failed)} failed", flush=True)
    time.sleep(0.2)
(out / "_failed.txt").write_text("\n".join(failed))
print(f"finished: {len(gauges) - len(failed)} outlines, {len(failed)} failed (listed in {out / '_failed.txt'})")
