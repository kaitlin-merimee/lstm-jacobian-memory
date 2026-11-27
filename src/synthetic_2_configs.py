"""synthetic_2_configs.py — build full synthetic-target configs from the benchmark config.

Each synthetic config = configs/camels531_daymet_benchmark.yml with ONLY these keys changed:
  experiment_name, additional_feature_files, target_variables, clip_targets_to_zero
Everything else (data, dates, inputs, statics, architecture, hyperparameters, seed) is copied
unchanged, so the real and synthetic models differ only in what they are trained to predict.

usage (from the project root):
  python src/synthetic_2_configs.py            # dry run: prints what would be written
  python src/synthetic_2_configs.py --write    # writes configs/camels531_syn_{SCEN}_n00.yml
"""
import argparse, shutil, sys
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "configs" / "camels531_daymet_benchmark.yml"
SYN_DIR = PROJECT / "data" / "synthetic"
SCENS = ["S1", "S2", "S3", "G"]
NOISE = "n00"
CHANGED = {"experiment_name", "additional_feature_files", "target_variables", "clip_targets_to_zero"}

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
args = ap.parse_args()

base = yaml.safe_load(BASE.read_text())
assert base.get("seed") == 1, f"benchmark seed is {base.get('seed')}, expected 1"
assert base.get("experiment_name", "").startswith("camels531_daymet_h256_seed1"), base.get("experiment_name")
missing_pickles = [s for s in SCENS if not (SYN_DIR / f"syn_{s}_{NOISE}.p").exists()]
if missing_pickles:
    print(f"WARNING: synthetic targets not generated yet for {missing_pickles} (run synthetic_1_generate first)")

for scen in SCENS:
    cfg = dict(base)
    cfg["experiment_name"] = f"camels531_syn_{scen}_{NOISE}_h256_seed1"
    cfg["additional_feature_files"] = [str(SYN_DIR / f"syn_{scen}_{NOISE}.p")]
    cfg["target_variables"] = ["q_syn"]
    cfg["clip_targets_to_zero"] = ["q_syn"]
    diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
    if not diff <= CHANGED:
        sys.exit(f"refusing: {scen} would change keys outside the allowed set: {sorted(diff - CHANGED)}")
    out = PROJECT / "configs" / f"camels531_syn_{scen}_{NOISE}.yml"
    added = sorted(set(cfg) - set(base))
    print(f"{scen}: {out.name}  changes {sorted(diff)}  | keys added (not in benchmark): {added}")
    if args.write:
        if out.exists() and not (out.with_suffix(".yml.partial_bak")).exists():
            shutil.copy(out, out.with_suffix(".yml.partial_bak"))
            print(f"   backed up existing {out.name} -> {out.name}.partial_bak")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("written." if args.write else "dry run only; add --write to write.")
