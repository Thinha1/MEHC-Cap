"""Merge the 4 shards of 1,250 images of an on/off configuration into one 5,000-image report and score the five-signal reranker.

Run from the repo root (CPU by default, so the GPU stays free for generation):
    python eval/onoff_postprocess.py --name cs_g3 --evidence union --spice
    python eval/onoff_postprocess.py --name e40_existing --merged results/e40_g3_yoloworld_merged.json
Output: results/onoff/<name>_merged.json (when merging) and results/onoff/<name>_eval.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dataset" / "retrieval_cache"
# every report covers the same 5,000 test images in the same order, so they share the retrieval neighbours
NB_SOURCE = CACHE / "neighbours_test_yoloworld_k5.npy"  # built in README step 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--evidence", choices=["union", "yolo", "retr"], default="union")
    ap.add_argument("--merged", type=Path, help="use an already merged report instead of merging 4 shards")
    ap.add_argument("--spice", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="allow the GPU (CPU by default)")
    a = ap.parse_args()
    out_dir = ROOT / "results" / "onoff"
    if a.merged:
        report = a.merged if a.merged.is_absolute() else ROOT / a.merged
    else:
        rows = []
        for start in (0, 1250, 2500, 3750):
            d = json.loads((out_dir / f"{a.name}_part{start}.json").read_text(encoding="utf-8"))
            run = [r for r in d["runs"] if r["config"] != "report"][0]
            rows += run["rows"]
        assert len(rows) == 5000, len(rows)
        report = out_dir / f"{a.name}_merged.json"
        report.write_text(json.dumps({"split": "test", "config": run["config"], "rows": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"merged {len(rows)} images -> {report}")
    nb = CACHE / f"neighbours_{report.stem}_k5.npy"
    if not nb.is_file():
        shutil.copyfile(NB_SOURCE, nb)
    env = dict(os.environ)
    if not a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "sobj_threshold_eval.py"), "--report", str(report),
           "--output", str(out_dir / f"{a.name}_eval.json"), "--thresholds", "3", "--evidence", a.evidence]
    if a.spice:
        cmd.append("--spice")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
