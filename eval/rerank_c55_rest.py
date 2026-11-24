"""Re-score on/off configurations with the 0.55 threshold for S_obj (candidates generated with the 0.25 blocking rule), GPU if available.

    python eval/rerank_c55_rest.py cs_g0|e40_g0|e40_yolo|signals|cl_g0|cl_g3|dl_g0|dl_g3
  cs_g0     baseline without blocking, five signals (r=3) + SPICE
  e40_g0    proposed without blocking, union evidence
  e40_yolo  proposed with YOLO-World-only blocking, yolo evidence
  signals   proposed (e40_g3), signal table (the "without S_obj" row) with signal_table.py
  cl_g0     contrastive learning only (CL), no blocking, five signals (r=3) + SPICE
  cl_g3     contrastive learning only (CL), blocking delta = 3, five signals (r=3) + SPICE
  dl_g0     Dynamic Logits only (DL), no blocking, five signals (r=3) + SPICE
  dl_g3     Dynamic Logits only (DL), blocking delta = 3, five signals (r=3) + SPICE
Writes results/confsweep/rr55/<name>_rr55_*.json
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dataset" / "retrieval_cache"
OUT = ROOT / "results" / "confsweep" / "rr55"
OUT.mkdir(parents=True, exist_ok=True)
T = 0.55
NB = "neighbours_test_yoloworld_k5.npy"  # all reports cover the same 5,000 test images in the same order
JOBS = {"cs_g0": ("results/onoff/cs_g0_merged.json", NB, "union", True),
        "e40_g0": ("results/onoff/e40_g0_merged.json", NB, "union", False),
        "e40_yolo": ("results/onoff/e40_yolo_merged.json", NB, "yolo", False),
        "cl_g0": ("results/onoff/cl_g0_merged.json", NB, "union", True),
        "cl_g3": ("results/onoff/cl_g3_merged.json", NB, "union", True),
        "dl_g0": ("results/onoff/dl_g0_merged.json", NB, "union", True),
        "dl_g3": ("results/onoff/dl_g3_merged.json", NB, "union", True),
        "signals": ("results/onoff/e40_g3_merged.json", NB, "union", False)}
name = sys.argv[1]
src, nbf, ev, spice = JOBS[name]
d = json.load(open(src, encoding="utf-8"))
rows = d["rows"] if "rows" in d else [r for r in d["runs"] if r["config"] != "report"][0]["rows"]
for r in rows:
    r["objects"] = [o for o in r["objects"] if o["confidence"] >= T]
rep = OUT / f"{name}_rr55_merged.json"
rep.write_text(json.dumps({"split": "test", "rows": rows}, ensure_ascii=False), encoding="utf-8")
nb = CACHE / f"neighbours_{rep.stem}_k5.npy"
if not nb.is_file(): shutil.copyfile(CACHE / nbf, nb)
env = dict(os.environ)
py = sys.executable
if name == "signals":
    out = OUT / "signals_rr55.json"
    if not out.is_file():
        subprocess.run([py, "eval/signal_table.py", "--report", str(rep), "--output", str(out)], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(name, "done", flush=True)
else:
    base = [py, "eval/sobj_threshold_eval.py", "--report", str(rep), "--evidence", ev, "--thresholds", "3"]
    e = OUT / f"{name}_rr55_eval.json"
    if not e.is_file():
        subprocess.run(base + ["--output", str(e)], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(name, "eval done", flush=True)
    if spice:
        s = OUT / f"{name}_rr55_spice_final.json"
        if not s.is_file():
            subprocess.run(base + ["--output", str(s), "--final-only", "--spice"], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(name, "SPICE done", flush=True)
rep.unlink()
