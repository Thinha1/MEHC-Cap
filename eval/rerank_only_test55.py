"""Run test once with the 0.55 threshold chosen on val: only the reranker uses the new threshold, candidates come from the 0.25 blocking rule. CPU.

    python eval/rerank_only_test55.py test    # proposed model (DL+CL + LB(Y+R)): all reranker variants, then SPICE for the five signals
    python eval/rerank_only_test55.py cs      # baseline with blocking: five signals (r=3), then SPICE
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dataset" / "retrieval_cache"
OUT = ROOT / "results" / "confsweep" / "rr55"
OUT.mkdir(parents=True, exist_ok=True)
T = 0.55
SETS = {"test": ("results/onoff/e40_g3_merged.json", "neighbours_test_yoloworld_k5.npy", []),
        "cs": ("results/onoff/cs_g3_merged.json", "neighbours_test_yoloworld_k5.npy", ["--thresholds", "3"])}
s = sys.argv[1]
src, nbf, extra = SETS[s]
d = json.load(open(src, encoding="utf-8"))
for r in d["rows"]:
    r["objects"] = [o for o in r["objects"] if o["confidence"] >= T]
rep = OUT / f"{s}_rr55_merged.json"
rep.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
nb = CACHE / f"neighbours_{rep.stem}_k5.npy"
if not nb.is_file(): shutil.copyfile(CACHE / nbf, nb)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
base = [sys.executable, "eval/sobj_threshold_eval.py", "--report", str(rep), "--evidence", "union"]
ev = OUT / f"{s}_rr55_eval.json"
if not ev.is_file():
    subprocess.run(base + ["--output", str(ev)] + extra, check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(s, "eval done", flush=True)
sp = OUT / f"{s}_rr55_spice_final.json"
if not sp.is_file():
    subprocess.run(base + ["--output", str(sp), "--final-only", "--spice", "--thresholds", "3"], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(s, "SPICE done", flush=True)
rep.unlink()
