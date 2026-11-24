"""Chạy test đúng một lần với ngưỡng 0,55 đã chọn trên val: chỉ bộ chọn lại câu dùng ngưỡng mới, câu đã sinh với luật chặn ở 0,25. CPU.

    python eval/rerank_only_test55.py test    # mô hình đề xuất (DL+CL + LB(Y+R)): các cấu hình chọn câu rồi SPICE cho năm tín hiệu
    python eval/rerank_only_test55.py cs      # mô hình cơ sở có chặn: năm tín hiệu (r=3) rồi SPICE
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
print(s, "eval xong", flush=True)
sp = OUT / f"{s}_rr55_spice_final.json"
if not sp.is_file():
    subprocess.run(base + ["--output", str(sp), "--final-only", "--spice", "--thresholds", "3"], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(s, "SPICE xong", flush=True)
rep.unlink()
