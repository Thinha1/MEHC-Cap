"""Các lượt chấm còn lại ở ngưỡng 0,55 cho chọn lại câu (câu đã sinh ở luật chặn 0,25), dùng GPU nếu có.

    python eval/rerank_c55_rest.py cs_g0|e40_g0|e40_yolo|signals|cl_g0|cl_g3|dl_g0|dl_g3
  cs_g0     mô hình cơ sở không chặn, bộ năm tín hiệu (r=3) + SPICE
  e40_g0    đề xuất không chặn, bằng chứng union
  e40_yolo  đề xuất chặn chỉ YOLO-World, bằng chứng yolo
  signals   đề xuất (e40_g3), bảng tín hiệu (dòng "bỏ S_obj") bằng signal_table.py
  cl_g0     chỉ học tương phản (CL), không chặn, bộ năm tín hiệu (r=3) + SPICE
  cl_g3     chỉ học tương phản (CL), luật chặn delta = 3, bộ năm tín hiệu (r=3) + SPICE
  dl_g0     chỉ Dynamic Logits (DL), không chặn, bộ năm tín hiệu (r=3) + SPICE
  dl_g3     chỉ Dynamic Logits (DL), luật chặn delta = 3, bộ năm tín hiệu (r=3) + SPICE
Ghi results/confsweep/rr55/<tên>_rr55_*.json
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
    print(name, "xong", flush=True)
else:
    base = [py, "eval/sobj_threshold_eval.py", "--report", str(rep), "--evidence", ev, "--thresholds", "3"]
    e = OUT / f"{name}_rr55_eval.json"
    if not e.is_file():
        subprocess.run(base + ["--output", str(e)], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(name, "eval xong", flush=True)
    if spice:
        s = OUT / f"{name}_rr55_spice_final.json"
        if not s.is_file():
            subprocess.run(base + ["--output", str(s), "--final-only", "--spice"], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(name, "SPICE xong", flush=True)
rep.unlink()
