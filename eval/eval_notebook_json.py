"""Re-score a result file of the coco2014-eval-reranking notebook (run on Kaggle) on CPU and add SPICE.

The file stores the selected caption of every image. The script scores two points: before_rerank (baseline_candidate) and five_score
(caption_five_score). It recomputes BLEU-1..4, METEOR, ROUGE-L, CIDEr, CHAIRs and CHAIRi exactly as the score() function
of the notebook and checks them against the metrics stored in the file. --spice adds SPICE through spice_per_caption.py: each
(image, caption) pair is scored once with Java 8, only for the two selected captions of each image, with checkpoints per chunk.

Usage (from the repo root, Java for METEOR, Java 8 for SPICE):
    python eval/eval_notebook_json.py results/kaggle/eval_rerank_<ckpt>_d0_test.json
    python eval/eval_notebook_json.py <file1>.json <file2>.json --spice
Writes <file>_cpu_eval.json next to the input. Per-caption SPICE is saved in <file>_spice_per_caption.json and reused on a rerun.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import chair_metric  # noqa: E402
from evaluate_reranked_quality import compute_metrics, simple_standardize  # noqa: E402

DATASET = ROOT / "dataset"
SELECTIONS = {"before_rerank": "baseline_candidate", "five_score": "caption_five_score"}
KEYS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE", "CHAIRs", "CHAIRi")


def evaluate(path: Path, spice: bool, objects, inv, fn2id, inst) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    refs = [r["references"] for r in rows]
    gt = {r["image"]: chair_metric.ground_truth_from_captions(r["references"], objects, inv) | inst.get(fn2id.get(r["image"]), set())
          for r in rows}
    sp = None
    if spice:
        per = path.with_name(path.stem + "_spice_per_caption.json")
        if not per.is_file():
            # send only the selected captions (not all 5 candidates): spice_per_caption reads baseline_candidate, caption, candidates
            sub = path.with_name(path.stem + "_spice_input.json")
            sub.write_text(json.dumps({"rows": [{"image": r["image"], "references": r["references"], "candidates": [],
                                                 "baseline_candidate": r["baseline_candidate"], "caption": r["caption_five_score"]}
                                                for r in rows]}, ensure_ascii=False), encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "eval" / "spice_per_caption.py"),
                            "--report", str(sub), "--output", str(per)], check=True, cwd=ROOT)
            sub.unlink()
        sp = json.loads(per.read_text(encoding="utf-8"))

    stored = data.get("metrics") or {}
    out = {}
    for name, key in SELECTIONS.items():
        if key not in rows[0]:
            print(f"[!] {path.name}: field {key} is missing, skipping {name}")
            continue
        caps = [r[key] for r in rows]
        m = compute_metrics(caps, refs, use_meteor=True, use_cider=True, use_spice=False, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: c for r, c in zip(rows, caps)}, gt, objects, inv)
        res = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}   # as in the notebook's score()
        res.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2))
        if sp is not None:
            res["SPICE"] = round(100 * float(np.mean([sp[r["image"]][simple_standardize(c)] for r, c in zip(rows, caps)])), 2)
        diff = {k: (stored[name][k], res[k]) for k in stored.get(name, {}) if k in res and abs(stored[name][k] - res[k]) > 0.011}
        print(f"{name:14s}", {k: res.get(k) for k in KEYS if k in res}, "| matches the stored metrics" if not diff else f"| [!] differs {diff}")
        out[name] = res
    return {"source": path.name, "delta": (data.get("run") or {}).get("delta"), "images": len(rows), "metrics": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--spice", action="store_true", help="add SPICE (Java 8, about 30 minutes per file)")
    args = ap.parse_args()

    objects, inv = chair_metric.load_synonyms()
    fn2id = chair_metric.load_filename_to_cocoid(DATASET / "dataset_coco.json")
    inst = chair_metric.load_instance_objects_by_cocoid(DATASET / "instances_val2014.json", inv)
    for f in args.files:
        print(f"===== {f.name}")
        res = evaluate(f, args.spice, objects, inv, fn2id, inst)
        out = f.with_name(f.stem + "_cpu_eval.json")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
