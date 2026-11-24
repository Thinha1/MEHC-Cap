"""Rebuild the three-seed MS COCO tables of the README from the released result JSONs (CPU only, no model or images).

    python eval/make_tables.py --dir results/kaggle

--dir holds the files written by notebooks/eval/coco2014-eval-reranking.ipynb (one per checkpoint and δ) and the
*_cpu_eval.json files that eval/eval_notebook_json.py --spice adds next to them (they carry SPICE).
Prints mean ± sample standard deviation over seeds 41, 42, 43.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import chair_metric  # noqa: E402

SEEDS = (41, 42, 43)
NAMES = {  # configuration -> file stem pattern
    "CE": "eval_rerank_baseline_epoch40_seed{s}_d0_test",
    "CE + LB(Y+R)": "eval_rerank_baseline_epoch40_seed{s}_d3_test",
    "DL+CL": "eval_rerank_seed{s}_dynlogits_contrastive_attnpool_epoch40_model_d0_test",
    "DL+CL + LB(Y)": "eval_rerank_seed{s}_dynlogits_contrastive_attnpool_epoch40_model_d3_yolo_test",
    "DL+CL + LB(R)": "eval_rerank_seed{s}_dynlogits_contrastive_attnpool_epoch40_model_d3_retr_test",
    "DL+CL + LB(Y+R)": "eval_rerank_seed{s}_dynlogits_contrastive_attnpool_epoch40_model_d3_test",
}


def fmt(v: list[float]) -> str:
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/kaggle"))
    args = ap.parse_args()
    objects, inv = chair_metric.load_synonyms()
    objs = lambda c: set(chair_metric.caption_to_words(c, objects, inv)[1])

    metrics, recall = {}, {}
    for name, pat in NAMES.items():
        for s in SEEDS:
            f = args.dir / (pat.format(s=s) + ".json")
            data = json.loads(f.read_text(encoding="utf-8"))
            cpu = f.with_name(f.stem + "_cpu_eval.json")
            m = data["metrics"]
            if cpu.is_file():  # CPU re-score: same BLEU/METEOR/ROUGE/CIDEr/CHAIR, plus SPICE
                m = {**m, **json.loads(cpu.read_text(encoding="utf-8"))["metrics"]}
            metrics[name, s] = m
            rec = []
            for r in data["rows"]:
                ref = set().union(*(objs(c) for c in r["references"]))
                if ref:
                    rec.append(len(objs(r["caption_five_score"]) & ref) / len(ref))
            recall[name, s] = 100 * float(np.mean(rec))

    col = lambda name, key, k: [metrics[name, s][key][k] for s in SEEDS]
    print("Main table (MS COCO test)")
    print("| Configuration | B-4 | METEOR | ROUGE-L | CIDEr | SPICE | CHs | CHi |")
    for label, name, key in (("CE", "CE", "before_rerank"), ("CE + RR", "CE", "five_score"),
                             ("DL+CL + LB(Y+R)", "DL+CL + LB(Y+R)", "before_rerank"),
                             ("DL+CL + LB(Y+R) + RR", "DL+CL + LB(Y+R)", "five_score")):
        print(f"| {label} | " + " | ".join(fmt(col(name, key, k)) for k in
                                           ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE", "CHAIRs", "CHAIRi")) + " |")

    print("\nReranking signals (generator DL+CL + LB(Y+R))")
    print("| Configuration | B-4 | CIDEr | CHs | CHi |")
    for label, key in (("Before reranking", "before_rerank"), ("Five signals (r = 3)", "five_score"),
                       ("Without S_obj", "plus_s_retr"), ("Without S_retr", "plus_s_obj"),
                       ("Five signals, r = 1", "five_score_1of5")):
        print(f"| {label} | " + " | ".join(fmt(col("DL+CL + LB(Y+R)", key, k)) for k in
                                           ("BLEU-4", "CIDEr", "CHAIRs", "CHAIRi")) + " |")

    print("\nOn/off rows with three seeds")
    print("| Configuration | CHAIRs before | CIDEr before | CHAIRs after | CHAIRi after | CIDEr after | Recall after |")
    for name in ("CE", "CE + LB(Y+R)", "DL+CL", "DL+CL + LB(Y)", "DL+CL + LB(R)", "DL+CL + LB(Y+R)"):
        cells = [fmt(col(name, "before_rerank", "CHAIRs")), fmt(col(name, "before_rerank", "CIDEr")),
                 fmt(col(name, "five_score", "CHAIRs")), fmt(col(name, "five_score", "CHAIRi")),
                 fmt(col(name, "five_score", "CIDEr")), fmt([recall[name, s] for s in SEEDS])]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
