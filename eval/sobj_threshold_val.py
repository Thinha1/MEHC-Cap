"""Choose the evidence threshold r_S of S_obj on the validation split, with weights w_obj = 1, w_retr = 0.1.

An object is not penalised when YOLO detects it (conf >= 0.25) or at least r of the K retrieved captions name it. Computes (CHAIRs, CHAIRi,
CIDEr, BLEU-4, no METEOR):
- sweep: K ∈ {3, 5, 10} × r ∈ {1..min(K, 5)}, five signals, plus a YOLO-only version ("yolo") with K = 5.
- obj_only: only 1·S_obj added (K = 5) for each r and the YOLO-only version.
- grid: w_retr ∈ {0, 0.1, 0.2, 0.3} × w_obj ∈ {0, 0.5, 1, 2} with K = 5, r = --grid-thr.
- table4: all metrics (with METEOR) for three signals / +0.1·S_retr / +1·S_obj / five signals with K = 5, r = --grid-thr.

Usage (from the repo root):
    python eval/sobj_threshold_val.py --report results/e35_k6_val_merged.json \
        --output results/e35_k6_sobj_threshold_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chair_metric  # noqa: E402
import sretr_experiment as se  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402

KS = [3, 5, 10]
YOLO_ONLY = 99  # a threshold no retrieval count reaches: YOLO is the only evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--grid-thr", type=int, default=3)
    args = ap.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data["rows"] if "rows" in data else [r for r in data["runs"] if r["config"] != "report"][0]["rows"]
    objects, inv = chair_metric.load_synonyms()
    clip = se.Clip()
    caps, emb = se.corpus(clip)
    gt = se.chair_gt(rows, objects, inv)
    refs = [r["references"] for r in rows]
    se.KS = KS
    nb = se.neighbours(clip, rows, emb, args.report.stem, max(KS))
    per_image = se.scores_for_report(clip, rows, nb, caps, emb, objects, inv)

    cap_objs = {}

    def objs(t):
        if t not in cap_objs:
            cap_objs[t] = chair_metric.caption_to_words(t, objects, inv)[1]
        return cap_objs[t]

    yolo = [{inv[o["label"]] for o in r.get("objects", []) if o["confidence"] >= 0.25 and o["label"] in inv} for r in rows]
    counts = {k: [Counter(o for j in idx[:k] for o in set(objs(caps[j]))) for idx in nb] for k in KS}

    def s_obj(k, thr):
        out = []
        for p, c, y in zip(per_image, counts[k], yolo):
            sup = y | {o for o, n in c.items() if n >= thr}
            out.append(np.array([-sum(o not in sup for o in objs(t)) for t in p["cands"]], dtype=np.float64))
        return out

    def pick(w_retr, w_obj, k, so):
        return [p["cands"][int(np.argmax(p["base"] + w_retr * se.minmax(p["s_retr"][k]) + w_obj * se.minmax(m)))]
                for p, m in zip(per_image, so)]

    def metrics(captions, meteor=False):
        m = compute_metrics(captions, refs, use_meteor=meteor, use_cider=True, use_spice=False, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: c for r, c in zip(rows, captions)}, gt, objects, inv)
        out = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()} if meteor else \
            {"CIDEr": round(100 * float(m["CIDEr"]), 2), "BLEU-4": round(float(m["BLEU-4"]), 2)}
        out.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2))
        se.log(str(out))
        return out

    res = {"three_score": metrics(se.select(per_image, None, 5, 0.0)), "sweep": {}, "obj_only": {}, "grid": {}}
    for k in KS:
        for thr in [t for t in (1, 2, 3, 4, 5) if t <= k] + ([YOLO_ONLY] if k == 5 else []):
            so = s_obj(k, thr)
            name = f"K{k}_r{thr}" if thr != YOLO_ONLY else f"K{k}_yolo"
            res["sweep"][name] = metrics(pick(0.1, 1.0, k, so))
            res["sweep"][name]["sobj_equal"] = round(100 * float(np.mean([m.max() == m.min() for m in so])), 1)
            if k == 5:
                res["obj_only"][name] = metrics(pick(0.0, 1.0, k, so))
    so = s_obj(5, args.grid_thr)
    for wr in (0.0, 0.1, 0.2, 0.3):
        for wo in (0.0, 0.5, 1.0, 2.0):
            res["grid"][f"{wr:g},{wo:g}"] = metrics(pick(wr, wo, 5, so))
    res["table4"] = {"three_score": metrics(se.select(per_image, None, 5, 0.0), meteor=True),
                     "plus_s_retr_0.1": metrics(pick(0.1, 0.0, 5, so), meteor=True),
                     "plus_s_obj_1": metrics(pick(0.0, 1.0, 5, so), meteor=True),
                     "five_score": metrics(pick(0.1, 1.0, 5, so), meteor=True)}
    res["grid_thr"] = args.grid_thr
    args.output.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    se.log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
