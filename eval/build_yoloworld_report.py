"""Build an evaluation report whose objects come from YOLO-World (yolov8l-worldv2).

Images, reference captions and baseline captions are kept. Only "objects" and "yolo_classes" change, so every
evaluation script that reads a report (retrieval_logit_bias_experiment.py, logit_bias_final_eval.py) can use the
YOLO-World evidence directly. YOLO runs per image, threshold 0.25, at most 20 objects. With --split, a new report is
built from dataset_coco.json (no generated captions yet).

Usage: python eval/build_yoloworld_report.py --report results/X.json --output results/X_yoloworld.json
       python eval/build_yoloworld_report.py --split test --output results/test_yoloworld.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from evaluate_reranked_quality import KARPATHY_JSON_FILE, load_karpathy_references  # noqa: E402
from retrieval_logit_bias_experiment import CACHE, IMAGE_DIR  # noqa: E402
from yolo_objects import YoloObjectExtractor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--report", type=Path, help="existing report (captions kept, only the objects change)")
    src.add_argument("--split", choices=["val", "test"], help="build a new report from dataset_coco.json (no generated captions)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--yolo-model", default="yolov8l-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-objects", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=None, help="keep only the first N images (e.g. the 1,000-image val subset)")
    args = ap.parse_args()

    if args.split:
        names, refs = load_karpathy_references(KARPATHY_JSON_FILE, args.split, IMAGE_DIR)
        data = {"split": args.split, "rows": [{"image": n, "references": refs[n], "baseline_candidate": "", "caption": ""}
                                              for n in names]}
    else:
        data = json.loads(args.report.read_text(encoding="utf-8"))
    if args.max_rows:
        data["rows"] = data["rows"][: args.max_rows]
    extractor = YoloObjectExtractor(args.yolo_model, args.conf, args.max_objects)
    data["yolo_classes"] = sorted(extractor.class_names())
    data["yolo_model"] = args.yolo_model
    t0 = time.time()
    for i, row in enumerate(data["rows"]):
        row["objects"] = [o.__dict__ | {"box_xyxy": list(o.box_xyxy) if o.box_xyxy else None}
                          for o in extractor.extract(IMAGE_DIR / row["image"])]
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(data['rows'])} ({(time.time() - t0) / (i + 1):.2f}s/img)", flush=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Retrieval neighbours do not depend on YOLO: reuse the cache of the source report.
    src = CACHE / f"neighbours_{args.report.stem}_k5.npy" if args.report else None
    dst = CACHE / f"neighbours_{args.output.stem}_k5.npy"
    if src and src.exists() and not dst.exists():
        shutil.copy(src, dst)
    print(f"wrote {args.output} ({len(data['rows'])} rows, {len(data['yolo_classes'])} classes)")


if __name__ == "__main__":
    main()
