"""Build the retrieval cache for a report: CLIP ViT-B/32 embeddings of the train + restval captions (once) and the
top-5 caption neighbours of every image in the report (neighbours_<report stem>_k5.npy in dataset/retrieval_cache/).

    python eval/build_retrieval_cache.py --report results/test_yoloworld.json
"""
import argparse
import json
from pathlib import Path

import sretr_experiment as se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.report.read_text(encoding="utf-8"))["rows"]
    clip = se.Clip()
    _, emb = se.corpus(clip)
    idx = se.neighbours(clip, rows, emb, args.report.stem, 5)
    assert idx.shape == (len(rows), 5), idx.shape
    se.log(f"{se.CACHE_DIR / f'neighbours_{args.report.stem}_k5.npy'}: {idx.shape}")


if __name__ == "__main__":
    main()
