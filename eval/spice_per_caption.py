"""SPICE of every candidate caption, running spice-1.0.jar once for the whole report.

The SPICE of a configuration is the mean per-image "All" F-score, and each image is scored independently, so it is enough
to score each (image, caption) pair once and look up the selected caption. Captions and references are normalised like the "simple" tokenizer of
evaluate_reranked_quality.compute_metrics, so the values match compute_metrics(use_spice=True).

Usage (from the repo root):
    python eval/spice_per_caption.py --report results/e35_k6_yoloworld_merged.json \
        --output results/spice_per_caption_e35_k6_test.json
    python eval/spice_per_caption.py --report results/val1000_yoloworld.json \
        --preds results/ksweep_e35_val1000_*.json --output results/spice_per_caption_ksweep_val1000.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from evaluate_reranked_quality import find_java8_bin_dir, simple_standardize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--preds", nargs="*", default=[], help="k-sweep files with runs[].predictions (first/best)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=2500, help="pairs per SPICE call, a checkpoint is saved after each chunk")
    args = ap.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data["rows"] if "rows" in data else [r for r in data["runs"] if r["config"] != "report"][0]["rows"]
    texts = {r["image"]: {r["baseline_candidate"], r["caption"], *(c["text"] for c in r["candidates"])} for r in rows}
    for pat in args.preds:
        for f in sorted(glob.glob(pat)):
            for run in json.loads(Path(f).read_text(encoding="utf-8"))["runs"]:
                for p in run.get("predictions", []):
                    texts[p["image"]].update((p["first"], p["best"]))

    entries, keys = [], []
    for r in rows:
        refs = [simple_standardize(x) for x in r["references"]]
        for t in sorted({simple_standardize(x) for x in texts[r["image"]]}):
            entries.append({"image_id": len(keys), "test": t, "refs": refs})
            keys.append((r["image"], t))
    print(f"{len(rows)} images, {len(entries)} (image, caption) pairs", flush=True)

    from pycocoevalcap.spice import spice as spice_mod
    from pycocoevalcap.spice.get_stanford_models import get_stanford_models

    get_stanford_models()
    java8 = find_java8_bin_dir()
    # Windows resolves executables with the parent process PATH, so call the Java 8 java.exe directly (needed for Nashorn).
    java = os.path.join(java8, "java.exe") if java8 else "java"
    threads = os.environ.get("SPICE_THREADS") or str(max(1, (os.cpu_count() or 4) // 3))
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if os.name == "nt" else 0

    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    # Checkpoint: score in chunks, write each finished chunk to a .partial file, and skip scored pairs on a rerun.
    partial = args.output.with_name(args.output.name + ".partial")
    out: dict[str, dict[str, float]] = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else {}
    todo = [k for k, (img, t) in enumerate(keys) if t not in out.get(img, {})]
    if len(todo) < len(keys):
        print(f"checkpoint: {len(keys) - len(todo)} pairs done, {len(todo)} left", flush=True)
    start = time.perf_counter()
    for c in range(0, len(todo), args.chunk):
        part = [entries[k] for k in todo[c: c + args.chunk]]
        with tempfile.TemporaryDirectory() as tmp:
            in_file, out_file = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
            Path(in_file).write_text(json.dumps(part), encoding="utf-8")
            subprocess.check_call([java, "-jar", "-Xmx8G", spice_mod.SPICE_JAR, in_file, "-out", out_file,
                                   "-threads", threads, "-subset", "-silent"],
                                  cwd=os.path.dirname(os.path.abspath(spice_mod.__file__)), creationflags=flags)
            results = json.loads(Path(out_file).read_text(encoding="utf-8"))
        for res in results:
            img, t = keys[int(res["image_id"])]
            out.setdefault(img, {})[t] = to_float(res["scores"]["All"]["f"])
        partial.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        done = min(c + args.chunk, len(todo))
        el = time.perf_counter() - start
        print(f"{done}/{len(todo)} pairs | {el / 60:.1f} min | about {el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
    assert all(t in out.get(img, {}) for img, t in keys), "some pairs have no SPICE score"
    args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"SPICE done after {time.perf_counter() - start:.0f}s, wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
