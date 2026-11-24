from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

KARPATHY_JSON_FILE = DATASET_DIR / "dataset_coco.json"
VAL_IMG_DIR = DATASET_DIR / "val2014"
DEFAULT_INSTANCES_JSON = DATASET_DIR / "instances_val2014.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "results" / "reranked_quality_report.json"

DEFAULT_SPLIT = "test"
DEFAULT_MAX_IMAGES = None
RANDOM_SEED = 40


def find_java8_bin_dir() -> str | None:
    """Find a JRE/JDK 8 on Windows, used only for SPICE.

    spice-1.0.jar (2016) uses Nashorn (a JS engine REMOVED from the JDK since version 15) and hits the internal
    reflection blocked by the JDK 16+ module system. Java 8 has neither problem. The JAVA8_HOME environment
    variable wins if set, otherwise the usual Windows install locations are probed.
    """
    override = os.environ.get("JAVA8_HOME")
    if override and os.path.isfile(os.path.join(override, "bin", "java.exe")):
        return os.path.join(override, "bin")

    import glob

    patterns = [
        r"C:\Program Files\Java\jdk1.8*\bin\java.exe",
        r"C:\Program Files\Java\jre1.8*\bin\java.exe",
        r"C:\Program Files (x86)\Java\jdk1.8*\bin\java.exe",
        r"C:\Program Files (x86)\Java\jre1.8*\bin\java.exe",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])
    return None


def log_step(message: str) -> None:
    print(message, flush=True)


def spice_score_no_cache(gts: dict, res: dict) -> float:
    """SPICE like pycocoevalcap.spice.Spice.compute_score, but WITHOUT `-cache`.

    With `-cache`, spice-1.0.jar opens an LMDB with a hard-coded map size of ~98 GB. On Windows data.mdb
    is not sparse and really takes ~98 GB of disk. Without the cache the scores are identical, captions are just parsed again on each run.
    Uses the `java` on PATH (compute_metrics puts Java 8 first on PATH when it finds one).
    """
    import json
    import tempfile

    import numpy as np
    from pycocoevalcap.spice import spice as spice_mod
    from pycocoevalcap.spice.get_stanford_models import get_stanford_models

    get_stanford_models()
    spice_dir = os.path.dirname(os.path.abspath(spice_mod.__file__))
    ids = sorted(gts.keys())
    data = [{"image_id": i, "test": res[i][0], "refs": gts[i]} for i in ids]
    with tempfile.TemporaryDirectory() as tmp:
        in_file, out_file = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        with open(in_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        # CoreNLP uses every CPU thread by default. SPICE_THREADS limits it (default: a third of the threads, to keep the machine cool).
        threads = os.environ.get("SPICE_THREADS") or str(max(1, (os.cpu_count() or 4) // 3))
        # Windows: run Java at BelowNormal priority so the machine stays usable.
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if os.name == "nt" else 0
        subprocess.check_call(["java", "-jar", "-Xmx8G", spice_mod.SPICE_JAR, in_file, "-out", out_file,
                               "-threads", threads, "-subset", "-silent"], cwd=spice_dir, creationflags=flags)
        with open(out_file, encoding="utf-8") as f:
            results = json.load(f)

    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return float(np.mean([to_float(r["scores"]["All"]["f"]) for r in results]))


def elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.2f}s"


def load_karpathy_references(
    karpathy_json_file: Path,
    split: str,
    image_dir: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    if not karpathy_json_file.is_file():
        raise FileNotFoundError(f"Karpathy JSON not found: {karpathy_json_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"val2014 image folder not found: {image_dir}")

    data = json.loads(karpathy_json_file.read_text(encoding="utf-8"))
    references_by_filename: dict[str, list[str]] = defaultdict(list)

    for image in data.get("images", []):
        if image.get("split") != split:
            continue
        filename = image.get("filename")
        if not filename:
            continue
        if not (image_dir / filename).is_file():
            continue
        for sentence in image.get("sentences", []):
            caption = sentence.get("raw") or " ".join(sentence.get("tokens", []))
            if caption:
                references_by_filename[filename].append(caption)

    filenames = sorted(references_by_filename)
    if not filenames:
        raise ValueError(f"No images/captions for split '{split}'.")
    return filenames, references_by_filename


def load_prediction_records(predictions_file: Path) -> list[dict[str, Any]]:
    if not predictions_file.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    data = json.loads(predictions_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "rows" in data and isinstance(data["rows"], list):
            return data["rows"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Predictions JSON must be an object, a list, or an object with an 'items' key.")


def get_image_filename(record: dict[str, Any]) -> str:
    value = (
        record.get("image")
        or record.get("filename")
        or record.get("image_path")
        or record.get("path")
    )
    if not value:
        raise ValueError(f"Record is missing image/image_path: {record.keys()}")
    return Path(str(value)).name


def get_nested_caption(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "caption" in value:
            return get_nested_caption(value["caption"])
    return None


def get_caption(record: dict[str, Any], field: str) -> str:
    if field in record:
        caption = get_nested_caption(record[field])
        if caption is not None:
            return caption

    if field == "auto":
        for key in [
            "caption",  # evaluate_reranked_quality.py's own row shape (--run-pipeline/--rerank-only)
            "best_caption",
            "llm_selected_caption",
            "selected_caption",
            "reranked",
            "reranked_caption",
            "final_caption",
            "generated",
        ]:
            if key in record:
                caption = get_nested_caption(record[key])
                if caption:
                    return caption
        ranking = record.get("ranking")
        if isinstance(ranking, list) and ranking:
            caption = get_nested_caption(ranking[0])
            if caption:
                return caption

    raise ValueError(f"Cannot read a caption from field '{field}' of the record.")


def get_first_candidate_caption(record: dict[str, Any]) -> str | None:
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    return get_nested_caption(candidates[0])


def simple_standardize(text: str) -> str:
    """Same normalization as the baseline eval script (mscoco.txt evaluation): lowercase + strip
    punctuation/digits via regex, no real tokenizer. Kept byte-for-byte identical to that script
    so --tokenizer simple gives numbers directly comparable to baseline-reported ones."""
    text = str(text).lower()
    strip_chars = r"""!"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890"""
    text = re.sub(f"[{re.escape(strip_chars)}]", "", text)
    return " ".join(text.split())


def compute_metrics(
    predictions: list[str],
    references: list[list[str]],
    use_meteor: bool = True,
    use_cider: bool = True,
    use_spice: bool = True,
    tokenizer: str = "ptb",
) -> dict[str, float]:
    """BLEU/METEOR/ROUGE-L/CIDEr/SPICE via the official pycocoevalcap scorers either way -- the
    two `tokenizer` modes only change what text those scorers see:
    - "ptb" (default): real Stanford PTBTokenizer, the standard COCO caption eval convention.
      Needs a JRE (`java`) on PATH -- METEOR needs it too regardless of this setting.
    - "simple": lowercase + strip punctuation/digits (see simple_standardize), matching the
      baseline eval script (mscoco.txt evaluation) exactly -- use this to compare fairly against
      numbers that script reported, since "ptb" tokenization alone shifts scores.
    ROUGE-L/CIDEr are pure Python/numpy either way and never need Java."""
    if not predictions:
        raise ValueError("No predictions to evaluate.")

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise ImportError("pycocoevalcap is missing. Install it with: pip install pycocoevalcap") from exc

    start = time.perf_counter()
    if tokenizer == "simple":
        log_step("[*] Tokenizing (simple, as the baseline: lowercase, punctuation and digits removed)...")
        gts = {i: [simple_standardize(ref) for ref in refs] for i, refs in enumerate(references)}
        res = {i: [simple_standardize(pred)] for i, pred in enumerate(predictions)}
    elif tokenizer == "ptb":
        log_step("[*] Tokenizing (Stanford PTBTokenizer, needs Java)...")
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        gts_raw = {i: [{"caption": ref} for ref in refs] for i, refs in enumerate(references)}
        res_raw = {i: [{"caption": pred}] for i, pred in enumerate(predictions)}
        try:
            ptb = PTBTokenizer()
            gts = ptb.tokenize(gts_raw)
            res = ptb.tokenize(res_raw)
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Cannot run PTBTokenizer -- pycocoevalcap needs 'java' (JRE) on PATH."
            ) from exc
    else:
        raise ValueError("--tokenizer accepts only 'ptb' or 'simple'.")
    log_step(f"[*] Tokenize done after {elapsed(start)}")

    metrics: dict[str, float] = {}

    start = time.perf_counter()
    log_step("[*] Computing BLEU-1..4 (pycocoevalcap)...")
    bleu_scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for n, score in enumerate(bleu_scores, start=1):
        metrics[f"BLEU-{n}"] = score * 100
    log_step(f"[*] BLEU done after {elapsed(start)}")

    if use_meteor:
        start = time.perf_counter()
        log_step("[*] Computing METEOR (pycocoevalcap, needs Java)...")
        from pycocoevalcap.meteor.meteor import Meteor

        meteor_value, _ = Meteor().compute_score(gts, res)
        metrics["METEOR"] = meteor_value * 100
        log_step(f"[*] METEOR done after {elapsed(start)}")
    else:
        log_step("[*] Skipping METEOR.")

    start = time.perf_counter()
    log_step("[*] Computing ROUGE-L (pycocoevalcap)...")
    rouge_value, _ = Rouge().compute_score(gts, res)
    metrics["ROUGE-L"] = rouge_value * 100
    log_step(f"[*] ROUGE-L done after {elapsed(start)}")

    if use_cider:
        start = time.perf_counter()
        log_step("[*] Computing CIDEr (pycocoevalcap)...")
        cider_value, _ = Cider().compute_score(gts, res)
        metrics["CIDEr"] = cider_value
        log_step(f"[*] CIDEr done after {elapsed(start)}")

    if use_spice:
        start = time.perf_counter()
        log_step("[*] Computing SPICE (pycocoevalcap, needs Java -- the first run can be slow)...")
        # spice-1.0.jar uses both Nashorn (a JS engine REMOVED from the JDK since version 15) and internal
        # java.base reflection through the FST library (blocked by the module system since JDK 16). Java 8 has
        # neither problem, so it is preferred WHEN installed. Only PATH is changed (JAVA_TOOL_OPTIONS
        # --add-opens is NOT set at the same time, because Java 8 does not know that flag and the JVM
        # would fail at start-up with "Unrecognized option"). This applies only to the SPICE subprocess,
        # PATH is restored right after, so METEOR/PTBTokenizer keep running on the current JDK.
        original_path = os.environ.get("PATH", "")
        java8_bin = find_java8_bin_dir()
        if java8_bin:
            os.environ["PATH"] = java8_bin + os.pathsep + original_path
            log_step(f"[*] Using Java 8 for SPICE (found at {java8_bin}) -- needed for Nashorn.")
        else:
            # No Java 8: fall back to --add-opens to at least get past the JPMS/FST error on JDK 16+
            # (a separate Nashorn error can still occur, see the error message if it fails).
            os.environ["JAVA_TOOL_OPTIONS"] = " ".join(
                [
                    os.environ.get("JAVA_TOOL_OPTIONS", ""),
                    "--add-opens=java.base/java.lang=ALL-UNNAMED",
                    "--add-opens=java.base/java.util=ALL-UNNAMED",
                    "--add-opens=java.base/java.math=ALL-UNNAMED",
                    "--add-opens=java.base/java.text=ALL-UNNAMED",
                    "--add-opens=java.base/java.time=ALL-UNNAMED",
                    "--add-opens=java.base/java.io=ALL-UNNAMED",
                    "--add-opens=java.base/java.net=ALL-UNNAMED",
                    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
                    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
                    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
                ]
            ).strip()
        try:
            spice_value = spice_score_no_cache(gts, res)
            metrics["SPICE"] = spice_value * 100
            log_step(f"[*] SPICE done after {elapsed(start)}")
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Cannot run SPICE -- pycocoevalcap needs 'java' (JRE) on PATH, like METEOR."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "SPICE (spice-1.0.jar) failed (see stderr above). Three causes seen "
                "in practice on Windows: (1) a recent JDK (16+) blocks the reflection of the FST library "
                "inside the jar (InaccessibleObjectException) -- --add-opens was added through "
                "JAVA_TOOL_OPTIONS above, (2) [handled] the ~98 GB LMDB cache: spice_score_no_cache "
                "does not pass -cache, so no data.mdb is created, (3) "
                "spice-1.0.jar uses Nashorn (a JS engine) that was REMOVED from the JDK in version 15 -- causing "
                "NullPointerException 'scriptEngine is null'. Fix for (3): install a JRE/JDK 8 "
                "and point JAVA8_HOME to its root folder (the script finds it and uses it ONLY for "
                "SPICE). Without Java 8, pass --no-spice to skip this metric on this machine."
            ) from exc
        finally:
            os.environ["PATH"] = original_path

    return metrics


def build_eval_rows_from_predictions(
    records: list[dict[str, Any]],
    references_by_filename: dict[str, list[str]],
    caption_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        filename = get_image_filename(record)
        refs = references_by_filename.get(filename)
        if not refs:
            skipped += 1
            continue
        caption = get_caption(record, caption_field)
        rows.append(
            {
                "image": filename,
                "caption": caption,
                "baseline_candidate": get_first_candidate_caption(record),
                "references": refs,
            }
        )
    if skipped:
        log_step(f"[!] Skipped {skipped} records with no reference for their filename.")
    return rows


def run_demo_pipeline(
    filenames: list[str],
    references_by_filename: dict[str, list[str]],
    image_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from config import ClipScoreConfig, GeneratorConfig, PipelineConfig, YoloConfig
    from pipeline import CaptionPipeline

    config = PipelineConfig(
        yolo=YoloConfig(
            model_name=args.yolo_model,
            confidence_threshold=args.yolo_conf,
            max_objects=args.max_objects,
        ),
        clipscore=ClipScoreConfig(model_name=args.clip_model, device=args.device),
        generator=GeneratorConfig(
            checkpoint_path=Path(args.caption_checkpoint) if args.caption_checkpoint else None,
            vocab_path=Path(args.vocab_file) if args.vocab_file else None,
            max_candidates=args.max_candidates,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            max_new_tokens=args.max_new_tokens,
            safe_beam_size=args.safe_beam_size,
        ),
    )
    pipeline = CaptionPipeline(config)
    # Real class names from the loaded YOLO checkpoint (set as a side effect of constructing
    # CaptionPipeline, see pipeline.py). Saved into the report so --rerank-only can rebuild the
    # same object-noun vocabulary later without loading YOLO again.
    yolo_classes = sorted(pipeline.object_extractor.class_names())
    rows: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        rows = existing.get("rows", []) if isinstance(existing, dict) else []
        log_step(f"[*] Resume: {len(rows)} rows already in {args.output}")

    done = {row.get("image") for row in rows}
    pending = [filename for filename in filenames if filename not in done]

    for index, filename in enumerate(tqdm(pending, desc="Running demo pipeline"), start=1):
        image_path = image_dir / filename
        result = pipeline.run(image_path)
        baseline = result.candidates[0].text if result.candidates else None
        rows.append(
            {
                "image": filename,
                "caption": result.best_caption,
                "baseline_candidate": baseline,
                "references": references_by_filename[filename],
                "objects": [obj.__dict__ for obj in result.objects],
                "candidates": [candidate.__dict__ for candidate in result.candidates],
                "ranking": [
                    {
                        "caption": item.caption.text,
                        "rank": item.rank,
                        "rerank_score": item.rerank_score,
                        "reason": item.reason,
                    }
                    for item in result.ranked
                ],
            }
        )
        if args.save_every > 0 and index % args.save_every == 0:
            save_partial_report(args.output, rows, args, yolo_classes=yolo_classes)
    return rows, yolo_classes


def load_report_rows(report_file: Path) -> list[dict[str, Any]]:
    rows, _ = load_report(report_file)
    return rows


def load_report(report_file: Path) -> tuple[list[dict[str, Any]], list[str] | None]:
    """Returns (rows, yolo_classes). yolo_classes is the real YOLO class list saved by
    run_demo_pipeline (see vocab.set_object_noun_vocab) -- None for older reports that predate
    this field, or for a plain list-of-rows file."""
    if not report_file.is_file():
        raise FileNotFoundError(f"Candidates file not found: {report_file}")
    data = json.loads(report_file.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        classes = data.get("yolo_classes")
        return data["rows"], (classes if isinstance(classes, list) else None)
    if isinstance(data, list):
        return data, None
    raise ValueError("Candidates file must be an object with a 'rows' key or a list of rows.")


def run_rerank_only_pipeline(
    candidates_file: Path,
    wanted_filenames: set[str],
    references_by_filename: dict[str, list[str]],
    image_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str] | None]:
    """Rerank captions from a report JSON already produced by --run-pipeline, reusing its saved
    'objects'/'candidates' (with clip_score already computed) instead of re-running YOLO/caption
    generation/CLIP. Only loads the lightweight formula reranker, so this is
    dramatically faster than --run-pipeline and needs no GPU -- useful for trying different
    FORMULA_* weights without paying for the full pipeline again. --concurrency > 1 runs
    images in a thread pool."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import vocab
    from formula_reranker import FormulaReranker
    from schema import CaptionCandidate, DetectedObject

    report_rows, yolo_classes = load_report(candidates_file)
    if yolo_classes:
        # Rebuild the exact same object-noun vocabulary the original --run-pipeline used
        # (S_det/hallucination flagging), without loading YOLO here at all.
        vocab.set_object_noun_vocab(set(yolo_classes))
    else:
        log_step(
            "[!] Candidates file has no 'yolo_classes' (old report). "
            "Using the default object-noun vocabulary (some COCO classes may be missing)."
        )
    cached_rows = {row["image"]: row for row in report_rows if row.get("image")}

    rows: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        rows = existing.get("rows", []) if isinstance(existing, dict) else []
        log_step(f"[*] Resume: {len(rows)} rows already in {args.output}")

    done = {row.get("image") for row in rows}
    pending = [filename for filename in wanted_filenames if filename not in done]
    missing = [filename for filename in pending if filename not in cached_rows]
    if missing:
        log_step(f"[!] Skipped {len(missing)} images that are not in the candidates file (not run with --run-pipeline yet).")
    pending = [filename for filename in pending if filename in cached_rows]

    def rerank_one(filename: str) -> dict[str, Any]:
        cached = cached_rows[filename]
        candidates = [
            CaptionCandidate(
                text=c["text"],
                source=c.get("source", "cached"),
                generation_score=c.get("generation_score"),
                clip_score=c.get("clip_score"),
                metadata=c.get("metadata") or {},
            )
            for c in cached.get("candidates", [])
        ]
        objects = [
            DetectedObject(
                label=o["label"],
                confidence=o.get("confidence", 0.0),
                box_xyxy=tuple(o["box_xyxy"]) if o.get("box_xyxy") else None,
            )
            for o in cached.get("objects", [])
        ]
        ranked = FormulaReranker().rerank(candidates, objects)
        best_caption = ranked[0].caption.text if ranked else (candidates[0].text if candidates else "")
        return {
            "image": filename,
            "caption": best_caption,
            "baseline_candidate": cached.get("baseline_candidate") or (candidates[0].text if candidates else None),
            "references": references_by_filename.get(filename, cached.get("references", [])),
            "objects": cached.get("objects", []),
            "candidates": cached.get("candidates", []),
            "ranking": [
                {
                    "caption": item.caption.text,
                    "rank": item.rank,
                    "rerank_score": item.rerank_score,
                    "reason": item.reason,
                }
                for item in ranked
            ],
        }

    concurrency = max(1, args.concurrency)
    if concurrency == 1:
        for index, filename in enumerate(tqdm(pending, desc="Reranking from cached candidates"), start=1):
            rows.append(rerank_one(filename))
            if args.save_every > 0 and index % args.save_every == 0:
                save_partial_report(args.output, rows, args, yolo_classes=yolo_classes)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(rerank_one, filename): filename for filename in pending}
            progress = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Reranking from cached candidates ({concurrency}x)",
            )
            for index, future in enumerate(progress, start=1):
                filename = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    log_step(f"[!] Rerank error for image {filename}: {exc}")
                if args.save_every > 0 and index % args.save_every == 0:
                    save_partial_report(args.output, rows, args, yolo_classes=yolo_classes)
    return rows, yolo_classes


def save_partial_report(
    output_file: Path, rows: list[dict[str, Any]], args: argparse.Namespace, yolo_classes: list[str] | None = None
) -> None:
    payload = {
        "partial": True,
        "split": args.split,
        "count": len(rows),
        "caption_field": "best_caption",
        "use_meteor": not args.no_meteor,
        "yolo_classes": yolo_classes,
        "rows": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_baseline(
    rows: list[dict[str, Any]],
    use_meteor: bool,
    use_cider: bool,
    use_spice: bool,
    tokenizer: str = "ptb",
) -> dict[str, float] | None:
    baseline_rows = [row for row in rows if row.get("baseline_candidate")]
    if not baseline_rows:
        return None
    log_step("\n=== EVALUATING FIRST-CANDIDATE CAPTIONS (BEFORE RERANK) ===")
    return compute_metrics(
        [row["baseline_candidate"] for row in baseline_rows],
        [row["references"] for row in baseline_rows],
        use_meteor=use_meteor,
        use_cider=use_cider,
        use_spice=use_spice,
        tokenizer=tokenizer,
    )


def build_chair_ground_truth(
    rows: list[dict[str, Any]],
    instances_json: Path | None,
    karpathy_json: Path,
) -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """Ground-truth MSCOCO objects per image filename for CHAIRs/CHAIRi.

    Uses the row's human reference captions for every image, then merges official COCO instance
    annotations when `instances_json` is available. This mirrors tune_formula_weights.py so the
    final eval report and the weight-search report measure hallucination the same way.
    """
    import chair_metric

    mscoco_objects, inverse_synonym_dict = chair_metric.load_synonyms()
    gt_by_filename: dict[str, set[str]] = {}
    for row in rows:
        filename = row.get("image")
        references = row.get("references")
        if not filename or not references:
            continue
        gt_by_filename[filename] = chair_metric.ground_truth_from_captions(
            references, mscoco_objects, inverse_synonym_dict
        )

    if instances_json and instances_json.is_file():
        filename_to_cocoid = chair_metric.load_filename_to_cocoid(karpathy_json)
        instance_objects_by_cocoid = chair_metric.load_instance_objects_by_cocoid(
            instances_json, inverse_synonym_dict
        )
        merged = 0
        for filename in gt_by_filename:
            cocoid = filename_to_cocoid.get(filename)
            if cocoid is not None and cocoid in instance_objects_by_cocoid:
                gt_by_filename[filename] |= instance_objects_by_cocoid[cocoid]
                merged += 1
        log_step(f"[*] CHAIR ground truth: reference captions + instance segmentation for {merged}/{len(gt_by_filename)} images.")
    else:
        log_step(
            "[!] No valid instances_val2014.json -- CHAIR ground truth uses only the 5 "
            "human reference captions (weaker than instance segmentation)."
        )
    return gt_by_filename, mscoco_objects, inverse_synonym_dict


def compute_chair_metrics(
    rows: list[dict[str, Any]],
    caption_field: str,
    gt_by_filename: dict[str, set[str]],
    mscoco_objects: set[str],
    inverse_synonym_dict: dict[str, str],
) -> dict[str, float]:
    import chair_metric

    predictions = {
        row["image"]: row[caption_field]
        for row in rows
        if row.get("image") and row.get(caption_field)
    }
    return chair_metric.compute_chair(predictions, gt_by_filename, mscoco_objects, inverse_synonym_dict)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for name, value in metrics.items():
        suffix = "" if name in {"CIDEr", "num_captions", "skipped_no_ground_truth"} else "%"
        print(f"{name:8s}: {value:.4f}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate captions after formula reranking.")
    parser.add_argument("--predictions", type=Path, help="JSON output from demo/run_pipeline.py or a batch JSON/list.")
    parser.add_argument("--run-pipeline", action="store_true", help="Run demo pipeline first, then evaluate reranked captions.")
    parser.add_argument(
        "--rerank-only",
        action="store_true",
        help="Rerank from a report JSON already produced by --run-pipeline (reuses its saved "
        "objects/candidates instead of re-running YOLO/caption generation/CLIP). Needs --candidates-file.",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        help="Report JSON from a previous --run-pipeline run, used as input for --rerank-only.",
    )
    parser.add_argument("--caption-field", default="auto", help="Caption field to evaluate. Default auto detects best_caption/reranked/generated.")
    parser.add_argument("--karpathy-json", type=Path, default=KARPATHY_JSON_FILE)
    parser.add_argument("--instances-json", type=Path, default=DEFAULT_INSTANCES_JSON)
    parser.add_argument("--image-dir", type=Path, default=VAL_IMG_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--sample", action="store_true", help="Random sample max-images instead of first max-images.")
    parser.add_argument(
        "--no-meteor",
        action="store_true",
        help="Skip METEOR (pycocoevalcap.meteor -- needs Java/JRE on PATH).",
    )
    parser.add_argument("--no-cider", action="store_true")
    parser.add_argument(
        "--no-spice",
        action="store_true",
        help="Skip SPICE (pycocoevalcap.spice -- needs Java/JRE on PATH, slower than the other metrics).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["ptb", "simple"],
        default="simple",
        help="'ptb' = Stanford PTBTokenizer, the standard COCO evaluation. 'simple' (default) = lowercase "
        "+ punctuation and digits removed, exactly as the baseline script (mscoco.txt evaluation) -- for a fair "
        "comparison with the reported baseline numbers.",
    )
    parser.add_argument("--compare-first-candidate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--resume", action="store_true", help="Continue from the partially saved report JSON.")
    parser.add_argument("--save-every", type=int, default=10, help="Save progress every N images with --run-pipeline.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of images reranked in parallel (only for --rerank-only).",
    )

    parser.add_argument("--yolo-model", default="yolov8l-worldv2.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--caption-checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "seed42_dynlogits_contrastive_attnpool_epoch40_model.pth"))
    parser.add_argument("--vocab-file", default=str(DATASET_DIR / "coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json"))
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--safe-beam-size",
        type=int,
        default=3,
        help="Add one 'safe' candidate from plain (non-diverse) beam search with this beam width "
        "-- always present, so the reranker is never worse than plain beam search. 0 = off, only "
        "the diverse candidates (old behavior).",
    )
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-beam-groups", type=int, default=5)
    parser.add_argument("--diversity-penalty", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    filenames, references_by_filename = load_karpathy_references(
        args.karpathy_json,
        split=args.split,
        image_dir=args.image_dir,
    )
    if args.max_images is not None:
        filenames = random.sample(filenames, min(args.max_images, len(filenames))) if args.sample else filenames[: args.max_images]

    yolo_classes: list[str] | None = None
    if args.run_pipeline:
        rows, yolo_classes = run_demo_pipeline(filenames, references_by_filename, args.image_dir, args)
    elif args.rerank_only:
        if not args.candidates_file:
            raise SystemExit("Pass --candidates-file report.json (from --run-pipeline) with --rerank-only.")
        rows, yolo_classes = run_rerank_only_pipeline(
            args.candidates_file, set(filenames), references_by_filename, args.image_dir, args
        )
    elif args.predictions:
        records = load_prediction_records(args.predictions)
        rows = build_eval_rows_from_predictions(records, references_by_filename, args.caption_field)
        if args.max_images is not None:
            rows = rows[: args.max_images]
    else:
        raise SystemExit("Pass --predictions file.json, --run-pipeline, or --rerank-only --candidates-file file.json.")

    if not rows:
        raise SystemExit("No rows to evaluate.")

    log_step(f"[*] Reranked captions to evaluate: {len(rows)}")
    reranked_metrics = compute_metrics(
        [row["caption"] for row in rows],
        [row["references"] for row in rows],
        use_meteor=not args.no_meteor,
        use_cider=not args.no_cider,
        use_spice=not args.no_spice,
        tokenizer=args.tokenizer,
    )
    gt_by_filename, mscoco_objects, inverse_synonym_dict = build_chair_ground_truth(
        rows,
        args.instances_json,
        args.karpathy_json,
    )
    reranked_metrics.update(
        compute_chair_metrics(rows, "caption", gt_by_filename, mscoco_objects, inverse_synonym_dict)
    )

    baseline_metrics = None
    if args.compare_first_candidate:
        baseline_metrics = compare_baseline(
            rows, not args.no_meteor, not args.no_cider, not args.no_spice, args.tokenizer
        )
        if baseline_metrics is not None:
            baseline_rows = [row for row in rows if row.get("baseline_candidate")]
            baseline_metrics.update(
                compute_chair_metrics(
                    baseline_rows,
                    "baseline_candidate",
                    gt_by_filename,
                    mscoco_objects,
                    inverse_synonym_dict,
                )
            )

    print_metrics("CAPTION RESULTS AFTER RERANKING", reranked_metrics)
    if baseline_metrics is not None:
        print_metrics("FIRST-CANDIDATE RESULTS BEFORE RERANK", baseline_metrics)

    payload = {
        "split": args.split,
        "count": len(rows),
        "caption_field": args.caption_field,
        "use_meteor": not args.no_meteor,
        "use_cider": not args.no_cider,
        "use_spice": not args.no_spice,
        "tokenizer": args.tokenizer,
        "instances_json": str(args.instances_json) if args.instances_json else None,
        "metrics_after_rerank": reranked_metrics,
        "metrics_first_candidate": baseline_metrics,
        "yolo_classes": yolo_classes,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"[*] Saved report: {args.output}")


if __name__ == "__main__":
    main()
