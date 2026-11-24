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
    """Tim JRE/JDK 8 tren Windows danh rieng cho SPICE.

    spice-1.0.jar (2016) dung Nashorn (JS engine bi GO KHOI JDK tu ban 15 tro di) va cham reflection
    noi bo cua JDK tu ban 16 (module system). Java 8 khong co ca 2 van de nay. Uu tien bien moi
    truong JAVA8_HOME neu nguoi dung tu khai bao, sau do do cac vi tri cai Windows pho bien.
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
    """SPICE giống pycocoevalcap.spice.Spice.compute_score nhưng KHÔNG truyền `-cache`.

    Với `-cache`, spice-1.0.jar mở một LMDB có map size hardcode ~98 GB; trên Windows file data.mdb
    không sparse nên chiếm thật ~98 GB đĩa. Bỏ cache thì điểm y hệt, chỉ phải parse lại câu mỗi lần chạy.
    Dùng `java` đang có trong PATH (compute_metrics đặt Java 8 lên đầu PATH nếu tìm thấy).
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
        # CoreNLP mặc định dùng mọi luồng CPU; SPICE_THREADS giới hạn lại (mặc định 1/3 số luồng để máy đỡ nóng).
        threads = os.environ.get("SPICE_THREADS") or str(max(1, (os.cpu_count() or 4) // 3))
        # Windows: chạy Java ở độ ưu tiên BelowNormal để nhường CPU khi người dùng đang dùng máy.
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
        raise FileNotFoundError(f"Khong tim thay Karpathy JSON: {karpathy_json_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay folder anh val2014: {image_dir}")

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
        raise ValueError(f"Khong co anh/caption nao cho split '{split}'.")
    return filenames, references_by_filename


def load_prediction_records(predictions_file: Path) -> list[dict[str, Any]]:
    if not predictions_file.is_file():
        raise FileNotFoundError(f"Khong tim thay predictions file: {predictions_file}")
    data = json.loads(predictions_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "rows" in data and isinstance(data["rows"], list):
            return data["rows"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Predictions JSON phai la object, list, hoac object co key 'items'.")


def get_image_filename(record: dict[str, Any]) -> str:
    value = (
        record.get("image")
        or record.get("filename")
        or record.get("image_path")
        or record.get("path")
    )
    if not value:
        raise ValueError(f"Record thieu image/image_path: {record.keys()}")
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

    raise ValueError(f"Khong lay duoc caption voi field '{field}' trong record.")


def get_first_candidate_caption(record: dict[str, Any]) -> str | None:
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    return get_nested_caption(candidates[0])


def simple_standardize(text: str) -> str:
    """Same normalization as the baseline eval script (danh gia mscoco.txt): lowercase + strip
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
      baseline eval script (danh gia mscoco.txt) exactly -- use this to compare fairly against
      numbers that script reported, since "ptb" tokenization alone shifts scores.
    ROUGE-L/CIDEr are pure Python/numpy either way and never need Java."""
    if not predictions:
        raise ValueError("Khong co prediction nao de danh gia.")

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise ImportError("Thieu pycocoevalcap. Cai bang: pip install pycocoevalcap") from exc

    start = time.perf_counter()
    if tokenizer == "simple":
        log_step("[*] Dang tokenize (don gian, giong baseline: lowercase + bo dau cau/so)...")
        gts = {i: [simple_standardize(ref) for ref in refs] for i, refs in enumerate(references)}
        res = {i: [simple_standardize(pred)] for i, pred in enumerate(predictions)}
    elif tokenizer == "ptb":
        log_step("[*] Dang tokenize (Stanford PTBTokenizer, can Java)...")
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        gts_raw = {i: [{"caption": ref} for ref in refs] for i, refs in enumerate(references)}
        res_raw = {i: [{"caption": pred}] for i, pred in enumerate(predictions)}
        try:
            ptb = PTBTokenizer()
            gts = ptb.tokenize(gts_raw)
            res = ptb.tokenize(res_raw)
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Khong chay duoc PTBTokenizer -- pycocoevalcap can 'java' (JRE) trong PATH."
            ) from exc
    else:
        raise ValueError("--tokenizer chi nhan 'ptb' hoac 'simple'.")
    log_step(f"[*] Xong tokenize sau {elapsed(start)}")

    metrics: dict[str, float] = {}

    start = time.perf_counter()
    log_step("[*] Dang tinh BLEU-1..4 (pycocoevalcap)...")
    bleu_scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for n, score in enumerate(bleu_scores, start=1):
        metrics[f"BLEU-{n}"] = score * 100
    log_step(f"[*] Xong BLEU sau {elapsed(start)}")

    if use_meteor:
        start = time.perf_counter()
        log_step("[*] Dang tinh METEOR (pycocoevalcap, can Java)...")
        from pycocoevalcap.meteor.meteor import Meteor

        meteor_value, _ = Meteor().compute_score(gts, res)
        metrics["METEOR"] = meteor_value * 100
        log_step(f"[*] Xong METEOR sau {elapsed(start)}")
    else:
        log_step("[*] Bo qua METEOR.")

    start = time.perf_counter()
    log_step("[*] Dang tinh ROUGE-L (pycocoevalcap)...")
    rouge_value, _ = Rouge().compute_score(gts, res)
    metrics["ROUGE-L"] = rouge_value * 100
    log_step(f"[*] Xong ROUGE-L sau {elapsed(start)}")

    if use_cider:
        start = time.perf_counter()
        log_step("[*] Dang tinh CIDEr (pycocoevalcap)...")
        cider_value, _ = Cider().compute_score(gts, res)
        metrics["CIDEr"] = cider_value
        log_step(f"[*] Xong CIDEr sau {elapsed(start)}")

    if use_spice:
        start = time.perf_counter()
        log_step("[*] Dang tinh SPICE (pycocoevalcap, can Java -- co the cham o lan chay dau)...")
        # spice-1.0.jar dung ca Nashorn (JS engine, bi GO KHOI JDK tu ban 15) lan reflection noi
        # bo java.base qua thu vien FST (chan boi module system tu JDK 16). Java 8 khong dinh ca
        # 2 van de nay nen la lua chon uu tien NEU may co san. Chi doi PATH (khong dong thoi set
        # JAVA_TOOL_OPTIONS --add-opens, vi Java 8 KHONG hieu flag do va se crash ngay luc khoi
        # dong JVM voi "Unrecognized option"). Chi ap dung rieng cho subprocess cua Spice, khoi
        # phuc PATH ngay sau, khong anh huong METEOR/PTBTokenizer da chay on dinh voi JDK hien tai.
        original_path = os.environ.get("PATH", "")
        java8_bin = find_java8_bin_dir()
        if java8_bin:
            os.environ["PATH"] = java8_bin + os.pathsep + original_path
            log_step(f"[*] Dung Java 8 cho SPICE (tim thay tai {java8_bin}) -- can cho Nashorn.")
        else:
            # Khong co Java 8: fallback --add-opens de it nhat qua duoc loi JPMS/FST tren JDK 16+
            # (van con nguy co loi Nashorn rieng, xem thong bao loi neu that bai).
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
            log_step(f"[*] Xong SPICE sau {elapsed(start)}")
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Khong chay duoc SPICE -- pycocoevalcap can 'java' (JRE) trong PATH, giong METEOR."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "SPICE (spice-1.0.jar) chay ra loi (xem stderr o tren). Ba nguyen nhan da gap "
                "thuc te tren Windows: (1) JDK qua moi (16+) chan reflection cua thu vien FST ben "
                "trong jar (InaccessibleObjectException) -- da tu them --add-opens qua "
                "JAVA_TOOL_OPTIONS o tren; (2) [da xu ly] cache LMDB ~98GB: spice_score_no_cache "
                "khong truyen -cache nen khong tao data.mdb; (3) "
                "spice-1.0.jar dung Nashorn (JS engine), da bi GO KHOI JDK tu ban 15 -- gay "
                "NullPointerException 'scriptEngine is null'. Cach sua (3): cai/dung mot JRE/JDK 8 "
                "va tro JAVA8_HOME toi thu muc goc cua no (script se tu do va uu tien no CHI cho "
                "SPICE); neu khong co Java 8, dung --no-spice de bo qua metric nay tren may nay."
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
        log_step(f"[!] Bo qua {skipped} record vi khong tim thay reference theo filename.")
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
        log_step(f"[*] Resume: da co {len(rows)} rows trong {args.output}")

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
        raise FileNotFoundError(f"Khong tim thay candidates file: {report_file}")
    data = json.loads(report_file.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        classes = data.get("yolo_classes")
        return data["rows"], (classes if isinstance(classes, list) else None)
    if isinstance(data, list):
        return data, None
    raise ValueError("Candidates file phai la object co key 'rows' hoac mot list cac row.")


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
            "[!] Candidates file khong co 'yolo_classes' (report cu). "
            "Dung tu dien object-noun mac dinh (co the thieu vai lop COCO)."
        )
    cached_rows = {row["image"]: row for row in report_rows if row.get("image")}

    rows: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        rows = existing.get("rows", []) if isinstance(existing, dict) else []
        log_step(f"[*] Resume: da co {len(rows)} rows trong {args.output}")

    done = {row.get("image") for row in rows}
    pending = [filename for filename in wanted_filenames if filename not in done]
    missing = [filename for filename in pending if filename not in cached_rows]
    if missing:
        log_step(f"[!] Bo qua {len(missing)} anh khong co trong candidates file (chua duoc chay --run-pipeline).")
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
                    log_step(f"[!] Loi rerank anh {filename}: {exc}")
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
    log_step("\n=== DANH GIA CAPTION UNG VIEN DAU TIEN (TRUOC RERANK) ===")
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
        log_step(f"[*] CHAIR ground truth: reference captions + instance segmentation cho {merged}/{len(gt_by_filename)} anh.")
    else:
        log_step(
            "[!] Khong co instances_val2014.json hop le -- CHAIR ground truth chi dung 5 cau "
            "reference nguoi viet (yeu hon instance segmentation)."
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
        help="Bo qua METEOR (pycocoevalcap.meteor -- can Java/JRE trong PATH).",
    )
    parser.add_argument("--no-cider", action="store_true")
    parser.add_argument(
        "--no-spice",
        action="store_true",
        help="Bo qua SPICE (pycocoevalcap.spice -- can Java/JRE trong PATH, cham hon cac metric khac).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["ptb", "simple"],
        default="simple",
        help="'ptb' (mac dinh) = Stanford PTBTokenizer, chuan danh gia COCO. 'simple' = lowercase "
        "+ bo dau cau/so giong het script baseline (danh gia mscoco.txt) -- dung de so sanh "
        "cong bang voi so lieu baseline da bao cao.",
    )
    parser.add_argument("--compare-first-candidate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--resume", action="store_true", help="Tiep tuc tu report JSON da luu tam.")
    parser.add_argument("--save-every", type=int, default=10, help="Luu tam sau moi N anh khi --run-pipeline.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="So anh rerank chay song song (chi cho --rerank-only).",
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
        help="Sinh them 1 candidate 'an toan' bang plain beam search (khong diverse), beam width "
        "nay -- luon co mat de reranker khong bao gio te hon plain beam search. 0 = tat, chi con "
        "cac candidate diverse (hanh vi cu).",
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
            raise SystemExit("Hay truyen --candidates-file report.json (tu --run-pipeline) khi dung --rerank-only.")
        rows, yolo_classes = run_rerank_only_pipeline(
            args.candidates_file, set(filenames), references_by_filename, args.image_dir, args
        )
    elif args.predictions:
        records = load_prediction_records(args.predictions)
        rows = build_eval_rows_from_predictions(records, references_by_filename, args.caption_field)
        if args.max_images is not None:
            rows = rows[: args.max_images]
    else:
        raise SystemExit("Hay truyen --predictions file.json, --run-pipeline, hoac --rerank-only --candidates-file file.json.")

    if not rows:
        raise SystemExit("Khong co row nao de danh gia.")

    log_step(f"[*] So caption reranked se danh gia: {len(rows)}")
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

    print_metrics("KET QUA CAPTION SAU RERANKING", reranked_metrics)
    if baseline_metrics is not None:
        print_metrics("KET QUA UNG VIEN DAU TIEN TRUOC RERANK", baseline_metrics)

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
    log_step(f"[*] Da luu report: {args.output}")


if __name__ == "__main__":
    main()
