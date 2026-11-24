"""Evaluate the Flickr30k models with the same flow as COCO: logit blocking at decoding time + reranking.

Per image (as in retrieval_logit_bias_experiment.py / sobj_threshold_eval.py, only the data changes to Flickr30k):
  1. YOLO-World (yolov8l-worldv2, threshold 0.25) -> detected objects.
  2. Retrieve the K = 5 nearest captions from the Flickr30k TRAIN captions (CLIP ViT-B/32, as for COCO).
  3. Evidence E(I) = YOLO  U  {objects named by >= r of the K retrieved captions} (r = 3).
  4. Fixed blocking rule: z' = z - delta * 1[w is a COCO object word, o(w) not in E(I)] at every decoding step (delta = 0 means no blocking).
  5. Generate 5 candidates (1 plain beam + diverse beam search), score them with CLIP ViT-B/16 and FormulaReranker (0.2 / 0.4 / 0.4).
  6. Rerank: three signals (S_base, S_align, S_det) and five signals (+ 1.0 x S_obj + 0.1 x S_retr, min-max per image).

Two stages, so the images can be split over at most 2 parallel processes:
  run    generate candidates for a range of images and a list of deltas, saved as a JSON "part" (supports --resume).
  merge  merge the parts, compute BLEU/METEOR/ROUGE-L/CIDEr(/SPICE) and CHAIRs/CHAIRi per delta and per reranking rule.
         With --select, choose delta by the paper's rule (lowest CHAIRs among the deltas whose CIDEr is at most 1 point below delta = 0).

Unlike COCO: Flickr30k has no COCO instances, so CHAIR takes the ground truth from the 5 reference captions of each image (higher values, NOT
directly comparable with CHAIR on COCO). The retrieval store is the Flickr30k train captions, not COCO.

Examples (from the repo root, Python with torch + transformers + ultralytics, Java for METEOR):
    # 1) choose delta on val (first 1,000 images, 2 parts in parallel)
    python eval/flickr30k_eval.py run --split val --start 0   --max-images 500 --deltas 0 1 2 3 4 \\
        --checkpoint flickr30k_..._best_model.pth --vocab-file flickr30k_..._vocab.json --output results/f30k_val_part0.json
    python eval/flickr30k_eval.py run --split val --start 500 --max-images 500 --deltas 0 1 2 3 4 ... --output results/f30k_val_part500.json
    python eval/flickr30k_eval.py merge --parts results/f30k_val_part0.json results/f30k_val_part500.json \\
        --select --output results/f30k_val_eval.json
    # 2) run test with the chosen delta (e.g. 3) and delta = 0 for comparison
    python eval/flickr30k_eval.py run --split test --deltas 0 3 ... --output results/f30k_test_part0.json
    python eval/flickr30k_eval.py merge --parts results/f30k_test_part*.json --output results/f30k_test_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(HERE))

DATASET = ROOT / "dataset"
FLICKR = DATASET / "flickr30k"
SPLIT_SIZES = (29000, 1014, 1000)   # Karpathy train / val / test, used when only a caption file exists (random split, as in the notebook)
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ Flickr30k data
def load_meta(karpathy_json: Path | None, captions_file: Path | None) -> list[dict]:
    """[{filename, split, sents}]: dataset_flickr30k.json (Karpathy) or captions.txt/results.csv split at random as in the notebook."""
    meta: list[dict] = []
    if karpathy_json and karpathy_json.is_file():
        data = json.loads(karpathy_json.read_text(encoding="utf-8"))
        for img in data["images"]:
            sents = [s["raw"] if "raw" in s else " ".join(s.get("tokens", [])) for s in img.get("sentences", [])]
            if img.get("filename") and sents:
                meta.append({"filename": img["filename"], "split": img.get("split"), "sents": sents})
        log(f"Karpathy split from {karpathy_json}")
        return meta
    if captions_file and captions_file.is_file():
        by_image: dict[str, list[str]] = {}
        with open(captions_file, "r", encoding="utf-8") as f:
            header = f.readline()
            sep = "|" if "|" in header else ","
            for line in f:
                parts = line.rstrip("\n").split(sep, 2 if sep == "|" else 1)
                if len(parts) >= 2:
                    by_image.setdefault(parts[0].strip(), []).append(parts[-1].strip())
        names = sorted(by_image)
        random.Random(SEED).shuffle(names)
        n_tr, n_va, n_te = SPLIT_SIZES
        for i, name in enumerate(names):
            split = "train" if i < n_tr else "val" if i < n_tr + n_va else "test" if i < n_tr + n_va + n_te else None
            if split:
                meta.append({"filename": name, "split": split, "sents": by_image[name]})
        log(f"RANDOM SPLIT from {captions_file} (not the Karpathy split, it must match the split used for training)")
        return meta
    raise SystemExit("Neither --karpathy-json (dataset_flickr30k.json) nor --captions-file (captions.txt/results.csv) was found.")


def retrieval_bank(clip, meta: list[dict], cache_dir: Path) -> tuple[list[str], np.ndarray]:
    """Retrieval store: every Flickr30k train caption (lowercased), encoded with CLIP ViT-B/32 and cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    caps_f, emb_f = cache_dir / "train_captions.json", cache_dir / "train_caption_emb_b32.npy"
    if caps_f.is_file() and emb_f.is_file():
        return json.loads(caps_f.read_text(encoding="utf-8")), np.load(emb_f)
    caps = [s.strip().lower() for m in meta if m["split"] == "train" for s in m["sents"]]
    log(f"encoding the retrieval store: {len(caps)} train captions")
    emb = clip.text(caps)
    caps_f.write_text(json.dumps(caps), encoding="utf-8")
    np.save(emb_f, emb)
    return caps, emb


# ------------------------------------------------------------------ logit blocking during decoding
CURRENT_BIAS: torch.Tensor | None = None


def install_logit_bias() -> None:
    import local_captioner

    orig = local_captioner.ImageCaptioningTransformer.decode_step

    def biased(self, memory, sequence):
        logits = orig(self, memory, sequence)
        return logits if CURRENT_BIAS is None else logits + CURRENT_BIAS

    local_captioner.ImageCaptioningTransformer.decode_step = biased


IRREGULAR = {"men": "man", "women": "woman", "kids": "kid", "guys": "guy", "mice": "mouse"}


def vocab_object_map(vocab: dict[str, int], objects: set[str], inv: dict[str, str]) -> dict[int, str]:
    """token id -> canonical COCO class, for single-token object words of the vocabulary (as in retrieval_logit_bias_experiment.py)."""
    import chair_metric

    lem = chair_metric._LEMMATIZER
    out = {}
    for word, idx in vocab.items():
        w = IRREGULAR.get(word) or lem.lemmatize(word, pos="n")
        if w in objects:
            out[idx] = inv[w]
    return out


# ------------------------------------------------------------------ run stage
def cmd_run(args) -> None:
    global CURRENT_BIAS
    if args.device == "cpu":
        torch.cuda.is_available = lambda: False   # the pipeline modules pick their device through this function
    import chair_metric
    import vocab as vocab_mod
    from clipscore import ClipScorer
    from formula_reranker import FormulaReranker
    from generate import CandidateCaptionGenerator
    from schema import DetectedObject
    from sretr_experiment import Clip
    from yolo_objects import YoloObjectExtractor

    meta = load_meta(args.karpathy_json, args.captions_file)
    split_meta = [m for m in meta if m["split"] == args.split][args.start: args.start + args.max_images]
    if not split_meta:
        raise SystemExit(f"Split '{args.split}' has no images in the selected range.")
    image_dir = args.image_dir
    paths = [image_dir / m["filename"] for m in split_meta]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing in {image_dir}, e.g. {missing[:3]}")

    objects_set, inv = chair_metric.load_synonyms()
    clip32 = Clip()
    caps, bank_emb = retrieval_bank(clip32, meta, args.cache_dir)

    # --- evidence (independent of delta): YOLO-World + K retrieved captions
    extractor = YoloObjectExtractor(str(args.yolo_model), args.conf, args.max_objects)
    vocab_mod.set_object_noun_vocab(extractor.class_names())
    log(f"YOLO-World on {len(paths)} images")
    yolo_objs = [extractor.extract(p) for p in paths]
    img_emb = torch.from_numpy(clip32.images(paths)).to(clip32.device)
    bank_t = torch.from_numpy(bank_emb).to(clip32.device)
    nb = np.concatenate([(img_emb[i: i + 256] @ bank_t.T).topk(args.k, dim=1).indices.cpu().numpy()
                         for i in range(0, len(img_emb), 256)])
    del bank_t

    evidence = []
    for objs, idx in zip(yolo_objs, nb):
        yolo = {inv[o.label] for o in objs if o.confidence >= args.conf and o.label in inv}
        counts = Counter(o for j in idx for o in set(chair_metric.caption_to_words(caps[j], objects_set, inv)[1]))
        maj = {o for o, n in counts.items() if n >= args.maj}
        evidence.append({"yolo": sorted(yolo), "counts": dict(counts), "support": yolo | maj})

    # --- caption model, CLIP ViT-B/16 and the reranker
    vocab = json.loads(args.vocab_file.read_text(encoding="utf-8"))
    tok2obj = vocab_object_map(vocab, objects_set, inv)
    log(f"{len(tok2obj)} vocabulary tokens map to COCO objects")
    install_logit_bias()
    generator = CandidateCaptionGenerator(
        checkpoint_path=args.checkpoint, vocab_path=args.vocab_file, max_candidates=5, num_beams=10,
        num_beam_groups=5, diversity_penalty=0.8, max_new_tokens=40, safe_beam_size=3)
    clip16 = ClipScorer(model_name="openai/clip-vit-base-patch16")
    alpha, beta, gamma = args.weights
    formula = FormulaReranker(alpha=alpha, beta=beta, gamma=gamma, formula_mode="linear", clip_scorer=clip16)

    out = {"split": args.split, "start": args.start, "images": len(split_meta), "maj": args.maj, "k": args.k,
           "checkpoint": str(args.checkpoint), "weights": list(args.weights), "runs": {}}
    if args.resume and args.output.is_file():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if old.get("images") == out["images"] and old.get("start") == out["start"]:
            out["runs"] = old["runs"]
            log("resume: " + ", ".join(f"delta {d}: {len(r['rows'])} images" for d, r in out["runs"].items()))

    for delta in args.deltas:
        key = f"{delta:g}"
        run = out["runs"].setdefault(key, {"delta": delta, "rows": []})
        t0, done0 = time.time(), len(run["rows"])
        for i in range(done0, len(split_meta)):
            m, path, objs, ev = split_meta[i], paths[i], yolo_objs[i], evidence[i]
            if delta:
                bias = torch.zeros(len(vocab))
                for tid, obj in tok2obj.items():
                    if obj not in ev["support"]:
                        bias[tid] = -delta
                CURRENT_BIAS = bias.to(generator.device)
            else:
                CURRENT_BIAS = None
            cands = generator.generate(path, objs)
            scored = clip16.score(path, cands)
            ranked = formula.rerank(scored, objs)
            rs = {r.caption.text: r.rerank_score for r in ranked}
            texts = [c.text for c in scored]
            if texts:
                ce = clip32.text(texts).astype(np.float32)
                s_retr = (ce @ bank_emb[nb[i]].astype(np.float32).T).mean(axis=1)
            else:
                s_retr = np.zeros(0)
            run["rows"].append({
                "image": m["filename"], "references": m["sents"], "first": texts[0] if texts else "",
                "yolo": ev["yolo"], "counts": ev["counts"],
                "objects": [{"label": o.label, "confidence": o.confidence} for o in objs],
                "candidates": [{"text": t, "rerank_score": rs.get(t, -1e9), "s_retr": float(sr),
                                "named": chair_metric.caption_to_words(t, objects_set, inv)[1]}
                               for t, sr in zip(texts, s_retr)]})
            n = i + 1 - done0
            if n % args.save_every == 0:
                args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                log(f"  delta={key}: {i + 1}/{len(split_meta)} ({(time.time() - t0) / n:.2f}s/image)")
        CURRENT_BIAS = None
        args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        log(f"done delta={key} ({len(run['rows'])} images)")
    log(f"wrote {args.output}")


# ------------------------------------------------------------------ merge stage
def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.full_like(v, 0.5) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def pick(rows: list[dict], mode: str, thr: int = 3, lam_obj: float = 1.0, lam_retr: float = 0.1) -> list[str]:
    """mode 'first' = plain beam caption, 'three' = three signals, 'five' = five signals (S_obj threshold thr/K, S_retr)."""
    out = []
    for r in rows:
        cands = r["candidates"]
        if not cands:
            out.append(r["first"])
            continue
        base = np.array([c["rerank_score"] for c in cands], dtype=np.float64)
        if mode == "first":
            out.append(r["first"])
            continue
        score = base
        if mode == "five":
            sup = set(r["yolo"]) | {o for o, n in r["counts"].items() if n >= thr}
            s_obj = np.array([-sum(o not in sup for o in c["named"]) for c in cands], dtype=np.float64)
            s_retr = np.array([c["s_retr"] for c in cands], dtype=np.float64)
            score = base + lam_retr * minmax(s_retr) + lam_obj * minmax(s_obj)
        out.append(cands[int(np.argmax(score))]["text"])
    return out


def cmd_merge(args) -> None:
    import chair_metric
    from evaluate_reranked_quality import compute_metrics

    runs: dict[str, dict] = {}
    info = {}
    for f in args.parts:
        part = json.loads(f.read_text(encoding="utf-8"))
        info = {k: part[k] for k in ("split", "maj", "k", "checkpoint", "weights")}
        for key, run in part["runs"].items():
            runs.setdefault(key, {"delta": run["delta"], "rows": {}})["rows"].update({r["image"]: r for r in run["rows"]})
    objects_set, inv = chair_metric.load_synonyms()

    def score(rows, preds, spice):
        refs = [r["references"] for r in rows]
        gt = {r["image"]: chair_metric.ground_truth_from_captions(r["references"], objects_set, inv) for r in rows}
        m = compute_metrics(preds, refs, use_meteor=not args.no_meteor, use_cider=True, use_spice=spice, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: p for r, p in zip(rows, preds)}, gt, objects_set, inv)
        res = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}   # as in sobj_threshold_eval.py
        res.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2),
                   mean_len=round(float(np.mean([len(p.split()) for p in preds])), 2))
        log(str(res))
        return res

    configs = [("before_rerank", "first", 0)] + [("three_score", "three", 0)] + [(f"five_score_{t}of{info['k']}", "five", t) for t in args.thresholds]
    result = {**info, "gt": "5 reference captions (Flickr30k has no COCO instances)", "deltas": {}}
    order = None
    for key in sorted(runs, key=lambda x: float(x)):
        rows_by_img = runs[key]["rows"]
        order = order or list(rows_by_img)
        rows = [rows_by_img[i] for i in order if i in rows_by_img]
        log(f"delta = {key}: {len(rows)} images")
        entry = {"images": len(rows)}
        for name, mode, thr in configs:
            entry[name] = score(rows, pick(rows, mode, thr, args.lam_obj, args.lam_retr), args.spice and name == args.spice_on)
        result["deltas"][key] = entry

    if args.select:
        sel_key = args.select_on
        base_key = "0"
        if base_key not in result["deltas"]:
            raise SystemExit("--select needs delta = 0 in the parts as the CIDEr reference.")
        ref_cider = result["deltas"][base_key][sel_key]["CIDEr"]
        ok = {k: v for k, v in result["deltas"].items() if float(k) > 0 and v[sel_key]["CIDEr"] >= ref_cider - args.max_cider_drop}
        result["selection"] = {"rule": f"lowest CHAIRs after '{sel_key}' among deltas > 0 with CIDEr >= {ref_cider} - {args.max_cider_drop}",
                               "eligible": sorted(ok, key=float),
                               "chosen": min(ok, key=lambda k: (ok[k][sel_key]["CHAIRs"], -ok[k][sel_key]["CIDEr"])) if ok else None}
        log(f"chosen delta: {result['selection']}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {args.output}")
    print_table(result)


def print_table(result: dict) -> None:
    cols = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE", "CHAIRs", "CHAIRi"]
    print("\n" + f"{'delta':>6} {'selection':<18}" + "".join(f"{c:>9}" for c in cols))
    for key, entry in result["deltas"].items():
        for name, vals in entry.items():
            if isinstance(vals, dict):
                print(f"{key:>6} {name:<18}" + "".join(f"{vals.get(c, '-'):>9}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate candidates for a range of images and deltas")
    r.add_argument("--split", choices=["val", "test"], required=True)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--max-images", type=int, default=100000, help="number of images in this part (split across 2 processes)")
    r.add_argument("--deltas", type=float, nargs="+", default=[0.0, 3.0], help="blocking strength delta, 0 = no blocking")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--vocab-file", type=Path, required=True)
    r.add_argument("--karpathy-json", type=Path, default=FLICKR / "dataset_flickr30k.json")
    r.add_argument("--captions-file", type=Path, default=None, help="fallback when there is no Karpathy json (must match the training split)")
    r.add_argument("--image-dir", type=Path, default=FLICKR / "images")
    r.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "flickr30k_cache")
    r.add_argument("--yolo-model", type=Path, default=Path("yolov8l-worldv2.pt"))
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--max-objects", type=int, default=20)
    r.add_argument("--k", type=int, default=5, help="number of retrieved captions")
    r.add_argument("--maj", type=int, default=3, help="majority threshold r for the blocking evidence")
    r.add_argument("--weights", type=float, nargs=3, default=(0.2, 0.4, 0.4), metavar=("ALPHA", "BETA", "GAMMA"))
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--save-every", type=int, default=25)
    r.add_argument("--output", type=Path, required=True)
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("merge", help="merge the parts and compute the metrics")
    m.add_argument("--parts", type=Path, nargs="+", required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--thresholds", type=int, nargs="+", default=[3, 1], help="S_obj thresholds r for the five-signal reranker")
    m.add_argument("--lam-obj", type=float, default=1.0)
    m.add_argument("--lam-retr", type=float, default=0.1)
    m.add_argument("--no-meteor", action="store_true", help="skip METEOR (without Java)")
    m.add_argument("--spice", action="store_true", help="also compute SPICE (Java 8, slow), only for the --spice-on selection")
    m.add_argument("--spice-on", default="five_score_3of5")
    m.add_argument("--select", action="store_true", help="choose delta by the paper's rule (run on val)")
    m.add_argument("--select-on", default="three_score", help="selection used to choose delta (the paper uses the three signals)")
    m.add_argument("--max-cider-drop", type=float, default=1.0)
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
