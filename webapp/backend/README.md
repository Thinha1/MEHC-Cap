# Backend — Branching Captioning Pipeline API

FastAPI wrapper around a **standalone copy** of the `demo/` captioning engine (see
`pipeline/`), extended with the proposed method: caption retrieval, the fixed logit-blocking
rule and the five-signal reranking (`pipeline/retrieval.py`). `demo/` is never imported; all
model files live in `weights/`.

Each pipeline stage is its own endpoint so the frontend (React Flow) can run the graph one node
at a time instead of all at once.

## Setup & run

Use a Python 3.10 with torch/ultralytics/transformers/fastapi/uvicorn (note: the default Python 3.11
on PATH does NOT have these packages):

```powershell
python -m pip install -r webapp/backend/requirements.txt
```

Start the server (from the folder that contains `webapp/`):

```powershell
python -m uvicorn webapp.backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Models and data (`weights/` folder)

All model files live in `webapp/backend/weights/` (nothing points outside the folder), so the whole
`webapp` folder can be copied anywhere and still runs. This folder is in `.gitignore`.

| File | Role |
|---|---|
| `models.json` | Captioning checkpoints listed in the model selector of the UI (id, name, file, vocab, default delta) |
| `restval_dynlogits_contrastive_attnpool_epoch40_model.pth` | Proposed model (default delta 3) |
| `restval_baseline_best_model.pth` | Baseline model (default delta 0, no blocking) |
| `coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json` | 10,031-word vocabulary shared by the two checkpoints above |
| `yolov8l-worldv2.pt` | YOLO-World object detection |
| `clip-vit-base-patch16/` | CLIP ViT-B/16 for CLIPScore and the encoder architecture of the captioner |
| `clip-vit-base-patch32/` | CLIP ViT-B/32 for caption retrieval and S_retr |
| `retrieval/train_captions.json`, `retrieval/train_caption_emb_b32.npy` | Retrieval store: 566,747 train/restval captions and their B/32 embeddings (fp16) |

To add a checkpoint: copy the `.pth` and its vocabulary into `weights/`, then add an entry to `models.json`.

The settings of the proposed pipeline are in `pipeline/config.py`: K = 5, r = 3, w_obj = 1, w_retr = 0.1,
alpha/beta/gamma = 0.2/0.4/0.4 (not read from `.env`).

## Environment variables (optional)

| Variable | Default | Meaning |
|---|---|---|
| `WEIGHTS_DIR` | `webapp/backend/weights` | Folder that holds the models |
| `YOLO_MODEL_PATH` | `weights/yolov8l-worldv2.pt` | Path to the YOLO checkpoint |
| `PIPELINE_DEVICE` | `cuda` if available, otherwise `cpu` | Torch device for every model |
| `PRELOAD_ALL_MODELS` | `1` | Load every checkpoint in `models.json` at start-up (`0` loads only the default model) |
| `AUTO_LOAD_RETRIEVAL` | `1` | Load the retrieval store at start-up (`0` to load it with the button in the UI) |

## Endpoints

| Step | Method & Path |
|---|---|
| List of models | `GET /api/models` |
| Retrieval store | `GET /api/retrieval` (status), `POST /api/retrieval/load` (load the store) |
| 1. Input | `POST /api/jobs` (multipart `image`) |
| 2A. YOLO-World | `POST /api/jobs/{job_id}/steps/detect` |
| 2B. Caption retrieval | `POST /api/jobs/{job_id}/steps/retrieve` (the store must be loaded first) |
| 3. Generate captions + logit blocking | `POST /api/jobs/{job_id}/steps/caption` (body `{"model_id": ..., "delta": ...}`, requires 2A + 2B) |
| 4. CLIPScore | `POST /api/jobs/{job_id}/steps/clipscore` |
| 5. Reranking | `POST /api/jobs/{job_id}/steps/rerank` (five-signal formula, no LLM) |
| 6. Final | `GET /api/jobs/{job_id}/steps/final` |
| — | `GET /api/jobs/{job_id}` (snapshot of the whole job) |
| — | `GET /api/jobs/{job_id}/image` (serves the uploaded image) |

Calling a step out of order returns HTTP 409 with the list of missing steps. Running step 3 again (for example with
another model) resets steps 4 to 6 to "not run".

State is kept in RAM (no database) and is lost on restart, which is enough for a single-user demo.
