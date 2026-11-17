# MEHC-Cap demo web app

An interactive demo of the MEHC-Cap inference pipeline. The pipeline is drawn as a graph (React Flow), and each stage is a node
you run one at a time to see its intermediate output: the detected objects, the retrieved captions, the candidate captions with
and without logit blocking, the CLIPScore of each candidate, the five reranking signals and the final caption.

- `backend/`: FastAPI server. `backend/pipeline/` is a standalone copy of the captioning engine (captioner, YOLO-World, CLIPScore,
  formula reranker, CHAIR) plus `retrieval.py` for caption retrieval, the logit-blocking rule and the five-signal reranker. It does
  not import the root modules of the repository.
- `frontend/`: Next.js (App Router) with `@xyflow/react`, zustand and Tailwind.

## Pipeline graph

| Node | Backend step | What it shows |
|---|---|---|
| 1. Input | `POST /api/jobs` | The uploaded image |
| 2A. Object detection | `POST /api/jobs/{id}/steps/detect` | YOLO-World boxes (threshold 0.25, at most 20 objects) |
| 2B. Caption retrieval | `POST /api/jobs/{id}/steps/retrieve` | The K = 5 nearest training captions (CLIP ViT-B/32) and the evidence set `E(I)` |
| 3. Caption generation | `POST /api/jobs/{id}/steps/caption` | Up to 5 candidates, with logit blocking `z' = z - δ` on object words outside `E(I)` (δ = 0 turns it off) |
| 4. Image-text match | `POST /api/jobs/{id}/steps/clipscore` | CLIPScore (CLIP ViT-B/16) of each candidate |
| 5. Reranking and hallucination check | `POST /api/jobs/{id}/steps/rerank` | `S_base`, `S_align`, `S_belief`, `S_obj`, `S_retr` and the total score of each candidate |
| 6. Final result | `GET /api/jobs/{id}/steps/final` | The selected caption |

Step 3 needs 2A and 2B. Calling a step before the steps it depends on returns HTTP 409 with the missing steps. Running step 3
again (for example with another model or δ) resets steps 4 to 6. Jobs are kept in memory, so they are lost when the server
restarts. This is enough for a single-user demo.

The reranker uses the same settings as the paper: weights `0.2 / 0.4 / 0.4` for `S_base / S_align / S_belief`, `w_obj = 1`,
`w_retr = 0.1`, `K = 5`, `r = 3`, and YOLO-World threshold 0.55 for the evidence set of `S_obj` only. They are set in
`backend/pipeline/config.py`.

## Model files (`backend/weights/`)

The backend loads every model file from `backend/weights/`, which is not in the repository (about 2.8 GB). Create it with:

| File | Role |
|---|---|
| `models.json` | The caption checkpoints listed in the model selector of the UI |
| `*.pth` + vocabulary `coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json` | Caption checkpoints from the training notebooks (`../notebooks/01`, `02`) |
| `yolov8l-worldv2.pt` | YOLO-World |
| `clip-vit-base-patch16/`, `clip-vit-base-patch32/` | CLIP models (optional, downloaded from Hugging Face when missing) |
| `retrieval/train_captions.json`, `retrieval/train_caption_emb_b32.npy` | Retrieval store: the train + restval captions and their CLIP ViT-B/32 embeddings |

Example `models.json` (the decoder layer count is read from each checkpoint):

```json
{
  "default": "mehc-cap",
  "models": [
    {"id": "mehc-cap", "label": "MEHC-Cap (DL+CL)", "checkpoint": "seed42_dynlogits_contrastive_attnpool_epoch40_model.pth",
     "vocab": "coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json", "default_delta": 3},
    {"id": "baseline", "label": "Baseline (CE)", "checkpoint": "baseline_epoch40_seed42.pth",
     "vocab": "coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json", "default_delta": 0}
  ]
}
```

`backend/README.md` lists the endpoints and the environment variables (`WEIGHTS_DIR`, `YOLO_MODEL_PATH`, `PIPELINE_DEVICE`,
`PRELOAD_ALL_MODELS`, `AUTO_LOAD_RETRIEVAL`).

## Run with Docker

From this folder, with an NVIDIA GPU, the NVIDIA driver and nvidia-container-toolkit:

```
docker compose up --build
```

Open http://localhost:3000 for the UI and http://localhost:8000/docs for the API. `backend/weights/` is mounted read-only. Without a
GPU, remove the `deploy` block in `docker-compose.yml` and the backend runs on CPU (slowly). The browser calls the backend directly,
so to open the UI from another machine, set the build argument `NEXT_PUBLIC_API_BASE_URL` to `http://<server-ip>:8000`.

## Run locally

Backend (Python 3.10, PyTorch; the Docker image pins `transformers==4.46.1` and `ultralytics==8.4.34`). The code uses relative
imports, so start it as the package `webapp.backend.main` from the folder that contains `webapp/`:

```
pip install -r webapp/backend/requirements.txt
python -m uvicorn webapp.backend.main:app --reload --port 8000
```

Frontend (Node 22), in `webapp/frontend/`:

```
npm install
npm run dev
```

The frontend reads the backend URL from `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`). Put it in
`frontend/.env.local` to change it.
