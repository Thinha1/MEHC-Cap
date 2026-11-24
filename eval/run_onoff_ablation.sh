#!/usr/bin/env bash
# Thí nghiệm bật/tắt thành phần trên tập kiểm tra (5.000 ảnh, chia 4 đoạn 1.250 ảnh, bỏ qua đoạn đã có kết quả).
# Hai luồng chạy song song (tối đa 2 tiến trình GPU):
#   bash eval/run_onoff_ablation.sh P   # mô hình đề xuất, chặn δ=3 (e40_g3): cấu hình cuối
#   bash eval/run_onoff_ablation.sh A   # cơ sở + chặn δ=3 (cs_g3), rồi đề xuất chỉ YOLO (e40_yolo)
#   bash eval/run_onoff_ablation.sh B   # đề xuất không chặn (e40_g0), rồi đề xuất chỉ truy hồi (e40_retr)
#   bash eval/run_onoff_ablation.sh C   # chỉ Dynamic Logits: không chặn (dl_g0), chặn δ=3 (dl_g3)
#   bash eval/run_onoff_ablation.sh D   # chỉ học tương phản: không chặn (cl_g0), chặn δ=3 (cl_g3)
#   bash eval/run_onoff_ablation.sh E   # cơ sở không chặn (cs_g0)
# Chạy từ thư mục gốc của repo. Kết quả: results/onoff/<tên>_part<start>.json
set -u
PY="${PY:-python}"  # override with PY=/path/to/python
REPORT="${REPORT:-results/test_yoloworld.json}"  # YOLO-World report of the test split (README, step 3)
VOCAB=dataset/coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json
E40="${E40:-checkpoints/seed42_dynlogits_contrastive_attnpool_epoch40_model.pth}"
BASE="${BASE:-checkpoints/baseline_epoch40_seed42.pth}"
DL="${DL:-checkpoints/seed42_dynlogits_epoch40_model.pth}"                 # Dynamic Logits only
CL="${CL:-checkpoints/seed42_contrastive_attnpool_epoch40_model.pth}"      # contrastive learning only
mkdir -p results/onoff

run() {  # name checkpoint gamma evidence
  local name=$1 ck=$2 gamma=$3 ev=$4
  for start in 0 1250 2500 3750; do
    out=results/onoff/${name}_part${start}.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    "$PY" eval/retrieval_logit_bias_experiment.py --report "$REPORT" --checkpoint "$ck" --vocab-file "$VOCAB" \
      --start "$start" --max-images 1250 --maj 3 --evidence "$ev" --configs "0,$gamma" --save-rows \
      --output "$out" > "results/onoff/${name}_part${start}.log" 2>&1
  done
}

case "${1:-}" in
  P) run e40_g3 "$E40" 3 union ;;
  A) run cs_g3 "$BASE" 3 union; run e40_yolo "$E40" 3 yolo ;;
  B) run e40_g0 "$E40" 0 union; run e40_retr "$E40" 3 retr ;;
  C) run dl_g0 "$DL" 0 union; run dl_g3 "$DL" 3 union ;;
  D) run cl_g0 "$CL" 0 union; run cl_g3 "$CL" 3 union ;;
  E) run cs_g0 "$BASE" 0 union ;;
  *) echo "usage: $0 P|A|B|C|D|E"; exit 1 ;;
esac
echo "done $1"
