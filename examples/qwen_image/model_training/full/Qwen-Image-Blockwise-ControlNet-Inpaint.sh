#!/usr/bin/env bash
set -euo pipefail

### Qwen Blockwise-ControlNet Inpaint full training example
### Dataset expected layout:
### /home/ubuntu/for_jjseol/qwen_eraser_dataset/
###   train_blockwise_controlnet_inpaint.jsonl
###   val_blockwise_controlnet_inpaint.jsonl
###   inputs/
###   targets/
###   blockwise_masks/
###     train/
###     val/

SAVE_STEPS=10000
DATASET_ROOT="/home/ubuntu/for_jjseol/qwen_eraser_dataset"
TRAIN_JSONL="${DATASET_ROOT}/train_blockwise_controlnet_inpaint.jsonl"
OUTPUT_ROOT="/mnt/lica-data-2/for_jjseol/diffsynth_models/train/Qwen-Image-Blockwise-ControlNet-Inpaint_full"

# Resume behavior (notebook-style):
# - When RESUME_TRAINING=true and PREFER_LATEST_TRAINED_CHECKPOINT=true,
#   choose latest step checkpoint from OUTPUT_ROOT.
# - Otherwise use MANUAL_CONTROLNET_CKPT.
RESUME_TRAINING=true
PREFER_LATEST_TRAINED_CHECKPOINT=true
CONTROLNET_CKPT_GLOB="${OUTPUT_ROOT}/step-*.safetensors"
MANUAL_CONTROLNET_CKPT="${OUTPUT_ROOT}/step-30000.safetensors"

BASE_MODEL_ID_WITH_ORIGIN_PATHS="Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
DEFAULT_CONTROLNET_MODEL_ID="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors"

MODEL_ID_WITH_ORIGIN_PATHS="${BASE_MODEL_ID_WITH_ORIGIN_PATHS},${DEFAULT_CONTROLNET_MODEL_ID}"
MODEL_PATHS_JSON=""

if [[ "${RESUME_TRAINING}" == "true" ]]; then
  if [[ "${PREFER_LATEST_TRAINED_CHECKPOINT}" == "true" ]]; then
    INIT_CONTROLNET_CKPT="$(
      python3 - "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = sorted(root.glob("step-*.safetensors"), key=lambda p: p.stat().st_mtime)
if not candidates:
    print("")
    raise SystemExit(0)
print(str(candidates[-1]))
PY
    )"
    if [[ -z "${INIT_CONTROLNET_CKPT}" ]]; then
      echo "[resume] No trained checkpoint found: ${CONTROLNET_CKPT_GLOB}"
      exit 1
    fi
  else
    INIT_CONTROLNET_CKPT="${MANUAL_CONTROLNET_CKPT}"
    if [[ ! -f "${INIT_CONTROLNET_CKPT}" ]]; then
      echo "[resume] Manual checkpoint not found: ${INIT_CONTROLNET_CKPT}"
      exit 1
    fi
  fi

  echo "[resume] controlnet ckpt: ${INIT_CONTROLNET_CKPT}"

  # Load base Qwen weights via model_id_with_origin_paths,
  # and load resumed controlnet via model_paths.
  MODEL_ID_WITH_ORIGIN_PATHS="${BASE_MODEL_ID_WITH_ORIGIN_PATHS}"
  MODEL_PATHS_JSON="$(
    python3 - "${INIT_CONTROLNET_CKPT}" <<'PY'
import json
import sys
print(json.dumps([sys.argv[1]]))
PY
  )"
fi

TRAIN_CMD=(
  accelerate launch
  --config_file examples/qwen_image/model_training/full/accelerate_config.yaml
  examples/qwen_image/model_training/train.py
  --dataset_base_path "${DATASET_ROOT}"
  --dataset_metadata_path "${TRAIN_JSONL}"
  --data_file_keys "image,blockwise_controlnet_image,blockwise_controlnet_inpaint_mask"
  --max_pixels 1048576
  --dataset_repeat 1
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}"
  --learning_rate 1e-4
  --num_epochs 5
  --save_steps "${SAVE_STEPS}"
  --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0."
  --output_path "${OUTPUT_ROOT}"
  --trainable_models "blockwise_controlnet"
  --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask"
  --use_gradient_checkpointing
  --find_unused_parameters
)

if [[ -n "${MODEL_PATHS_JSON}" ]]; then
  TRAIN_CMD+=(--model_paths "${MODEL_PATHS_JSON}")
fi

DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
"${TRAIN_CMD[@]}"

# If you want to pre-train a Inpaint Blockwise ControlNet from scratch,
# please run the following script to first generate the initialized model weights file,
# and then start training with a high learning rate (1e-3).

# python examples/qwen_image/model_training/scripts/Qwen-Image-Blockwise-ControlNet-Inpaint-Initialize.py

# accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
#   --dataset_base_path "${DATASET_ROOT}" \
#   --dataset_metadata_path "${TRAIN_JSONL}" \
#   --data_file_keys "image,blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
#   --model_paths '["models/blockwise_controlnet_inpaint.safetensors"]' \
#   --learning_rate 1e-3 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0." \
#   --output_path "./models/train/Qwen-Image-Blockwise-ControlNet-Inpaint_full" \
#   --trainable_models "blockwise_controlnet" \
#   --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
#   --use_gradient_checkpointing \
#   --find_unused_parameters
