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
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "${DATASET_ROOT}" \
  --dataset_metadata_path "${TRAIN_JSONL}" \
  --data_file_keys "image,blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps "${SAVE_STEPS}" \
  --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0." \
  --output_path "${OUTPUT_ROOT}" \
  --trainable_models "blockwise_controlnet" \
  --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
  --use_gradient_checkpointing \
  --find_unused_parameters

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
