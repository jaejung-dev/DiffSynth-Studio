### Qwen object eraser full training example
### Dataset expected layout:
### /home/ubuntu/for_jjseol/qwen_eraser_dataset/
###   train.jsonl
###   inputs/
###   masks/
###   targets/

SAVE_STEPS=1000
DATASET_ROOT="/home/ubuntu/for_jjseol/qwen_eraser_dataset"
OUTPUT_ROOT="/mnt/lica-data-2/for_jjseol/diffsynth_models/train/Qwen-Image-Edit-2511_full_layered_vae_object_eraser"

DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "${DATASET_ROOT}" \
  --dataset_metadata_path "${DATASET_ROOT}/train.jsonl" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image/tokenizer" \
  --processor_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image-Edit/processor" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --save_steps "${SAVE_STEPS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_ROOT}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --rgba_keys "image,edit_image" \
  --zero_cond_t
