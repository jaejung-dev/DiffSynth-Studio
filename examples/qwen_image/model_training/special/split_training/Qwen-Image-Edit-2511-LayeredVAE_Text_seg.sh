DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "/home/ubuntu/for_jjseol/qwen_text_seg" \
  --dataset_metadata_path "/home/ubuntu/for_jjseol/qwen_text_seg/train.json" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image/tokenizer" \
  --processor_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image-Edit/processor" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/local/Qwen-Image-Edit-2511_full_layered_vae_text_seg_cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --rgba_keys "image,edit_image" \
  --zero_cond_t \
  --task "sft:data_process"

DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "/mnt/local/Qwen-Image-Edit-2511_full_layered_vae_text_seg_cache" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2511_full_layered_vae_text_seg" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t \
  --task "sft:train"
  --init_dit_ckpt "/home/ubuntu/DiffSynth-Studio/models/train/Qwen-Image-Edit-2511_full_layered_vae_text_seg/epoch-0-20k.safetensors" \