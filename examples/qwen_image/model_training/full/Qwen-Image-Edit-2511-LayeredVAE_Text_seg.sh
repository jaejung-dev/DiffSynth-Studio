### Optional: add --init_dit_ckpt "<path>" to resume
SAVE_STEPS=1000
DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "/home/ubuntu/for_jjseol/qwen_text_seg,/home/ubuntu/for_jjseol/qwen_fg_seg" \
  --dataset_metadata_path "/home/ubuntu/for_jjseol/qwen_text_seg/train100k.json,/home/ubuntu/for_jjseol/qwen_fg_seg/train100k.json" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --init_dit_ckpt "/mnt/lica-data-2/for_jjseol/diffsynth_models/train/Qwen-Image-Edit-2511_full_layered_vae_text_fg_seg/step-5000-1.safetensors" \
  --tokenizer_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image/tokenizer" \
  --processor_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image-Edit/processor" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --save_steps "${SAVE_STEPS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/train/Qwen-Image-Edit-2511_full_layered_vae_text_fg_seg" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --rgba_keys "image,edit_image" \
  --zero_cond_t # This is a special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.
