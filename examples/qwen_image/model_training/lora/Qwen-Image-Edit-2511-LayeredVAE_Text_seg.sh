DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models \
DIFFSYNTH_SKIP_DOWNLOAD=true \
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "/home/ubuntu/for_jjseol/qwen_text_seg" \
  --dataset_metadata_path "/home/ubuntu/for_jjseol/qwen_text_seg/train100k.json" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image/tokenizer" \
  --processor_path "/mnt/lica-data-2/for_jjseol/diffsynth_models/Qwen/Qwen-Image-Edit/processor" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2511_lora_layered_vae_text_seg" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 256 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --rgba_keys "image,edit_image" \
  --zero_cond_t # This is a special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.
