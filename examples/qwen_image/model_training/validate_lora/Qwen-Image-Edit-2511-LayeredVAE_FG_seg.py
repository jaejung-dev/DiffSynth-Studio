import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


# Optional: point to local cached models to avoid downloads.
# export DIFFSYNTH_MODEL_BASE_PATH=/mnt/lica-data-2/for_jjseol/diffsynth_models
# export DIFFSYNTH_SKIP_DOWNLOAD=true

LORA_PATH = "models/train/Qwen-Image-Edit-2511_lora_layered_vae_fg_seg/epoch-1.safetensors"

# Example sample (update as needed)
SAMPLE_ID = "0A1fpH5vp6T8dy0VU9xi"
DATA_ROOT = Path("/mnt/data/for_jjseol/qwen_text_seg")
IMAGE_PATH = DATA_ROOT / "images" / f"{SAMPLE_ID}.png"
EDIT_IMAGE_PATH = DATA_ROOT / "edit_images" / f"{SAMPLE_ID}.png"

PROMPT = "Extract the text layer."
SEED = 123
STEPS = 40


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA checkpoint not found: {LORA_PATH}")
    pipe.load_lora(pipe.dit, LORA_PATH)

    input_image = Image.open(IMAGE_PATH).convert("RGBA")
    edit_image = Image.open(EDIT_IMAGE_PATH).convert("RGBA")
    width, height = input_image.size

    result = pipe(
        PROMPT,
        input_image=input_image,
        edit_image=edit_image,
        seed=SEED,
        num_inference_steps=STEPS,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
    )
    out_path = Path("image_fg_seg.png")
    result.save(out_path)
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
