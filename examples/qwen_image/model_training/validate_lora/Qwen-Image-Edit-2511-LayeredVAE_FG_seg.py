import glob
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


DEFAULT_MODEL_BASE = "/mnt/lica-data-2/for_jjseol/diffsynth_models"

LORA_PATH = "models/train/Qwen-Image-Edit-2511_lora_layered_vae_fg_seg/epoch-0-fail.safetensors"

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

    model_base = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", DEFAULT_MODEL_BASE))
    qwen_root = model_base / "Qwen"

    def _model_config_local_only(local_glob: str, label: str) -> ModelConfig:
        paths = sorted(glob.glob(local_glob))
        if not paths:
            raise FileNotFoundError(f"[model] Missing local weights for {label}: {local_glob}")
        path_value = paths if len(paths) > 1 else paths[0]
        print(f"[model] Using local weights: {path_value}")
        return ModelConfig(path=path_value)

    def _aux_config_local_only(local_dir: Path, label: str) -> ModelConfig:
        if not local_dir.exists():
            raise FileNotFoundError(f"[aux] Missing local assets for {label}: {local_dir}")
        print(f"[aux] Using local assets: {local_dir}")
        return ModelConfig(path=str(local_dir))

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            _model_config_local_only(
                str(qwen_root / "Qwen-Image-Edit-2511" / "transformer" / "diffusion_pytorch_model-*.safetensors"),
                "Qwen-Image-Edit-2511 transformer",
            ),
            _model_config_local_only(
                str(qwen_root / "Qwen-Image" / "text_encoder" / "model*.safetensors"),
                "Qwen-Image text encoder",
            ),
            _model_config_local_only(
                str(qwen_root / "Qwen-Image-Layered" / "vae" / "diffusion_pytorch_model.safetensors"),
                "Qwen-Image-Layered VAE",
            ),
        ],
        tokenizer_config=None,
        processor_config=_aux_config_local_only(qwen_root / "Qwen-Image-Edit" / "processor", "Qwen-Image-Edit processor"),
    )

    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA checkpoint not found: {LORA_PATH}")
    pipe.load_lora(pipe.dit, LORA_PATH)

    edit_image = Image.open(EDIT_IMAGE_PATH).convert("RGBA")
    #edit_image = Image.open(IMAGE_PATH).convert("RGBA")
    width, height = edit_image.size

    result = pipe(
        PROMPT,
        edit_image=edit_image,
        seed=SEED,
        num_inference_steps=STEPS,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
    )
    input_image = Image.open(IMAGE_PATH).convert("RGBA")

    # Resize input/edit to match output size
    target_size = result.size
    if input_image.size != target_size:
        input_image = input_image.resize(target_size, Image.LANCZOS)
    if edit_image.size != target_size:
        edit_image = edit_image.resize(target_size, Image.LANCZOS)

    def _checkerboard_rgba(size, tile=32):
        w, h = size
        light = (200, 200, 200, 255)
        dark = (160, 160, 160, 255)
        bg = Image.new("RGBA", size, light)
        tile_img = Image.new("RGBA", (tile, tile), dark)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if (x // tile + y // tile) % 2 == 1:
                    bg.paste(tile_img, (x, y))
        return bg

    # Composite output on checkerboard background for RGBA visibility
    output_rgba = result.convert("RGBA")
    checker_bg = _checkerboard_rgba(output_rgba.size, tile=32)
    output_composite = output_rgba#Image.alpha_composite(checker_bg, output_rgba).convert("RGB")

    # Compose side-by-side: edit_image | output | input_image
    edit_rgb = edit_image.convert("RGB")
    input_rgb = input_image.convert("RGB")
    out_w, out_h = target_size
    canvas = Image.new("RGB", (out_w * 3, out_h), (0, 0, 0))
    canvas.paste(edit_rgb, (0, 0))
    canvas.paste(output_composite, (out_w, 0))
    canvas.paste(input_rgb, (out_w * 2, 0))

    out_path = Path("image_fg_seg_triplet.png")
    canvas.save(out_path)
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
