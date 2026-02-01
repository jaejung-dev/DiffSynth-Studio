from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


DEFAULT_MODEL_BASE = "/mnt/lica-data-2/for_jjseol/diffsynth_models"


def _resolve_paths(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def _model_config_from_local_or_remote(
    *,
    local_glob: str,
    model_id: str,
    origin_file_pattern: str,
    vram_config: dict,
) -> ModelConfig:
    paths = _resolve_paths(local_glob)
    if paths:
        path_value = paths if len(paths) > 1 else paths[0]
        print(f"[model] Using local weights: {path_value}")
        return ModelConfig(path=path_value, **vram_config)
    print(f"[model] Falling back to remote: {model_id}:{origin_file_pattern}")
    return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config)


def _aux_config_from_local_or_remote(
    *,
    local_dir: Path,
    model_id: str,
    origin_file_pattern: str,
) -> ModelConfig:
    if local_dir.exists():
        print(f"[aux] Using local assets: {local_dir}")
        return ModelConfig(path=str(local_dir))
    print(f"[aux] Falling back to remote: {model_id}:{origin_file_pattern}")
    return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 inference with Layered RGBA VAE.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument(
        "--edit_images",
        type=str,
        required=True,
        help="Comma-separated edit image paths (use at least one).",
    )
    parser.add_argument("--output", type=str, default="image_edit_2511_rgba.png", help="Output image path.")
    parser.add_argument("--height", type=int, default=None, help="Output height (multiple of 16).")
    parser.add_argument("--width", type=int, default=None, help="Output width (multiple of 16).")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--model_base_path", type=str, default=None, help="Shared model base path.")
    parser.add_argument(
        "--edit_image_auto_resize",
        action="store_true",
        default=True,
        help="Auto-resize edit images (default: True).",
    )
    parser.add_argument(
        "--no_edit_image_auto_resize",
        action="store_false",
        dest="edit_image_auto_resize",
        help="Disable auto-resize for edit images.",
    )
    parser.add_argument(
        "--zero_cond_t",
        action="store_true",
        default=True,
        help="Enable zero_cond_t for 2511 (default: True).",
    )
    parser.add_argument(
        "--no_zero_cond_t",
        action="store_false",
        dest="zero_cond_t",
        help="Disable zero_cond_t.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_base = Path(args.model_base_path or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", DEFAULT_MODEL_BASE))
    qwen_root = model_base / "Qwen"

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    edit_transformer = _model_config_from_local_or_remote(
        local_glob=str(qwen_root / "Qwen-Image-Edit-2511" / "transformer" / "diffusion_pytorch_model-*.safetensors"),
        model_id="Qwen/Qwen-Image-Edit-2511",
        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
        vram_config=vram_config,
    )
    text_encoder = _model_config_from_local_or_remote(
        local_glob=str(qwen_root / "Qwen-Image" / "text_encoder" / "model*.safetensors"),
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="text_encoder/model*.safetensors",
        vram_config=vram_config,
    )
    layered_vae = _model_config_from_local_or_remote(
        local_glob=str(qwen_root / "Qwen-Image-Layered" / "vae" / "diffusion_pytorch_model.safetensors"),
        model_id="Qwen/Qwen-Image-Layered",
        origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
        vram_config=vram_config,
    )

    tokenizer_cfg = _aux_config_from_local_or_remote(
        local_dir=qwen_root / "Qwen-Image" / "tokenizer",
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="tokenizer/",
    )
    processor_cfg = _aux_config_from_local_or_remote(
        local_dir=qwen_root / "Qwen-Image-Edit" / "processor",
        model_id="Qwen/Qwen-Image-Edit",
        origin_file_pattern="processor/",
    )

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[edit_transformer, text_encoder, layered_vae],
        tokenizer_config=tokenizer_cfg,
        processor_config=processor_cfg,
    )

    edit_paths = [p.strip() for p in args.edit_images.split(",") if p.strip()]
    if not edit_paths:
        raise ValueError("No edit images provided. Use --edit_images path1,path2,...")

    edit_images = [Image.open(path).convert("RGBA") for path in edit_paths]

    if args.height is None or args.width is None:
        width, height = edit_images[0].size
    else:
        height = args.height
        width = args.width

    result = pipe(
        args.prompt,
        edit_image=edit_images,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=height,
        width=width,
        edit_image_auto_resize=args.edit_image_auto_resize,
        zero_cond_t=args.zero_cond_t,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"[done] saved to {output_path}")


if __name__ == "__main__":
    main()
