from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

DEFAULT_MODEL_BASE = "/mnt/lica-data-2/for_jjseol/diffsynth_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct images with Qwen-Image-Layered RGBA VAE.")
    parser.add_argument("--input", type=str, required=True, help="Input image path.")
    parser.add_argument("--output", type=str, default="vae_recon.png", help="Reconstruction output path.")
    parser.add_argument(
        "--side_by_side",
        type=str,
        default=None,
        help="Optional side-by-side output path.",
    )
    parser.add_argument("--height", type=int, default=None, help="Resize height (multiple of 16).")
    parser.add_argument("--width", type=int, default=None, help="Resize width (multiple of 16).")
    parser.add_argument(
        "--model_base_path",
        type=str,
        default=None,
        help="Base path for local diffsynth models.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable disk offload (may be slower, avoids GPU memory spikes).",
    )
    parser.add_argument(
        "--alpha_in_minus1_1",
        action="store_true",
        help="If set, map alpha to [-1,1] (default: False, keep alpha in [0,1]).",
    )
    return parser.parse_args()


def _build_vram_config(device: str, *, low_vram: bool) -> dict:
    if device == "cuda" and low_vram:
        return {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    if device == "cuda":
        return {
            "offload_dtype": None,
            "offload_device": None,
            "onload_dtype": torch.bfloat16,
            "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    return {
        "offload_dtype": None,
        "offload_device": None,
        "onload_dtype": torch.float32,
        "onload_device": "cpu",
        "preparing_dtype": torch.float32,
        "preparing_device": "cpu",
        "computation_dtype": torch.float32,
        "computation_device": "cpu",
    }


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-10))


def _pil_to_input_tensor(
    img: Image.Image,
    *,
    device: str,
    dtype: torch.dtype,
    alpha_in_minus1_1: bool,
) -> torch.Tensor:
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * 2.0 - 1.0
    alpha = arr[..., 3:4]
    if alpha_in_minus1_1:
        alpha = alpha * 2.0 - 1.0
    stacked = np.concatenate([rgb, alpha], axis=-1)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _tensor_to_rgba_pil(x: torch.Tensor, *, alpha_in_minus1_1: bool) -> Image.Image:
    if x.ndim == 5:
        x = x[:, :, 0, :, :]
    x = x[0].permute(1, 2, 0).detach().float().cpu().numpy()
    rgb = (np.clip(x[..., :3], -1.0, 1.0) + 1.0) * 0.5
    alpha = x[..., 3:4]
    if alpha_in_minus1_1:
        alpha = (np.clip(alpha, -1.0, 1.0) + 1.0) * 0.5
    else:
        alpha = np.clip(alpha, 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if args.low_vram and device != "cuda":
        print("[warn] --low_vram requested on CPU; disabling low_vram.")
        args.low_vram = False

    model_base = Path(args.model_base_path or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", DEFAULT_MODEL_BASE))
    vae_path = model_base / "Qwen" / "Qwen-Image-Layered" / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE not found at {vae_path}")

    vram_config = _build_vram_config(device, low_vram=args.low_vram)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[ModelConfig(path=str(vae_path), **vram_config)],
        tokenizer_config=None,
        processor_config=None,
    )

    input_image = Image.open(args.input).convert("RGBA")
    height, width = input_image.size[1], input_image.size[0]
    if args.height is not None and args.width is not None:
        height, width = args.height, args.width
    height, width = pipe.check_resize_height_width(height, width)
    if (input_image.size[1], input_image.size[0]) != (height, width):
        input_image = input_image.resize((width, height))

    input_tensor = _pil_to_input_tensor(
        input_image,
        device=device,
        dtype=torch_dtype,
        alpha_in_minus1_1=args.alpha_in_minus1_1,
    )
    with torch.no_grad():
        latents = pipe.vae.encode(input_tensor)
        recon = pipe.vae.decode(latents)

    recon_img = _tensor_to_rgba_pil(recon, alpha_in_minus1_1=args.alpha_in_minus1_1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon_img.save(output_path)

    if args.side_by_side:
        canvas = Image.new("RGBA", (input_image.width * 2, input_image.height))
        canvas.paste(input_image, (0, 0))
        canvas.paste(recon_img, (input_image.width, 0))
        side_path = Path(args.side_by_side)
        side_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(side_path)

    rgb_in = ((input_tensor[:, :3] + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb_out = recon[:, :3].clamp(0.0, 1.0)#((recon[:, :3] + 1.0) * 0.5).clamp(0.0, 1.0)
    if args.alpha_in_minus1_1:
        alpha_in = ((input_tensor[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
        alpha_out = ((recon[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        alpha_in = input_tensor[:, 3:4].clamp(0.0, 1.0)
        alpha_out = recon[:, 3:4].clamp(0.0, 1.0)

    mse_rgb = torch.mean((rgb_in - rgb_out) ** 2).item()
    mse_alpha = torch.mean((alpha_in - alpha_out) ** 2).item()

    white = torch.ones_like(rgb_in)
    black = torch.zeros_like(rgb_in)
    comp_in_white = rgb_in * alpha_in + white * (1.0 - alpha_in)
    comp_out_white = rgb_out * alpha_out + white * (1.0 - alpha_out)
    comp_in_black = rgb_in * alpha_in + black * (1.0 - alpha_in)
    comp_out_black = rgb_out * alpha_out + black * (1.0 - alpha_out)

    mse_white = torch.mean((comp_in_white - comp_out_white) ** 2).item()
    mse_black = torch.mean((comp_in_black - comp_out_black) ** 2).item()

    print(f"[metrics] mse_rgb={mse_rgb:.6f} psnr_rgb={_psnr(mse_rgb):.2f}dB")
    print(f"[metrics] mse_alpha={mse_alpha:.6f}")
    print(f"[metrics] psnr_white={_psnr(mse_white):.2f}dB psnr_black={_psnr(mse_black):.2f}dB")
    print(f"[done] recon saved to {output_path}")


if __name__ == "__main__":
    main()
