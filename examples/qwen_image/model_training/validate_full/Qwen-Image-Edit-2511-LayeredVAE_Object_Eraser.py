import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


DEFAULT_MODEL_BASE = "/mnt/lica-data-2/for_jjseol/diffsynth_models"
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/for_jjseol/qwen_eraser_dataset")
DEFAULT_METADATA_FILE = "train.jsonl"
DEFAULT_TRAIN_OUTPUT_ROOT = Path(
    "/mnt/lica-data-2/for_jjseol/diffsynth_models/train/Qwen-Image-Edit-2511_full_layered_vae_object_eraser"
)

PROMPT = "[TASK_ERASE] Remove masked object and fill naturally."
SEED = 123
STEPS = 40

# If set, this stem will be used directly.
# Example: "05ZRvPt1aGLNDswT75zI_00"
SAMPLE_STEM: str | None = None


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


def _resolve_checkpoint(train_output_root: Path) -> Path:
    # Priority: explicit env path -> latest epoch -> latest step
    ckpt_override = os.environ.get("FULL_CKPT")
    if ckpt_override:
        ckpt_path = Path(ckpt_override)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[ckpt] FULL_CKPT does not exist: {ckpt_path}")
        return ckpt_path

    candidates: list[Path] = []
    candidates.extend(train_output_root.glob("epoch-*.safetensors"))
    candidates.extend(train_output_root.glob("step-*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"[ckpt] No checkpoint found under {train_output_root}. "
            "Expected epoch-*.safetensors or step-*.safetensors."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _infer_stem_from_src(src_rel: str) -> str:
    # inputs/<stem>_src.png -> <stem>
    src_name = Path(src_rel).name
    if not src_name.endswith("_src.png"):
        raise ValueError(f"[data] Unexpected source filename format: {src_name}")
    return src_name[: -len("_src.png")]


def _resolve_sample_paths(dataset_root: Path, metadata_file: str, sample_stem: str | None):
    if sample_stem is not None:
        src_path = dataset_root / "inputs" / f"{sample_stem}_src.png"
        rmv_path = dataset_root / "masks" / f"{sample_stem}_mask_rgba.png"
        tgt_path = dataset_root / "targets" / f"{sample_stem}_erased.png"
        if not src_path.exists() or not rmv_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(
                f"[data] Missing one of sample files for stem={sample_stem}:\n"
                f"  {src_path}\n  {rmv_path}\n  {tgt_path}"
            )
        return sample_stem, src_path, rmv_path, tgt_path

    meta_path = dataset_root / metadata_file
    if not meta_path.exists():
        raise FileNotFoundError(f"[data] Metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        first_line = None
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break
    if first_line is None:
        raise ValueError(f"[data] Metadata is empty: {meta_path}")

    entry = json.loads(first_line)
    edit_image = entry.get("edit_image")
    image_rel = entry.get("image")
    if not isinstance(edit_image, list) or len(edit_image) < 2:
        raise ValueError(f"[data] edit_image must be a list with 2 items: {entry}")
    if not isinstance(image_rel, str):
        raise ValueError(f"[data] image field must be a string: {entry}")

    src_path = dataset_root / edit_image[0]
    rmv_path = dataset_root / edit_image[1]
    tgt_path = dataset_root / image_rel
    stem = _infer_stem_from_src(edit_image[0])

    if not src_path.exists() or not rmv_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"[data] Missing one of metadata-referenced files:\n"
            f"  {src_path}\n  {rmv_path}\n  {tgt_path}"
        )
    return stem, src_path, rmv_path, tgt_path


def _resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.LANCZOS)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_base = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", DEFAULT_MODEL_BASE))
    dataset_root = Path(os.environ.get("ERASER_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
    metadata_file = os.environ.get("ERASER_METADATA_FILE", DEFAULT_METADATA_FILE)
    train_output_root = Path(os.environ.get("ERASER_TRAIN_OUTPUT_ROOT", str(DEFAULT_TRAIN_OUTPUT_ROOT)))

    qwen_root = model_base / "Qwen"
    ckpt_path = _resolve_checkpoint(train_output_root)
    print(f"[ckpt] Using checkpoint: {ckpt_path}")

    sample_stem, src_path, rmv_path, tgt_path = _resolve_sample_paths(dataset_root, metadata_file, SAMPLE_STEM)
    print(f"[data] Sample stem: {sample_stem}")
    print(f"[data] source: {src_path}")
    print(f"[data] removed-layer: {rmv_path}")
    print(f"[data] target: {tgt_path}")

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

    state_dict = load_state_dict(str(ckpt_path))
    pipe.dit.load_state_dict(state_dict)

    src_rgba = Image.open(src_path).convert("RGBA")
    removed_layer_rgba = Image.open(rmv_path).convert("RGBA")
    target_rgba = Image.open(tgt_path).convert("RGBA")
    width, height = src_rgba.size

    pred = pipe(
        PROMPT,
        edit_image=[src_rgba, removed_layer_rgba],
        seed=SEED,
        num_inference_steps=STEPS,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,
    ).convert("RGBA")

    src_rgba = _resize_to(src_rgba, pred.size)
    removed_layer_rgba = _resize_to(removed_layer_rgba, pred.size)
    target_rgba = _resize_to(target_rgba, pred.size)

    w, h = pred.size
    canvas = Image.new("RGBA", (w * 4, h), (0, 0, 0, 255))
    canvas.paste(src_rgba, (0, 0))
    canvas.paste(removed_layer_rgba, (w, 0))
    canvas.paste(pred, (w * 2, 0))
    canvas.paste(target_rgba, (w * 3, 0))

    output_dir = Path(__file__).resolve().parents[4] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{sample_stem}_object_eraser_pred.png"
    panel_path = output_dir / f"{sample_stem}_object_eraser_panel.png"
    pred.save(pred_path)
    canvas.save(panel_path)
    print(f"[done] saved prediction: {pred_path}")
    print(f"[done] saved panel: {panel_path}")


if __name__ == "__main__":
    main()
