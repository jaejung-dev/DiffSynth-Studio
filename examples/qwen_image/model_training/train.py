import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--init_dit_ckpt", type=str, default=None, help="Path to a DiT checkpoint to warm-start training.")
    parser.add_argument(
        "--rgba_keys",
        type=str,
        default="",
        help="Comma-separated data keys to load as RGBA (e.g. image,edit_image).",
    )
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    rgba_keys = {key.strip() for key in args.rgba_keys.split(",") if key.strip()}

    def parse_csv(value):
        if value is None:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    base_paths = parse_csv(args.dataset_base_path)
    if not base_paths:
        raise ValueError("dataset_base_path is required.")
    if args.dataset_metadata_path:
        metadata_paths = parse_csv(args.dataset_metadata_path)
        if len(metadata_paths) == 1 and len(base_paths) > 1:
            metadata_paths = metadata_paths * len(base_paths)
        elif len(metadata_paths) != len(base_paths):
            raise ValueError(
                "dataset_base_path and dataset_metadata_path must have the same number of items,"
                " or provide a single metadata path to reuse for all datasets."
            )
    else:
        metadata_paths = [None] * len(base_paths)

    def rgba_image_operator(base_path):
        return RouteByType(
            operator_map=[
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> LoadImage(convert_RGB=False, convert_RGBA=True)
                    >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
                ),
                (
                    list,
                    SequencialProcess(
                        ToAbsolutePath(base_path)
                        >> LoadImage(convert_RGB=False, convert_RGBA=True)
                        >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)
                    ),
                ),
            ]
        )

    def build_dataset(base_path, metadata_path):
        special_operator_map = {
            # Qwen-Image-Layered
            "layer_input_image": ToAbsolutePath(base_path)
            >> LoadImage(convert_RGB=False, convert_RGBA=True)
            >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
            "image": rgba_image_operator(base_path)
            if "image" in rgba_keys
            else RouteByType(
                operator_map=[
                    (
                        str,
                        ToAbsolutePath(base_path)
                        >> LoadImage()
                        >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
                    ),
                    (
                        list,
                        SequencialProcess(
                            ToAbsolutePath(base_path)
                            >> LoadImage(convert_RGB=False, convert_RGBA=True)
                            >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)
                        ),
                    ),
                ]
            ),
        }

        if "edit_image" in rgba_keys:
            special_operator_map["edit_image"] = rgba_image_operator(base_path)

        return UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            ),
            special_operator_map=special_operator_map,
        )

    datasets = [
        build_dataset(base_path, metadata_path)
        for base_path, metadata_path in zip(base_paths, metadata_paths)
    ]
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        zero_cond_t=args.zero_cond_t,
    )
    if args.init_dit_ckpt:
        state_dict = load_state_dict(args.init_dit_ckpt)
        load_result = model.pipe.dit.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"[init] Missing keys when loading DiT ckpt: {load_result.missing_keys[:10]}")
        if load_result.unexpected_keys:
            print(f"[init] Unexpected keys when loading DiT ckpt: {load_result.unexpected_keys[:10]}")
        print(f"[init] DiT checkpoint loaded: {args.init_dit_ckpt}")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
