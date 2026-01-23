from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_argument("--force-shard", action="store_true", help="Force model sharding across devices even with small batch sizes. Useful for memory-constrained scenarios.")  # fmt: off
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE_KONTEXT

    # 1. Initialize distributed group early to determine strategy
    import mlx.core as mx
    group = mx.distributed.init()

    # Check for incompatible configuration
    # Quantization after sharding doesn't work because to_quantized() creates
    # new QuantizedLinear layers that lose sharding metadata, causing deadlock
    if group.size() > 1 and args.quantize is not None:
        print("\n" + "="*60)
        print("ERROR: Quantization + Distributed Sharding Not Supported")
        print("="*60)
        print("\nMLX limitation: Quantization replaces sharded Linear layers")
        print("with QuantizedLinear layers that don't preserve distributed")
        print("metadata, causing inference to deadlock.")
        print("\nOptions:")
        print("  1. Use distributed sharding WITHOUT quantization")
        print("     → Memory savings: ~40-50%")
        print("     → Works perfectly, no deadlock")
        print()
        print("  2. Use quantization on SINGLE device")
        print("     → Memory savings: ~20-30% (depending on bit level)")
        print("     → Works perfectly")
        print()
        print("Recommendation: Distributed sharding provides better memory")
        print("savings than quantization and works reliably. Use sharding")
        print("for multi-device runs, quantization for single-device runs.")
        print("="*60 + "\n")
        import sys
        sys.exit(1)

    # 2. Load the model (quantize only for single device)
    qwen = QwenImageEdit(
        quantize=args.quantize if group.size() == 1 else None,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 3. Setup distributed processing if multiple devices available
    should_gather = False
    original_seed = args.seed

    if group.size() > 1:
        print(f"\n{'='*60}")
        print(f"Distributed Processing Enabled: {group.size()} devices detected")
        print(f"{'='*60}\n")

        # Determine strategy: model sharding vs data parallelism
        # For Qwen Image Edit, we always use model sharding since it's
        # a single-image editing task (no batch dimension to parallelize)
        if args.force_shard:
            print("Strategy: FORCED MODEL SHARDING")
            print("  → Model will be split across devices")
            qwen.transformer.shard(group)
        else:
            print("Strategy: MODEL SHARDING (default for Qwen Image Edit)")
            print("  → Model will be split across devices")
            qwen.transformer.shard(group)

        # Seed coordination for sharding: all devices use same seed
        # Note: args.seed is a list, we'll handle coordination per seed in the loop
        print(f"  → Using seed list: {args.seed}")
        print()
    else:
        print("Single device mode - no distributed processing\n")

    # 4. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=QwenLatentCreator,
    )

    try:
        for seed in args.seed:
            # 3. Prepare image paths
            image_paths = [str(p) for p in args.image_paths]

            # 4. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                image_path=image_paths[0],  # Use first image for metadata
                image_paths=image_paths,
                num_inference_steps=args.steps,
            )

            # 5. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.save(path=output_path, export_json_metadata=args.metadata)

            # Print distributed stats for rank 0 only
            if group.size() > 1 and group.rank() == 0:
                print(f"\n{'='*60}")
                print(f"Distributed Generation Complete")
                print(f"  Devices used: {group.size()}")
                print(f"  Seed: {seed}")
                print(f"{'='*60}\n")

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
