"""Generate a Kandinsky 6 video with the published OrbitQuant runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

DEFAULT_REPO = "WaveCut/Kandinsky-6.0-Lite-distill-5s-OrbitQuant-W4A4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("kandinsky-orbitquant.mp4"))
    parser.add_argument("--profile", choices=("fast", "low-memory", "exact"), default="fast")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--allocator-cap-gib", type=float)
    parser.add_argument("--warmup", action="store_true", help="Discard one full generation first")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    if any(value <= 0 or value % 16 for value in (args.width, args.height)):
        parser.error("width and height must be positive multiples of 16")
    if args.frames <= 0 or args.frames % 4 != 1:
        parser.error("frames must be a positive number of the form 4n + 1")
    if args.allocator_cap_gib is not None and not 0 < args.allocator_cap_gib < float("inf"):
        parser.error("allocator-cap-gib must be finite and positive")
    return args


def configure_pipeline(pipe, profile):
    from orbitquant.runtime.kandinsky6 import (
        install_kandinsky6_attention,
        install_kandinsky6_norm_fusion,
        install_kandinsky6_pointwise,
        install_kandinsky6_projection_reuse,
        install_kandinsky6_qk_norm,
    )
    from orbitquant.runtime.kandinsky6_video import (
        install_kandinsky6_video_io,
        install_kandinsky6_video_norm,
    )

    if profile not in ("fast", "low-memory", "exact"):
        raise ValueError(f"unsupported profile: {profile}")
    if profile != "exact":
        install_kandinsky6_attention(pipe.transformer)
    install_kandinsky6_pointwise(pipe.transformer)
    install_kandinsky6_projection_reuse(pipe.transformer)
    if profile != "exact":
        install_kandinsky6_norm_fusion(pipe.transformer)
        install_kandinsky6_qk_norm(pipe.transformer)
    install_kandinsky6_video_io(pipe)
    if profile == "exact":
        pipe.vae.enable_tiling()
    else:
        install_kandinsky6_video_norm(pipe)
        pipe.vae.enable_tiling(
            tile_sample_min_height=272,
            tile_sample_min_width=320,
            tile_sample_stride_height=240,
            tile_sample_stride_width=288,
        )
    if profile == "low-memory":
        original_decode = pipe.vae.decode

        def decode_after_reclaim(*args, **kwargs):
            pipe.maybe_free_model_hooks()
            return original_decode(*args, **kwargs)

        pipe.vae.decode = decode_after_reclaim
    pipe.enable_model_cpu_offload()
    return pipe


def load_pipeline(args):
    from diffusers import Kandinsky6TI2VAPipeline, Kandinsky6Transformer3DModel
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import Qwen2_5_VLForConditionalGeneration

    import orbitquant  # noqa: F401 -- registers Hugging Face quantizers

    revision = HfApi().model_info(args.repo, revision=args.revision).sha
    metadata_path = hf_hub_download(args.repo, "artifact_metadata.json", revision=revision)
    metadata = json.loads(Path(metadata_path).read_text())
    source = metadata["source_model"]
    transformer = Kandinsky6Transformer3DModel.from_pretrained(
        args.repo, revision=revision, torch_dtype=torch.bfloat16
    )
    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.repo, revision=revision, subfolder="text_encoder", dtype=torch.bfloat16
    )
    pipe = Kandinsky6TI2VAPipeline.from_pretrained(
        source["repo_id"], revision=source["revision"], transformer=transformer,
        text_encoder=encoder, torch_dtype=torch.bfloat16,
    )
    return configure_pipeline(pipe, args.profile)


@torch.inference_mode()
def generate(pipe, args):
    return pipe(
        prompt=args.prompt, height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=10, guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")
    cap = args.allocator_cap_gib
    if cap is None and args.profile == "low-memory":
        cap = 7.25
    if cap is not None:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = cap * 1024**3 / total
        if fraction > 1:
            raise ValueError("allocator cap exceeds the device's total memory")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.backends.cudnn.benchmark = False
    pipe = load_pipeline(args)
    if args.warmup:
        print("Warming the full pipeline", flush=True)
        generate(pipe, args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = generate(pipe, args)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    from diffusers.utils import encode_video

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        output.frames[0], fps=24, output_path=str(args.output),
        audio=output.audio[0][None], audio_sample_rate=pipe.audio_sample_rate,
    )
    print(json.dumps({
        "output": str(args.output), "seconds": seconds,
        "torch_allocated_peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "profile": args.profile, "width": args.width, "height": args.height,
    }), flush=True)


if __name__ == "__main__":
    main()
