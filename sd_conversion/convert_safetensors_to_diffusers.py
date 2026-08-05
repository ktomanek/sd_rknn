#!/usr/bin/env python3
"""Step 1 of the custom-model pipeline: single .safetensors -> diffusers folder.

Runs on an x86 PC / Mac (needs torch + diffusers). NOT on the Orange Pi.

Civitai models come as one .safetensors file. The ONNX exporter (step 2) needs the
unpacked diffusers directory layout (unet/, vae/, text_encoder/, ...), so we load the
single file and re-save it as a folder.

IMPORTANT — run steps 1 and 2 in the SAME environment (the one described by
requirements-pc.txt). The text_encoder weights are written using the *installed*
transformers' CLIP key convention; if step 2 later runs under a different transformers
major version, the loader silently falls back to random weights ("Some weights ... were
newly initialized ... You should probably TRAIN this model") and the exported ONNX text
encoder is garbage. Install everything up front, then run step 1 -> step 2.

The export-readiness normalization (tokenizer vocab files, nulling feature_extractor /
safety_checker in model_index.json) lives in step 2 (convert_diffusers_to_onnx.py), so it
applies no matter how the diffusers folder was produced — this step just unpacks.

Example:
    python convert_safetensors_to_diffusers.py \
        --input ~/models/dreamshaper_8LCM.safetensors \
        --output ~/models/dreamshaper_diffusers
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="Path to the .safetensors (or .ckpt) file")
    ap.add_argument("--output", "-o", required=True,
                    help="Output directory for the diffusers folder")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()

    if not in_path.is_file():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 1
    if out_path.exists() and any(out_path.iterdir()):
        print(f"ERROR: output dir exists and is not empty: {out_path}", file=sys.stderr)
        return 1

    # Imported here so --help works without torch installed.
    from diffusers import StableDiffusionPipeline

    print(f"Loading single-file checkpoint: {in_path}")
    # No safety_checker arg: modern diffusers (>=0.30) attaches no NSFW filter by
    # default, which is what we want (dream images, and it drops a dependency). The old
    # load_safety_checker=False kwarg is deprecated and routes through a legacy loader
    # that needs torchvision, so we deliberately omit it.
    # torch_dtype=None keeps original precision; ONNX export + RKNN handle dtype later.
    pipe = StableDiffusionPipeline.from_single_file(
        str(in_path),
        torch_dtype=None,
    )

    print(f"Saving diffusers folder: {out_path}")
    pipe.save_pretrained(str(out_path))

    print(f"Done. Next: convert_diffusers_to_onnx.py -i {out_path} -o <onnx_dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
