#!/usr/bin/env python3
"""Export the TAESD decoder (Tiny AutoEncoder for SD) to ONNX.

To be used as drop-in replacement for the SD1.5 VAE decoder in the LCM runners (rknn/onnx).

Example:
    python export_taesd.py --output ~/models/taesd_onnx --size 256x256
    # -> ~/models/taesd_onnx/model.onnx  +  config.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderTiny


DEFAULT_REPO = "madebyollin/taesd"


class TaesdDecoder(torch.nn.Module):
    """Wraps AutoencoderTiny.decoder so forward(latents) -> image in the SD [-1, 1] range.

    See module docstring, convention (1): the decoder is applied directly to the raw
    denoised latents (TAESD's config scaling_factor is 1.0, so no unscaling). The
    decoder ALREADY emits the SD [-1, 1] range that the runner's postprocess expects,
    so no output remapping is applied — this matches diffusers' AutoencoderTiny.decode,
    which returns self.decoder(x) unchanged.
    """

    def __init__(self, taesd: AutoencoderTiny):
        super().__init__()
        self.decoder = taesd.decoder

    def forward(self, latent_sample):
        return self.decoder(latent_sample)


def parse_size(size: str):
    """'256x256' -> (height, width). Latents are 1/8 the image size, 4 channels."""
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like WIDTHxHEIGHT, got: {size!r}")
    return h, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", "-o", required=True,
                    help="Output directory for model.onnx + config.json")
    ap.add_argument("--repo", "-r", default=DEFAULT_REPO,
                    help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--size", "-s", default="256x256",
                    help="Image size WxH the decoder is exported for (default: 256x256). "
                         "Shape is resolution-locked; re-export for other sizes.")
    ap.add_argument("--opset", default=18, type=int,
                    help="ONNX opset (default: 18). The dynamo exporter emits >=18; "
                         "requesting 17 triggers a Resize down-convert that has no adapter.")
    args = ap.parse_args()

    height, width = parse_size(args.size)
    latent_h, latent_w = height // 8, width // 8

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    print(f"Loading {args.repo} ...")
    taesd = AutoencoderTiny.from_pretrained(args.repo)
    taesd.eval()

    model = TaesdDecoder(taesd)
    model.eval()   # the wrapper is a fresh nn.Module; eval() the whole thing, not just taesd
    # SD1.5 latents: (batch, 4, H/8, W/8)
    dummy = torch.randn(1, 4, latent_h, latent_w)

    print(f"Exporting decoder to {onnx_path} (latents {list(dummy.shape)}) ...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path.as_posix(),
            input_names=["latent_sample"], output_names=["sample"],
            opset_version=args.opset,
        )

    # scaling_factor 1.0: the runner's `denoised /= scaling_factor` must be a no-op
    # because unscaling is NOT part of TAESD's decode path (see module docstring).
    config = {"scaling_factor": 1.0}
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {config_path}: {config}")

    print("\nDone. Next:")
    print(f"  1. Verify on PC:  python runners/run_onnx_lcm.py --prompt '...' "
          f"-i <model_onnx> --vae-dir {out_dir} -s {args.size}")
    print(f"  2. On the board:  python sd_conversion/convert_onnx_to_rknn.py "
          f"-m {out_dir} -c model -r {args.size}")
    print(f"     then copy {config_path.name} next to the produced model.rknn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
