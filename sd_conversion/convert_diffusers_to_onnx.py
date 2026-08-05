#!/usr/bin/env python3
"""Step 2 of the custom-model pipeline: diffusers folder -> ONNX (LCM task).

Runs on an x86 PC / Mac (needs optimum[onnxruntime]). NOT on the Orange Pi.

Thin, checked wrapper around:
    optimum-cli export onnx --model <DIR> --task latent-consistency <OUT>

The --task latent-consistency is what makes the LCM scheduler / few-step path export
correctly. Using the wrong task silently produces a model that needs 25-50 steps.

Before exporting, this script normalizes the diffusers folder so optimum can load it,
regardless of where the folder came from (our step 1, or a step-0 HF download):
  1. Nulls feature_extractor + safety_checker in model_index.json. Neither is part of a
     text-to-image export; keeping them makes optimum try to load weights we don't ship
     (safety_checker) or an unresolvable torchvision-less processor name.
  2. Ensures the slow-tokenizer vocab.json + merges.txt exist (the exporter instantiates
     the slow CLIPTokenizer). Both edits are idempotent.

Install first (on the PC):
    pip install optimum
    pip install --upgrade --upgrade-strategy eager "optimum[onnxruntime]"

Example:
    python convert_diffusers_to_onnx.py \
        --input ~/models/dreamshaper_diffusers \
        --output ~/models/dreamshaper_onnx
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


class _Skip(Exception):
    """Non-fatal: nothing to patch (already rank-1, or unexpected graph)."""


def patch_unet_timestep_to_rank1(unet_onnx: str) -> str:
    """Rewrite <unet>/model.onnx so `timestep` is a rank-1 [1] input. Returns a status
    message. Idempotent; writes a .orig backup once. Raises FileNotFoundError if missing;
    raises _Skip (caught by the caller) when there is nothing safe to do.

    Why: optimum exports the unet with a 0-d scalar `timestep` (the graph starts with an
    Unsqueeze scalar->[1]). That runs fine under onnxruntime on a PC, but RKNN cannot accept
    a rank-0 input (rknn.build dies in fold_constant: "'numpy.float32' object is not
    iterable"). The fix is on the ONNX side: make `timestep` rank-1 [1] and delete the
    leading Unsqueeze, rewiring its consumers to the input — structurally identical to the
    vendor's hand-made ONNX. Weights live in external model.onnx_data and are never touched.
    After patching, runners must feed np.array([t]) (rank-1) and convert_onnx_to_rknn.py
    uses [1] for the timestep in input_size_list.
    """
    path = Path(unet_onnx).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    import onnx

    # load_external_data=False: keep the big weights external + untouched.
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph

    ts = next((i for i in g.input if i.name == "timestep"), None)
    if ts is None:
        raise _Skip("no input named 'timestep' — is this a unet graph?")

    dims = ts.type.tensor_type.shape.dim
    if len(dims) == 1:
        return "timestep already rank-1; nothing to do (idempotent)."
    if len(dims) != 0:
        raise _Skip(f"timestep has rank {len(dims)} (expected 0-d scalar); not patching.")

    # 1) scalar -> [1]
    del ts.type.tensor_type.shape.dim[:]
    ts.type.tensor_type.shape.dim.add().dim_value = 1

    # 2) delete the leading Unsqueeze(scalar->[1]) and rewire its output to 'timestep'
    uns = next((n for n in g.node
                if n.op_type == "Unsqueeze" and "timestep" in n.input), None)
    if uns is None:
        raise _Skip("timestep is scalar but no Unsqueeze consumes it; graph differs "
                    "from the expected optimum export; not patching.")
    out = uns.output[0]
    g.node.remove(uns)
    rewired = 0
    for node in g.node:
        for idx, name in enumerate(node.input):
            if name == out:
                node.input[idx] = "timestep"
                rewired += 1

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

    onnx.save(m, str(path))
    return f"patched timestep -> [1] (removed {uns.name}, rewired {rewired} consumer[s])."


def make_export_ready(in_path: Path) -> None:
    """Normalize a diffusers folder in place so the optimum ONNX export succeeds.

    Idempotent. Covers folders from either entry path (step 1, or a step-0 HF download).
    """
    # 1. Slow-tokenizer vocab files (vocab.json + merges.txt). Some folders ship only the
    #    fast tokenizer.json; the exporter needs the slow CLIPTokenizer's files.
    tok_dir = in_path / "tokenizer"
    if tok_dir.is_dir():
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(str(tok_dir)).save_pretrained(str(tok_dir))
        print("  normalized tokenizer: ensured vocab.json + merges.txt")

    # 2. Drop feature_extractor + safety_checker from model_index.json — not used by a
    #    txt2img export, and a common cause of load failures (missing safety_checker
    #    weights, or a torchvision-less 'CLIPImageProcessorPil' the exporter can't find).
    idx_path = in_path / "model_index.json"
    idx = json.loads(idx_path.read_text())
    changed = False
    for key in ("feature_extractor", "safety_checker"):
        if idx.get(key) not in (None, [None, None]):
            idx[key] = [None, None]
            changed = True
    if idx.get("requires_safety_checker") is not False:
        idx["requires_safety_checker"] = False
        changed = True
    if changed:
        idx_path.write_text(json.dumps(idx, indent=2))
        print("  normalized model_index.json: nulled feature_extractor + safety_checker")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="diffusers folder from step 1")
    ap.add_argument("--output", "-o", required=True,
                    help="output directory for the ONNX model")
    ap.add_argument("--task", default="latent-consistency",
                    help="optimum export task (default: latent-consistency)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()

    if not (in_path / "model_index.json").is_file():
        print(f"ERROR: {in_path} does not look like a diffusers folder "
              f"(no model_index.json). Run step 1 first.", file=sys.stderr)
        return 1

    print(f"Normalizing {in_path} for export ...")
    make_export_ready(in_path)

    if shutil.which("optimum-cli") is None:
        print("ERROR: optimum-cli not found. Install with:\n"
              '  pip install optimum\n'
              '  pip install --upgrade --upgrade-strategy eager "optimum[onnxruntime]"',
              file=sys.stderr)
        return 1

    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", str(in_path),
        "--task", args.task,
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("optimum-cli export failed.", file=sys.stderr)
        return result.returncode

    # optimum exports the unet with a 0-d scalar timestep, which RKNN's build rejects
    # (RKNN needs rank>=1 inputs). Patch it to rank-1 [1] here so the ONNX is RKNN-ready
    # straight out of step 2 (and the runners' np.array([t]) matches). See
    # patch_unet_timestep_to_rank1() above.
    unet_onnx = out_path / "unet" / "model.onnx"
    if unet_onnx.is_file():
        try:
            print("unet timestep:", patch_unet_timestep_to_rank1(str(unet_onnx)))
        except _Skip as e:
            print(f"WARNING: unet timestep not patched: {e}", file=sys.stderr)

    print(f"\nDone. ONNX model at: {out_path}")
    print("Next (optional, recommended): validate on PC before RKNN conversion —")
    print(f"  python ../runners/run_onnx_lcm.py -i {out_path} -o ./images \\")
    print("    --prompt 'a photo of an astronaut riding a horse' --num-inference-steps 4 -s 256x256")
    print("Then copy the ONNX dir to the box that runs step 3 (ONNX->RKNN).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
