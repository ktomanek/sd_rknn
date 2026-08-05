#!/usr/bin/env python3
"""Step 0 of the pipeline: download a model repo from the Hugging Face Hub.

Runs on the PC / Mac. Uses plain HTTPS (urllib) to stream files straight into the output
folder. Deliberately does NOT use huggingface_hub / snapshot_download, because that:
  - pulls every weight format at once (.bin AND .safetensors AND fp16 variants) — e.g.
    ~9 GB for a model whose diffusers weights are ~4 GB, and
  - populates a shared cache dir (~/.cache/huggingface and a .cache/ inside the output).
Here you get exactly the files you ask for, no cache, no duplicates.

The file list comes from the public HF API; select with --exclude / --include globs
(matched against repo-relative paths) or pin an explicit --files list.

Two common shapes:
  * a diffusers-format repo (model_index.json + unet/ vae/ text_encoder/ ...) -> feed
    into step 2 (convert_diffusers_to_onnx.py). Default repo: SimianLuo/LCM_Dreamshaper_v7
    (a distilled LCM model — the vendor-compatible 4-input kind).
  * a pre-converted ONNX repo -> skip to step 3 (ONNX->RKNN) on the Pi.

Single-file civitai .safetensors aren't downloaded here — grab those from the civitai UI
and start at step 1 (convert_safetensors_to_diffusers.py).

Examples:
    # default: LCM_Dreamshaper_v7 diffusers weights only (drops bundled onnx, the
    # redundant single-file checkpoint, safety_checker, docs/images)
    python download_model.py -o ~/models/lcm_dreamshaper_v7_diffusers

    # grab a pre-converted ONNX repo instead
    python download_model.py -r TheyCallMeHex/LCM-Dreamshaper-V7-ONNX \
        -o ~/models/lcm_dreamshaper_v7_onnx --exclude '*.md' '*.png'

    # pin an exact file list
    python download_model.py -o ~/out --files model_index.json unet/config.json
"""
import argparse
import fnmatch
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_REPO = "SimianLuo/LCM_Dreamshaper_v7"

# For the default (diffusers) case: skip bundled ONNX (we convert ourselves), the
# redundant top-level single-file checkpoint, safety_checker (nulled downstream), and
# non-model cruft. Override with --exclude to replace this list.
DEFAULT_EXCLUDES = [
    "*.onnx", "*.onnx_data",          # we run our own step-2 ONNX export
    "*_4k.safetensors",               # redundant merged single-file (~4 GB)
    "safety_checker/*",               # not used by txt2img; nulled in model_index.json
    "*.png", "*.jpg", "*.jpeg",       # teaser images
    "*.md", ".gitattributes",         # docs
    "inference.py", "lcm_pipeline.py", "lcm_scheduler.py",  # repo's own demo scripts
]


def list_repo_files(repo: str, revision: str) -> list:
    url = f"https://huggingface.co/api/models/{repo}"
    if revision:
        url += f"/revision/{revision}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return sorted(f["rfilename"] for f in data.get("siblings", []))


def select(files: list, includes: list, excludes: list) -> list:
    # Many diffusers repos ALSO ship a redundant top-level single-file checkpoint
    # (e.g. DreamShaper8_LCM.safetensors) next to the unpacked unet/ vae/ text_encoder/
    # weights. For a diffusers repo we only need the subfolder weights, so skip root-level
    # single-file checkpoints (they'd otherwise double the download).
    is_diffusers = "model_index.json" in files
    weight_ext = (".safetensors", ".ckpt", ".bin", ".pt", ".pth")
    chosen = []
    for f in files:
        if includes and not any(fnmatch.fnmatch(f, p) for p in includes):
            continue
        if any(fnmatch.fnmatch(f, p) for p in excludes):
            continue
        if is_diffusers and "/" not in f and f.endswith(weight_ext):
            print(f"  (skipping redundant top-level checkpoint: {f})")
            continue
        chosen.append(f)
    return chosen


def download_one(repo: str, revision: str, rel: str, out_path: Path) -> int:
    rev = revision or "main"
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{rel}"
    dest = out_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tty = sys.stdout.isatty()  # only draw the live progress bar on a real terminal
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if tty and total:
                    pct = 100 * done / total
                    print(f"\r  {rel}  {done>>20}/{total>>20} MiB ({pct:4.1f}%)",
                          end="", flush=True)
    tmp.rename(dest)
    print(f"\r  {rel}  {done>>20} MiB  done" + (" " * 20 if tty else ""))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", "-r", default=DEFAULT_REPO,
                    help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--output", "-o", required=True,
                    help="Local directory to download into (must be empty/new)")
    ap.add_argument("--revision", default=None, help="Optional branch / tag / commit")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Only download paths matching these globs (default: all)")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help=f"Skip paths matching these globs (default: {DEFAULT_EXCLUDES})")
    ap.add_argument("--files", nargs="*", default=None,
                    help="Explicit repo-relative paths; bypasses include/exclude")
    args = ap.parse_args()

    out_path = Path(args.output).expanduser()
    if out_path.exists() and any(out_path.iterdir()):
        print(f"ERROR: output dir exists and is not empty: {out_path}", file=sys.stderr)
        return 1

    excludes = DEFAULT_EXCLUDES if args.exclude is None else args.exclude

    if args.files:
        chosen = args.files
    else:
        print(f"Listing files in {args.repo}"
              + (f"@{args.revision}" if args.revision else "") + " ...")
        all_files = list_repo_files(args.repo, args.revision)
        chosen = select(all_files, args.include or [], excludes)

    if not chosen:
        print("ERROR: no files selected (check --include/--exclude).", file=sys.stderr)
        return 1

    print(f"Downloading {len(chosen)} file(s) -> {out_path}")
    total_bytes = 0
    for rel in chosen:
        total_bytes += download_one(args.repo, args.revision, rel, out_path)
    print(f"Done. {total_bytes>>20} MiB into {out_path}")

    if (out_path / "model_index.json").is_file():
        print(f"Looks like a diffusers folder. Next: "
              f"convert_diffusers_to_onnx.py -i {out_path} -o <onnx_dir>")
    elif any(out_path.rglob("*.onnx")):
        print("Looks like a pre-converted ONNX model. Next: scp to the Pi, run step 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
