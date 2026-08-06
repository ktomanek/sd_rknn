# Stable Diffusion 1.5 LCM on RKNN 3588

Note: based on code from https://huggingface.co/thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2/tree/main

Text-to-image with an SD 1.5 LCM model, running on the RK3588 NPU. The pipeline
converts a SD 1.5 LCM safetensor checkpoint to diffusers format and then to ONNX, then ONNX to RKNN.

## Layout

- `sd_conversion/` — model conversion scripts
- `runners/` — inference: `run_onnx_lcm.py` (PC/CPU), `run_rknn_lcm.py` (board/NPU)

## Setup

```bash
# On the PC (conversion + ONNX inference)
pip install -r requirements-convert_to_onnx.txt

# On the board (RKNN conversion + NPU inference)
pip install -r requirements-convert_to_rknn.txt
```

`rknn-toolkit2` and `rknn-toolkit-lite2` (>= 2.3.0) are not on PyPI — install the
aarch64 wheels manually from Rockchip's release repo. Versions < 2.3.0 have a
fp16 precision bug that produces garbage images.

## Convert a model

```bash
# 1. Download a base model (default: SimianLuo/LCM_Dreamshaper_v7)
python sd_conversion/download_model.py -o ./model_src

# --- if you start from a single .safetensors/.ckpt instead of a diffusers folder ---
python sd_conversion/convert_safetensors_to_diffusers.py -i model.safetensors -o ./model_diffusers

# 2. diffusers -> ONNX
python sd_conversion/convert_diffusers_to_onnx.py -i ./model_src -o ./model_onnx

# 3. ONNX -> RKNN (run on the RK3588 board)
python sd_conversion/convert_onnx_to_rknn.py -m ./model_onnx \
    -c "text_encoder,unet,vae_decoder" -r 256x256
```

## Run inference

```bash
# On PC (ONNX / CPU)
python runners/run_onnx_lcm.py --prompt "a cat astronaut" -i ./model_onnx -o ./images

# On board (RKNN / NPU)
python runners/run_rknn_lcm.py --prompt "a cat astronaut" -i ./model_rknn -o ./images
```

Common flags (both runners): `-s 256x256` (size), `--num-inference-steps 4`,
`--guidance-scale 7.5`, `--seed 93`, `--vae-dir` (alternate VAE, e.g. TAESD),
`--output-file` (exact output path).

## Optional faster VAE decode with TAESD

We can use another VAE decoder to cut SD inference time. [TAESD](https://huggingface.co/madebyollin/taesd) 
is such a tiny distilled SD1.5 VAE decoder and can be used as a drop-in replacement for the `vae_decoder` component, with
negligible quality loss at 256px (tested). One TAESD decoder works across all SD1.5 LCM models
(shared latent space), so you build it once.

Export to onnx first:

```
python sd_conversion/export_taesd.py -o ~models/taesd -s 256x256
```

Convert to RKNN:
```
python sd_conversion/convert_onnx_to_rknn.py -m ~models/taesd -c vae_decoder -r 256x256
```

Use as drop in replacement:
```
python runners/run_rknn_lcm.py --prompt "a cat astronaut" \
    -i ./model_rknn --vae-dir ~/models/vae_decoder -s 256x256
```
