#!/usr/bin/env python
# coding: utf-8

from typing import List
from rknn.api import RKNN
from math import exp
from sys import exit
import argparse
import json
import os


def convert_pipeline_component(onnx_path: str, resolution_list: List[List[int]], target_platform: str = 'rk3588', multi_core: bool = False):
    print(f'Converting {onnx_path} to RKNN model')
    print(f'with target platform {target_platform}')
    print(f'with resolutions:')
    for res in resolution_list:
        print(f'- {res[0]}x{res[1]}')
    use_dynamic_shape = False
    if(len(resolution_list) > 1):
        print("Warning: RKNN dynamic shape support is probably broken, may throw errors")
        use_dynamic_shape = True

    batch_size = 1
    LATENT_RESIZE_FACTOR = 8
    # build shape list
    if "text_encoder" in onnx_path:
        input_size_list = [[[1,77]]]
        inputs=['input_ids']
        use_dynamic_shape = False
    elif "unet" in onnx_path:
        # batch_size = 2  # for classifier free guidance # broken for rknn python api

        # 4-input distilled LCM vs 3-input LCM-LoRA-merged: detect via the unet config's
        # time_cond_proj_dim (256 -> has the guidance-scale `timestep_cond`; None -> 3-input,
        # no w-embedding). This lets the same converter handle both model kinds.
        cfg_path = os.path.join(os.path.dirname(onnx_path), "config.json")
        time_cond_proj_dim = None
        if os.path.exists(cfg_path):
            time_cond_proj_dim = json.load(open(cfg_path)).get("time_cond_proj_dim")
        has_timestep_cond = time_cond_proj_dim is not None
        print(f'    unet: {"4-input (distilled LCM)" if has_timestep_cond else "3-input (non-distilled LCM)"}'
              f' (time_cond_proj_dim={time_cond_proj_dim})')

        input_size_list = []
        for res in resolution_list:
            shapes = [
                [1, 4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR],
                # timestep: rank-1. RKNN can't take a 0-d scalar, and our optimum ONNX
                # originally exported timestep as a scalar, so the unet graph is patched
                # (patch_unet_timestep_to_rank1() in convert_diffusers_to_onnx.py) to
                # accept [1]. See the insights doc.
                [1],
                [1, 77, 768],
            ]
            if has_timestep_cond:
                shapes.append([1, time_cond_proj_dim])  # guidance-scale w-embedding
            input_size_list.append(shapes)
        inputs = ['sample', 'timestep', 'encoder_hidden_states']
        if has_timestep_cond:
            inputs.append('timestep_cond')
    elif "vae_decoder" in onnx_path:
        input_size_list = []
        for res in resolution_list:
            input_size_list.append(
                [[1,4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR]]
            )
        inputs=['latent_sample']
    else:
        print("Unknown component: ", onnx_path)
        exit(1)

    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # single_core_mode=True pins the model to 1 of the RK3588's 3 NPU cores (the vendor
    # default). --multi-core sets it False to let the compiler use multiple cores. NOTE:
    # the vendor annotated run_rknn_lcm.py with "Multi-core will cause kernel crash" for
    # this SD graph, and RK3588 multi-core mainly helps throughput (parallel requests), not
    # single-image latency — so treat multi-core as an experiment, not a guaranteed win.
    print(f'    core mode: {"multi-core" if multi_core else "single-core"}')
    rknn.config(target_platform='rk3588', optimization_level=3, single_core_mode=not multi_core,
                dynamic_input= input_size_list if use_dynamic_shape else None)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_path,
                         inputs=None if use_dynamic_shape else inputs,
                         input_size_list= None if use_dynamic_shape else input_size_list[0])   
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, rknn_batch_size=batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    #export
    print('--> Export RKNN model')
    ret = rknn.export_rknn(onnx_path.replace('.onnx', '.rknn'))
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()
    print('RKNN model is converted successfully!')


def parse_resolution_list(resolution: str) -> List[List[int]]:
    resolution_pairs = resolution.split(',')
    parsed_resolutions = []
    for pair in resolution_pairs:
        width, height = map(int, pair.split('x'))
        parsed_resolutions.append([width, height])
    
    return parsed_resolutions
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Stable Diffusion ONNX models to RKNN models')
    parser.add_argument('-m','--model-dir', type=str, help='Directory containing the Stable Diffusion ONNX models', required=True)
    parser.add_argument('-c','--components', type=str, help='Name of the components to convert, e.g. "text_encoder,unet,vae_decoder"', default='text_encoder, unet, vae_decoder')
    parser.add_argument('-r','--resolutions', type=str, help='Comma-separated list of resolutions for the model, e.g. "256x256,512x512"', default='256x256')
    parser.add_argument('--target_platform', type=str, help='Target platform for the RKNN model, default is "rk3588"', default='rk3588')
    parser.add_argument('--multi-core', dest='multi_core', action='store_true',
                        help='Build for multiple NPU cores (single_core_mode=False). EXPERIMENTAL: '
                             'the vendor noted multi-core crashes this SD graph, and it mainly helps '
                             'throughput, not single-image latency. Default: single-core.')
    args = parser.parse_args()

    components = args.components.split(',')

    for component in components:
        onnx_path = f'{args.model_dir}/{component.strip()}/model.onnx'
        resolution_list = parse_resolution_list(args.resolutions)
        if(len(resolution_list) == 0):
            print("Error: No resolutions specified")
            exit(1)

        convert_pipeline_component(onnx_path, resolution_list, args.target_platform, args.multi_core)
