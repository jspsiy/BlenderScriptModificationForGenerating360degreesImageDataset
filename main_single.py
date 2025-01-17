import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str

import argparse

def handle_found_object(
    object_path: str,
    num_renders:int,
    target_directory:str,
    only_northern_hemisphere:bool,
    render_timeout:int,
    gpu_devices: Optional[Union[int, List[int]]] = None,
) -> bool:

    args = f"--object_path '{object_path}' --num_renders {num_renders}"

    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"



        # get the command to run
        command = f"blender-3.2.2-linux-x64/blender --background --python blender_script.py -- {args} "

        # if using_gpu:
        #     command = f"export DISPLAY=:0.{gpu_i} && {command}"

        # render the object (put in dev null)
        try:
            result = subprocess.run(["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,)
            print(result.stdout.decode())  # Print the standard outp
            print(result.stderr.decode())   # Print any errors
        except Exception as e:
            print(f"An error occurred while running Blender: {e}")
            return False

        return True



def render_objects(
    input_file: str,
    output_dir: str,
    num_renders: int = 16,
    processes: Optional[int] = None,
    save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 3000,
    gpu_devices: Optional[Union[int, List[int]]] = None
) -> None:
    # Get the GPU devices to use
    parsed_gpu_devices: Union[int, List[int]] = gpu_devices if gpu_devices is not None else len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    # Set number of processes if not specified
    if processes is None:
        processes = multiprocessing.cpu_count() * 3

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the object name (without extension) to create a subdirectory in the output folder
    object_name = os.path.splitext(os.path.basename(input_file))[0]
    object_render_dir = os.path.join(output_dir, object_name)
    
    # Create the output subdirectory if it doesn't exist
    os.makedirs(object_render_dir, exist_ok=True)
    supported_extensions = [".glb", ".obj", ".fbx", ".vrm"]
    if any(input_file.endswith(ext) for ext in supported_extensions):
    # Render the object
        handle_found_object(
            object_path=input_file,
            num_renders=num_renders,
            target_directory=object_render_dir,
            only_northern_hemisphere=only_northern_hemisphere,
            render_timeout=render_timeout,
            gpu_devices=parsed_gpu_devices
        )

def main():
    parser = argparse.ArgumentParser(description="Render a 3D object with Blender.")
    
    # Define command-line arguments with default values
    parser.add_argument("input_file", type=str, help="Path to the input .glb file.")
    parser.add_argument("output_dir", type=str, help="Directory to save rendered outputs.")
    parser.add_argument("--num_renders", type=int, default=16, help="Number of renders per object.")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count() * 3, help="Number of processes to use for rendering.")
    parser.add_argument("--save_repo_format", choices=["zip", "tar", "tar.gz", "files"], default=None, help="Format for saving output.")
    parser.add_argument("--only_northern_hemisphere", action="store_true", default=False, help="Limit renders to the northern hemisphere.")
    parser.add_argument("--render_timeout", type=int, default=None, help="Timeout for each render process.")
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="GPU device IDs to use for rendering.")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the render_objects function with the parsed arguments
    render_objects(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_renders=args.num_renders,
        processes=args.processes,
        save_repo_format=args.save_repo_format,
        only_northern_hemisphere=args.only_northern_hemisphere,
        render_timeout=args.render_timeout,
        gpu_devices=args.gpu_devices
    )

if __name__ == "__main__":
    main()