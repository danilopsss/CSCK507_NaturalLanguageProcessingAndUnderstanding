"""
Device configuration utility.
Provides consistent device selection across all model files.
"""

import torch
import os


def get_device(force_cpu=False, force_cuda=False, force_mps=False):
    """
    Get the appropriate device for training/inference.

    Args:
        force_cpu (bool): Force CPU usage
        force_cuda (bool): Force CUDA usage (will raise error if not available)
        force_mps (bool): Force MPS usage (will raise error if not available)

    Returns:
        torch.device: The selected device
    """
    if force_cpu:
        device = torch.device("cpu")
        print("Device forced to CPU")
    elif force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA forced but not available!")
        device = torch.device("cuda")
        print("Device forced to CUDA")
    elif force_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS forced but not available!")
        device = torch.device("mps")
        print("Device forced to MPS (Apple Silicon)")
    else:
        # Check environment variable first
        device_env = os.getenv("TORCH_DEVICE", "").lower()
        if device_env == "cpu":
            device = torch.device("cpu")
            print("Device set to CPU via environment variable")
        elif device_env == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Device set to CUDA via environment variable")
            else:
                print(
                    "CUDA requested via environment but not available, falling back to auto-detect"
                )
                device = _auto_detect_device()
        elif device_env == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Device set to MPS via environment variable")
            else:
                print(
                    "MPS requested via environment but not available, falling back to auto-detect"
                )
                device = _auto_detect_device()
        else:
            # Auto-detect
            device = _auto_detect_device()

    _print_device_info(device)
    return device


def _auto_detect_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Auto-detected device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Auto-detected device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Auto-detected device: CPU")
    return device


def _print_device_info(device):
    """Print information about the selected device"""
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB"
        )
    elif device.type == "mps":
        print("Using Apple Silicon GPU acceleration")
        # MPS doesn't have direct memory query, but we can check if it's built
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device.type == "cpu":
        print("Using CPU for computation")


# Default device for the project
device = get_device()
