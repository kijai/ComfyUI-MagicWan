import torch
import os

def debug_enabled():
    """Check if debugging is enabled via environment variable"""
    return os.environ.get("MAGICWAN_DEBUG", "0") == "1"

def debug_print(*args, **kwargs):
    """Print debug messages if debugging is enabled"""
    if debug_enabled():
        print("[MagicWan Debug]", *args, **kwargs)

def debug_tensor(name, tensor):
    """Print tensor info for debugging"""
    if debug_enabled():
        if tensor is None:
            print(f"[MagicWan Debug] {name} is None")
        elif isinstance(tensor, torch.Tensor):
            print(f"[MagicWan Debug] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            print(f"  - min/max/mean: {tensor.min().item():.4f}/{tensor.max().item():.4f}/{tensor.mean().item():.4f}")
            if tensor.numel() < 10:
                print(f"  - values: {tensor.tolist()}")
            elif tensor.dim() <= 2:
                print(f"  - first few values: {tensor.flatten()[:5].tolist()}")
        else:
            print(f"[MagicWan Debug] {name}: {type(tensor)} (not a tensor)")
