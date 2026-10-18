"""Shared DA3 loading for the bench and the server.

DA3 imports pycolmap (COLMAP export only; unused here) and pycolmap bundles a second libomp, which
aborts the process next to torch's ("OMP: Error #15"). Stub it before DA3 is imported instead of
setting the unsafe KMP_DUPLICATE_LIB_OK.
"""
import sys
import types

import torch

sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))


def pick_device() -> str:
    """Apple GPU (MPS) if present, else CUDA, else CPU (works, but is slow)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = "depth-anything/DA3-BASE", device: str | None = None):
    from depth_anything_3.api import DepthAnything3

    device = device or pick_device()
    return DepthAnything3.from_pretrained(name).to(device).eval(), device
