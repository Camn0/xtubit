from __future__ import annotations
import importlib.util
import platform
import sys

OPTIONAL = [
    "rdkit", "torch", "torch_geometric", "faenet", "botorch",
    "streamlit", "py3Dmol", "vina", "meeko", "ortools"
]
print("Python:", sys.version)
print("Platform:", platform.platform())
for name in OPTIONAL:
    print(f"{name:18s}", "OK" if importlib.util.find_spec(name) else "MISSING")
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", repr(exc))
