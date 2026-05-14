from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _to_yaml_serializable(obj: Any) -> Any:
    """Recursively coerce values so PyYAML safe_dump never hits unknown types (e.g. TorchVersion)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_serializable(v) for v in obj]
    return str(obj)


def save_yaml(path: str | Path, data: Any) -> None:
    safe = _to_yaml_serializable(data)
    Path(path).write_text(yaml.safe_dump(safe, sort_keys=False), encoding="utf-8")


def resolve_device(device: str) -> torch.device:
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def env_info() -> dict[str, Any]:
    cuda_v = torch.version.cuda
    return {
        "python": str(os.sys.version),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": str(cuda_v) if cuda_v is not None else None,
        "device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }

