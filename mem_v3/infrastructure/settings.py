from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
