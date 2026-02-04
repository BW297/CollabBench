from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_log(pik_path: Path) -> Optional[Dict[str, Any]]:
    if not pik_path.exists():
        return None
    try:
        if pik_path.stat().st_size == 0:
            return None
    except Exception:
        return None
    try:
        data = pickle.loads(pik_path.read_bytes())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def iter_log_paths(input_path: Path, max_files: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    candidates: List[Path] = []
    candidates.extend(sorted(input_path.glob("logs_agent_*.pik")))
    if not candidates:
        candidates.extend(sorted(input_path.glob("*/logs_agent_*.pik")))
    if not candidates:
        candidates.extend(sorted(input_path.rglob("logs_agent_*.pik")))

    if max_files is not None:
        candidates = candidates[:max_files]
    return candidates
