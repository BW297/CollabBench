from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def truncate(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def extract_json_obj(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in response.")
    return json.loads(m.group(0))


def plan_to_high_level_action(plan: str) -> str:
    plan = (plan or "").strip()
    if not plan:
        return ""
    m = re.match(r"^\s*(\[[^\]]+\])\s*(.*)$", plan)
    if not m:
        return plan
    return f"{m.group(1)} {m.group(2)}".strip()

