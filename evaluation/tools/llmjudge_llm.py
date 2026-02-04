from __future__ import annotations

import json
from typing import Any, Dict, Optional

from llmjudge_utils import extract_json_obj, safe_int


def openai_client(api_key: str, api_base: str):
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key, base_url=api_base)
    except Exception:
        # Fallback: send OpenAI-compatible HTTP requests directly.
        return {"_transport": "requests", "api_key": api_key, "api_base": api_base}


def chat_completion(client, *, model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # openai>=1.0 client
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # OpenAI-compatible HTTP via requests
    if isinstance(client, dict) and client.get("_transport") == "requests":
        import requests

        api_base = str(client.get("api_base", "")).rstrip("/")
        api_key = str(client.get("api_key", ""))
        url = f"{api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    raise RuntimeError(f"Unsupported client type for chat completion: {type(client)}")


def judge_one_dimension(
    *,
    client,
    model: str,
    dimension_prompt: str,
    window_text: str,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            out = chat_completion(
                client,
                model=model,
                system_prompt=dimension_prompt,
                user_prompt=window_text,
                max_tokens=max_tokens,
            )
            obj = extract_json_obj(out)
            score = safe_int(obj.get("score"), -1)
            if score < 0 or score > 5:
                raise ValueError(f"Invalid score: {obj.get('score')}")
            v = obj.get("violations", [])
            if not isinstance(v, list):
                raise ValueError("`violations` must be a list.")
            return obj
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"LLM judge failed after retries={retries}: {last_err}")
