"""Phrase → (intent, params) lookup populated by operator confirmations.

Stored as a flat JSON file (`scripts/data/learned_overrides.json`)
keyed by lowercased + whitespace-collapsed transcription. Each entry
remembers the intent the operator confirmed and the slot params they
edited, so future utterances of the same phrase short-circuit straight
to that result — no model query, no confirmation modal.

The file is updated atomically on every feedback confirmation that
includes slot edits. On the read side we mtime-cache the parsed dict
so `lookup()` is effectively free for the hot path.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Same root resolution as training_orchestrator: parents[2] from this file
# = repo root (backend/services/learned_overrides.py → repo).
ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = ROOT / "scripts" / "data" / "learned_overrides.json"

_cache: dict[str, dict[str, Any]] = {}
_cache_mtime: float = -1.0


def _normalize(text: str) -> str:
    """Same canonical form used for keys + lookups: NFKC + collapse
    whitespace + lowercase + strip. Trailing punctuation is also stripped
    so «крутись.» / «Крутись!» / «крутись» all hit the same bucket."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" .,!?;:")


def _load_if_changed() -> None:
    global _cache, _cache_mtime
    try:
        mtime = OVERRIDES_PATH.stat().st_mtime
    except FileNotFoundError:
        if _cache_mtime != -1.0:
            _cache = {}
            _cache_mtime = -1.0
        return
    if mtime == _cache_mtime:
        return
    try:
        raw = OVERRIDES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            _cache = {str(k): v for k, v in data.items() if isinstance(v, dict)}
            _cache_mtime = mtime
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("learned_overrides parse failed (%s) — ignoring file", exc)


def lookup(text: str) -> Optional[dict[str, Any]]:
    """Return `{"intent": ..., "params": {...}}` if `text` matches a
    learned override (after canonicalisation), else None."""
    key = _normalize(text)
    if not key:
        return None
    _load_if_changed()
    entry = _cache.get(key)
    if not isinstance(entry, dict):
        return None
    intent = entry.get("intent")
    params = entry.get("params") or {}
    if not isinstance(intent, str) or not isinstance(params, dict):
        return None
    return {"intent": intent, "params": params}


def upsert(text: str, intent: str, params: dict[str, Any]) -> None:
    """Persist a phrase→(intent, params) mapping. Atomic temp+rename
    write, then bump cache."""
    key = _normalize(text)
    if not key or not intent:
        return
    _load_if_changed()
    cleaned_params = {
        k: v
        for k, v in params.items()
        if not k.startswith("_") and k != "original_text"
    }
    _cache[key] = {
        "intent": intent,
        "params": cleaned_params,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDES_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, OVERRIDES_PATH)
    # Refresh mtime so next read sees this write as the source of truth.
    try:
        global _cache_mtime
        _cache_mtime = OVERRIDES_PATH.stat().st_mtime
    except OSError:
        pass
    _log.info("learned override upserted: %r → %s %s", key, intent, cleaned_params)


def all_overrides() -> dict[str, dict[str, Any]]:
    """Read-only view of the current cache, for diagnostics / scripts."""
    _load_if_changed()
    return dict(_cache)
