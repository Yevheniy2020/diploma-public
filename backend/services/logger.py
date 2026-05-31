import json
from typing import Any, Optional

from sqlmodel import Session

from tables import CommandLogTable


async def log_command(
    db: Session,
    map_id: Optional[int],
    audio_path: Optional[str],
    intent: str,
    params: dict[str, Any],
    success: bool,
    error: Optional[str],
    latency_ms: Optional[int],
    transcription: Optional[str] = None,
) -> CommandLogTable:
    row = CommandLogTable(
        map_id=map_id,
        audio_path=audio_path,
        transcription=transcription,
        intent=intent,
        params_json=json.dumps(params, ensure_ascii=False),
        success=success,
        error_message=error,
        latency_ms=latency_ms,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
