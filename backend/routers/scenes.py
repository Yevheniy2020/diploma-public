import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from config import settings
from models import SceneInfo

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["scenes"])


@router.get("/scenes", response_model=list[SceneInfo])
def list_scenes() -> list[SceneInfo]:
    manifest = Path(settings.scenes_dir) / "scenes.json"
    if not manifest.exists():
        _log.info("scenes manifest missing at %s — returning empty list", manifest)
        return []
    try:
        raw: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _log.error("scenes.json invalid JSON: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"scenes.json malformed: {e}",
        )
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="scenes.json must be a JSON array",
        )
    return [SceneInfo.model_validate(item) for item in raw]
