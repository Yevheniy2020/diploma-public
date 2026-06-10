import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from models import RobotPose
from services.robot_adapter import RobotAdapter, get_adapter

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/robot", tags=["robot"])


class RobotStatusResponse(BaseModel):
    mode: str
    connected: bool
    last_pose: Optional[RobotPose] = None
    age_s: Optional[float] = None
    last_error: Optional[str] = None


@router.post("/pose", status_code=204)
async def report_pose(pose: RobotPose) -> None:
    adapter: RobotAdapter = get_adapter()
    adapter.update_pose(pose)


@router.get("/status", response_model=RobotStatusResponse)
async def status() -> RobotStatusResponse:
    adapter = get_adapter()
    s = adapter.status()
    return RobotStatusResponse(
        mode=s.mode,
        connected=s.connected,
        last_pose=s.last_pose,
        age_s=s.age_s,
        last_error=s.last_error,
    )


@router.post("/stop", status_code=204)
async def stop() -> None:
    adapter = get_adapter()
    await adapter.stop()
