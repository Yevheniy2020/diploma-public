from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx

from config import settings
from models import Point2D, RobotPose

_log = logging.getLogger(__name__)


@dataclass
class RobotStatus:
    mode: str
    connected: bool
    last_pose: Optional[RobotPose]
    age_s: Optional[float]
    last_error: Optional[str]


class RobotAdapter(ABC):

    mode: str

    @abstractmethod
    async def send_path(self, waypoints: list[Point2D]) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def update_pose(self, pose: RobotPose) -> None: ...

    @abstractmethod
    def status(self) -> RobotStatus: ...


class SimRobotAdapter(RobotAdapter):

    mode = "sim"

    def __init__(self) -> None:
        self._last_pose: Optional[RobotPose] = None
        self._last_ts: Optional[float] = None

    async def send_path(self, waypoints: list[Point2D]) -> None:
        return None

    async def stop(self) -> None:
        return None

    def update_pose(self, pose: RobotPose) -> None:
        self._last_pose = pose
        self._last_ts = time.monotonic()

    def status(self) -> RobotStatus:
        age = (time.monotonic() - self._last_ts) if self._last_ts else None
        return RobotStatus(
            mode=self.mode,
            connected=True,
            last_pose=self._last_pose,
            age_s=age,
            last_error=None,
        )


class HttpEsp32Adapter(RobotAdapter):

    mode = "http"
    _STALE_FACTOR = 3.0

    def __init__(self, url: str, timeout_s: float) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout_s
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._last_pose: Optional[RobotPose] = None
        self._last_ts: Optional[float] = None
        self._last_error: Optional[str] = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send_path(self, waypoints: list[Point2D]) -> None:
        payload = {"waypoints": [[wp.x, wp.y] for wp in waypoints]}
        await self._post("/cmd", payload)

    async def stop(self) -> None:
        await self._post("/cmd", {"stop": True})

    async def _post(self, path: str, payload: dict) -> None:
        try:
            r = await self._client.post(f"{self._url}{path}", json=payload)
            r.raise_for_status()
            self._last_error = None
        except Exception as exc:  # noqa: BLE001 — any wire failure must not crash voice flow
            self._last_error = str(exc)
            _log.warning("robot %s POST %s failed: %s", self._url, path, exc)

    def update_pose(self, pose: RobotPose) -> None:
        self._last_pose = pose
        self._last_ts = time.monotonic()

    def status(self) -> RobotStatus:
        age = (time.monotonic() - self._last_ts) if self._last_ts else None
        connected = age is not None and age < self._STALE_FACTOR * self._timeout
        return RobotStatus(
            mode=self.mode,
            connected=connected,
            last_pose=self._last_pose,
            age_s=age,
            last_error=self._last_error,
        )


_adapter: Optional[RobotAdapter] = None


def init_adapter() -> RobotAdapter:
    global _adapter
    if settings.robot_mode == "http":
        if not settings.robot_url:
            raise RuntimeError("ROBOT_MODE=http requires ROBOT_URL to be set")
        _adapter = HttpEsp32Adapter(settings.robot_url, settings.robot_timeout_s)
        _log.info("robot adapter: http → %s", settings.robot_url)
    else:
        _adapter = SimRobotAdapter()
        _log.info("robot adapter: sim")
    return _adapter


def get_adapter() -> RobotAdapter:
    if _adapter is None:
        return init_adapter()
    return _adapter
