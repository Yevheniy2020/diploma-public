"""Mock robot for demoing the real-robot adapter without hardware.

Implements the contract from docs/hardware/esp32.md:
  - Listens on http://0.0.0.0:<port>/cmd for waypoints / stop
  - Posts pose to <backend>/api/robot/pose at ~10 Hz
  - Simulates diff-drive kinematics (same model as the browser simulator)

Run with the backend already up:

    .venv/bin/python scripts/mock_robot.py \\
        --backend http://localhost:8000 \\
        --listen-port 8080 \\
        --start 1.5,1.5,0.0

Then set the backend's .env to ROBOT_MODE=http, ROBOT_URL=http://localhost:8080
and restart it. Click on the map (or speak a command) — the mock robot
moves and the backend's /api/robot/status flips to connected.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import Config, Server

_log = logging.getLogger("mock_robot")

# Mirror of frontend/src/sim/kinematics.ts constants. Keep these in
# sync if the simulator's behaviour is retuned.
LINEAR_SPEED = 0.5
ANGULAR_SPEED = 1.5
ARRIVAL_THRESHOLD = 0.05
ANGLE_THRESHOLD = 0.05
FORWARD_GATE = 0.7
DT = 0.1  # 10 Hz physics + pose report


@dataclass
class Pose:
    x: float
    y: float
    theta: float


class Command(BaseModel):
    waypoints: list[list[float]] | None = None
    stop: bool = False


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class MockRobot:
    def __init__(self, start: Pose, backend_url: str) -> None:
        self.pose = start
        self.backend_url = backend_url.rstrip("/")
        self.queue: list[tuple[float, float]] = []
        self.client = httpx.AsyncClient(timeout=2.0)

    def set_path(self, waypoints: list[list[float]]) -> None:
        self.queue = [(float(p[0]), float(p[1])) for p in waypoints]
        _log.info("queued %d waypoints", len(self.queue))

    def stop(self) -> None:
        self.queue = []
        _log.info("STOP")

    def step(self, dt: float) -> None:
        # Same loop as the JS kinematics: rotate-toward-target first, only
        # creep forward when the heading is within FORWARD_GATE. Pops the
        # waypoint when within ARRIVAL_THRESHOLD.
        if not self.queue:
            return
        gx, gy = self.queue[0]
        dx = gx - self.pose.x
        dy = gy - self.pose.y
        dist = math.hypot(dx, dy)
        if dist < ARRIVAL_THRESHOLD:
            self.queue.pop(0)
            return
        target_theta = math.atan2(dy, dx)
        err = wrap_angle(target_theta - self.pose.theta)
        step_a = max(-ANGULAR_SPEED * dt, min(ANGULAR_SPEED * dt, err))
        self.pose.theta = wrap_angle(self.pose.theta + step_a)
        if abs(err) < FORWARD_GATE:
            step_d = min(LINEAR_SPEED * dt, dist)
            self.pose.x += step_d * math.cos(self.pose.theta)
            self.pose.y += step_d * math.sin(self.pose.theta)

    async def report_loop(self) -> None:
        # 10 Hz: step physics + push pose to backend.
        url = f"{self.backend_url}/api/robot/pose"
        while True:
            self.step(DT)
            try:
                await self.client.post(
                    url, json={"x": self.pose.x, "y": self.pose.y, "theta": self.pose.theta}
                )
            except Exception as exc:  # noqa: BLE001 — backend may be transiently down
                _log.debug("pose POST failed: %s", exc)
            await asyncio.sleep(DT)


def build_app(robot: MockRobot) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(robot.report_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await robot.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post("/cmd", status_code=204)
    async def cmd(c: Command) -> None:
        if c.stop:
            robot.stop()
            return
        if c.waypoints:
            robot.set_path(c.waypoints)

    return app


def parse_pose(s: str) -> Pose:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,theta")
    return Pose(float(parts[0]), float(parts[1]), float(parts[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock robot for diploma-robot HAL")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument(
        "--start",
        type=parse_pose,
        default=Pose(1.5, 1.5, 0.0),
        help="x,y,theta (meters, radians)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    robot = MockRobot(args.start, args.backend)
    app = build_app(robot)
    config = Config(app, host=args.listen_host, port=args.listen_port, log_level="warning")
    Server(config).run()


if __name__ == "__main__":
    main()
