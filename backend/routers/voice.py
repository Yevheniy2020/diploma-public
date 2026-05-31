import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiofiles
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from config import settings
from db import get_session
from models import (
    CorrectionRow,
    EditCorrectionRequest,
    FeedbackStats,
    Intent,
    Point2D,
    RetrainStatus,
    RobotPose,
    VoiceFeedbackRequest,
    VoiceFeedbackResponse,
    VoiceFollowUp,
    VoiceResponse,
)
from services import training_orchestrator, voice_intent
from services.geometry import (
    cell_to_meters,
    nearest_free_cell_in_polygon,
    polygon_centroid,
)
from services.grid import decode_grid
from services.inflation import inflate
from services.logger import log_command
from services.planner import plan as planner_plan
from services.robot_adapter import get_adapter
from tables import CommandCorrectionTable, CommandLogTable, MapTable, SpaceTable

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["voice"])


_MIME_TO_EXT = {
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
}


def _ext_for_mime(mime: str) -> str:
    base = mime.split(";")[0].strip().lower()
    return _MIME_TO_EXT.get(base) or _MIME_TO_EXT.get(mime.lower(), "bin")


async def _save_upload(audio_bytes: bytes, mime: str) -> str:
    ext = _ext_for_mime(mime)
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{int(time.time())}_{uuid.uuid4().hex}.{ext}"
    fpath = Path(settings.uploads_dir) / fname
    async with aiofiles.open(fpath, "wb") as f:
        await f.write(audio_bytes)
    return str(fpath)


def _grid_for(map_row: MapTable) -> np.ndarray:
    if map_row.grid_data is None:
        return np.zeros((map_row.height_cells, map_row.width_cells), dtype=np.uint8)
    return decode_grid(map_row.grid_data, map_row.width_cells, map_row.height_cells)


def _find_space(db: Session, map_id: int, name: str) -> Optional[SpaceTable]:
    return db.exec(
        select(SpaceTable).where(SpaceTable.map_id == map_id, SpaceTable.name == name)
    ).first()


def _find_home_space(db: Session, map_id: int) -> Optional[SpaceTable]:
    """Return the home-tagged space of this map, or None if none flagged
    (RETURN_HOME falls back to the first space in that case)."""
    home = db.exec(
        select(SpaceTable).where(SpaceTable.map_id == map_id, SpaceTable.is_home == True)  # noqa: E712
    ).first()
    if home is not None:
        return home
    return db.exec(
        select(SpaceTable).where(SpaceTable.map_id == map_id).order_by(SpaceTable.id)
    ).first()


def _parse_polygon(s: str) -> Optional[list[tuple[float, float]]]:
    """Parse a space's vertices_json into a polygon. Returns None on
    malformed input — caller treats it as `no_path` rather than 500."""
    try:
        raw = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    out: list[tuple[float, float]] = []
    for v in raw:
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            return None
        x, y = v
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return None
        out.append((float(x), float(y)))
    return out


def _navigate_to_space(
    db: Session,
    map_row: MapTable,
    pose: RobotPose,
    space: SpaceTable,
) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Plan A* from the robot to the closest free cell inside the space's
    polygon. Shared by NAVIGATE and RETURN_HOME."""
    grid = _grid_for(map_row)
    polygon = _parse_polygon(space.vertices_json)
    if polygon is None:
        return False, {"target": space.name, "space_id": space.id}, (
            f"space '{space.name}' has invalid polygon"
        )
    inflated = inflate(grid)
    cell = nearest_free_cell_in_polygon(
        inflated_grid=inflated,
        polygon_m=polygon,
        robot_m=(pose.x, pose.y),
        cell_size=map_row.cell_size_m,
    )
    if cell is None:
        return False, {
            "target": space.name,
            "space_id": space.id,
            "reason": "no_path",
        }, "no_path"
    goal_m = cell_to_meters(
        cell[0], cell[1], map_row.cell_size_m, grid.shape[0]
    )
    waypoints = planner_plan(
        grid=grid,
        start_m=(pose.x, pose.y),
        goal_m=goal_m,
        cell_size=map_row.cell_size_m,
    )
    meta: dict[str, Any] = {
        "target": space.name,
        "space_id": space.id,
        "waypoints": [{"x": x, "y": y} for x, y in waypoints],
    }
    if not waypoints:
        meta["reason"] = "no_path"
        return False, meta, "no_path"
    return True, meta, None


def _navigate(
    db: Session, map_row: MapTable, pose: RobotPose, target_name: str
) -> tuple[bool, dict[str, Any], Optional[str]]:
    name = target_name.strip().lower()
    space = _find_space(db, map_row.id, name)  # type: ignore[arg-type]
    if space is None:
        return False, {"target": target_name}, f"space '{target_name}' not found"
    return _navigate_to_space(db, map_row, pose, space)


_DIRECTION_TO_UK = {
    "forward": "вперед",
    "backward": "назад",
    "left": "ліворуч",
    "right": "праворуч",
}

# Offset direction → unit vector in world frame (X=east, Y=north).
_OFFSET_TO_VECTOR: dict[str, tuple[float, float]] = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    # When phrased relative to a landmark, "ліворуч від X" / "перед X" — treat
    # them as world-frame because the user is reading the map top-down.
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "forward": (0.0, 1.0),
    "backward": (0.0, -1.0),
}

_OFFSET_DISPLAY_UK = {
    "north": "вище",
    "south": "нижче",
    "east": "правіше",
    "west": "лівіше",
    "up": "вище",
    "down": "нижче",
    "left": "лівіше",
    "right": "правіше",
    "forward": "перед",
    "backward": "за",
}


def _clamp_goal_to_map(
    map_row: MapTable, gx: float, gy: float
) -> tuple[float, float]:
    """Pull a goal back inside the map. The user can ask for "10m forward"
    on a 10×6 m map; without clamping, the goal lands outside the grid and
    A* returns []. We trim to one cell of margin from each edge — robot then
    drives as far as it physically can in that direction.
    """
    margin = map_row.cell_size_m
    map_w = map_row.width_cells * map_row.cell_size_m
    map_h = map_row.height_cells * map_row.cell_size_m
    return (
        max(margin, min(map_w - margin, gx)),
        max(margin, min(map_h - margin, gy)),
    )


def _resolve_offset(
    anchor_x: float, anchor_y: float, offset: dict[str, Any]
) -> tuple[float, float, str]:
    """Apply an offset vector to an anchor point. Returns (gx, gy, suffix)
    or raises ValueError on bad input."""
    direction = offset.get("direction", "")
    if not isinstance(direction, str) or direction not in _OFFSET_TO_VECTOR:
        raise ValueError(f"unknown offset direction '{direction}'")
    try:
        distance = float(offset.get("distance_m", 1.0))
    except (TypeError, ValueError):
        distance = 1.0
    distance = max(0.05, min(10.0, distance))
    dx, dy = _OFFSET_TO_VECTOR[direction]
    gx = anchor_x + dx * distance
    gy = anchor_y + dy * distance
    suffix = f"{_OFFSET_DISPLAY_UK[direction]} на {distance:.2f} м"
    return gx, gy, suffix


def _navigate_with_offset(
    db: Session,
    map_row: MapTable,
    pose: RobotPose,
    target_name: str,
    offset: dict[str, Any],
) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Navigate to an offset from a space's centroid. If the offset point is
    inside a wall (or outside the map after clamping has no path), we fall
    back to the nearest free cell — keeps "above the kitchen by half a meter"
    usable even when the centroid pushes into furniture geometry."""
    space = _find_space(db, map_row.id, target_name.strip().lower())  # type: ignore[arg-type]
    if space is None:
        return False, {"target": target_name, "offset": offset}, f"space '{target_name}' not found"
    polygon = _parse_polygon(space.vertices_json)
    if polygon is None:
        return False, {"target": space.name, "space_id": space.id, "offset": offset}, (
            f"space '{space.name}' has invalid polygon"
        )
    cx, cy = polygon_centroid(polygon)
    try:
        gx, gy, suffix = _resolve_offset(cx, cy, offset)
    except ValueError as e:
        return False, {"target": space.name, "offset": offset}, str(e)
    # An offset near the edge of the map can push the goal outside the grid.
    gx, gy = _clamp_goal_to_map(map_row, gx, gy)

    grid = _grid_for(map_row)
    waypoints = planner_plan(
        grid=grid,
        start_m=(pose.x, pose.y),
        goal_m=(gx, gy),
        cell_size=map_row.cell_size_m,
    )
    display = f"{space.name} ({suffix})"
    result: dict[str, Any] = {
        "target": display,
        "space_id": space.id,
        "offset": offset,
        "goal": {"x": gx, "y": gy},
        "waypoints": [{"x": x, "y": y} for x, y in waypoints],
    }
    if not waypoints:
        result["reason"] = "no_path"
        return False, result, "no_path"
    return True, result, None


def _drive_relative(
    map_row: MapTable,
    pose: RobotPose,
    direction: str,
    distance_m: float,
) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Free-form motion relative to robot heading. Computes a goal coordinate
    in the world frame, then runs A* so walls are still avoided. Falls back
    to a direct two-point path for very short moves where A* may not produce
    a useful path on a coarse grid.
    """
    if direction == "forward":
        heading = pose.theta
    elif direction == "backward":
        heading = pose.theta + math.pi
    elif direction == "left":
        # ROS convention: robot's left = +90° CCW from heading. Looking at a
        # top-down map with N up: when robot faces east (theta=0), its left
        # is north (screen-up). Same convention as the new ROTATE intent.
        heading = pose.theta + math.pi / 2
    elif direction == "right":
        heading = pose.theta - math.pi / 2
    else:
        return False, {"direction": direction}, f"unknown direction '{direction}'"

    # Snap the minimum distance to one cell so a tiny "5 см" command still
    # produces visible motion. The user's expectation is calibrated to the
    # visual scale of the map; sub-cell motion looks broken even when the
    # math is correct.
    cell = map_row.cell_size_m
    distance_m = max(cell, min(10.0, float(distance_m)))
    raw_gx = pose.x + distance_m * math.cos(heading)
    raw_gy = pose.y + distance_m * math.sin(heading)
    # Clamp goal so it can't sit outside the grid — the user can legitimately
    # ask for more meters than the map is wide.
    gx, gy = _clamp_goal_to_map(map_row, raw_gx, raw_gy)
    actual_distance = math.hypot(gx - pose.x, gy - pose.y)
    actual_cells = round(actual_distance / cell)

    grid = _grid_for(map_row)
    waypoints = planner_plan(
        grid=grid,
        start_m=(pose.x, pose.y),
        goal_m=(gx, gy),
        cell_size=map_row.cell_size_m,
    )
    if not waypoints and actual_distance < 0.3:
        # Short hops can fail planning when start and goal share a cell.
        # Fall back to a direct segment — caller's problem if it crosses a wall.
        waypoints = [(pose.x, pose.y), (gx, gy)]

    display = f"{_DIRECTION_TO_UK.get(direction, direction)} на {actual_distance:.2f} м"
    result: dict[str, Any] = {
        "target": display,
        "direction": direction,
        "distance_m": actual_distance,
        "distance_cells": actual_cells,
        "cell_size_m": cell,
        "requested_distance_m": distance_m,
        "goal": {"x": gx, "y": gy},
        "waypoints": [{"x": x, "y": y} for x, y in waypoints],
    }
    if not waypoints:
        result["reason"] = "no_path"
        return False, result, "no_path"
    return True, result, None


def _navigate_anchor(
    map_row: MapTable,
    pose: RobotPose,
    memory: Optional[dict[str, Any]],
    anchor: str,
) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Resolve 'previous_start' / 'previous_goal' from the frontend's memory
    blob and plan a path to the resolved coordinate. Distinct from _navigate
    because there's no space — the destination is a free-floating point.
    """
    if not isinstance(memory, dict):
        return False, {"anchor": anchor}, "no previous navigation"

    if anchor == "previous_start":
        anchor_data = memory.get("startedAt")
        display = "назад"
    elif anchor == "previous_goal":
        anchor_data = memory.get("goal")
        if isinstance(anchor_data, dict) and isinstance(anchor_data.get("name"), str):
            display = anchor_data["name"]
        else:
            display = "ще раз"
    else:
        return False, {"anchor": anchor}, f"unknown anchor '{anchor}'"

    if not isinstance(anchor_data, dict):
        return False, {"anchor": anchor}, "memory missing coordinates"
    gx = anchor_data.get("x")
    gy = anchor_data.get("y")
    if not isinstance(gx, (int, float)) or not isinstance(gy, (int, float)):
        return False, {"anchor": anchor}, "memory has invalid coordinates"

    grid = _grid_for(map_row)
    waypoints = planner_plan(
        grid=grid,
        start_m=(pose.x, pose.y),
        goal_m=(float(gx), float(gy)),
        cell_size=map_row.cell_size_m,
    )
    result: dict[str, Any] = {
        "target": display,
        "anchor": anchor,
        "waypoints": [{"x": x, "y": y} for x, y in waypoints],
    }
    if not waypoints:
        result["reason"] = "no_path"
        return False, result, "no_path"
    return True, result, None


def _delete_space(
    db: Session, map_row: MapTable, params: dict[str, Any]
) -> tuple[bool, dict[str, Any], Optional[str]]:
    name_raw = params.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        return False, {}, "missing 'name'"
    name = name_raw.strip().lower()
    row = _find_space(db, map_row.id, name)  # type: ignore[arg-type]
    if row is None:
        return False, {"name": name}, f"space '{name}' not found"
    if row.is_home:
        return False, {"name": name, "space_id": row.id}, (
            f"space '{name}' is the home space — assign another space as home before deleting"
        )
    db.delete(row)
    db.commit()
    return True, {"deleted": name}, None


def _rename_space(
    db: Session, map_row: MapTable, params: dict[str, Any]
) -> tuple[bool, dict[str, Any], Optional[str]]:
    old_raw = params.get("old_name")
    new_raw = params.get("new_name")
    if not isinstance(old_raw, str) or not old_raw.strip():
        return False, {}, "missing 'old_name'"
    if not isinstance(new_raw, str) or not new_raw.strip():
        return False, {}, "missing 'new_name'"
    old_name = old_raw.strip().lower()
    new_name = new_raw.strip().lower()
    row = _find_space(db, map_row.id, old_name)  # type: ignore[arg-type]
    if row is None:
        return False, {"old_name": old_name}, f"space '{old_name}' not found"
    row.name = new_name
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return False, {"old_name": old_name, "new_name": new_name}, (
            f"space '{new_name}' already exists"
        )
    db.refresh(row)
    return True, {"old_name": old_name, "new_name": new_name, "space_id": row.id}, None


def _dispatch_intent_one(
    db: Session,
    map_row: MapTable,
    pose: RobotPose,
    parsed_memory: Optional[dict[str, Any]],
    intent: Intent,
    params: dict[str, Any],
) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Run a single intent given a (possibly simulated) pose. Returns
    (success, action_result, error) — same shape every per-intent helper
    already uses. Extracted from post_voice so multi-step sequences can
    re-run the dispatcher with updated poses."""
    if intent is Intent.NAVIGATE:
        anchor = params.get("anchor")
        target_raw = params.get("target")
        offset = params.get("offset")
        has_target = isinstance(target_raw, str) and target_raw.strip()
        if isinstance(anchor, str) and anchor in ("previous_start", "previous_goal"):
            return _navigate_anchor(map_row, pose, parsed_memory, anchor)
        if has_target and isinstance(offset, dict):
            return _navigate_with_offset(db, map_row, pose, target_raw, offset)
        if has_target:
            return _navigate(db, map_row, pose, target_raw)
        return False, {}, "missing 'target' or 'anchor'"

    if intent is Intent.DRIVE_RELATIVE:
        direction_raw = params.get("direction")
        distance_raw = params.get("distance_m", 1.0)
        if not isinstance(direction_raw, str) or direction_raw not in _DIRECTION_TO_UK:
            return False, {}, "missing or invalid 'direction'"
        try:
            distance = float(distance_raw)
        except (TypeError, ValueError):
            distance = 1.0
        return _drive_relative(map_row, pose, direction_raw, distance)

    if intent is Intent.ROTATE:
        delta_raw = params.get("delta_deg")
        try:
            delta_deg = float(delta_raw) if delta_raw is not None else 0.0
        except (TypeError, ValueError):
            delta_deg = 0.0
        delta_deg = max(-360.0, min(360.0, delta_deg))
        delta_rad = math.radians(delta_deg)
        target_theta = pose.theta + delta_rad
        return True, {
            "target": f"повернути на {delta_deg:+.0f}°",
            "delta_deg": delta_deg,
            "delta_rad": delta_rad,
            "target_theta": target_theta,
        }, None

    if intent is Intent.DELETE_SPACE:
        return _delete_space(db, map_row, params)
    if intent is Intent.RENAME_SPACE:
        return _rename_space(db, map_row, params)
    if intent is Intent.STOP:
        return True, {"stopped": True}, None
    if intent is Intent.RETURN_HOME:
        home = _find_home_space(db, map_row.id)  # type: ignore[arg-type]
        if home is None:
            return False, {}, "no spaces on this map"
        return _navigate_to_space(db, map_row, pose, home)

    # Voice-perimeter drafting — backend just acknowledges; the frontend
    # accumulates the polygon and POSTs to /api/spaces on FINISH_SPACE.
    if intent is Intent.START_SPACE:
        name_raw = params.get("name")
        space_name = name_raw.strip().lower() if isinstance(name_raw, str) else ""
        if not space_name:
            return False, {}, "missing 'name'"
        return True, {"space_name": space_name}, None
    if intent is Intent.FINISH_SPACE:
        return True, {}, None
    if intent is Intent.CANCEL_SPACE:
        return True, {}, None

    return False, {"original_text": params.get("original_text", "")}, "intent not recognized"


async def _forward_to_robot(intent: Intent, action_result: dict[str, Any]) -> None:
    """Forward motion intents to the real-robot adapter.

    Movement intents publish waypoints; STOP triggers an e-stop; every
    other intent (space editing, drafts, unknown) is a no-op here because
    the device only cares about motion.
    """
    adapter = get_adapter()
    if intent is Intent.STOP:
        await adapter.stop()
        return
    if intent not in (Intent.NAVIGATE, Intent.DRIVE_RELATIVE, Intent.RETURN_HOME):
        return
    raw = action_result.get("waypoints")
    if not isinstance(raw, list) or not raw:
        return
    parsed: list[Point2D] = []
    for wp in raw:
        if not isinstance(wp, dict):
            continue
        try:
            parsed.append(Point2D(x=float(wp["x"]), y=float(wp["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    if parsed:
        await adapter.send_path(parsed)


def _simulate_pose_after(
    pose: RobotPose, intent: Intent, action_result: dict[str, Any]
) -> RobotPose:
    """Compute the pose the robot will end up at AFTER the given action
    completes, so the next action in a multi-step sequence can be planned
    from the right starting point. ROTATE updates theta only; movement
    intents jump to the last waypoint with theta along the final segment;
    everything else leaves pose unchanged."""
    if intent is Intent.ROTATE:
        tt = action_result.get("target_theta")
        if isinstance(tt, (int, float)):
            return RobotPose(x=pose.x, y=pose.y, theta=float(tt))
        return pose
    if intent in (Intent.NAVIGATE, Intent.DRIVE_RELATIVE, Intent.RETURN_HOME):
        wp = action_result.get("waypoints")
        if isinstance(wp, list) and wp:
            last = wp[-1]
            try:
                new_x = float(last["x"])
                new_y = float(last["y"])
            except (KeyError, TypeError, ValueError):
                return pose
            new_theta = pose.theta
            if len(wp) >= 2:
                p2 = wp[-2]
                try:
                    dx = new_x - float(p2["x"])
                    dy = new_y - float(p2["y"])
                    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                        new_theta = math.atan2(dy, dx)
                except (KeyError, TypeError, ValueError):
                    pass
            return RobotPose(x=new_x, y=new_y, theta=new_theta)
    return pose


@router.post("/voice", response_model=VoiceResponse)
async def post_voice(
    audio: UploadFile = File(...),
    robot_pos: str = Form(...),
    map_id: int = Form(...),
    voice_memory: Optional[str] = Form(None),
    db: Session = Depends(get_session),
) -> VoiceResponse:
    map_row = db.get(MapTable, map_id)
    if map_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")

    try:
        pose = RobotPose.model_validate_json(robot_pos)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid robot_pos JSON: {e}",
        )

    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/webm"
    audio_path = await _save_upload(audio_bytes, mime)

    space_rows = db.exec(select(SpaceTable).where(SpaceTable.map_id == map_id)).all()
    active_spaces = [{"name": r.name} for r in space_rows]

    parsed_memory: Optional[dict[str, Any]] = None
    if voice_memory:
        try:
            candidate = json.loads(voice_memory)
            if isinstance(candidate, dict):
                parsed_memory = candidate
        except json.JSONDecodeError:
            _log.warning("voice_memory was not valid JSON; ignoring")

    context = {
        "active_spaces": active_spaces,
        "robot_pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
        "map_name": map_row.name,
        "previous_navigation": parsed_memory,
    }

    started = time.monotonic()
    try:
        parsed = await voice_intent.parse_voice_command(audio_bytes, mime, context)
    except Exception as e:
        latency_ms = int((time.monotonic() - started) * 1000)
        await log_command(
            db,
            map_id=map_id,
            audio_path=audio_path,
            intent="UNKNOWN",
            params={"error": str(e)},
            success=False,
            error=f"voice_intent call failed: {e}",
            latency_ms=latency_ms,
        )
        return VoiceResponse(
            intent=Intent.UNKNOWN,
            params={"error": str(e)},
            action_result={},
        )

    intent_str = str(parsed.get("intent", "UNKNOWN"))
    params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
    params = params or {}

    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.UNKNOWN

    pending = parsed.get("_pending") or []
    if not isinstance(pending, list):
        pending = []

    # Mid-confidence band: don't dispatch. Log the row so the frontend can
    # POST /api/voice/feedback with command_log_id once the operator
    # decides; the actual action runs only on confirmation.
    if intent is Intent.UNCERTAIN:
        latency_ms = int((time.monotonic() - started) * 1000)
        log_row = await log_command(
            db,
            map_id=map_id,
            audio_path=audio_path,
            intent=intent.value,
            params=params,
            success=False,
            error="awaiting_confirmation",
            latency_ms=latency_ms,
            transcription=params.get("original_text") if isinstance(params, dict) else None,
        )
        return VoiceResponse(
            intent=Intent.UNCERTAIN,
            params=params,
            action_result={},
            command_log_id=log_row.id,
        )

    success, action_result, error = _dispatch_intent_one(
        db, map_row, pose, parsed_memory, intent, params
    )
    if intent is Intent.UNKNOWN and not action_result:
        action_result = {
            "original_text": params.get("original_text", ""),
            "error": error or "intent not recognized",
        }

    # Hardware bridge. In sim mode the adapter is a no-op so this is free;
    # in http mode it forwards waypoints / stop to the device alongside
    # returning them to the browser. The frontend still animates the sim
    # robot — operators see two layers if they look closely (commanded
    # path vs. real-robot reported pose).
    await _forward_to_robot(intent, action_result)

    # Multi-step: pre-simulate subsequent actions so the frontend can queue
    # and dispatch each as the robot becomes idle. Pose carries forward
    # through ROTATE / movement intents.
    follow_ups: list[VoiceFollowUp] = []
    sim_pose = _simulate_pose_after(pose, intent, action_result)
    for item in pending:
        if not isinstance(item, dict):
            continue
        nxt_intent_str = str(item.get("intent", "UNKNOWN"))
        try:
            nxt_intent = Intent(nxt_intent_str)
        except ValueError:
            nxt_intent = Intent.UNKNOWN
        nxt_params_raw = item.get("params")
        nxt_params: dict[str, Any] = nxt_params_raw if isinstance(nxt_params_raw, dict) else {}
        _ok, nxt_result, nxt_err = _dispatch_intent_one(
            db, map_row, sim_pose, parsed_memory, nxt_intent, nxt_params
        )
        if nxt_err and isinstance(nxt_result, dict):
            nxt_result = {**nxt_result, "error": nxt_err}
        follow_ups.append(
            VoiceFollowUp(intent=nxt_intent, params=nxt_params, action_result=nxt_result)
        )
        sim_pose = _simulate_pose_after(sim_pose, nxt_intent, nxt_result)

    latency_ms = int((time.monotonic() - started) * 1000)
    log_row = await log_command(
        db,
        map_id=map_id,
        audio_path=audio_path,
        intent=intent.value,
        params=params,
        success=success,
        error=error,
        latency_ms=latency_ms,
        transcription=params.get("original_text") if isinstance(params, dict) else None,
    )

    # Always expose command_log_id so the frontend can offer a manual
    # «виправити останню команду» button even when the model was confident.
    return VoiceResponse(
        intent=intent,
        params=params,
        action_result=action_result,
        follow_ups=follow_ups,
        command_log_id=log_row.id,
    )


@router.post("/voice/feedback", response_model=VoiceFeedbackResponse)
async def post_voice_feedback(
    body: VoiceFeedbackRequest,
    db: Session = Depends(get_session),
) -> VoiceFeedbackResponse:
    """Operator's verdict on an UNCERTAIN prediction. Stores a row in
    command_corrections; if was_correct=True, dispatches the originally
    predicted intent so the frontend doesn't need a second voice round."""
    log_row = db.get(CommandLogTable, body.command_log_id)
    if log_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="command_log_id not found"
        )

    try:
        params_blob = json.loads(log_row.params_json or "{}")
    except json.JSONDecodeError:
        params_blob = {}

    # Two paths converge here: (a) UNCERTAIN log row that's awaiting
    # confirmation — predicted intent + params live under the
    # _predicted_* keys; (b) a previously-dispatched command being
    # corrected post-hoc — log_row.intent IS the intent and params_blob
    # IS the params. Normalise so downstream logic doesn't care.
    if log_row.intent == Intent.UNCERTAIN.value:
        predicted_intent_str = str(params_blob.get("_predicted_intent", "UNKNOWN"))
        predicted_params = params_blob.get("_predicted_params") or {}
        predicted_confidence = float(params_blob.get("_confidence") or 0.0)
    else:
        predicted_intent_str = log_row.intent
        # Strip private/_metadata keys (_confidence etc.) — they were
        # added by _try_classify_local and shouldn't pollute slot edits.
        predicted_params = {
            k: v
            for k, v in params_blob.items()
            if not k.startswith("_") and k != "original_text"
        }
        predicted_confidence = float(params_blob.get("_confidence") or 1.0)

    # Operator may have edited slot values in the modal — those win
    # over the model's predicted_params for both dispatch and DB record.
    effective_intent_str = (
        body.corrected_intent.value
        if body.corrected_intent
        else predicted_intent_str
    )
    effective_params: dict[str, Any] = (
        dict(body.corrected_params)
        if body.corrected_params is not None
        else dict(predicted_params)
    )

    correction = CommandCorrectionTable(
        command_log_id=body.command_log_id,
        transcription=log_row.transcription or "",
        predicted_intent=predicted_intent_str,
        predicted_confidence=predicted_confidence,
        predicted_params_json=json.dumps(predicted_params, ensure_ascii=False),
        was_correct=body.was_correct,
        corrected_intent=body.corrected_intent.value if body.corrected_intent else None,
        corrected_params_json=(
            json.dumps(body.corrected_params, ensure_ascii=False)
            if body.corrected_params is not None
            else None
        ),
    )
    db.add(correction)
    db.commit()
    db.refresh(correction)

    # Persist the (transcription → intent + params) mapping into
    # learned_overrides whenever the operator either confirmed-with-edits
    # or pushed a different intent. Pure rejections without slot edits
    # don't generate a positive rule, so we skip them.
    transcription = log_row.transcription or ""
    teach_intent = effective_intent_str
    teach_params = effective_params
    should_teach = bool(transcription) and teach_intent in Intent.__members__ and (
        body.was_correct
        or body.corrected_intent is not None
        or body.corrected_params is not None
    )
    if should_teach:
        try:
            from services import learned_overrides as _lo  # noqa: PLC0415
            _lo.upsert(transcription, teach_intent, teach_params)
        except Exception as exc:  # noqa: BLE001
            _log.warning("learned_overrides upsert failed: %s", exc)

    if not body.was_correct:
        return VoiceFeedbackResponse(recorded=True)

    map_row = db.get(MapTable, log_row.map_id) if log_row.map_id is not None else None
    if map_row is None:
        return VoiceFeedbackResponse(
            recorded=True,
            intent=Intent(effective_intent_str) if effective_intent_str in Intent.__members__ else Intent.UNKNOWN,
            params=effective_params,
            action_result={"error": "map context lost; replay the command"},
        )

    try:
        intent = Intent(effective_intent_str)
    except ValueError:
        return VoiceFeedbackResponse(
            recorded=True,
            intent=Intent.UNKNOWN,
            params=effective_params,
            action_result={"error": f"unknown intent {effective_intent_str!r}"},
        )

    # Pose comes from the feedback request (frontend re-sends current
    # robot pose). For NAVIGATE / DELETE_SPACE / RENAME_SPACE pose is unused; for
    # ROTATE / DRIVE_RELATIVE it determines the goal pose.
    pose = body.robot_pose or RobotPose(x=0.0, y=0.0, theta=0.0)
    success, action_result, error = _dispatch_intent_one(
        db, map_row, pose, None, intent, effective_params
    )
    if error and isinstance(action_result, dict):
        action_result = {**action_result, "error": error}
    return VoiceFeedbackResponse(
        recorded=True,
        intent=intent,
        params=effective_params,
        action_result=action_result,
    )


@router.get("/voice/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats(db: Session = Depends(get_session)) -> FeedbackStats:
    """Snapshot for the training panel: how many corrections await
    retraining, when retraining last finished, and current state."""
    return FeedbackStats(
        pending_count=training_orchestrator.get_pending_count(db),
        last_retrained_at=training_orchestrator.get_last_retrained_at(),
        retrain_state=training_orchestrator.get_status()["state"],
    )


@router.post("/voice/feedback/retrain", status_code=status.HTTP_202_ACCEPTED,
             response_model=RetrainStatus)
async def trigger_retrain() -> RetrainStatus:
    """Kick off scripts/retrain_from_feedback.py in a background task.
    Returns 202 (accepted) on success or 409 (conflict) if a retrain is
    already running. The model is hot-reloaded in-process when the
    subprocess exits cleanly."""
    if not training_orchestrator.start_retrain():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="retrain already in progress",
        )
    return RetrainStatus(**training_orchestrator.get_status())


@router.get("/voice/feedback/retrain/status", response_model=RetrainStatus)
async def get_retrain_status() -> RetrainStatus:
    return RetrainStatus(**training_orchestrator.get_status())


# ---------------------------------------------------------------------------
# Correction list / edit / delete (operator-facing management)
# ---------------------------------------------------------------------------


def _row_to_model(row: CommandCorrectionTable) -> CorrectionRow:
    """Pack a DB row into the CorrectionRow shape — including the
    «effective» (intent, params) the row contributes to retraining."""
    try:
        pred_params = json.loads(row.predicted_params_json or "{}")
    except json.JSONDecodeError:
        pred_params = {}
    try:
        corr_params = (
            json.loads(row.corrected_params_json) if row.corrected_params_json else None
        )
    except json.JSONDecodeError:
        corr_params = None

    eff_intent = row.corrected_intent or row.predicted_intent
    eff_params = corr_params if isinstance(corr_params, dict) else (pred_params if isinstance(pred_params, dict) else {})

    return CorrectionRow(
        id=row.id or 0,
        transcription=row.transcription or "",
        predicted_intent=row.predicted_intent or "",
        predicted_confidence=row.predicted_confidence or 0.0,
        predicted_params=pred_params if isinstance(pred_params, dict) else {},
        was_correct=bool(row.was_correct),
        corrected_intent=row.corrected_intent,
        corrected_params=corr_params if isinstance(corr_params, dict) else None,
        created_at=row.created_at.isoformat() if row.created_at else "",
        effective_intent=eff_intent,
        effective_params=eff_params,
    )


def _resync_override_for(transcription: str, db: Session) -> None:
    """After an edit/delete that may have affected `transcription`'s
    learned override, find the most recent remaining «active» correction
    for that text and rewrite the JSON entry accordingly. If no active
    rows remain, drop the entry entirely."""
    from services import learned_overrides as _lo  # noqa: PLC0415
    if not transcription:
        return
    stmt = (
        select(CommandCorrectionTable)
        .where(CommandCorrectionTable.transcription == transcription)
        .order_by(CommandCorrectionTable.id.desc())  # type: ignore[union-attr]
    )
    rows = db.exec(stmt).all()
    for r in rows:
        # «active» = either confirmed or carries an override. Pure
        # rejections without override don't teach anything.
        is_active = bool(r.was_correct) or bool(r.corrected_intent) or bool(
            r.corrected_params_json
        )
        if not is_active:
            continue
        intent = r.corrected_intent or r.predicted_intent
        try:
            params = (
                json.loads(r.corrected_params_json)
                if r.corrected_params_json
                else json.loads(r.predicted_params_json or "{}")
            )
        except json.JSONDecodeError:
            params = {}
        if isinstance(params, dict) and intent:
            _lo.upsert(transcription, intent, params)
            return
    # Nothing left to teach — remove the entry from the override file.
    _drop_override(transcription)


def _drop_override(transcription: str) -> None:
    """Remove a single phrase from learned_overrides.json, atomic write."""
    from services import learned_overrides as _lo  # noqa: PLC0415
    overrides = dict(_lo.all_overrides())
    key = _lo._normalize(transcription)  # noqa: SLF001
    if key not in overrides:
        return
    overrides.pop(key)
    path = _lo.OVERRIDES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    import os as _os
    _os.replace(tmp, path)


@router.get("/voice/feedback/corrections", response_model=list[CorrectionRow])
async def list_corrections(
    pending_only: bool = True,
    db: Session = Depends(get_session),
) -> list[CorrectionRow]:
    """Pending corrections (those that landed after the last retrain) by
    default. Pass `pending_only=false` to inspect the full history."""
    stmt = select(CommandCorrectionTable).order_by(CommandCorrectionTable.id.desc())  # type: ignore[union-attr]
    if pending_only:
        last = training_orchestrator.get_last_retrained_at()
        if last:
            try:
                from datetime import datetime as _dt
                cutoff = _dt.fromisoformat(last)
                stmt = stmt.where(CommandCorrectionTable.created_at > cutoff)
            except ValueError:
                pass
    rows = db.exec(stmt).all()
    return [_row_to_model(r) for r in rows]


@router.patch("/voice/feedback/corrections/{correction_id}",
              response_model=CorrectionRow)
async def patch_correction(
    correction_id: int,
    body: EditCorrectionRequest,
    db: Session = Depends(get_session),
) -> CorrectionRow:
    """Edit the operator's verdict on an existing correction. Both the
    DB row and the learned-override entry stay in sync."""
    row = db.get(CommandCorrectionTable, correction_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="correction not found"
        )

    if body.corrected_intent is not None:
        row.corrected_intent = body.corrected_intent.value
        # Editing the intent counts as the operator owning this row's
        # verdict — flip was_correct off so it's interpreted as a
        # "different intent" override during retraining.
        row.was_correct = False if body.corrected_intent.value != row.predicted_intent else True
    if body.corrected_params is not None:
        row.corrected_params_json = json.dumps(body.corrected_params, ensure_ascii=False)
        # Slot edit alone counts as confirming intent + adjusting params.
        row.was_correct = True
    db.add(row)
    db.commit()
    db.refresh(row)

    # Re-derive the override for this transcription from the current
    # (possibly edited) state of all corrections that share its text.
    _resync_override_for(row.transcription or "", db)

    return _row_to_model(row)


@router.delete("/voice/feedback/corrections/{correction_id}",
               status_code=status.HTTP_204_NO_CONTENT)
async def delete_correction(
    correction_id: int,
    db: Session = Depends(get_session),
) -> None:
    row = db.get(CommandCorrectionTable, correction_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="correction not found"
        )
    transcription = row.transcription or ""
    db.delete(row)
    db.commit()
    _resync_override_for(transcription, db)
    return None
