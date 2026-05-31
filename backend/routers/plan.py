import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from db import get_session
from models import PlanRequest, PlanResponse, Point2D
from services.grid import decode_grid
from services.planner import plan as planner_plan
from services.robot_adapter import get_adapter
from tables import MapTable

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["plan"])


@router.post("/plan", response_model=PlanResponse)
async def post_plan(payload: PlanRequest, db: Session = Depends(get_session)) -> PlanResponse:
    map_row = db.get(MapTable, payload.map_id)
    if map_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")

    if map_row.grid_data is None:
        grid = np.zeros((map_row.height_cells, map_row.width_cells), dtype=np.uint8)
    else:
        grid = decode_grid(map_row.grid_data, map_row.width_cells, map_row.height_cells)

    waypoints = planner_plan(
        grid=grid,
        start_m=(payload.from_pos.x, payload.from_pos.y),
        goal_m=(payload.to_pos.x, payload.to_pos.y),
        cell_size=map_row.cell_size_m,
    )

    if not waypoints:
        return PlanResponse(waypoints=[], reason="no_path")

    points = [Point2D(x=x, y=y) for x, y in waypoints]
    await get_adapter().send_path(points)
    return PlanResponse(waypoints=points)
