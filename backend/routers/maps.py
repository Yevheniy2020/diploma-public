import base64
import binascii
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db import get_session
from models import MapCreate, MapResponse, MapUpdate
from services.image_to_grid import image_to_grid
from tables import MapTable

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/maps", tags=["maps"])


class MapSummary(BaseModel):
    id: int
    name: str
    width_cells: int
    height_cells: int
    cell_size_m: float
    scene_glb_path: Optional[str] = None


def _to_response(row: MapTable) -> MapResponse:
    grid_b64: Optional[str] = None
    if row.grid_data is not None:
        grid_b64 = base64.b64encode(row.grid_data).decode("ascii")
    return MapResponse(
        id=row.id,  # type: ignore[arg-type]
        name=row.name,
        width_cells=row.width_cells,
        height_cells=row.height_cells,
        cell_size_m=row.cell_size_m,
        grid_data_b64=grid_b64,
        scene_glb_path=row.scene_glb_path,
        scene_origin_x=row.scene_origin_x,
        scene_origin_y=row.scene_origin_y,
        scene_scale=row.scene_scale,
    )


@router.get("", response_model=list[MapSummary])
def list_maps(db: Session = Depends(get_session)) -> list[MapSummary]:
    rows = db.exec(select(MapTable).order_by(MapTable.id)).all()
    return [
        MapSummary(
            id=r.id,  # type: ignore[arg-type]
            name=r.name,
            width_cells=r.width_cells,
            height_cells=r.height_cells,
            cell_size_m=r.cell_size_m,
            scene_glb_path=r.scene_glb_path,
        )
        for r in rows
    ]


@router.get("/{map_id}", response_model=MapResponse)
def get_map(map_id: int, db: Session = Depends(get_session)) -> MapResponse:
    row = db.get(MapTable, map_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")
    return _to_response(row)


@router.post("", response_model=MapResponse, status_code=status.HTTP_201_CREATED)
def create_map(payload: MapCreate, db: Session = Depends(get_session)) -> MapResponse:
    row = MapTable(
        name=payload.name,
        width_cells=payload.width_cells,
        height_cells=payload.height_cells,
        cell_size_m=payload.cell_size_m,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"map name '{payload.name}' already exists",
        )
    db.refresh(row)
    return _to_response(row)


@router.post("/from-image", response_model=MapResponse, status_code=status.HTTP_201_CREATED)
async def create_map_from_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    cell_size_m: float = Form(0.05),
    max_cells: int = Form(200),
    invert: bool = Form(False),
    dilate: int = Form(0),
    db: Session = Depends(get_session),
) -> MapResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"expected PNG or JPEG, got {file.content_type}",
        )
    if not 20 <= max_cells <= 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="max_cells must be between 20 and 500",
        )
    if not 0.01 <= cell_size_m <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cell_size_m must be between 0.01 and 1.0",
        )
    if not 0 <= dilate <= 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="dilate must be between 0 and 5",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB",
        )

    try:
        result = image_to_grid(
            data,
            max_cells=max_cells,
            invert=invert,
            dilate=dilate,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    row = MapTable(
        name=name,
        width_cells=result.width_cells,
        height_cells=result.height_cells,
        cell_size_m=cell_size_m,
        grid_data=result.grid_bytes,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"map name '{name}' already exists",
        )
    db.refresh(row)
    return _to_response(row)


@router.put("/{map_id}", response_model=MapResponse)
def update_map(map_id: int, payload: MapUpdate, db: Session = Depends(get_session)) -> MapResponse:
    row = db.get(MapTable, map_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")

    if payload.name is not None:
        row.name = payload.name
    if payload.grid_data_b64 is not None:
        try:
            decoded = base64.b64decode(payload.grid_data_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"invalid base64 grid_data: {e}",
            )
        expected = row.width_cells * row.height_cells
        if len(decoded) != expected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"grid_data length {len(decoded)} != width*height {expected}",
            )
        row.grid_data = decoded
    if payload.scene_glb_path is not None:
        row.scene_glb_path = payload.scene_glb_path
    if payload.scene_origin_x is not None:
        row.scene_origin_x = payload.scene_origin_x
    if payload.scene_origin_y is not None:
        row.scene_origin_y = payload.scene_origin_y
    if payload.scene_scale is not None:
        row.scene_scale = payload.scene_scale

    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="map name conflicts with existing map",
        )
    db.refresh(row)
    return _to_response(row)


@router.delete("/{map_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_map(map_id: int, db: Session = Depends(get_session)) -> None:
    row = db.get(MapTable, map_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")
    db.delete(row)
    db.commit()
    return None
