import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db import get_session
from models import SpaceCreate, SpaceResponse, SpaceUpdate
from tables import MapTable, SpaceTable

_log = logging.getLogger(__name__)

router = APIRouter(tags=["spaces"])


def _to_response(row: SpaceTable) -> SpaceResponse:
    try:
        verts_raw = json.loads(row.vertices_json)
    except (json.JSONDecodeError, TypeError):
        _log.warning("space %d has malformed vertices_json; serving empty", row.id)
        verts_raw = []
    vertices: list[tuple[float, float]] = []
    if isinstance(verts_raw, list):
        for v in verts_raw:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    vertices.append((float(v[0]), float(v[1])))
                except (TypeError, ValueError):
                    continue
    return SpaceResponse(
        id=row.id,  # type: ignore[arg-type]
        map_id=row.map_id,
        name=row.name,
        vertices=vertices,
        is_home=bool(row.is_home),
    )


def _clear_other_homes(db: Session, map_id: int, keep_id: int | None) -> None:
    """Unset is_home on every other space of this map. Caller flushes."""
    rows = db.exec(
        select(SpaceTable).where(
            SpaceTable.map_id == map_id, SpaceTable.is_home == True  # noqa: E712
        )
    ).all()
    for r in rows:
        if r.id != keep_id:
            r.is_home = False
            db.add(r)


@router.get("/api/maps/{map_id}/spaces", response_model=list[SpaceResponse])
def list_spaces(map_id: int, db: Session = Depends(get_session)) -> list[SpaceResponse]:
    if db.get(MapTable, map_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")
    rows = db.exec(
        select(SpaceTable).where(SpaceTable.map_id == map_id).order_by(SpaceTable.id)
    ).all()
    return [_to_response(r) for r in rows]


@router.post("/api/spaces", response_model=SpaceResponse, status_code=status.HTTP_201_CREATED)
def create_space(payload: SpaceCreate, db: Session = Depends(get_session)) -> SpaceResponse:
    """Create a space from a voice-demonstrated perimeter (or any caller).
    If is_home=True, atomically clears the flag on any prior home-space of
    this map."""
    if db.get(MapTable, payload.map_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="map not found")
    polygon = [(float(x), float(y)) for x, y in payload.vertices]
    row = SpaceTable(
        map_id=payload.map_id,
        name=payload.name,
        vertices_json=json.dumps(polygon),
        is_home=payload.is_home,
    )
    if payload.is_home:
        _clear_other_homes(db, payload.map_id, keep_id=None)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"space '{payload.name}' already exists on map {payload.map_id}",
        )
    db.refresh(row)
    return _to_response(row)


@router.patch("/api/spaces/{space_id}", response_model=SpaceResponse)
def patch_space(
    space_id: int, payload: SpaceUpdate, db: Session = Depends(get_session)
) -> SpaceResponse:
    """Edit a space: rename or assign as home. Setting is_home=True
    atomically clears the flag on any prior home-space of the same map."""
    row = db.get(SpaceTable, space_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="space not found")

    if payload.name is not None:
        row.name = payload.name
    if payload.is_home is True:
        _clear_other_homes(db, row.map_id, keep_id=row.id)
        row.is_home = True
    elif payload.is_home is False:
        row.is_home = False

    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"space '{payload.name}' already exists on map {row.map_id}",
        )
    db.refresh(row)
    return _to_response(row)


@router.delete("/api/spaces/{space_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_space(space_id: int, db: Session = Depends(get_session)) -> None:
    row = db.get(SpaceTable, space_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="space not found")
    if row.is_home:
        # Refuse to leave a map without a home space — caller should assign
        # another space as home first via PATCH.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="cannot delete the home space — assign another space as home first",
        )
    db.delete(row)
    db.commit()
    return None
