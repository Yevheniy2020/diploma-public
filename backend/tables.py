from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MapTable(SQLModel, table=True):
    __tablename__ = "maps"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    width_cells: int
    height_cells: int
    cell_size_m: float = 0.1
    grid_data: Optional[bytes] = None
    scene_glb_path: Optional[str] = None
    scene_origin_x: float = 0.0
    scene_origin_y: float = 0.0
    scene_scale: float = 1.0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    spaces: list["SpaceTable"] = Relationship(
        back_populates="map",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class SpaceTable(SQLModel, table=True):
    __tablename__ = "spaces"
    __table_args__ = (UniqueConstraint("map_id", "name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    map_id: int = Field(foreign_key="maps.id", index=True)
    name: str
    vertices_json: str
    # Exactly one space per map should have is_home=True (enforced by a
    # partial unique index in schema.sql). The robot spawns at this space's
    # centroid and RETURN_HOME navigates to it.
    is_home: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    map: Optional[MapTable] = Relationship(back_populates="spaces")


class CommandLogTable(SQLModel, table=True):
    __tablename__ = "command_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    map_id: Optional[int] = Field(default=None, foreign_key="maps.id")
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    intent: str
    params_json: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=_now)


class CommandCorrectionTable(SQLModel, table=True):
    """Operator feedback for low-confidence intent predictions. Each row
    captures the model's guess, the operator's verdict (was it right?),
    and an optional override intent. `scripts/retrain_from_feedback.py`
    folds confirmed/overridden rows back into the training corpus."""
    __tablename__ = "command_corrections"

    id: Optional[int] = Field(default=None, primary_key=True)
    command_log_id: int = Field(foreign_key="command_log.id", index=True)
    transcription: str
    predicted_intent: str
    predicted_confidence: float
    predicted_params_json: Optional[str] = None
    was_correct: bool
    corrected_intent: Optional[str] = None
    corrected_params_json: Optional[str] = None
    created_at: datetime = Field(default_factory=_now, index=True)
