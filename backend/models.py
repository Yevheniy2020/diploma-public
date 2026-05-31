import math
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    NAVIGATE = "NAVIGATE"
    DRIVE_RELATIVE = "DRIVE_RELATIVE"
    ROTATE = "ROTATE"
    DELETE_SPACE = "DELETE_SPACE"
    RENAME_SPACE = "RENAME_SPACE"
    # Voice-perimeter space drafting. Recording itself happens in the
    # frontend; these three intents just signal "begin", "save", "discard".
    START_SPACE = "START_SPACE"
    FINISH_SPACE = "FINISH_SPACE"
    CANCEL_SPACE = "CANCEL_SPACE"
    STOP = "STOP"
    RETURN_HOME = "RETURN_HOME"
    UNKNOWN = "UNKNOWN"
    # Mid-confidence prediction awaiting operator confirmation. Frontend
    # opens the CorrectionDialog and POSTs to /api/voice/feedback once the
    # operator decides; the action is dispatched only on confirmation.
    UNCERTAIN = "UNCERTAIN"


def _normalize_name(v: str) -> str:
    v = v.strip().lower()
    if not v:
        raise ValueError("name must be non-empty")
    return v


def _normalize_theta(v: float) -> float:
    return ((v + math.pi) % (2 * math.pi)) - math.pi


class Point2D(BaseModel):
    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


class RobotPose(BaseModel):
    x: float
    y: float
    theta: float

    @field_validator("x", "y")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v

    @field_validator("theta")
    @classmethod
    def _wrap(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("theta must be a finite number")
        return _normalize_theta(v)


class MapCreate(BaseModel):
    name: str = Field(min_length=1)
    width_cells: int = Field(ge=1)
    height_cells: int = Field(ge=1)
    cell_size_m: float = Field(default=0.1, gt=0)


class MapUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    grid_data_b64: Optional[str] = None
    scene_glb_path: Optional[str] = None
    scene_origin_x: Optional[float] = None
    scene_origin_y: Optional[float] = None
    scene_scale: Optional[float] = None


class MapResponse(BaseModel):
    id: int
    name: str
    width_cells: int
    height_cells: int
    cell_size_m: float
    grid_data_b64: Optional[str] = None
    scene_glb_path: Optional[str] = None
    scene_origin_x: float
    scene_origin_y: float
    scene_scale: float


class SpaceCreate(BaseModel):
    map_id: int
    name: str
    vertices: list[tuple[float, float]]
    is_home: bool = False

    @field_validator("name")
    @classmethod
    def _norm_name(cls, v: str) -> str:
        return _normalize_name(v)

    @field_validator("vertices")
    @classmethod
    def _check_vertices(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        for x, y in v:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("vertex coordinates must be finite")
        return v


class SpaceUpdate(BaseModel):
    """PATCH body for /api/spaces/{id}. Any subset of fields may be present;
    omitted fields are left untouched. Setting `is_home=True` atomically
    clears the flag on any other space of the same map."""
    name: Optional[str] = None
    is_home: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def _norm_name(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else _normalize_name(v)


class SpaceResponse(BaseModel):
    id: int
    map_id: int
    name: str
    vertices: list[tuple[float, float]]
    is_home: bool = False


class PlanRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    map_id: int
    from_pos: Point2D = Field(alias="from")
    to_pos: Point2D = Field(alias="to")


class PlanResponse(BaseModel):
    waypoints: list[Point2D]
    reason: Optional[str] = None


class VoiceFollowUp(BaseModel):
    """One pre-simulated step in a multi-action voice command. Frontend
    queues these and dispatches each when the robot becomes idle."""
    intent: Intent
    params: dict[str, Any] = Field(default_factory=dict)
    action_result: dict[str, Any] = Field(default_factory=dict)


class VoiceResponse(BaseModel):
    intent: Intent
    params: dict[str, Any] = Field(default_factory=dict)
    action_result: dict[str, Any] = Field(default_factory=dict)
    follow_ups: list[VoiceFollowUp] = Field(default_factory=list)
    # When intent == UNCERTAIN we expose the row id so the frontend can
    # POST it back via /api/voice/feedback alongside the operator's verdict.
    command_log_id: Optional[int] = None


class VoiceFeedbackRequest(BaseModel):
    """Operator's verdict on an UNCERTAIN prediction. `was_correct=True`
    causes the backend to dispatch the (optionally edited) intent + slot
    params and return the resulting action_result; the `robot_pose`
    carries the pose to dispatch from. `was_correct=False` records the
    correction (with optional `corrected_intent` override) without
    dispatching anything.

    `corrected_params` lets the operator edit slot values before
    confirming — e.g. predicted ROTATE delta_deg=90 but user really
    wanted 360. When it differs from the model's prediction, the
    (transcription → intent + params) pair is also saved into
    learned_overrides so future occurrences short-circuit straight to
    the corrected behaviour."""
    command_log_id: int
    was_correct: bool
    corrected_intent: Optional[Intent] = None
    corrected_params: Optional[dict[str, Any]] = None
    robot_pose: Optional[RobotPose] = None


class VoiceFeedbackResponse(BaseModel):
    """Mirror of VoiceResponse for the confirmation path. When the
    operator confirms, this carries the dispatched action; when they
    reject, only the bookkeeping fields are populated."""
    recorded: bool
    intent: Optional[Intent] = None
    params: dict[str, Any] = Field(default_factory=dict)
    action_result: dict[str, Any] = Field(default_factory=dict)


class FeedbackStats(BaseModel):
    """Snapshot consumed by the operator console's training panel."""
    pending_count: int
    last_retrained_at: Optional[str] = None
    retrain_state: str  # idle | running | completed | failed


class CorrectionRow(BaseModel):
    """One row from command_corrections, normalised for the UI list."""
    id: int
    transcription: str
    predicted_intent: str
    predicted_confidence: float
    predicted_params: dict[str, Any] = Field(default_factory=dict)
    was_correct: bool
    corrected_intent: Optional[str] = None
    corrected_params: Optional[dict[str, Any]] = None
    created_at: str
    # Effective (intent, params) — what will go into the next retrain.
    effective_intent: str
    effective_params: dict[str, Any] = Field(default_factory=dict)


class EditCorrectionRequest(BaseModel):
    """PATCH body — both fields optional, missing = leave unchanged."""
    corrected_intent: Optional[Intent] = None
    corrected_params: Optional[dict[str, Any]] = None


class RetrainStatus(BaseModel):
    state: str
    phase: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class SceneInfo(BaseModel):
    id: str
    name: str
    glb_url: str
    default_map_id: int
