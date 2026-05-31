from typing import Annotated, Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str
    database_url: str = "sqlite:///./db.sqlite3"
    uploads_dir: str = "./uploads"
    scenes_dir: str = "./public/scenes"
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    # Set VOICE_USE_TOOLS=true to route voice intent classification through
    # Llama's native function-calling (services.voice_intent._classify_with_tools).
    # Default is the legacy free-form JSON path.
    voice_use_tools: bool = False
    # Intent recognition engine. "groq" → Groq Llama (legacy / fallback).
    # "xlm_roberta" → trained classifier from intent_model_dir; falls back to
    # Groq when the model is missing or confidence < intent_confidence_threshold.
    voice_classifier_engine: Literal["groq", "xlm_roberta"] = "groq"
    intent_model_dir: str = "./models/intent_classifier"
    # Confidence band for the joint XLM-RoBERTa classifier:
    #   conf >= HIGH                           → execute immediately
    #   LOW <= conf < HIGH                     → return UNCERTAIN, ask operator
    #   conf < LOW                             → fall back to Groq
    # Lowering HIGH increases the active-learning trigger rate during demos.
    intent_low_confidence_threshold: float = 0.50
    intent_high_confidence_threshold: float = 0.85
    # Motion backend selection. "sim" → browser owns kinematics (default,
    # current behavior). "http" → backend forwards waypoints to a real
    # device via services.robot_adapter.HttpEsp32Adapter; the device must
    # implement the contract documented in docs/hardware/esp32.md.
    robot_mode: Literal["sim", "http"] = "sim"
    robot_url: Optional[str] = None
    robot_timeout_s: float = 2.0

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


settings = Settings()
