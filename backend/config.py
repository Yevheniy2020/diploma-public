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
    voice_classifier_engine: Literal["xlm_roberta"] = "xlm_roberta"
    intent_model_dir: str = "./models/intent_classifier"
    intent_high_confidence_threshold: float = 0.85
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
