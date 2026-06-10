import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from config import settings
from db import engine, init_db
from routers import maps as maps_router
from routers import plan as plan_router
from routers import robot as robot_router
from routers import spaces as spaces_router
from routers import scenes as scenes_router
from routers import voice as voice_router
from seed import seed
from services import intent_classifier as intent_classifier_module
from services import robot_adapter

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    with Session(engine) as session:
        seed(session)
    if settings.voice_classifier_engine == "xlm_roberta":
        try:
            clf = intent_classifier_module.IntentClassifier.load(settings.intent_model_dir)
            intent_classifier_module.set_classifier(clf)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "intent classifier failed to load (%s) — voice commands will "
                "return UNKNOWN until a model is present", exc
            )
    adapter = robot_adapter.init_adapter()
    try:
        yield
    finally:
        aclose = getattr(adapter, "aclose", None)
        if aclose is not None:
            await aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(maps_router.router)
app.include_router(spaces_router.router)
app.include_router(plan_router.router)
app.include_router(voice_router.router)
app.include_router(scenes_router.router)
app.include_router(robot_router.router)

app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
