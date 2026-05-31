"""Background orchestration of `scripts/retrain_from_feedback.py`.

The voice frontend pokes us via three endpoints (`/voice/feedback/stats`,
`POST /voice/feedback/retrain`, `/voice/feedback/retrain/status`); this
module spawns the script as a subprocess, watches its sidecar status
file, and on success hot-reloads the joint classifier in the running
process — no `uvicorn` restart.

Process-wide singleton state lives in `_state`. The orchestrator does
not import torch / transformers itself; the hot-reload step pulls them
in lazily through `intent_classifier.IntentClassifier.load`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlmodel import Session, func, select

from config import settings
from tables import CommandCorrectionTable

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RETRAIN_SCRIPT = ROOT / "scripts" / "retrain_from_feedback.py"
DATA_DIR = ROOT / "scripts" / "data"
STATUS_FILE = DATA_DIR / "retrain_status.json"
LAST_RETRAINED_FILE = DATA_DIR / "last_retrained_at.txt"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class _RetrainState:
    """Mutable singleton — current retrain status. Read by /status, written
    by the watchdog task spawned alongside the subprocess."""

    def __init__(self) -> None:
        self.state: str = "idle"  # idle | running | completed | failed
        self.phase: Optional[str] = None  # preparing | paraphrasing | training | done | failed
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = 0.0
        if self.started_at and not self.finished_at:
            try:
                t0 = datetime.fromisoformat(self.started_at)
                elapsed = max(0.0, (datetime.now(timezone.utc) - t0).total_seconds())
            except ValueError:
                pass
        return {
            "state": self.state,
            "phase": self.phase,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 1),
            "error": self.error,
        }


_state = _RetrainState()


def get_status() -> dict[str, Any]:
    return _state.as_dict()


# ---------------------------------------------------------------------------
# Stats: pending corrections + last retrained
# ---------------------------------------------------------------------------


def get_last_retrained_at() -> Optional[str]:
    if not LAST_RETRAINED_FILE.exists():
        return None
    return LAST_RETRAINED_FILE.read_text(encoding="utf-8").strip() or None


def get_pending_count(db: Session) -> int:
    """How many command_corrections rows landed AFTER the last retrain.
    Falls back to the full count if no retrain has ever happened."""
    last = get_last_retrained_at()
    stmt = select(func.count(CommandCorrectionTable.id))  # type: ignore[arg-type]
    if last:
        try:
            cutoff = datetime.fromisoformat(last)
            stmt = stmt.where(CommandCorrectionTable.created_at > cutoff)
        except ValueError:
            pass
    return int(db.exec(stmt).one() or 0)


# ---------------------------------------------------------------------------
# Subprocess + watchdog
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _watch_sidecar(stop: asyncio.Event) -> None:
    """Poll the script's status sidecar every second and copy the phase
    into _state.phase. Exits when `stop` is set (subprocess finished)."""
    last_phase: Optional[str] = None
    while not stop.is_set():
        try:
            if STATUS_FILE.exists():
                raw = STATUS_FILE.read_text(encoding="utf-8")
                payload = json.loads(raw or "{}")
                phase = payload.get("phase")
                if isinstance(phase, str) and phase != last_phase:
                    _state.phase = phase
                    last_phase = phase
        except Exception as exc:  # noqa: BLE001
            _log.debug("sidecar read failed: %s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


def _hot_reload_classifier() -> None:
    """Replace the in-memory IntentClassifier with the freshly trained
    weights. Lazy import so we don't spin up torch on app startup."""
    from services import intent_classifier as ic  # noqa: PLC0415
    try:
        clf = ic.IntentClassifier.load(settings.intent_model_dir)
        ic.set_classifier(clf)
        _log.info("classifier hot-reloaded from %s", settings.intent_model_dir)
    except Exception as exc:  # noqa: BLE001
        _log.exception("hot-reload failed: %s", exc)
        raise


async def _run_subprocess() -> tuple[int, str]:
    """Spawn `python retrain_from_feedback.py`, capture the tail of
    stderr for diagnostics, return (exit_code, stderr_tail)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    backend_dir = Path(__file__).resolve().parents[1]
    py = str(backend_dir / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable  # fallback to whatever python is running uvicorn

    log_path = LOG_DIR / f"retrain_{int(time.time())}.log"
    proc = await asyncio.create_subprocess_exec(
        py,
        str(RETRAIN_SCRIPT),
        cwd=str(backend_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    log_path.write_bytes(b"=== STDOUT ===\n" + (stdout or b"") + b"\n=== STDERR ===\n" + (stderr or b""))
    tail = "\n".join((stderr or b"").decode("utf-8", "ignore").splitlines()[-5:])
    return proc.returncode or 0, tail


async def _retrain_task() -> None:
    """Whole life-cycle of one retrain: subprocess + sidecar watcher +
    hot-reload on success. Updates _state through every transition."""
    started = _now_iso()
    _state.state = "running"
    _state.phase = "preparing"
    _state.started_at = started
    _state.finished_at = None
    _state.error = None

    stop = asyncio.Event()
    watcher = asyncio.create_task(_watch_sidecar(stop))

    try:
        rc, tail = await _run_subprocess()
        stop.set()
        await watcher
    except Exception as exc:  # noqa: BLE001
        stop.set()
        await watcher
        _state.state = "failed"
        _state.phase = "failed"
        _state.error = f"orchestrator error: {exc}"
        _state.finished_at = _now_iso()
        return

    if rc != 0:
        _state.state = "failed"
        _state.phase = "failed"
        _state.error = tail or f"exit code {rc}"
        _state.finished_at = _now_iso()
        return

    try:
        _hot_reload_classifier()
    except Exception as exc:  # noqa: BLE001
        _state.state = "failed"
        _state.phase = "failed"
        _state.error = f"reload failed: {exc}"
        _state.finished_at = _now_iso()
        return

    _state.state = "completed"
    _state.phase = "done"
    _state.finished_at = _now_iso()


def start_retrain() -> bool:
    """Kick off the retrain. Returns False if one is already running."""
    if _state.state == "running":
        return False
    asyncio.get_event_loop().create_task(_retrain_task())
    return True
