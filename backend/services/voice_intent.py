import asyncio
import logging
import re
import time
from typing import Any, Optional

from groq import AsyncGroq

from config import settings


_logger = logging.getLogger(__name__)
_client = AsyncGroq(api_key=settings.groq_api_key)

_STT_MODEL = "whisper-large-v3"

_INTENT_VALUES = {
    "NAVIGATE",
    "DRIVE_RELATIVE",
    "ROTATE",
    "DELETE_SPACE",
    "RENAME_SPACE",
    "START_SPACE",
    "FINISH_SPACE",
    "CANCEL_SPACE",
    "STOP",
    "RETURN_HOME",
    "UNKNOWN",
}


def _space_names(context: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in context.get("active_spaces") or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _whisper_prompt(context: dict[str, Any]) -> str:
    names = _space_names(context)
    space_part = ", ".join(names) if names else "кухня, спальня, база"
    bias = (
        f"Команди українською для мобільного робота. "
        f"Назви ділянок: {space_part}. "
        f"Ключові слова: кімната, зона, область, місце, мітка, ділянка, "
        f"метр, метри, сантиметри, "
        f"вперед, назад, праворуч, ліворуч, наліво, направо, "
        f"вище, нижче, правіше, лівіше, перед, за, над, під, пів метра."
    )
    return bias[:880]


def _filename_for_mime(mime: str) -> str:
    base = (mime or "").split(";")[0].strip().lower()
    if "webm" in base:
        return "audio.webm"
    if "ogg" in base:
        return "audio.ogg"
    if "wav" in base:
        return "audio.wav"
    if "mp4" in base or "m4a" in base:
        return "audio.m4a"
    if "mpeg" in base or "mp3" in base:
        return "audio.mp3"
    return "audio.webm"


async def _transcribe(
    audio_bytes: bytes, mime_type: str, context: dict[str, Any]
) -> str:
    fname = _filename_for_mime(mime_type)
    resp = await _client.audio.transcriptions.create(
        file=(fname, audio_bytes),
        model=_STT_MODEL,
        prompt=_whisper_prompt(context),
        response_format="text",
        temperature=0.0,
    )
    if isinstance(resp, str):
        return resp.strip()
    return getattr(resp, "text", str(resp)).strip()


def _classify_one_segment(
    clf: Any, slots_mod: Any, text: str, context: dict[str, Any]
) -> tuple[str, dict[str, Any], float]:
    pred = clf.predict(text)
    intent, confidence = pred[0], pred[1]
    model_spans = pred[2] if len(pred) > 2 else {}
    if intent not in _INTENT_VALUES:
        intent = "UNKNOWN"
    params = slots_mod.extract_slots(
        text,
        intent,
        context.get("active_spaces"),
        model_spans=model_spans,
    )
    params["original_text"] = text
    params["_confidence"] = round(float(confidence), 3)
    return intent, params, float(confidence)


def _split_at_offsets(text: str, offsets: list[int]) -> list[str]:
    if not offsets:
        return [text]
    points = sorted(set(o for o in offsets if 0 < o < len(text)))
    parts: list[str] = []
    last = 0
    for p in points:
        seg = text[last:p].strip(" .,!?;:")
        if seg:
            parts.append(seg)
        last = p
    tail = text[last:].strip(" .,!?;:")
    if tail:
        parts.append(tail)
    cleaned: list[str] = [parts[0]] if parts else []
    for seg in parts[1:]:
        m = re.match(r"^\s*\S+\s+", seg)
        cleaned.append(seg[m.end():].strip() if m else seg)
    return [c for c in cleaned if c]


def _try_classify_local(
    transcription: str, context: dict[str, Any]
) -> Optional[dict[str, Any]]:
    from services import intent_classifier as _ic
    from services import slot_extractor as _slots

    clf = _ic.get_classifier()
    if clf is None:
        return None
    t0 = time.monotonic()
    pred = clf.predict(transcription)
    intent = pred[0]
    confidence = float(pred[1])
    model_spans = pred[2] if len(pred) > 2 else {}
    boundary_offsets: list[int] = pred[3] if len(pred) > 3 else []

    if boundary_offsets:
        segments = _split_at_offsets(transcription, boundary_offsets)
        if len(segments) >= 2:
            frames: list[tuple[str, dict[str, Any], float]] = [
                _classify_one_segment(clf, _slots, seg, context)
                for seg in segments
            ]
            confs = [f[2] for f in frames]
            min_conf = min(confs)
            elapsed_ms = (time.monotonic() - t0) * 1000

            head_intent, head_params, _ = frames[0]
            head_params["_confidence"] = round(min_conf, 3)
            head_params["original_text"] = transcription
            head_params["_source"] = "xlm_boundary"
            pending = [
                {"intent": i, "params": p} for (i, p, _) in frames[1:]
                if i in _INTENT_VALUES
            ]

            if min_conf < settings.intent_high_confidence_threshold:
                return {
                    "intent": "UNCERTAIN",
                    "params": {
                        "original_text": transcription,
                        "_confidence": round(min_conf, 3),
                        "_predicted_intent": head_intent,
                        "_predicted_params": head_params,
                        "_predicted_pending": pending,
                    },
                    "_ms": elapsed_ms,
                }

            result: dict[str, Any] = {
                "intent": head_intent,
                "params": head_params,
                "_ms": elapsed_ms,
            }
            if pending:
                result["_pending"] = pending
            return result

    if intent not in _INTENT_VALUES:
        intent = "UNKNOWN"
    params = _slots.extract_slots(
        transcription,
        intent,
        context.get("active_spaces"),
        model_spans=model_spans,
    )
    params["original_text"] = transcription
    params["_confidence"] = round(confidence, 3)
    elapsed_ms = (time.monotonic() - t0) * 1000

    if confidence < settings.intent_high_confidence_threshold:
        return {
            "intent": "UNCERTAIN",
            "params": {
                "original_text": transcription,
                "_confidence": round(confidence, 3),
                "_predicted_intent": intent,
                "_predicted_params": params,
            },
            "_ms": elapsed_ms,
        }
    return {"intent": intent, "params": params, "_ms": elapsed_ms}


async def parse_voice_command(
    audio_bytes: bytes,
    mime_type: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    if len(audio_bytes) < 1000:
        return {
            "intent": "UNKNOWN",
            "params": {"original_text": "", "error": "audio too short"},
        }

    last_err: Exception | None = None
    transcription = ""

    for attempt in range(2):
        try:
            stt_start = time.monotonic()
            transcription = await _transcribe(audio_bytes, mime_type, context)
            stt_ms = (time.monotonic() - stt_start) * 1000

            if not transcription:
                _logger.info("groq stt empty transcription attempt=%d", attempt + 1)
                return {
                    "intent": "UNKNOWN",
                    "params": {"original_text": "", "error": "empty transcription"},
                }

            clf_result = await asyncio.to_thread(
                _try_classify_local, transcription, context
            )
            if clf_result is None:
                _logger.warning(
                    "local classifier unavailable — returning UNKNOWN for %r",
                    transcription,
                )
                return {
                    "intent": "UNKNOWN",
                    "params": {
                        "original_text": transcription,
                        "error": "classifier unavailable",
                    },
                }

            clf_ms = clf_result.pop("_ms", 0.0)
            steps = 1 + len(clf_result.get("_pending", []))
            _logger.info(
                "local parse_voice attempt=%d stt_ms=%.0f clf_ms=%.0f "
                "intent=%s conf=%.2f steps=%d",
                attempt + 1,
                stt_ms,
                clf_ms,
                clf_result["intent"],
                clf_result["params"].get("_confidence", 0.0),
                steps,
            )
            return clf_result

        except Exception as e:
            last_err = e
            _logger.warning("voice pipeline error attempt=%d: %s", attempt + 1, e)
            if attempt < 1:
                await asyncio.sleep(1.0)

    _logger.error("voice pipeline failed after retries: %s", last_err)
    return {
        "intent": "UNKNOWN",
        "params": {
            "original_text": transcription,
            "error": str(last_err) if last_err else "unknown error",
        },
    }
