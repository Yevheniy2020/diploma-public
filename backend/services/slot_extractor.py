from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

_log = logging.getLogger(__name__)


_DIGITS_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

_NUMBER_WORDS: dict[str, float] = {
    "нуль": 0,
    "один": 1, "одна": 1, "одне": 1, "одного": 1, "одної": 1,
    "два": 2, "дві": 2, "двох": 2,
    "три": 3, "трьох": 3,
    "чотири": 4, "чотирьох": 4,
    "п'ять": 5, "пять": 5, "п’ять": 5, "пʼять": 5,
    "шість": 6, "шести": 6,
    "сім": 7, "семи": 7,
    "вісім": 8, "восьми": 8,
    "дев'ять": 9, "девять": 9, "дев’ять": 9, "девʼять": 9,
    "десять": 10, "десяти": 10,
    "пів": 0.5, "півтора": 1.5, "півтори": 1.5,
    "zero": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "half": 0.5,
}

_MULTIPLIER_WORDS: dict[str, int] = {
    "двічі": 2, "тричі": 3, "чотири рази": 4, "п'ять разів": 5,
    "twice": 2, "thrice": 3,
}


def _normalize_apostrophes(s: str) -> str:
    return s.replace("ʼ", "'").replace("’", "'").replace("`", "'")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_apostrophes(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def _first_number(text: str) -> float | None:
    t = _norm(text)
    m = _DIGITS_RE.search(t)
    if m:
        return float(m.group(1).replace(",", "."))
    for word, val in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return float(val)
    return None


def _multiplier(text: str) -> int:
    t = _norm(text)
    m = re.search(r"\b(\d+)\s*(?:раз(?:и|ів|у)?|times?)\b", t)
    if m:
        return max(1, int(m.group(1)))
    for word, val in _MULTIPLIER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return val
    m2 = re.search(
        r"\b(один|два|три|чотири|п'ять|one|two|three|four|five)\s+(?:раз(?:и|ів|у)?|times?)\b",
        t,
    )
    if m2:
        return int(_NUMBER_WORDS.get(m2.group(1), 1))
    return 1


_DIST_PATTERNS = [
    (re.compile(r"\bпів[\s-]?(?:метра|метру|метр)\b", re.I), 0.5),
    (re.compile(r"\bhalf\s+(?:a\s+)?(?:meter|metre)\b", re.I), 0.5),
    (re.compile(r"\bпівтор[аи]\s+метр", re.I), 1.5),
]

_CM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:см|cm|сантиметр\w*|centim(?:etre|eter)s?)\b", re.I)
_M_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:м|m|метр\w*|met(?:re|er)s?)\b", re.I)
_REPETITION_NUMBER_RE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|"
    r"один|два|три|чотири|п'ять|пять|шість|сім|вісім|"
    r"one|two|three|four|five|six|seven|eight)"
    r"\s*(?:раз(?:и|ів|у)?|times?)\b",
    re.I,
)


def _parse_distance(text: str) -> float | None:
    t = _norm(text)
    for pat, val in _DIST_PATTERNS:
        if pat.search(t):
            return val
    t_no_rep = _REPETITION_NUMBER_RE.sub(" ", t)
    if (m := _CM_RE.search(t_no_rep)):
        return float(m.group(1).replace(",", ".")) / 100.0
    if (m := _M_RE.search(t_no_rep)):
        return float(m.group(1).replace(",", "."))
    if t_no_rep != t and _first_number(t_no_rep) is None:
        return None
    n = _first_number(t_no_rep)
    if n is not None:
        return n
    return None


_FORWARD_RE = re.compile(
    r"\b(вперед|уперед|forward|ahead|straight)\b", re.I
)
_BACKWARD_RE = re.compile(
    r"\b(назад|back|backwards?|reverse|behind)\b", re.I
)
_LEFT_RE = re.compile(
    r"\b(ліворуч|наліво|вліво|лівіше|left|leftwards?)\b", re.I
)
_RIGHT_RE = re.compile(
    r"\b(праворуч|направо|вправо|правіше|right|rightwards?)\b", re.I
)


def _direction(text: str) -> str | None:
    t = _norm(text)
    if _FORWARD_RE.search(t):
        return "forward"
    if _BACKWARD_RE.search(t):
        return "backward"
    if _LEFT_RE.search(t):
        return "left"
    if _RIGHT_RE.search(t):
        return "right"
    return None


_HALF_TURN_RE = re.compile(
    r"\b(?:розверн|пів\s*оберт|пол[уi][-\s]?оберт|180\s*град|spin\s+around|half\s+turn|180\s*degrees?|u[-\s]?turn)",
    re.I,
)
_QUARTER_TURN_RE = re.compile(
    r"\b(?:чверть\s*оберт|quarter\s+turn)\b", re.I
)
_DEG_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:°|град(?:ус(?:а|ів|у)?)?|deg(?:rees?)?)\b", re.I
)


def _rotate_slots(text: str) -> dict[str, float]:
    t = _norm(text)
    sign = 0
    if _LEFT_RE.search(t):
        sign = +1
    elif _RIGHT_RE.search(t):
        sign = -1
    if re.search(r"counter[-\s]?clockwise|проти\s+годинник", t):
        sign = +1
    elif re.search(r"\bclockwise\b|за\s+годинник", t):
        sign = -1

    if _HALF_TURN_RE.search(t):
        magnitude = 180.0
        if sign == 0:
            sign = +1
    elif _QUARTER_TURN_RE.search(t):
        magnitude = 90.0
        if sign == 0:
            sign = +1
    else:
        m = _DEG_RE.search(t)
        if m:
            magnitude = float(m.group(1).replace(",", "."))
        else:
            magnitude = 90.0

    mult = _multiplier(t)
    delta_deg = sign * magnitude * mult if sign else magnitude * mult
    delta_deg = max(-360.0, min(360.0, delta_deg))
    return {"delta_deg": delta_deg}


def _name_list(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("name"), str):
            out.append(it["name"])
        elif isinstance(it, str):
            out.append(it)
    return out


def _fuzzy_name(text: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    t = _norm(text)
    for name in candidates:
        n = _norm(name)
        if n and (n in t or _stem(n) in t):
            return name
    try:
        from rapidfuzz import fuzz  # type: ignore[import-untyped]
    except ImportError:
        _log.warning("rapidfuzz not installed — fuzzy name match disabled")
        return None
    best_name: str | None = None
    best_score = 0.0
    for name in candidates:
        score = fuzz.partial_ratio(_norm(name), t)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score >= 80 else None


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    while word and word[-1] in "ауоиієїяюе":
        word = word[:-1]
        if len(word) <= 3:
            break
    return word


_MIN_SPAN_LEN = 2


def _name_spans(text: str, model_spans: dict[str, list[tuple[int, int]]] | None) -> list[str]:
    if not model_spans:
        return []
    out: list[str] = []
    for s, e in model_spans.get("NAME", []):
        chunk = text[s:e].strip()
        if len(chunk) >= _MIN_SPAN_LEN:
            out.append(chunk)
    return out


def _drive_relative_slots(text: str) -> dict[str, Any]:
    direction = _direction(text)
    distance = _parse_distance(text)
    if distance is None:
        distance = 1.0
    mult = _multiplier(text)
    distance *= mult
    out: dict[str, Any] = {}
    if direction is not None:
        out["direction"] = direction
    out["distance_m"] = distance
    return out


def _navigate_slots(
    text: str,
    candidates: list[str],
    name_spans: list[str],
) -> dict[str, Any]:
    target = _fuzzy_name(text, candidates)
    if target is None:
        for span_text in name_spans:
            target = _fuzzy_name(span_text, candidates)
            if target:
                break
    out: dict[str, Any] = {}
    if target:
        out["target"] = target
    return out


_START_SPACE_HEAD_RE = re.compile(
    r"^\s*(?:почни|почати|починаю|починати|розпочни|розпочати|"
    r"стартуй|стартую|начни|начинаю|"
    r"малюй|малювати|намалюй|намалювати|"
    r"створи|створити|створюй|"
    r"познач|познач(?:ити|у)?|"
    r"додай|додати|"
    r"start|create|add|mark|draw)\s+"
    r"(?:малювати\s+)?"
    r"(?:(?:кімнат[ауиі]|зон[ауи]|област[ьі]|місц[еяю]|мітк[ауи]|"
    r"ділянк[ауи]|регіон[ауі]?|space|area|zone|place|label)\s+)?",
    re.I,
)


def _start_space_slots(text: str, name_spans: list[str]) -> dict[str, Any]:
    if name_spans:
        return {"name": _norm(name_spans[0])}
    t = _norm(text)
    m = _START_SPACE_HEAD_RE.match(t)
    if m:
        t = t[m.end():]
    t = re.sub(r"\s+", " ", t).strip(" .,!?")
    return {"name": t} if t else {}


def _delete_space_slots(
    text: str,
    candidates: list[str],
    name_spans: list[str],
) -> dict[str, Any]:
    name = _fuzzy_name(text, candidates)
    if name is None:
        for span_text in name_spans:
            name = _fuzzy_name(span_text, candidates)
            if name:
                break
    return {"name": name} if name else {}


def _rename_space_slots(
    text: str,
    candidates: list[str],
    name_spans: list[str],
) -> dict[str, Any]:
    t = _norm(text)
    rename_verbs = (
        r"переназви(?:ти)?|переназва(?:ти)?|"
        r"перейменуй(?:те)?|перейменува(?:ти)?|"
        r"переіменуй(?:те)?|переіменува(?:ти)?|"
        r"зміни(?:ти)?\s+(?:назву|ім[\'ʼ’]я|найменування)|"
        r"змін(?:а|и)\s+назв(?:и|у)"
    )
    m = re.search(
        rf"(?:{rename_verbs})(?:\s+(?:кімнату|кімнат[ауи]|зон[ауи]|"
        r"област[ьі]|місц[еяю]|мітк[ауи]|ділянк[ауи]|регіон[ауі]?|"
        r"space|area|zone|place|label))?"
        r"\s+(.+?)\s+(?:на|у|в)\s+(.+)$",
        t,
    )
    if not m:
        m = re.search(r"rename\s+(?:the\s+)?(?:space\s+)?(.+?)\s+to\s+(.+)$", t)
    if not m:
        m = re.search(r"change\s+(?:the\s+)?(?:name\s+of\s+)?(.+?)\s+to\s+(.+)$", t)
    if m:
        old_raw = m.group(1).strip(" .,!?")
        new_raw = m.group(2).strip(" .,!?")
        old_name = _fuzzy_name(old_raw, candidates) or old_raw
        if old_name and new_raw:
            return {"old_name": old_name, "new_name": new_raw}

    if len(name_spans) == 2 and len(name_spans[1].strip()) >= 3:
        old_raw, new_raw = name_spans[0], name_spans[1]
        old_name = _fuzzy_name(old_raw, candidates) or _norm(old_raw)
        new_name = _norm(new_raw)
        if old_name and new_name:
            return {"old_name": old_name, "new_name": new_name}
    return {}


def extract_slots(
    text: str,
    intent: str,
    active_spaces: Iterable[Any] | None = None,
    model_spans: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    text = text or ""

    spaces = _name_list(active_spaces or [])
    name_spans = _name_spans(text, model_spans)

    if intent == "ROTATE":
        return _rotate_slots(text)
    if intent == "DRIVE_RELATIVE":
        return _drive_relative_slots(text)
    if intent == "NAVIGATE":
        return _navigate_slots(text, spaces, name_spans)
    if intent == "DELETE_SPACE":
        return _delete_space_slots(text, spaces, name_spans)
    if intent == "RENAME_SPACE":
        return _rename_space_slots(text, spaces, name_spans)
    if intent == "START_SPACE":
        return _start_space_slots(text, name_spans)
    if intent in {"FINISH_SPACE", "CANCEL_SPACE", "STOP", "RETURN_HOME"}:
        return {}
    return {"original_text": text}
