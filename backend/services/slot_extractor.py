"""Deterministic slot extraction for the trained intent classifier.

The classifier only decides which intent a command belongs to. Slots
(target label, distance, sign, etc.) are extracted here from the raw
transcription using regex and fuzzy-match against the live label list.

Why deterministic instead of a learned slot tagger:
- Rotation sign («наліво» = +90) is the dominant LLM failure today; a
  one-line lexical rule fixes 100 % of cases.
- New labels appear at runtime — there's nothing to retrain. The fuzzy
  matcher reads the `active_labels` list from each request's context.

Public surface:
    extract_slots(text, intent, active_labels) -> dict[str, Any]
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Number parsing (UK + EN words + digits)
# ---------------------------------------------------------------------------

_DIGITS_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

# Word → numeric value. Covers the forms we actually encounter in seed
# transcriptions; longer numerals («двадцять п'ять») can be added later
# but are uncommon for distance/angle commands.
_NUMBER_WORDS: dict[str, float] = {
    # uk
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
    # en
    "zero": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "half": 0.5,
}

# Multiplier words («N разів / N times»).
_MULTIPLIER_WORDS: dict[str, int] = {
    # uk
    "двічі": 2, "тричі": 3, "чотири рази": 4, "п'ять разів": 5,
    # en
    "twice": 2, "thrice": 3,
}


def _normalize_apostrophes(s: str) -> str:
    return s.replace("ʼ", "'").replace("’", "'").replace("`", "'")


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace + normalize apostrophes."""
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_apostrophes(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def _first_number(text: str) -> float | None:
    """Return the first numeric mention (digits or word). None if absent."""
    t = _norm(text)
    # Compound: «N разів» / «N times» — return N (caller handles multiplier separately).
    m = _DIGITS_RE.search(t)
    if m:
        return float(m.group(1).replace(",", "."))
    for word, val in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return float(val)
    return None


def _multiplier(text: str) -> int:
    """Return repetition count if present («тричі», «3 рази», «twice»). Default 1."""
    t = _norm(text)
    # «N разів / N раз / N times»
    m = re.search(r"\b(\d+)\s*(?:раз(?:и|ів|у)?|times?)\b", t)
    if m:
        return max(1, int(m.group(1)))
    # word-based: «двічі», «тричі», «twice», «thrice»
    for word, val in _MULTIPLIER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return val
    # «N times» with worded N
    m2 = re.search(
        r"\b(один|два|три|чотири|п'ять|one|two|three|four|five)\s+(?:раз(?:и|ів|у)?|times?)\b",
        t,
    )
    if m2:
        return int(_NUMBER_WORDS.get(m2.group(1), 1))
    return 1


# ---------------------------------------------------------------------------
# Distance (returns metres)
# ---------------------------------------------------------------------------

_DIST_PATTERNS = [
    # «півметра / півметру / half a meter»
    (re.compile(r"\bпів[\s-]?(?:метра|метру|метр)\b", re.I), 0.5),
    (re.compile(r"\bhalf\s+(?:a\s+)?(?:meter|metre)\b", re.I), 0.5),
    # «півтора метри / one and a half meters»
    (re.compile(r"\bпівтор[аи]\s+метр", re.I), 1.5),
]

_CM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:см|cm|сантиметр\w*|centim(?:etre|eter)s?)\b", re.I)
_M_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:м|m|метр\w*|met(?:re|er)s?)\b", re.I)
# A number that is followed by «раз / times» is a repetition counter, not
# a distance — used to rule it out of distance parsing so the multiplier
# isn't double-counted («два рази вперед» = 2× default 1 m, not 2 × 2 m).
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
    # Strip «N раз / times» before searching so its number doesn't masquerade
    # as the distance.
    t_no_rep = _REPETITION_NUMBER_RE.sub(" ", t)
    if (m := _CM_RE.search(t_no_rep)):
        return float(m.group(1).replace(",", ".")) / 100.0
    if (m := _M_RE.search(t_no_rep)):
        return float(m.group(1).replace(",", "."))
    # Bare number with no unit — only count it as distance if no repetition
    # token swallowed it; otherwise the user meant pure repetition with the
    # default distance.
    if t_no_rep != t and _first_number(t_no_rep) is None:
        return None
    n = _first_number(t_no_rep)
    if n is not None:
        return n
    return None


# ---------------------------------------------------------------------------
# Direction (DRIVE_RELATIVE)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ROTATE: sign + magnitude
# ---------------------------------------------------------------------------

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
    # 1) sign from lexicon — left = +, right = -
    sign = 0
    if _LEFT_RE.search(t):
        sign = +1
    elif _RIGHT_RE.search(t):
        sign = -1
    # «counterclockwise» / «проти годинникової»
    if re.search(r"counter[-\s]?clockwise|проти\s+годинник", t):
        sign = +1
    elif re.search(r"\bclockwise\b|за\s+годинник", t):
        sign = -1

    # 2) magnitude
    if _HALF_TURN_RE.search(t):
        magnitude = 180.0
        if sign == 0:
            sign = +1  # «розвернись» without a side — pick CCW
    elif _QUARTER_TURN_RE.search(t):
        magnitude = 90.0
        if sign == 0:
            sign = +1
    else:
        m = _DEG_RE.search(t)
        if m:
            magnitude = float(m.group(1).replace(",", "."))
        else:
            magnitude = 90.0  # default for «поверни ліворуч»

    # 3) repetition: «3 рази / тричі / twice»
    mult = _multiplier(t)
    delta_deg = sign * magnitude * mult if sign else magnitude * mult
    # Clamp to dispatcher's accepted range.
    delta_deg = max(-360.0, min(360.0, delta_deg))
    return {"delta_deg": delta_deg}


# ---------------------------------------------------------------------------
# Space name fuzzy-match
# ---------------------------------------------------------------------------


def _name_list(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("name"), str):
            out.append(it["name"])
        elif isinstance(it, str):
            out.append(it)
    return out


def _fuzzy_name(text: str, candidates: list[str]) -> str | None:
    """Return the closest candidate name found in `text`, or None.

    Two-tier strategy: exact substring first (catches «їдь до кухні»
    when name is «кухня»), then rapidfuzz partial ratio."""
    if not candidates:
        return None
    t = _norm(text)
    # Exact substring (case-insensitive, with cheap stem variants).
    for name in candidates:
        n = _norm(name)
        if n and (n in t or _stem(n) in t):
            return name
    # Fuzzy: split text into tokens, score each against each candidate.
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
    """Crude Ukrainian stem — drop 1-2 trailing vowels so «кухні / кухню»
    matches «кухня». Conservative enough not to collide between distinct
    names that share short stems."""
    if len(word) <= 3:
        return word
    while word and word[-1] in "ауоиієїяюе":
        word = word[:-1]
        if len(word) <= 3:
            break
    return word


# ---------------------------------------------------------------------------
# Per-intent extractors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model-span helpers
# ---------------------------------------------------------------------------

# Drop very short spans — the slot tagger occasionally emits 1-char
# false positives on punctuation tokens.
_MIN_SPAN_LEN = 2


def _name_spans(text: str, model_spans: dict[str, list[tuple[int, int]]] | None) -> list[str]:
    """Return the substrings the slot tagger marked as NAME, in order,
    filtering out implausibly short ones."""
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
    # `candidates` is the canonical space-name list — fuzzy-match the whole
    # utterance first; fall back to fuzzy-matching the model's NAME span
    # text (helps when the user's wording happens to miss the name in
    # active form, e.g. an inflected mention not in our stem table).
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


# Verb head for START_SPACE — strips «почни малювати <synonym> NAME» / «створи
# мітку NAME» / «start space NAME». The synonym group accepts every word the
# operator might reach for: кімнат*, зон*, област*, місц*, мітк*, ділянк*,
# регіон*. Optional, so plain «почни NAME» also strips cleanly.
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
    """Fallback name extraction for START_SPACE. The regex preempt in
    `voice_intent.py` normally handles this path, but if Groq emits
    START_SPACE without a name slot (or the trained classifier ever picks
    it up), this strips the verb head and returns the rest as the space
    name."""
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
    """Resolve the space name to delete. Prefer fuzzy-match against the
    canonical space list; fall back to fuzzy-matching the slot tagger's
    NAME span."""
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
    """Best-effort split: «переназви X на Y» / «rename X to Y». Try the
    Ukrainian rename-verb regex first — it covers the canonical phrasings
    and is robust to inflection. Only fall back to the slot tagger's NAME
    spans when the regex matches nothing, because the joint model tends
    to emit sub-word fragments for the «new_name» part (e.g. «їдальню»
    splits into ['ї','даль','ню'] which would produce a useless slot)."""
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

    # Last-ditch: trust the tagger only when it gave exactly two spans
    # AND the second one is at least 3 chars (filters out subword noise
    # like 'ню' / 'ї'). Otherwise return empty so the dispatcher can ask
    # the operator for confirmation rather than commit a garbage name.
    if len(name_spans) == 2 and len(name_spans[1].strip()) >= 3:
        old_raw, new_raw = name_spans[0], name_spans[1]
        old_name = _fuzzy_name(old_raw, candidates) or _norm(old_raw)
        new_name = _norm(new_raw)
        if old_name and new_name:
            return {"old_name": old_name, "new_name": new_name}
    return {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_slots(
    text: str,
    intent: str,
    active_spaces: Iterable[Any] | None = None,
    model_spans: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Extract intent-specific parameters from a transcription. Returns
    the same shape that `backend.routers.voice._dispatch_intent_one`
    expects.

    `active_spaces` is the canonical list of {name, description} dicts
    that the voice router builds — newly created spaces appear there on
    the next request, so no retraining is required for the model to use
    them.

    `model_spans` is the slot-tagger output for the joint classifier:
    a dict mapping entity type to character-level (start, end) tuples
    in `text`. May be None / empty for legacy intent-only checkpoints,
    in which case all routing falls back to regex + fuzzy_match.

    Before any other logic, an exact match against `learned_overrides`
    short-circuits and returns the operator-taught params verbatim —
    that's how «крутись» → ROTATE +360 sticks across requests."""
    text = text or ""
    # Operator-taught override wins when intent agrees. We don't accept
    # cross-intent matches: an override stored as ROTATE shouldn't apply
    # if the classifier (or caller) currently asserts NAVIGATE.
    try:
        from services import learned_overrides as _lo  # noqa: PLC0415
        match = _lo.lookup(text)
        if match is not None and match.get("intent") == intent:
            params = match.get("params") or {}
            return dict(params)
    except Exception as exc:  # noqa: BLE001
        _log.warning("learned_overrides lookup failed: %s", exc)

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
