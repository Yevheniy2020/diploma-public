import ast
import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

from groq import AsyncGroq

from config import settings


_logger = logging.getLogger(__name__)
_client = AsyncGroq(api_key=settings.groq_api_key)

_STT_MODEL = "whisper-large-v3"
_LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Conjunctions that signal "user chained two commands". The trained
# classifier only emits ONE intent per utterance, so when these appear
# we skip it and route the request to the Groq tools-path, which can
# split the utterance into a multi-step VoiceFollowUp queue.
_CHAIN_MARKER_RE = re.compile(
    r"\b(?:і|й|та|потім|після\s+цього|після\s+того|"
    r"and\s+then|then|and)\b",
    re.I,
)

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


SYSTEM_PROMPT = """\
You are a voice command parser for an indoor mobile robot.
You receive a Ukrainian transcription and the current map context.
Identify the intent and extract parameters. Return JSON only, no extra text.

The transcription may be imperfect (mishearings, wrong word forms, partial words).
The semantic primitive is a *space* — a named polygon (region) on the map.
A space can be a whole room (кухня, спальня) but also a smaller area (зона
шафи, зарядна область, мітка біля столу). The user may refer to it in
Ukrainian as "кімната", "зона", "область", "місце", "мітка", "ділянка",
"регіон" — treat all of these as synonyms for the same primitive.
When matching target names for NAVIGATE, pick the closest match from
CONTEXT.active_spaces. The robot drives INTO the space (to the nearest free
cell inside the polygon), not to a fixed point. Do not invent names that are
not on the map. If nothing reasonable matches, return UNKNOWN.

SEMANTIC INFERENCE — be helpful, not strict:
The user may describe a place by FUNCTION instead of naming the space.
Match the user's phrasing to the closest existing space name in
CONTEXT.active_spaces by meaning.

Examples of function → typical space (only pick a space that ACTUALLY exists
in CONTEXT.active_spaces):
  - "де я їм" / "де готую" / "куди ходимо їсти" → kitchen-like space (кухня, їдальня)
  - "де я сплю" / "де ліжко" / "куди йду спати" → bedroom-like space (спальня)
  - "де телевізор" / "де відпочиваю" / "вітальня" → living-space-like space
  - "де миюся" / "ванна" / "туалет" → bathroom-like space
  - "робоче місце" / "де працюю" / "кабінет" → office-like space
  - "вхід" / "початкова точка" / "стартова" / "база" → the home space
Rules:
  1. The inferred space MUST already exist in CONTEXT.active_spaces.
     Never invent a space.
  2. If MULTIPLE spaces could fit, pick the single most natural one.
  3. If NO space name in active_spaces plausibly fits, return UNKNOWN.
  4. Prefer making a reasonable inference over UNKNOWN — being helpful is the
     priority. The user can correct with "стоп" if the guess is wrong.

AVAILABLE INTENTS:
- NAVIGATE: go to a known space. Either params.target (space name string) OR
  params.anchor ("previous_start" | "previous_goal") for anaphoric references.
- DRIVE_RELATIVE: free-form motion relative to current robot heading.
  Required params.direction ("forward" | "backward" | "left" | "right").
  Optional params.distance_m (number in METERS, default 1.0). Always convert
  centimeters to meters (10 см → 0.1, "пів метра" → 0.5, "сорок сантиметрів" → 0.4).
- DELETE_SPACE: delete a space by name. Field "name".
- RENAME_SPACE: rename a space. Fields "old_name", "new_name".
- START_SPACE: begin recording a space perimeter via voice demonstration. The
  operator will then drive the robot along the space boundary (or click corners
  on the map). Field "name" (space name string).
- FINISH_SPACE: close the in-progress perimeter and save it as a polygon. No
  params. Only meaningful if a START_SPACE is currently in progress.
- CANCEL_SPACE: discard the in-progress perimeter without saving. No params.
- STOP: halt motion. No params.
- RETURN_HOME: return to the home space. No params.
- UNKNOWN: if not understood, empty, or off-topic.

NORMALIZATION:
- Reduce space names to nominative singular ("кухні", "кухнею", "на кухню" → "кухня").
- Numbers from Ukrainian words: "два" → 2, "три" → 3, "півтора" → 1.5,
  "два з половиною" → 2.5, "сорок сантиметрів" → 0.4.
- "додому" / "на базу" / "повертайся" alone (no "назад") → RETURN_HOME.
- "стій" / "стоп" / "зупинись" → STOP.

ANAPHORA — references to the PREVIOUS navigation in CONTEXT:
- "назад" / "йди назад" / "повертайся назад" / "поверни назад" (no distance)
  → NAVIGATE with params.anchor="previous_start".
- "ще раз" / "повтори" / "знову туди" / "до тієї ж точки"
  → NAVIGATE with params.anchor="previous_goal".
- These are valid only when CONTEXT shows a Previous navigation. If
  CONTEXT says "Previous navigation: none" and the user uses an
  anaphoric phrase, return UNKNOWN with original_text preserved.

OFFSETS — position relative to a space (NAVIGATE only):
The user can describe a goal as "near space X, in direction Y, by distance Z".
Add params.offset = {"direction": <enum>, "distance_m": <number>} alongside
params.target. The offset is anchored at the space's centroid. Default distance
is 1.0 m if not stated. Distance words follow the same DISTANCE VOCABULARY
rules as DRIVE_RELATIVE (метр / см / пів метра).

Offset direction vocabulary (Ukrainian → enum):
- "вище" / "над" / "вгорі" / "зверху" → "north"
- "нижче" / "під" / "знизу" → "south"
- "праворуч від" / "правіше" / "справа від" → "east"
- "ліворуч від" / "лівіше" / "зліва від" → "west"
- "перед" / "попереду (кімнати)" → "forward"
- "за" / "позаду" / "ззаду" → "backward"

Use OFFSET only when the phrasing actually references a space
("нижче ціль-зона", "праворуч від кухні"). Plain "вперед" without a space name
is DRIVE_RELATIVE, not NAVIGATE+offset.

DIRECTION VOCABULARY (for DRIVE_RELATIVE):
- "вперед" / "прямо" / "уперед" → forward
- "назад" WITH a distance qualifier OR motion verb like "поїдь / здай / зміст"
  → backward (distinguishes from anaphoric "назад"; e.g.
  "поїдь назад на метр" = DRIVE_RELATIVE backward 1.0,
  but "йди назад" alone = NAVIGATE anchor=previous_start)
- "праворуч" / "направо" / "вправо" / "правіше" → right
- "ліворуч" / "наліво" / "вліво" / "лівіше" → left

DISTANCE VOCABULARY:
- "метр" / "метра" / "метри" / "метрів" / "м" → meters (×1.0)
- "сантиметр" / "сантиметра" / "сантиметри" / "сантиметрів" / "см" → ×0.01
- "пів метра" / "півметра" / "половина метра" → 0.5
- "трохи" / "ледь" without number → 0.3
- If no distance given for DRIVE_RELATIVE → use 1.0.

OUTPUT JSON SCHEMA: {"intent": "<INTENT>", "params": {...}}

EXAMPLES (transcription → JSON):
"йди на кухню" → {"intent":"NAVIGATE","params":{"target":"кухня"}}
"йди назад" → {"intent":"NAVIGATE","params":{"anchor":"previous_start"}}
"повертайся назад" → {"intent":"NAVIGATE","params":{"anchor":"previous_start"}}
"ще раз туди" → {"intent":"NAVIGATE","params":{"anchor":"previous_goal"}}
"повтори" → {"intent":"NAVIGATE","params":{"anchor":"previous_goal"}}
"поїдь вперед" → {"intent":"DRIVE_RELATIVE","params":{"direction":"forward","distance_m":1.0}}
"проїдь п'ять метрів" → {"intent":"DRIVE_RELATIVE","params":{"direction":"forward","distance_m":5.0}}
"проїдь 10 сантиметрів" → {"intent":"DRIVE_RELATIVE","params":{"direction":"forward","distance_m":0.1}}
"проїдь 10 см" → {"intent":"DRIVE_RELATIVE","params":{"direction":"forward","distance_m":0.1}}
"трохи направо" → {"intent":"DRIVE_RELATIVE","params":{"direction":"right","distance_m":0.3}}
"вправо на пів метра" → {"intent":"DRIVE_RELATIVE","params":{"direction":"right","distance_m":0.5}}
"ліворуч на два метри" → {"intent":"DRIVE_RELATIVE","params":{"direction":"left","distance_m":2.0}}
"здай назад на метр" → {"intent":"DRIVE_RELATIVE","params":{"direction":"backward","distance_m":1.0}}
"поїдь назад трохи" → {"intent":"DRIVE_RELATIVE","params":{"direction":"backward","distance_m":0.3}}
"стоп" → {"intent":"STOP","params":{}}
"зупинись" → {"intent":"STOP","params":{}}
"повертайся додому" → {"intent":"RETURN_HOME","params":{}}
"видали спальню" → {"intent":"DELETE_SPACE","params":{"name":"спальня"}}
"видали зону шафи" → {"intent":"DELETE_SPACE","params":{"name":"шафа"}}
"прибери мітку зарядна" → {"intent":"DELETE_SPACE","params":{"name":"зарядна"}}
"забери кухню з мапи" → {"intent":"DELETE_SPACE","params":{"name":"кухня"}}
"перейменуй кухню на їдальню" → {"intent":"RENAME_SPACE","params":{"old_name":"кухня","new_name":"їдальня"}}
"перейменуй зону шафи на склад" → {"intent":"RENAME_SPACE","params":{"old_name":"шафа","new_name":"склад"}}
"почни малювати кімнату спальня" → {"intent":"START_SPACE","params":{"name":"спальня"}}
"почни зону шафи" → {"intent":"START_SPACE","params":{"name":"шафа"}}
"стартуй область кухня" → {"intent":"START_SPACE","params":{"name":"кухня"}}
"намалюй мітку зарядна" → {"intent":"START_SPACE","params":{"name":"зарядна"}}
"створи місце біля столу" → {"intent":"START_SPACE","params":{"name":"стіл"}}
"почати ділянку вітальня" → {"intent":"START_SPACE","params":{"name":"вітальня"}}
"готово" [when drafting a space] → {"intent":"FINISH_SPACE","params":{}}
"завершити кімнату" → {"intent":"FINISH_SPACE","params":{}}
"зберегти зону" → {"intent":"FINISH_SPACE","params":{}}
"зберегти область" → {"intent":"FINISH_SPACE","params":{}}
"скасуй кімнату" → {"intent":"CANCEL_SPACE","params":{}}
"скасуй мітку" → {"intent":"CANCEL_SPACE","params":{}}
"відмінити" [when drafting a space] → {"intent":"CANCEL_SPACE","params":{}}
"піди нижче ціль-зона на метр" → {"intent":"NAVIGATE","params":{"target":"ціль-зона","offset":{"direction":"south","distance_m":1.0}}}
"правіше кухні на пів метра" → {"intent":"NAVIGATE","params":{"target":"кухня","offset":{"direction":"east","distance_m":0.5}}}
"над базою на 30 сантиметрів" → {"intent":"NAVIGATE","params":{"target":"база","offset":{"direction":"north","distance_m":0.3}}}
"перед спальнею" → {"intent":"NAVIGATE","params":{"target":"спальня","offset":{"direction":"forward","distance_m":1.0}}}
"ліворуч від ціль-зона" → {"intent":"NAVIGATE","params":{"target":"ціль-зона","offset":{"direction":"west","distance_m":1.0}}}
"піди туди, де я їм" [active_spaces has "кухня"] → {"intent":"NAVIGATE","params":{"target":"кухня"}}
"відведи мене у кімнату де я сплю" [active_spaces has "спальня"] → {"intent":"NAVIGATE","params":{"target":"спальня"}}
"повернись на стартову точку" → {"intent":"RETURN_HOME","params":{}}
"йди в кімнату де телевізор" [active_spaces = ["кухня","спальня","база-зона"]] → {"intent":"UNKNOWN","params":{"original_text":"<as transcribed>"}}
"""


def _space_names(context: dict[str, Any]) -> list[str]:
    """Extract space name strings from context. Used both for Whisper
    prompt biasing and for the tool-call enum so the model can't propose
    spaces that aren't on the active map."""
    names: list[str] = []
    for item in context.get("active_spaces") or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _whisper_prompt(context: dict[str, Any]) -> str:
    """Initial-prompt biasing for Whisper: vocabulary + space names.

    Whisper treats this as a phonetic / lexical hint, not an instruction.
    Listing the expected command vocabulary plus the spaces actually on the
    current map dramatically reduces mishearing of domain words ("кімната",
    "база") and makes space-name transcription robust to noise.
    """
    names = _space_names(context)
    space_part = ", ".join(names) if names else "кухня, спальня, база"
    # Whisper hard-limits the `prompt` parameter to 896 characters. Bias is
    # most effective for proper nouns (space names) and domain jargon — common
    # verbs (йди, поїдь) are already in Whisper's vocabulary, so we drop the
    # verb-heavy phrasing from earlier versions.
    bias = (
        f"Команди українською для мобільного робота. "
        f"Назви ділянок: {space_part}. "
        f"Ключові слова: кімната, зона, область, місце, мітка, ділянка, "
        f"метр, метри, сантиметри, "
        f"вперед, назад, праворуч, ліворуч, наліво, направо, "
        f"вище, нижче, правіше, лівіше, перед, за, над, під, пів метра."
    )
    # Defensive truncation in case space_part itself blows the budget.
    return bias[:880]


def _format_active_spaces(context: dict[str, Any]) -> str:
    """Render the active space names as a comma-separated list for the LLM
    context. Used by both the Groq SYSTEM_PROMPT and the Groq tools-path
    enum for space_name."""
    items = context.get("active_spaces") or []
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(f'"{item}"')
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            parts.append(f'"{item["name"]}"')
    return "[" + ", ".join(parts) + "]"


def _context_message(context: dict[str, Any]) -> str:
    pose = context.get("robot_pose", {"x": 0.0, "y": 0.0, "theta": 0.0})
    map_name = context.get("map_name", "")
    prev = context.get("previous_navigation")

    lines = [
        "CONTEXT:",
        f"- Map: {map_name}",
        f"- Active spaces on map: {_format_active_spaces(context)}",
        f"- Current robot position: x={pose.get('x', 0):.2f} y={pose.get('y', 0):.2f} theta={pose.get('theta', 0):.2f}",
    ]
    if isinstance(prev, dict):
        sa = prev.get("startedAt") if isinstance(prev.get("startedAt"), dict) else None
        g = prev.get("goal") if isinstance(prev.get("goal"), dict) else None
        if sa and g:
            lines.append(
                f'- Previous navigation: from ({sa.get("x", 0):.2f}, {sa.get("y", 0):.2f}) '
                f'to "{g.get("name", "?")}" at ({g.get("x", 0):.2f}, {g.get("y", 0):.2f})'
            )
        else:
            lines.append("- Previous navigation: none")
    else:
        lines.append("- Previous navigation: none")
    return "\n".join(lines)


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
    # No language hint: Whisper-large-v3 auto-detects between UK and EN.
    # The space-name prompt still steers transcription toward the active map.
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


# --- Function-calling path (gated by settings.voice_use_tools) ---------------

SHORT_SYSTEM_PROMPT = """\
You are a voice command parser for an indoor mobile robot. The user
speaks Ukrainian. Call one or more tools that together carry out the
command. If the user chains actions ("повернись ліворуч, поїдь вперед
на метр, потім поверни праворуч"), emit ONE tool call PER action in
the order the user spoke them — they will be executed sequentially on
the robot. If nothing fits, do not call a tool — reply briefly with
plain text.

Trust your own knowledge of Ukrainian morphology, number words, and
unit conversion (centimeters to meters, fractions, etc.). One arithmetic
case you must handle explicitly: REPETITION COUNTERS. When the user
says "N разів" / "тричі" / "двічі" / "чотири рази", multiply the
relevant numeric parameter by N before emitting one tool call — never
emit a single-step call when the user asked for a repeat. Examples:
"поверни ліворуч три рази" → rotate_in_place delta_deg=+270 (90×3);
"двічі направо" → delta_deg=-180; "проїдь два рази по метру" →
drive_relative distance_m=2.0.

The notes below cover the remaining domain-specific decisions you
cannot derive from tool signatures alone:

- The semantic primitive is a *space* — a named polygon. A space can be a
  whole room (кухня, спальня) or a smaller area (зона шафи, мітка
  зарядна, область вікна). The user may refer to a space as "кімната",
  "зона", "область", "місце", "мітка", "ділянка", or "регіон" — these
  are all synonyms and map to the same primitive. NAVIGATE drives the
  robot INTO a space (planner picks the nearest free cell inside the
  polygon), it does not go to one fixed point.
- "йди / поїдь / здай / рухайся назад" — plain reverse motion → use
  drive_relative with direction="backward". navigate_anchor is only for
  explicit references to a prior journey ("ще раз туди", "повернись
  туди де був"), and only when CONTEXT actually shows a previous
  navigation.
- "видали / забери / прибери" + a space name → delete_space (NOT stop or
  navigate). The verb removes the space from the map.
- "перейменуй / назви по-новому" → rename_space.
- Space creation is a special multi-turn flow: start_space begins a draft,
  the operator drives or clicks the perimeter, finish_space saves it,
  cancel_space aborts.
- rotate_in_place sign: ліворуч / наліво / вліво is POSITIVE delta_deg
  (CCW). праворуч / направо / вправо is NEGATIVE delta_deg (CW).
- Tool argument values MUST be final numbers, never expressions. You must
  compute multiplications and additions yourself before emitting the call.
  WRONG: {"delta_deg": -90*3}. RIGHT: {"delta_deg": -270}.
"""


def _build_tools(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the tool registry for the current request. space_name enums are
    bound to the active spaces so the model cannot hallucinate non-existent
    spaces at the structural level.
    """
    space_names = _space_names(context)
    has_prev_nav = isinstance(context.get("previous_navigation"), dict) and (
        isinstance(context["previous_navigation"].get("startedAt"), dict)
        or isinstance(context["previous_navigation"].get("goal"), dict)
    )
    space_enum: dict[str, Any] = {"enum": space_names} if space_names else {}

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "navigate_to_space",
                "description": (
                    "Drive the robot into a named space. The planner picks the "
                    "nearest free cell inside the polygon."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "space_name": {"type": "string", **space_enum},
                    },
                    "required": ["space_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "navigate_with_offset",
                "description": (
                    "Drive to a position offset from a known space's centroid "
                    "(e.g. 'south of кухня by 1m', 'нижче ціль-зона на метр'). "
                    "Use only when the user specifies a space plus a direction "
                    "and distance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "space_name": {"type": "string", **space_enum},
                        "offset_direction": {
                            "type": "string",
                            "enum": ["north", "south", "east", "west", "forward", "backward"],
                            "description": "World-frame direction (X=east, Y=north).",
                        },
                        "offset_distance_m": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 10.0,
                        },
                    },
                    "required": ["space_name", "offset_direction", "offset_distance_m"],
                },
            },
        },
    ]

    if has_prev_nav:
        tools.append({
            "type": "function",
            "function": {
                "name": "navigate_anchor",
                "description": (
                    "Re-do the previous navigation. previous_start sends "
                    "the robot to where it began, previous_goal repeats "
                    "the destination. Only for explicit references to a "
                    "prior journey, NOT plain 'йди назад' (which is reverse "
                    "motion → drive_relative)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor": {
                            "type": "string",
                            "enum": ["previous_start", "previous_goal"],
                        },
                    },
                    "required": ["anchor"],
                },
            },
        })

    tools.extend([
        {
            "type": "function",
            "function": {
                "name": "drive_relative",
                "description": (
                    "Drive a distance in a direction relative to the robot's "
                    "heading. forward / backward / left / right are robot-frame. "
                    "Use for any 'поїдь / йди / здай / рухайся' + direction. "
                    "Plain 'йди назад' (reverse motion) belongs here, NOT in "
                    "navigate_anchor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right"],
                        },
                        "distance_m": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 10.0,
                            "description": "Default 1.0 if the user did not specify.",
                        },
                    },
                    "required": ["direction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rotate_in_place",
                "description": (
                    "Rotate the robot in place by delta_deg degrees "
                    "(positive = CCW, i.e. left). Use for any 'поверни / "
                    "повернись / розвернись' imperative. Translation is not "
                    "involved — for that use drive_relative."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "delta_deg": {
                            "type": "number",
                            "minimum": -360,
                            "maximum": 360,
                        },
                    },
                    "required": ["delta_deg"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "return_home",
                "description": "Return to the home space (the space flagged is_home).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop",
                "description": "Halt all motion immediately.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_space",
                "description": "Delete an existing space by name.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", **space_enum}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rename_space",
                "description": "Rename an existing space.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "old_name": {"type": "string", **space_enum},
                        "new_name": {"type": "string"},
                    },
                    "required": ["old_name", "new_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_space",
                "description": (
                    "Begin recording a new space perimeter via voice "
                    "demonstration. The operator will then drive the robot "
                    "along the boundary (or click corners). Pair with "
                    "finish_space or cancel_space."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Ukrainian, nominative singular.",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_space",
                "description": "Close the in-progress space perimeter and save it.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_space",
                "description": "Discard the in-progress space perimeter without saving.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ])

    return tools


def _tool_call_to_intent(fn_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Map a tool call back to the {intent, params} shape that
    routers/voice.py already understands. Keeps the router unaware of which
    classification path produced the result.
    """
    if fn_name == "navigate_to_space":
        return {"intent": "NAVIGATE", "params": {"target": args.get("space_name", "")}}
    if fn_name == "navigate_with_offset":
        return {
            "intent": "NAVIGATE",
            "params": {
                "target": args.get("space_name", ""),
                "offset": {
                    "direction": args.get("offset_direction"),
                    "distance_m": args.get("offset_distance_m", 1.0),
                },
            },
        }
    if fn_name == "navigate_anchor":
        return {"intent": "NAVIGATE", "params": {"anchor": args.get("anchor")}}
    if fn_name == "rotate_in_place":
        return {"intent": "ROTATE", "params": {"delta_deg": args.get("delta_deg", 0)}}
    if fn_name == "return_home":
        return {"intent": "RETURN_HOME", "params": {}}
    if fn_name == "drive_relative":
        return {
            "intent": "DRIVE_RELATIVE",
            "params": {
                "direction": args.get("direction"),
                "distance_m": args.get("distance_m", 1.0),
            },
        }
    if fn_name == "stop":
        return {"intent": "STOP", "params": {}}
    if fn_name == "delete_space":
        return {"intent": "DELETE_SPACE", "params": {"name": args.get("name", "")}}
    if fn_name == "rename_space":
        return {
            "intent": "RENAME_SPACE",
            "params": {
                "old_name": args.get("old_name", ""),
                "new_name": args.get("new_name", ""),
            },
        }
    if fn_name == "start_space":
        return {"intent": "START_SPACE", "params": {"name": args.get("name", "")}}
    if fn_name == "finish_space":
        return {"intent": "FINISH_SPACE", "params": {}}
    if fn_name == "cancel_space":
        return {"intent": "CANCEL_SPACE", "params": {}}
    return {"intent": "UNKNOWN", "params": {"error": f"unknown tool '{fn_name}'"}}


# --- Recovery for malformed tool calls ---------------------------------------
# Small LLMs occasionally emit arithmetic expressions ("-90*3") instead of
# computed values ("-270"). Groq's tool-call validator rejects those with
# code='tool_use_failed', exposing the raw text in failed_generation. We
# salvage that text: regex out the function call, replace numeric-looking
# expressions with their computed values, and re-route through the normal
# tool→intent mapper. Better than UNKNOWN after a cooked-but-malformed call.

_FUNC_CALL_RE = re.compile(r"<function=(\w+)>\s*(\{.*\})\s*</function>", re.DOTALL)
_VALUE_RE = re.compile(r'("(?:\w+)")\s*:\s*([^,}\]\n]+?)(?=\s*[,}\]\n])')

_SAFE_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Constant, ast.Num,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.USub, ast.UAdd, ast.FloorDiv, ast.Mod,
)


def _safe_arith_eval(expr: str) -> Optional[float]:
    """Compile and evaluate a simple numeric expression. Returns None unless
    the AST contains only constants and basic arithmetic ops."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            return None
    try:
        return float(eval(compile(tree, "<expr>", "eval")))  # noqa: S307 — sandboxed by AST whitelist
    except Exception:
        return None


def _recover_tool_use_failed(err: Exception) -> Optional[dict[str, Any]]:
    body = getattr(err, "body", None)
    inner = body.get("error", {}) if isinstance(body, dict) else {}
    if inner.get("code") != "tool_use_failed":
        return None
    failed = inner.get("failed_generation")
    if not isinstance(failed, str):
        return None

    m = _FUNC_CALL_RE.search(failed)
    if not m:
        return None
    fn_name = m.group(1)
    args_text = m.group(2)

    try:
        args = json.loads(args_text)
    except json.JSONDecodeError:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            raw = match.group(2).strip()
            # Fast path: already a literal number.
            try:
                return f"{key}: {float(raw)}"
            except ValueError:
                pass
            evaluated = _safe_arith_eval(raw)
            if evaluated is not None:
                return f"{key}: {evaluated}"
            return match.group(0)

        fixed = _VALUE_RE.sub(replace, args_text)
        try:
            args = json.loads(fixed)
        except json.JSONDecodeError:
            _logger.warning("recover: still bad JSON after arithmetic eval: %r", fixed)
            return None

    if not isinstance(args, dict):
        return None
    _logger.info("recovered tool_use_failed: %s args=%s", fn_name, args)
    return _tool_call_to_intent(fn_name, args)


async def _classify_with_tools(
    transcription: str, context: dict[str, Any]
) -> dict[str, Any]:
    tools = _build_tools(context)
    user_msg = (
        f"{_context_message(context)}\n\n"
        f'Transcription: "{transcription}"'
    )
    try:
        resp = await _client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": SHORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=200,
        )
    except Exception as e:
        # tool_use_failed: model produced a malformed function call (most
        # often an arithmetic expression instead of a number). Try to salvage.
        salvaged = _recover_tool_use_failed(e)
        if salvaged is not None:
            return salvaged
        raise
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        # Model didn't call any tool → nothing matched.
        return {"intent": "UNKNOWN", "params": {}}

    # Cap at 5 actions per utterance — guards against a runaway model
    # emitting a long chain. 5 is plenty for "повернись, поїдь, ще раз
    # повернись, поверни мітку, стоп" style sequences.
    tool_calls = tool_calls[:5]

    actions: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        actions.append(_tool_call_to_intent(fn_name, args))

    if not actions:
        return {"intent": "UNKNOWN", "params": {}}
    if len(actions) == 1:
        return actions[0]
    # Multi-step: primary action carries the rest in a private field that
    # the router will materialize into VoiceFollowUp[].
    primary = actions[0]
    primary["_pending"] = actions[1:]
    return primary


# --- Free-form JSON path (legacy default) ------------------------------------


async def _classify(transcription: str, context: dict[str, Any]) -> dict[str, Any]:
    user_msg = (
        f"{_context_message(context)}\n\n"
        f'Transcription: "{transcription}"\n\n'
        "Return JSON only."
    )
    resp = await _client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=200,
    )
    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


def _classify_one_segment(
    clf: Any, slots_mod: Any, text: str, context: dict[str, Any]
) -> tuple[str, dict[str, Any], float]:
    """Classify one already-split segment. Returns (intent, params,
    confidence). Boundary predictions from this call are ignored — we
    don't recurse further; one split-pass is enough for the chains we
    care about. Slot extraction runs on the segment text only, so
    direction / distance / target are scoped correctly per step."""
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
    """Cut `text` at each char offset and return the segments. Offsets
    are assumed sorted ascending; result is `len(offsets) + 1` strings.
    Whitespace around split points is stripped. Empty segments (e.g. two
    adjacent markers) are dropped — the caller treats anything missing
    as UNKNOWN, which falls through to Groq."""
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
    # Drop the marker word itself from the start of each non-first segment.
    # We do this by skipping leading "word + whitespace" — the marker word
    # is whatever runs from `last` up to the next whitespace.
    if tail:
        parts.append(tail)
    # Trim leading marker word from segments after the first one.
    cleaned: list[str] = [parts[0]] if parts else []
    for seg in parts[1:]:
        # Strip a leading marker word like "і", "потім", "and then".
        # We rely on the fact that markers are short and end at whitespace.
        m = re.match(r"^\s*\S+\s+", seg)
        cleaned.append(seg[m.end():].strip() if m else seg)
    return [c for c in cleaned if c]


def _try_classify_local(
    transcription: str, context: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Run the trained XLM-RoBERTa intent+slot+boundary classifier.

    Outcomes:
    - **single-step confident** (conf ≥ HIGH, no boundary): returns
      `{"intent", "params", "_ms"}` — caller dispatches as usual.
    - **multi-step confident**: returns the same shape plus `_pending`
      (list of `{intent, params}` follow-ups, one per extra segment).
    - **uncertain** (LOW ≤ conf < HIGH): returns the same shape but with
      `intent="UNCERTAIN"` and the model's actual guess kept under
      `params["_predicted_intent"]` / `params["_predicted_params"]`.
    - **fallback** (conf < LOW or classifier missing): returns None,
      caller falls back to Groq.

    Multi-step detection comes from the boundary head's per-token output
    (`boundary_offsets`). When the head is absent (legacy checkpoint),
    the path collapses to single-step exactly like before.
    """
    from services import intent_classifier as _ic
    from services import slot_extractor as _slots

    # Operator-taught phrase short-circuit: if this exact transcription
    # was confirmed before with a specific intent + params, dispatch it
    # straight away with confidence=1.0. No model query, no UNCERTAIN
    # band, no modal — that's what "the system learned this phrase" means.
    from services import learned_overrides as _lo
    override = _lo.lookup(transcription)
    if override is not None:
        intent_str = override["intent"]
        if intent_str in _INTENT_VALUES:
            t_lo = time.monotonic()
            params = dict(override.get("params") or {})
            params["original_text"] = transcription
            params["_confidence"] = 1.0
            params["_source"] = "learned_override"
            elapsed_ms = (time.monotonic() - t_lo) * 1000
            return {"intent": intent_str, "params": params, "_ms": elapsed_ms}

    clf = _ic.get_classifier()
    if clf is None:
        return None
    t0 = time.monotonic()
    pred = clf.predict(transcription)
    # Backward-compat: legacy classifiers return a 3-tuple; new ones return 4.
    intent = pred[0]
    confidence = float(pred[1])
    model_spans = pred[2] if len(pred) > 2 else {}
    boundary_offsets: list[int] = pred[3] if len(pred) > 3 else []

    # --- multi-step path ---------------------------------------------------
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

            if min_conf < settings.intent_low_confidence_threshold:
                _logger.info(
                    "local multi-step below LOW (%.2f < %.2f) — falling back to Groq",
                    min_conf, settings.intent_low_confidence_threshold,
                )
                return None

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

    # --- single-step path (unchanged behaviour) ---------------------------
    if confidence < settings.intent_low_confidence_threshold:
        _logger.info(
            "local classifier below LOW threshold (%.2f < %.2f) — falling back to Groq",
            confidence,
            settings.intent_low_confidence_threshold,
        )
        return None
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
        # Uncertain band: stash the model's guess for the frontend dialog,
        # surface UNCERTAIN at the top level so the router knows not to
        # dispatch yet.
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
    """Two-step voice→intent: Whisper transcription, then Llama JSON classification.

    Returns {"intent": "<INTENT>", "params": {..., "original_text": str}}.
    On failure returns UNKNOWN with an "error" field — never raises.
    """
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

            # XLM-RoBERTa classifier path: deterministic, ~30 ms per
            # forward pass, no LLM. Multi-step utterances are now handled
            # natively by the trained boundary head — _try_classify_local
            # will return a `_pending` list when the model predicts a
            # chain marker. The legacy regex guard is kept only as a
            # diagnostic log, since the boundary head supersedes it.
            if settings.voice_classifier_engine == "xlm_roberta":
                if _CHAIN_MARKER_RE.search(transcription) is not None:
                    _logger.debug(
                        "chain marker present in %r — boundary head decides",
                        transcription,
                    )
                clf_result = await asyncio.to_thread(
                    _try_classify_local, transcription, context
                )
                if clf_result is not None:
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

            llm_start = time.monotonic()
            if settings.voice_use_tools:
                parsed = await _classify_with_tools(transcription, context)
            else:
                parsed = await _classify(transcription, context)
            llm_ms = (time.monotonic() - llm_start) * 1000

            intent = str(parsed.get("intent", "UNKNOWN")).upper()
            if intent not in _INTENT_VALUES:
                intent = "UNKNOWN"
            params_raw = parsed.get("params")
            params = params_raw if isinstance(params_raw, dict) else {}
            params["original_text"] = transcription

            # Preserve the multi-step queue if _classify_with_tools produced one.
            # The router materializes _pending into VoiceFollowUp[].
            pending = parsed.get("_pending")
            result: dict[str, Any] = {"intent": intent, "params": params}
            if isinstance(pending, list) and pending:
                result["_pending"] = pending

            _logger.info(
                "groq parse_voice attempt=%d stt_ms=%.0f llm_ms=%.0f intent=%s mode=%s steps=%d",
                attempt + 1,
                stt_ms,
                llm_ms,
                intent,
                "tools" if settings.voice_use_tools else "json",
                1 + (len(pending) if isinstance(pending, list) else 0),
            )
            return result

        except Exception as e:
            last_err = e
            _logger.warning("groq error attempt=%d: %s", attempt + 1, e)
            if attempt < 1:
                await asyncio.sleep(1.0)

    _logger.error("groq failed after retries: %s", last_err)
    return {
        "intent": "UNKNOWN",
        "params": {
            "original_text": transcription,
            "error": str(last_err) if last_err else "unknown error",
        },
    }
