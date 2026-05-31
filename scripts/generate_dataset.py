"""Augment hand-written seeds with paraphrases from Groq, then write
intents.jsonl + intents_eval.jsonl for fine-tuning a joint intent + slot
model.

Usage (from repo root):
    cd backend && .venv/bin/python ../scripts/generate_dataset.py

Env: GROQ_API_KEY must be set (read from backend/.env via config.py).

Pipeline:
    1. Read scripts/data/seeds.jsonl. Label-CRUD rows carry a `slots`
       field whose values are literal substrings of `text`.
    2. For each seed, ask Groq for N paraphrases — prompt explicitly
       requires the label name(s) to appear *verbatim*.
    3. Auto-annotate each paraphrase: locate every slot value as
       case-insensitive substring; drop the paraphrase if any slot value
       is missing or, for RENAME, if old_name no longer precedes new_name.
    4. Stratified 90/10 split → intents.jsonl + intents_eval.jsonl.

Output schema per JSONL row:
    {"text": "переназви кухню на їдальня", "intent": "RENAME_LABEL",
     "lang": "uk", "slots": {"old_name": "кухню", "new_name": "їдальня"}}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import settings  # noqa: E402
from groq import AsyncGroq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_dataset")

DATA_DIR = ROOT / "scripts" / "data"
SEEDS_PATH = DATA_DIR / "seeds.jsonl"
FEEDBACK_SEEDS_PATH = DATA_DIR / "seeds_from_feedback.jsonl"
TRAIN_OUT = DATA_DIR / "intents.jsonl"
EVAL_OUT = DATA_DIR / "intents_eval.jsonl"
DB_PATH = ROOT / "backend" / "db.sqlite3"

PARAPHRASE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_PARAPHRASES_PER_SEED = 8
DEFAULT_CONCURRENCY = 2  # Groq free tier: 30 RPM on llama-4-scout
EVAL_FRACTION = 0.10
RNG_SEED = 1337

# Slot keys whose values must appear verbatim in the paraphrase. RADIUS
# is excluded — wording varies wildly («пів метра / 0.8 / half a meter»)
# and the regex slot extractor handles it; the model sees these tokens
# as O during training.
NAME_SLOT_KEYS = {"name", "old_name", "new_name"}

# Lexical signals that must survive paraphrasing for an intent to remain
# itself. Without this guard, the LLM occasionally produces a paraphrase
# that keeps the label name verbatim but drops the action verb, so the
# row ends up labelled DELETE_LABEL while the text actually expresses
# NAVIGATE / CREATE / nothing. We require at least one regex hit per row.
_INTENT_SIGNALS: dict[str, re.Pattern[str]] = {
    "DELETE_SPACE": re.compile(
        r"\b(?:видал|забер|прибер|позбу|забира|вилуч|delete|remove|"
        r"erase|get\s+rid|drop)\w*", re.I),
    "RENAME_SPACE": re.compile(
        r"\b(?:переназв|перейменув|перейменуй|переіменув|переіменуй|"
        r"змін(?:и|ити)\s+(?:назв|ім|найм)|"
        r"rename|change\s+(?:the\s+)?name|"
        r"called|називат|буде\s+називат)", re.I),
    "START_SPACE": re.compile(
        r"\b(?:почн|почат|починаю|стартуй|стартую|малюй|малювати|намалюй|"
        r"обвед|обводь|обводи|креслит|кресли|розпочн|розпочат|"
        r"створ|додай|запиши|запусти|познач|позначити|давай\s+намалю|"
        r"новий\s+полігон|"
        r"start(?:\s+drawing)?\s+(?:room|space|area|zone|place|label)|"
        r"begin\s+(?:room|space|area|zone|place|label)|"
        r"draw\s+(?:room|space|area|zone|place|label)|"
        r"new\s+(?:room|space|area|zone|place|label)|"
        r"trace\s+(?:room|space|area|zone|place|label)|"
        r"outline\s+(?:room|space|area|zone|place|label)|"
        r"record\s+(?:room|space|area|zone|place|label)|"
        r"mark\s+(?:room|space|area|zone|place|label)|"
        r"add\s+(?:room|space|area|zone|place|label)|"
        r"create\s+(?:room|space|area|zone|place|label))", re.I),
    "FINISH_SPACE": re.compile(
        r"\b(?:готов|заверш|закінч|збер|досить|хватить|стоп\s+малю|"
        r"це\s+все|ось\s+(?:і\s+)?все|оце\s+вона|ну\s+(?:і\s+)?все|"
        r"done|finish|save|complete|submit|that['’ʼ]?s\s+it)", re.I),
    "CANCEL_SPACE": re.compile(
        r"\b(?:скас|відмін|забуд|видали\s+чернет|видалити\s+чернет|"
        r"забери\s+чернет|стер|не\s+треба|не\s+зберіг|помилив|помилила|"
        r"хибно|ой\s+ні|ну\s+ні|"
        r"cancel|discard|nevermind|abort|scrap|forget\s+it)", re.I),
}


def intent_signal_present(text: str, intent: str) -> bool:
    """For label-CRUD intents that also share a label name, verify the
    paraphrase still carries an intent-discriminating verb. Returns
    True for intents we don't gate (NAVIGATE, ROTATE, etc.)."""
    pattern = _INTENT_SIGNALS.get(intent)
    if pattern is None:
        return True
    return pattern.search(text) is not None

PARAPHRASE_PROMPT_UK = """\
Ти — генератор перифразів для тренування класифікатора голосових команд для домашнього мобільного робота.

Я даю тобі команду українською. Згенеруй РІВНО {n} різних перифразів цієї команди — обов'язково українською, так, як міг би сказати реальний користувач.

Зберігай той самий намір (intent). Можеш міняти словоформи відмінків, порядок слів, синоніми (їдь / поїдь / рухайся / прямуй), додавати ввічливість («будь ласка»), розмовні форми, незначні мовні неточності.

ОБОВ'ЯЗКОВО:
- НАЗВИ ЛЕЙБЛІВ повинні з'являтися У ПЕРИФРАЗІ ДОСЛІВНО, у тому ж порядку. Якщо в команді є «кухня», у перифразі ТЕЖ має бути слово «кухня». Не міняй назви на синоніми, не схиляй у інший відмінок (якщо в оригіналі «кухню» — пиши «кухню»).
- Не змінюй ЗНАК НАПРЯМКУ повороту (наліво лишається наліво).
- Не плутай інтенти.

Поверни рівно {n} рядків. По одному перифразу на рядок. Без нумерації, без лапок, без коментарів."""

PARAPHRASE_PROMPT_EN = """\
You are a paraphrase generator for training a joint intent + slot classifier for an indoor mobile robot voice interface.

I will give you one English command. Generate EXACTLY {n} distinct paraphrases of it — in English — that a real user might say.

Keep the same intent. You may vary word order, contractions, synonyms (drive / go / move / head), politeness ("please"), casual forms, minor disfluencies.

MANDATORY:
- LABEL NAMES must appear in the paraphrase VERBATIM, in the same order. If the command says "kitchen", the paraphrase MUST also contain "kitchen" as a word — don't substitute synonyms, don't paraphrase the label.
- Do NOT change the rotation direction sign (left stays left).
- Do NOT change the intent.

Return exactly {n} lines. One paraphrase per line. No numbering, no quotes, no commentary."""


def load_seeds(path: Path) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seed = {
                "text": row["text"].strip(),
                "intent": row["intent"],
                "lang": row.get("lang", "uk"),
            }
            slots = row.get("slots")
            if isinstance(slots, dict) and slots:
                seed["slots"] = {k: str(v) for k, v in slots.items()}
            seeds.append(seed)
    return seeds


def load_command_log(db_path: Path) -> list[dict[str, Any]]:
    """Pull (transcription, intent) from command_log. No slots — these
    rows train intent only; slot heads see them as all-O."""
    if not db_path.exists():
        log.warning("db.sqlite3 not found at %s — skipping command_log enrichment", db_path)
        return []
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT transcription, intent FROM command_log "
            "WHERE success = 1 AND transcription IS NOT NULL AND TRIM(transcription) != ''"
        )
        for tx, intent in cur.fetchall():
            tx = tx.strip()
            lang = "uk" if re.search(r"[а-яіїєґ]", tx.lower()) else "en"
            rows.append({"text": tx, "intent": intent, "lang": lang})
    log.info("loaded %d rows from command_log", len(rows))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def slots_present_in(text: str, slots: dict[str, str]) -> bool:
    """Verify every name-bearing slot value appears as a substring of
    `text` (case-insensitive). Returns False if any value is missing.
    For RENAME, also enforce that old_name appears before new_name."""
    if not slots:
        return True
    t = text.lower()
    positions: dict[str, int] = {}
    for k, v in slots.items():
        if k not in NAME_SLOT_KEYS:
            continue
        v_norm = v.lower().strip()
        if not v_norm:
            continue
        idx = t.find(v_norm)
        if idx < 0:
            return False
        positions[k] = idx
    if "old_name" in positions and "new_name" in positions:
        if positions["old_name"] >= positions["new_name"]:
            return False
    return True


async def paraphrase_one(
    client: AsyncGroq,
    text: str,
    lang: str,
    n: int,
    sem: asyncio.Semaphore,
) -> list[str]:
    prompt = (PARAPHRASE_PROMPT_UK if lang == "uk" else PARAPHRASE_PROMPT_EN).format(n=n)
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=PARAPHRASE_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.9,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content or ""
                lines = [ln.strip().strip("\"'`–-—•*").strip() for ln in raw.splitlines()]
                lines = [ln for ln in lines if ln and len(ln) <= 200]
                await asyncio.sleep(4.0)  # Self-throttle to fit 30 RPM
                return lines[:n]
            except Exception as exc:  # noqa: BLE001
                wait = min(30, 2 ** (attempt + 1))
                log.warning("paraphrase failed (%s) — retry in %ds: %s", text[:40], wait, exc)
                await asyncio.sleep(wait)
        log.error("giving up on: %s", text[:60])
        return []


async def expand(
    seeds: list[dict[str, Any]],
    n_per_seed: int,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    client = AsyncGroq(api_key=settings.groq_api_key)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = list(seeds)  # always keep originals
    drop_stats: dict[str, int] = defaultdict(int)

    async def task(seed: dict[str, Any]) -> list[dict[str, Any]]:
        paras = await paraphrase_one(client, seed["text"], seed["lang"], n_per_seed, sem)
        out: list[dict[str, Any]] = []
        slots = seed.get("slots") or {}
        for p in paras:
            if slots and not slots_present_in(p, slots):
                drop_stats[seed["intent"]] += 1
                continue
            if not intent_signal_present(p, seed["intent"]):
                drop_stats[seed["intent"] + "_signal"] += 1
                continue
            row: dict[str, Any] = {"text": p, "intent": seed["intent"], "lang": seed["lang"]}
            if slots:
                row["slots"] = dict(slots)
            out.append(row)
        return out

    coros = [task(s) for s in seeds]
    log.info("requesting paraphrases for %d seeds (concurrency=%d, n=%d) ...",
             len(seeds), concurrency, n_per_seed)
    done = 0
    for fut in asyncio.as_completed(coros):
        new = await fut
        results.extend(new)
        done += 1
        if done % 25 == 0:
            log.info("  progress: %d / %d seeds done, %d total examples (drops: %s)",
                     done, len(seeds), len(results), dict(drop_stats))
    return results, dict(drop_stats)


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = normalize(r["text"])
        if not key:
            continue
        if key in seen and seen[key]["intent"] != r["intent"]:
            log.warning("intent conflict for %r: kept %s, dropped %s",
                        key, seen[key]["intent"], r["intent"])
            continue
        if key not in seen:
            seen[key] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# Chain row synthesis
# ---------------------------------------------------------------------------

# (full marker text, byte-offset within the marker where the FIRST marker
# word starts). The boundary head is trained to fire on the first subword
# of this word. We pick from this list uniformly at random per chain.
_UK_CHAIN_MARKERS: list[tuple[str, int]] = [
    (" і ", 1),
    (" та ", 1),
    (" потім ", 1),
    (", потім ", 2),
    (", а потім ", 2),   # comma + filler «а» + потім
    (", а далі ", 2),    # comma + filler «а» + далі
    (" і далі ", 1),
    (" після цього ", 1),
    (" після того ", 1),
    (" а далі ", 1),
    (" а потім ", 1),
    (" наступним ", 1),
]
_EN_CHAIN_MARKERS: list[tuple[str, int]] = [
    (" and ", 1),
    (" and then ", 1),
    (" then ", 1),
    (", then ", 2),
]

# Optional leading interjection prepended to ~25 % of synthesized chain
# rows. Teaches the model to tolerate filler words («А йди ... »,
# «Ну рухайся ... »). Stored as plain strings to prepend; we re-uppercase
# the first letter to match natural sentence-cased transcripts.
_UK_LEADING_INTERJECTIONS: list[str] = ["А ", "Ну ", "Ой ", "От ", "Так "]

# Intent combos for synthesized chains. Weighted by what's actually useful
# in practice — ROTATE+DRIVE dominates because that's the canonical demo
# («поверни ліворуч і вперед на 2 метри»).
_CHAIN_COMBOS_TWO: list[tuple[float, tuple[str, str]]] = [
    (0.50, ("ROTATE", "DRIVE_RELATIVE")),
    (0.15, ("DRIVE_RELATIVE", "NAVIGATE")),
    (0.10, ("NAVIGATE", "STOP")),
    (0.10, ("NAVIGATE", "NAVIGATE")),   # «йди до бета а потім до альфа»
    (0.05, ("ROTATE", "NAVIGATE")),
    (0.05, ("DRIVE_RELATIVE", "ROTATE")),  # «вперед і поверни ліворуч»
    (0.05, ("NAVIGATE", "RETURN_HOME")),
]
_CHAIN_COMBO_THREE: tuple[str, str, str] = ("ROTATE", "DRIVE_RELATIVE", "NAVIGATE")
_FRACTION_THREE_STEP = 0.05  # 5 % of chain rows are 3-step

_CHAIN_USEFUL_INTENTS = {"NAVIGATE", "DRIVE_RELATIVE", "ROTATE", "STOP", "RETURN_HOME"}


def _detect_lang(text: str) -> str:
    """Cheap language sniff: any Cyrillic letter wins."""
    t = text.lower()
    return "uk" if any("а" <= ch <= "я" or ch in "іїєґ" for ch in t) else "en"


def _strip_tail_punct(text: str) -> str:
    """Drop trailing punctuation so chained text doesn't double up
    separators."""
    return text.rstrip(" .,!?:;")


def _lower_first(text: str) -> str:
    """Lowercase only the first letter so the chained part reads as a
    continuation, not a new sentence."""
    return text[:1].lower() + text[1:] if text else text


def _pick_marker(lang: str, rng: random.Random) -> tuple[str, int]:
    pool = _UK_CHAIN_MARKERS if lang == "uk" else _EN_CHAIN_MARKERS
    return rng.choice(pool)


def _pick_combo(rng: random.Random) -> tuple[str, str]:
    r = rng.random()
    cum = 0.0
    for w, combo in _CHAIN_COMBOS_TWO:
        cum += w
        if r < cum:
            return combo
    return _CHAIN_COMBOS_TWO[-1][1]


def _segment_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Ground-truth (intent, slots) for one segment, used by eval. Slots
    default to {} when the source row doesn't carry them."""
    return {"intent": row["intent"], "slots": dict(row.get("slots") or {})}


def synthesize_chains(
    rows: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Pair single-intent rows into multi-step chain rows.

    The returned rows carry an extra `segments` list with per-step
    ground truth, plus a `chain_markers` list of char offsets pointing
    at the first letter of each marker word in `text`. The top-level
    `intent` is set to the FIRST segment's intent so `stratified_split`
    and `deduplicate` continue to work; training-time code recognises
    chain rows by `segments` and ignores intent/slot loss for them."""
    if n <= 0:
        return []
    pool: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["intent"] not in _CHAIN_USEFUL_INTENTS:
            continue
        if r.get("segments"):
            # already a chain row — skip re-chaining
            continue
        pool[r.get("lang", "uk")][r["intent"]].append(r)

    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 4:
        attempts += 1
        lang = "uk" if rng.random() < 0.7 else "en"
        per_lang = pool.get(lang)
        if not per_lang:
            continue
        is_three = rng.random() < _FRACTION_THREE_STEP
        intents = _CHAIN_COMBO_THREE if is_three else _pick_combo(rng)
        parts: list[dict[str, Any]] = []
        for intent in intents:
            bucket = per_lang.get(intent)
            if not bucket:
                parts = []
                break
            parts.append(rng.choice(bucket))
        if not parts:
            continue
        # Optional leading interjection (~25 % of UK chains). Shifts all
        # marker offsets right by len(prefix). Teaches the boundary head
        # not to over-anchor on text starting with an action verb.
        prefix = ""
        if lang == "uk" and rng.random() < 0.25:
            prefix = rng.choice(_UK_LEADING_INTERJECTIONS)
            # «А йди» reads more naturally with lowercase action verb.
            parts = [
                {**parts[0], "text": _lower_first(parts[0]["text"])},
                *parts[1:],
            ]

        # Stitch text + record marker offsets.
        text = prefix + _strip_tail_punct(parts[0]["text"])
        markers: list[int] = []
        for nxt in parts[1:]:
            marker_text, word_offset = _pick_marker(lang, rng)
            markers.append(len(text) + word_offset)
            text = text + marker_text + _lower_first(_strip_tail_punct(nxt["text"]))
        segments = [_segment_payload(p) for p in parts]
        out.append({
            "text": text,
            "intent": segments[0]["intent"],  # for stratified_split bucketing only
            "lang": lang,
            "segments": segments,
            "chain_markers": markers,
        })
    return out


def stratified_split(
    rows: list[dict[str, Any]], eval_frac: float, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        k = max(1, int(round(len(items) * eval_frac)))
        eval_.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append-only sibling of write_jsonl. Creates the file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(rows: list[dict[str, Any]], title: str) -> None:
    by_intent: dict[str, int] = defaultdict(int)
    by_lang: dict[str, int] = defaultdict(int)
    with_slots = 0
    chain_rows = 0
    for r in rows:
        by_intent[r["intent"]] += 1
        by_lang[r["lang"]] += 1
        if r.get("slots"):
            with_slots += 1
        if r.get("segments"):
            chain_rows += 1
    log.info("=== %s: %d rows (%d with slots, %d chains) ===",
             title, len(rows), with_slots, chain_rows)
    log.info("  by lang: %s", dict(by_lang))
    for k in sorted(by_intent):
        log.info("  %-15s %d", k, by_intent[k])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_PARAPHRASES_PER_SEED,
                    help="paraphrases per seed (default 8)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--use-command-log", action="store_true",
                    help="enrich seeds with command_log rows (no slots, intent only)")
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="skip Groq calls (output seeds only)")
    ap.add_argument("--append-from", type=Path, default=None,
                    help="paraphrase ONLY the seeds in this file and APPEND the "
                         "result to existing intents.jsonl / intents_eval.jsonl. "
                         "Skips re-paraphrasing seeds.jsonl entirely — much faster "
                         "for incremental retrains. Falls back to full rebuild if "
                         "intents.jsonl doesn't exist yet.")
    ap.add_argument("--with-chains", type=int, default=0,
                    help="synthesize N multi-step rows by pairing existing "
                         "single-intent rows with chain markers («і», «потім», …). "
                         "Each chain row carries `segments` and `chain_markers` "
                         "fields used by the boundary head during training. "
                         "Ignored in --no-paraphrase mode.")
    args = ap.parse_args()

    rng = random.Random(RNG_SEED)

    # Append mode: only the operator's brand-new corrections get
    # paraphrased; everything we've already trained on stays in
    # intents.jsonl untouched. Saves ~6 minutes and ~125k Groq tokens
    # on every retrain after the first.
    append_mode = (
        args.append_from is not None
        and args.append_from.exists()
        and TRAIN_OUT.exists()
        and EVAL_OUT.exists()
    )

    if append_mode:
        seeds = load_seeds(args.append_from)
        log.info("APPEND mode: paraphrasing %d new seeds from %s",
                 len(seeds), args.append_from)
    else:
        if args.append_from is not None:
            log.info("append-from given but intents.jsonl missing — falling back to full rebuild")
        seeds = load_seeds(SEEDS_PATH)
        log.info("loaded %d hand-written seeds (%d carry slots)",
                 len(seeds), sum(1 for s in seeds if s.get("slots")))
        # Active-learning seeds collected by retrain_from_feedback.py from
        # operator-confirmed corrections. Optional file; absent on first run.
        if FEEDBACK_SEEDS_PATH.exists():
            feedback = load_seeds(FEEDBACK_SEEDS_PATH)
            log.info("loaded %d feedback seeds", len(feedback))
            seeds = deduplicate(seeds + feedback)
            log.info("merged: %d unique seeds total", len(seeds))
        if args.use_command_log:
            log_rows = load_command_log(DB_PATH)
            seeds = deduplicate(seeds + log_rows)
            log.info("merged: %d unique seeds", len(seeds))

    drop_stats: dict[str, int] = {}
    if args.no_paraphrase:
        rows = list(seeds)
    else:
        rows, drop_stats = await expand(seeds, args.n, args.concurrency)
    rows = deduplicate(rows)

    if drop_stats:
        log.info("paraphrases dropped due to missing/misordered slots: %s", drop_stats)

    chain_rows: list[dict[str, Any]] = []
    if args.with_chains > 0:
        chain_rows = synthesize_chains(rows, args.with_chains, rng)
        log.info("synthesized %d chain rows (target %d)", len(chain_rows), args.with_chains)
        rows = rows + chain_rows

    train, eval_ = stratified_split(rows, EVAL_FRACTION, rng)

    if append_mode:
        append_jsonl(TRAIN_OUT, train)
        append_jsonl(EVAL_OUT, eval_)
        log.info("APPENDED %d to %s and %d to %s",
                 len(train), TRAIN_OUT.name, len(eval_), EVAL_OUT.name)
    else:
        write_jsonl(TRAIN_OUT, train)
        write_jsonl(EVAL_OUT, eval_)
        log.info("wrote %s and %s", TRAIN_OUT.name, EVAL_OUT.name)

    report(train, "NEW TRAIN" if append_mode else "TRAIN")
    report(eval_, "NEW EVAL" if append_mode else "EVAL")


if __name__ == "__main__":
    asyncio.run(main())
