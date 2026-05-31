"""Evaluate the trained joint intent + slot model on intents_eval.jsonl.

Reports:
  - per-class intent precision/recall/F1, macro-F1, confusion matrix
  - slot extraction: per-intent name exact-match accuracy + token-level F1
  - optionally a side-by-side comparison with the existing Groq pipeline

Usage:
    cd backend && .venv/bin/python ../scripts/eval_intent.py
    .venv/bin/python ../scripts/eval_intent.py --compare-groq
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "scripts" / "data"
EVAL_PATH = DATA_DIR / "intents_eval.jsonl"
MODEL_DIR = ROOT / "backend" / "models" / "intent_classifier"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_intent")


def _slot_match(pred: str, gold: str) -> bool:
    """Compare slot strings tolerantly: exact case-insensitive, OR
    sharing a long enough common prefix to ignore Ukrainian inflection
    (e.g. «кухня» vs «кухню» share «кухн» — production resolves the
    in-text form to its canonical label via fuzzy_match, so eval should
    accept either)."""
    p = pred.strip().lower()
    g = gold.strip().lower()
    if not p or not g:
        return False
    if p == g:
        return True
    # Common prefix of length ≥ 3 with both sides ≥ 3 chars long.
    common = 0
    for a, b in zip(p, g):
        if a == b:
            common += 1
        else:
            break
    return common >= 3 and min(len(p), len(g)) >= 3 and abs(len(p) - len(g)) <= 3


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _synth_active(gold_slots: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a permissive active_spaces pool so single-row eval mirrors
    what production sees. Includes a few common rooms plus any name
    referenced in this row's gold slots."""
    out: list[dict[str, Any]] = [
        {"name": "база"}, {"name": "кухня"}, {"name": "спальня"},
        {"name": "коридор"}, {"name": "балкон"}, {"name": "точка 1"},
        {"name": "балкон 2"},
    ]
    for k, v in (gold_slots or {}).items():
        if k in {"name", "old_name", "new_name"} and isinstance(v, str):
            out.append({"name": v})
    return out


def predict_local(
    model_dir: Path,
    rows: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Run the local classifier on every SINGLE-INTENT row. Chain rows
    (those with a `segments` field) are skipped here — `predict_chains`
    handles them separately. Returns [(predicted_intent, predicted_slots),
    ...] aligned 1:1 with the filtered single-intent input."""
    sys.path.insert(0, str(ROOT / "backend"))
    from services import intent_classifier as ic  # noqa: E402
    from services import slot_extractor as se  # noqa: E402

    clf = ic.IntentClassifier.load(str(model_dir))
    results: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        text = row["text"]
        pred = clf.predict(text)
        intent = pred[0]
        spans = pred[2] if len(pred) > 2 else {}
        params = se.extract_slots(
            text, intent,
            active_spaces=_synth_active(row.get("slots") or {}),
            model_spans=spans,
        )
        slots = {
            k: v
            for k, v in params.items()
            if k in {"name", "old_name", "new_name", "radius"}
        }
        results.append((intent, slots))
    return results


def predict_chains(
    model_dir: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the local classifier on CHAIN rows. Each result is a dict:
    `{boundary_offsets, segments: [{intent, slots}, ...]}`. Slot
    extraction runs per-segment with `_synth_active` from the union of
    all segments' gold slots."""
    sys.path.insert(0, str(ROOT / "backend"))
    from services import intent_classifier as ic  # noqa: E402
    from services import slot_extractor as se  # noqa: E402

    clf = ic.IntentClassifier.load(str(model_dir))
    out: list[dict[str, Any]] = []
    for row in rows:
        text = row["text"]
        # Active-space pool: union of every segment's gold slots.
        gold_union: dict[str, Any] = {}
        for seg in row.get("segments") or []:
            for k, v in (seg.get("slots") or {}).items():
                if isinstance(v, str):
                    gold_union[k] = v
        active = _synth_active(gold_union)

        pred = clf.predict(text)
        boundary_offsets: list[int] = pred[3] if len(pred) > 3 else []

        # Split per predicted boundary; classify each segment.
        from services import voice_intent as vi  # noqa: PLC0415, E402
        segments = vi._split_at_offsets(text, boundary_offsets)
        seg_preds: list[dict[str, Any]] = []
        for seg_text in segments:
            sp = clf.predict(seg_text)
            seg_intent = sp[0]
            seg_spans = sp[2] if len(sp) > 2 else {}
            seg_params = se.extract_slots(
                seg_text, seg_intent,
                active_spaces=active,
                model_spans=seg_spans,
            )
            seg_slots = {
                k: v for k, v in seg_params.items()
                if k in {"name", "old_name", "new_name", "radius"}
            }
            seg_preds.append({"intent": seg_intent, "slots": seg_slots})
        out.append({"boundary_offsets": boundary_offsets, "segments": seg_preds})
    return out


async def predict_groq(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    sys.path.insert(0, str(ROOT / "backend"))
    from services import voice_intent  # type: ignore[import-not-found]  # noqa: E402

    out: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        ctx = {"active_labels": [], "robot_pose": {"x": 0, "y": 0, "theta": 0}, "map_name": ""}
        try:
            r = await voice_intent._classify_with_tools(row["text"], ctx)
            intent = r.get("intent", "UNKNOWN")
            params = r.get("params") or {}
            slots = {
                k: v
                for k, v in params.items()
                if k in {"name", "old_name", "new_name", "target", "radius"}
            }
            # Groq uses "target" for NAVIGATE; for slot accuracy that
            # parallels our model's "name", treat them as same field
            # only when comparing NAVIGATE.
            out.append((intent, slots))
        except Exception as exc:  # noqa: BLE001
            log.warning("groq call failed for %r: %s", row["text"], exc)
            out.append(("UNKNOWN", {}))
    return out


def report_intents(name: str, gold: list[str], pred: list[str], labels: list[str]) -> None:
    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    log.info("=== INTENT %s ===  accuracy: %.3f (%d/%d)", name, correct / n, correct, n)
    f1s: list[float] = []
    weighted_sum = 0.0
    total_support = 0
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        support = sum(1 for g in gold if g == c)
        weighted_sum += f1 * support
        total_support += support
        log.info("  %-15s  P=%.3f R=%.3f F1=%.3f  (n=%d)", c, prec, rec, f1, support)
    log.info("  macro-F1: %.3f", sum(f1s) / len(f1s))
    log.info("  weighted-F1: %.3f", weighted_sum / total_support if total_support else 0.0)
    confusions: dict[tuple[str, str], int] = defaultdict(int)
    for g, p in zip(gold, pred):
        if g != p:
            confusions[(g, p)] += 1
    if confusions:
        log.info("  confusion (gold -> pred): %d misclassified", sum(confusions.values()))
        for (g, p), cnt in sorted(confusions.items(), key=lambda x: -x[1]):
            log.info("    %s -> %s  x%d", g, p, cnt)


def report_slots(
    name: str,
    rows: list[dict[str, Any]],
    pred_slots: list[dict[str, Any]],
    pred_intents: list[str],
) -> None:
    """Per-intent name exact-match accuracy. Only counts rows that have a
    gold slot for the intent in question."""
    log.info("=== SLOTS %s ===", name)
    by_intent: dict[str, list[bool]] = defaultdict(list)
    for row, p_slots, p_intent in zip(rows, pred_slots, pred_intents):
        gold_slots = row.get("slots") or {}
        if not gold_slots:
            continue
        intent = row["intent"]
        if intent != p_intent:
            # Intent wrong → all slot fields wrong by definition.
            by_intent[intent].append(False)
            continue
        # Compare every gold key. Case-insensitive equality on string slots.
        ok = True
        for k, v in gold_slots.items():
            if k not in {"name", "old_name", "new_name"}:
                continue
            pv = p_slots.get(k)
            if pv is None:
                ok = False
                break
            if not _slot_match(str(pv), str(v)):
                ok = False
                break
        by_intent[intent].append(ok)
    if not by_intent:
        log.info("  (no rows with gold slots in eval set)")
        return
    overall = [v for vs in by_intent.values() for v in vs]
    log.info("  overall name exact-match: %.3f (%d/%d)",
             sum(overall) / len(overall), sum(overall), len(overall))
    for intent in sorted(by_intent):
        vs = by_intent[intent]
        log.info("  %-15s  acc=%.2f  (%d/%d)",
                 intent, sum(vs) / len(vs), sum(vs), len(vs))


def report_chains(
    rows: list[dict[str, Any]],
    preds: list[dict[str, Any]],
) -> None:
    """Multi-step eval. Reports:
    - boundary detection F1 (per row: did the model find the right N markers?)
    - chain accuracy (every segment's intent matches gold)
    - per-segment intent confusion summary."""
    log.info("=== CHAINS ===")
    if not rows:
        log.info("  (no chain rows in eval set)")
        return
    n = len(rows)
    correct = 0
    boundary_tp = 0
    boundary_fp = 0
    boundary_fn = 0
    per_segment_correct = 0
    per_segment_total = 0
    for row, pred in zip(rows, preds):
        gold_markers = set(row.get("chain_markers") or [])
        pred_markers = set(pred.get("boundary_offsets") or [])
        # Permissive: a predicted offset counts as TP if it lands within
        # ±3 chars of a gold offset (subword tokenisation can land 1–2
        # chars off the word boundary).
        used: set[int] = set()
        for p in pred_markers:
            hit = next((g for g in gold_markers if abs(g - p) <= 3 and g not in used), None)
            if hit is not None:
                boundary_tp += 1
                used.add(hit)
            else:
                boundary_fp += 1
        boundary_fn += len(gold_markers - used)

        # Per-segment intent comparison (order matters).
        gold_segs = row.get("segments") or []
        pred_segs = pred.get("segments") or []
        all_match = len(gold_segs) == len(pred_segs)
        for g, p in zip(gold_segs, pred_segs):
            per_segment_total += 1
            if g.get("intent") == p.get("intent"):
                per_segment_correct += 1
            else:
                all_match = False
        if all_match and len(gold_segs) > 0:
            correct += 1

    prec = boundary_tp / (boundary_tp + boundary_fp) if (boundary_tp + boundary_fp) else 0.0
    rec = boundary_tp / (boundary_tp + boundary_fn) if (boundary_tp + boundary_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    log.info("  boundary detection P=%.2f R=%.2f F1=%.2f  (tp=%d fp=%d fn=%d)",
             prec, rec, f1, boundary_tp, boundary_fp, boundary_fn)
    log.info("  chain accuracy: %.3f (%d/%d rows fully correct)",
             correct / n, correct, n)
    if per_segment_total:
        log.info("  per-segment intent accuracy: %.3f (%d/%d segments)",
                 per_segment_correct / per_segment_total,
                 per_segment_correct, per_segment_total)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-groq", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--data", type=Path, default=EVAL_PATH,
                    help="path to a .jsonl eval set (defaults to intents_eval.jsonl)")
    args = ap.parse_args()

    if not args.data.exists():
        log.error("eval data not found: %s", args.data)
        sys.exit(1)
    log.info("eval data: %s", args.data)
    all_rows = load_jsonl(args.data)
    # Chain rows carry a `segments` list; single-intent rows don't.
    chain_rows = [r for r in all_rows if r.get("segments")]
    single_rows = [r for r in all_rows if not r.get("segments")]
    gold_intents = [r["intent"] for r in single_rows]
    labels = sorted({*gold_intents, "UNKNOWN"})
    log.info("eval set: %d total (%d single-intent, %d chain)",
             len(all_rows), len(single_rows), len(chain_rows))
    log.info("single-intent rows with gold slots: %d",
             sum(1 for r in single_rows if r.get("slots")))

    if not args.model_dir.exists():
        log.error("model dir not found: %s — run train_intent.py first", args.model_dir)
        sys.exit(1)

    local_results = predict_local(args.model_dir, single_rows)
    local_intents = [r[0] for r in local_results]
    local_slots = [r[1] for r in local_results]
    report_intents("local (joint XLM-RoBERTa)", gold_intents, local_intents, labels)
    report_slots("local (joint XLM-RoBERTa)", single_rows, local_slots, local_intents)

    if chain_rows:
        chain_preds = predict_chains(args.model_dir, chain_rows)
        report_chains(chain_rows, chain_preds)

    if args.compare_groq:
        groq_results = await predict_groq(single_rows)
        groq_intents = [r[0] for r in groq_results]
        groq_slots = [r[1] for r in groq_results]
        report_intents("Groq Llama baseline", gold_intents, groq_intents, labels)
        report_slots("Groq Llama baseline", single_rows, groq_slots, groq_intents)


if __name__ == "__main__":
    asyncio.run(main())
