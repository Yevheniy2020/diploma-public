from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrain_from_feedback")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "scripts" / "data"
FEEDBACK_SEEDS = DATA_DIR / "seeds_from_feedback.jsonl"
CURSOR_FILE = DATA_DIR / "last_processed_correction.txt"
LAST_RETRAINED_FILE = DATA_DIR / "last_retrained_at.txt"
STATUS_FILE = DATA_DIR / "retrain_status.json"
PENDING_SEEDS_FILE = DATA_DIR / "pending_seeds.jsonl"
DB_PATH = ROOT / "backend" / "db.sqlite3"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(phase: str, started_at: str, error: str | None = None) -> None:
    payload = {
        "phase": phase,
        "started_at": started_at,
        "updated_at": _now_iso(),
        "error": error,
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATUS_FILE)


def detect_lang(text: str) -> str:
    return "uk" if re.search(r"[а-яіїєґА-ЯІЇЄҐ]", text) else "en"


def read_cursor() -> int:
    if not CURSOR_FILE.exists():
        return 0
    try:
        return int(CURSOR_FILE.read_text(encoding="utf-8").strip() or "0")
    except (ValueError, OSError):
        return 0


def write_cursor(value: int) -> None:
    CURSOR_FILE.write_text(str(value), encoding="utf-8")


def fetch_new_corrections(db_path: Path, after_id: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        log.error("db.sqlite3 not found at %s", db_path)
        sys.exit(1)
    out: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT id, transcription, predicted_intent, predicted_params_json, "
            "was_correct, corrected_intent, corrected_params_json "
            "FROM command_corrections "
            "WHERE id > ? ORDER BY id ASC",
            (after_id,),
        )
        for row in cur.fetchall():
            cid, tx, pred_intent, pred_params_json, was_correct, override, corr_params_json = row
            tx = (tx or "").strip()
            if not tx:
                continue
            try:
                pred_params = json.loads(pred_params_json or "{}")
            except json.JSONDecodeError:
                pred_params = {}
            try:
                corr_params = json.loads(corr_params_json or "null") if corr_params_json else None
            except json.JSONDecodeError:
                corr_params = None
            out.append({
                "id": cid,
                "transcription": tx,
                "predicted_intent": pred_intent,
                "predicted_params": pred_params if isinstance(pred_params, dict) else {},
                "was_correct": bool(was_correct),
                "corrected_intent": override,
                "corrected_params": corr_params if isinstance(corr_params, dict) else None,
            })
    return out


def derive_seed_row(c: dict[str, Any]) -> dict[str, Any] | None:
    text = c["transcription"]
    if c["was_correct"]:
        intent = c["corrected_intent"] or c["predicted_intent"]
        params = c["corrected_params"] or c["predicted_params"] or {}
        slots: dict[str, str] = {}
        for k in ("name", "old_name", "new_name"):
            v = params.get(k)
            if isinstance(v, str) and v.strip():
                slots[k] = v.strip()
        row: dict[str, Any] = {
            "text": text,
            "intent": intent,
            "lang": detect_lang(text),
        }
        if slots:
            row["slots"] = slots
        return row
    if c["corrected_intent"]:
        return {
            "text": text,
            "intent": c["corrected_intent"],
            "lang": detect_lang(text),
        }
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chain_retrain(started_at: str) -> None:
    backend_dir = ROOT / "backend"
    py = str(backend_dir / ".venv" / "bin" / "python")
    write_status("paraphrasing", started_at)
    log.info("→ generating dataset (generate_dataset.py)")
    cmd = [py, str(ROOT / "scripts" / "generate_dataset.py")]
    if PENDING_SEEDS_FILE.exists() and PENDING_SEEDS_FILE.stat().st_size > 0:
        cmd += ["--append-from", str(PENDING_SEEDS_FILE)]
        log.info("  using --append-from (incremental mode)")
    subprocess.check_call(cmd, cwd=backend_dir)
    write_status("training", started_at)
    log.info("→ training joint model (train_intent.py)")
    subprocess.check_call(
        [py, str(ROOT / "scripts" / "train_intent.py"), "--epochs", "8"],
        cwd=backend_dir,
    )
    PENDING_SEEDS_FILE.unlink(missing_ok=True)
    LAST_RETRAINED_FILE.write_text(_now_iso(), encoding="utf-8")
    write_status("done", started_at)
    log.info("retrain complete — backend will hot-reload weights")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="write seeds_from_feedback.jsonl but skip paraphrase+train")
    ap.add_argument("--reset-cursor", action="store_true",
                    help="reprocess all corrections from scratch")
    args = ap.parse_args()

    started_at = _now_iso()
    write_status("preparing", started_at)
    try:
        cursor = 0 if args.reset_cursor else read_cursor()
        log.info("processing corrections with id > %d", cursor)
        corrections = fetch_new_corrections(DB_PATH, cursor)
        if not corrections:
            log.info("no new corrections to process — exiting")
            write_status("done", started_at)
            return

        seed_rows: list[dict[str, Any]] = []
        skipped = 0
        for c in corrections:
            seed = derive_seed_row(c)
            if seed is None:
                skipped += 1
                continue
            seed_rows.append(seed)

        log.info("→ converted %d corrections into %d seed rows (%d skipped — pure rejections)",
                 len(corrections), len(seed_rows), skipped)

        if seed_rows:
            append_jsonl(FEEDBACK_SEEDS, seed_rows)
            write_jsonl(PENDING_SEEDS_FILE, seed_rows)
            log.info("appended to %s, wrote scratch %s",
                     FEEDBACK_SEEDS.name, PENDING_SEEDS_FILE.name)
        write_cursor(corrections[-1]["id"])

        if args.dry:
            log.info("--dry: skipping paraphrase + train")
            write_status("done", started_at)
            return

        if not seed_rows:
            log.info("no new positive seeds; skipping retrain")
            write_status("done", started_at)
            return

        chain_retrain(started_at)
    except Exception as exc:  # noqa: BLE001
        write_status("failed", started_at, error=str(exc))
        raise


if __name__ == "__main__":
    main()
