from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (  # type: ignore[import-untyped]
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_intent")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "scripts" / "data"
TRAIN_PATH = DATA_DIR / "intents.jsonl"
EVAL_PATH = DATA_DIR / "intents_eval.jsonl"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"
OUT_DIR = ROOT / "backend" / "models" / "intent_classifier"

BASE_MODEL = "xlm-roberta-base"
MAX_LEN = 64
SEED = 1337

SLOT_LABELS = ["O", "B-NAME", "I-NAME"]
SLOT_TO_ID = {tag: i for i, tag in enumerate(SLOT_LABELS)}
ID_TO_SLOT = {i: tag for i, tag in enumerate(SLOT_LABELS)}
IGNORE_INDEX = -100
NAME_SLOT_KEYS = ("name", "old_name", "new_name")

NUM_BOUNDARY_TAGS = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def char_spans_for(text: str, slots: dict[str, str]) -> list[tuple[int, int]]:
    if not slots:
        return []
    t = text.lower()
    spans: list[tuple[int, int]] = []
    cursor = 0
    keys_in_order = [k for k in ("old_name", "new_name", "name") if k in slots]
    for k in keys_in_order:
        v = str(slots[k]).lower().strip()
        if not v:
            continue
        idx = t.find(v, cursor)
        if idx < 0:
            return []
        spans.append((idx, idx + len(v)))
        cursor = idx + len(v)
    return sorted(spans)


def build_token_labels(
    offsets: list[tuple[int, int]],
    char_spans: list[tuple[int, int]],
) -> list[int]:
    labels: list[int] = []
    for start, end in offsets:
        if start == 0 and end == 0:
            labels.append(IGNORE_INDEX)
            continue
        in_span = False
        is_first = False
        for s_start, s_end in char_spans:
            if start >= s_start and end <= s_end and start < s_end:
                in_span = True
                is_first = (start == s_start)
                break
        if in_span:
            labels.append(SLOT_TO_ID["B-NAME" if is_first else "I-NAME"])
        else:
            labels.append(SLOT_TO_ID["O"])
    return labels


def build_boundary_labels(
    offsets: list[tuple[int, int]],
    marker_offsets: list[int],
) -> list[int]:
    labels: list[int] = []
    marker_set = set(marker_offsets)
    for start, end in offsets:
        if start == 0 and end == 0:
            labels.append(IGNORE_INDEX)
            continue
        labels.append(1 if start in marker_set else 0)
    return labels


class JointDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        rows: list[dict[str, Any]],
        intent_to_id: dict[str, int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.rows = rows
        self.intent_to_id = intent_to_id
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = row["text"]
        is_chain = bool(row.get("segments"))
        slots = row.get("slots") or {}
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        attn = enc["attention_mask"][0].tolist()

        if is_chain:
            slot_labels = [IGNORE_INDEX] * len(offsets)
        else:
            char_spans = char_spans_for(text, slots) if slots else []
            slot_labels = build_token_labels(offsets, char_spans)
            slot_labels = [
                IGNORE_INDEX if a == 0 else lbl
                for a, lbl in zip(attn, slot_labels)
            ]

        if is_chain:
            intent_label = IGNORE_INDEX
        else:
            intent_label = self.intent_to_id[row["intent"]]

        marker_offsets: list[int] = list(row.get("chain_markers") or [])
        boundary_labels = build_boundary_labels(offsets, marker_offsets)
        boundary_labels = [
            IGNORE_INDEX if a == 0 else lbl
            for a, lbl in zip(attn, boundary_labels)
        ]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_labels": torch.tensor(intent_label, dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
            "boundary_labels": torch.tensor(boundary_labels, dtype=torch.long),
        }


class XLMRobertaForIntentSlot(XLMRobertaPreTrainedModel):

    def __init__(
        self,
        config: XLMRobertaConfig,
        num_intents: int,
        num_slot_tags: int,
        num_boundary_tags: int = NUM_BOUNDARY_TAGS,
    ) -> None:
        super().__init__(config)
        self.num_intents = num_intents
        self.num_slot_tags = num_slot_tags
        self.num_boundary_tags = num_boundary_tags
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        dropout_p = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_p)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_tags)
        self.boundary_classifier = nn.Linear(config.hidden_size, num_boundary_tags)
        config.num_intents = num_intents
        config.num_slot_tags = num_slot_tags
        config.num_boundary_tags = num_boundary_tags
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        intent_labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
        boundary_labels: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        seq = outputs.last_hidden_state
        pooled = self.dropout(seq[:, 0])
        intent_logits = self.intent_classifier(pooled)
        slot_logits = self.slot_classifier(self.dropout(seq))
        boundary_logits = self.boundary_classifier(self.dropout(seq))
        loss = None
        if intent_labels is not None and slot_labels is not None:
            loss_i = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
                intent_logits, intent_labels
            )
            loss_s = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
                slot_logits.view(-1, self.num_slot_tags),
                slot_labels.view(-1),
            )
            loss = loss_i + loss_s
            if boundary_labels is not None:
                loss_b = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
                    boundary_logits.view(-1, self.num_boundary_tags),
                    boundary_labels.view(-1),
                )
                loss = loss + loss_b
        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "boundary_logits": boundary_logits,
        }


def compute_metrics(eval_pred):  # type: ignore[no-untyped-def]
    (intent_logits, slot_logits, boundary_logits), (
        intent_labels, slot_labels, boundary_labels
    ) = eval_pred

    intent_pred = np.argmax(intent_logits, axis=-1)
    intent_mask = intent_labels != IGNORE_INDEX
    in_pred = intent_pred[intent_mask]
    in_gold = intent_labels[intent_mask]
    if len(in_gold) == 0:
        intent_acc = 0.0
        intent_f1 = 0.0
    else:
        intent_acc = float((in_pred == in_gold).mean())
        classes = np.unique(in_gold)
        f1s = []
        for c in classes:
            tp = int(((in_pred == c) & (in_gold == c)).sum())
            fp = int(((in_pred == c) & (in_gold != c)).sum())
            fn = int(((in_pred != c) & (in_gold == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        intent_f1 = float(np.mean(f1s))

    slot_pred = np.argmax(slot_logits, axis=-1)
    mask = slot_labels != IGNORE_INDEX
    sl_pred = slot_pred[mask]
    sl_gold = slot_labels[mask]
    if len(sl_gold) == 0:
        slot_f1 = 0.0
    else:
        f1s2 = []
        for c in range(len(SLOT_LABELS)):
            tp = int(((sl_pred == c) & (sl_gold == c)).sum())
            fp = int(((sl_pred == c) & (sl_gold != c)).sum())
            fn = int(((sl_pred != c) & (sl_gold == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s2.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        slot_f1 = float(np.mean(f1s2))

    bd_pred = np.argmax(boundary_logits, axis=-1)
    bd_mask = boundary_labels != IGNORE_INDEX
    bd_pred_m = bd_pred[bd_mask]
    bd_gold_m = boundary_labels[bd_mask]
    tp = int(((bd_pred_m == 1) & (bd_gold_m == 1)).sum())
    fp = int(((bd_pred_m == 1) & (bd_gold_m == 0)).sum())
    fn = int(((bd_pred_m == 0) & (bd_gold_m == 1)).sum())
    bd_prec = tp / (tp + fp) if (tp + fp) else 0.0
    bd_rec = tp / (tp + fn) if (tp + fn) else 0.0
    bd_f1 = 2 * bd_prec * bd_rec / (bd_prec + bd_rec) if (bd_prec + bd_rec) else 0.0

    return {
        "intent_accuracy": intent_acc,
        "intent_f1_macro": intent_f1,
        "slot_f1_macro": slot_f1,
        "boundary_precision": float(bd_prec),
        "boundary_recall": float(bd_rec),
        "boundary_f1": float(bd_f1),
    }


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--base-model", type=str, default=BASE_MODEL)
    args = ap.parse_args()

    set_seed(SEED)
    device = pick_device()
    log.info("device: %s", device)

    intent_to_id = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    id_to_intent = {int(v): k for k, v in intent_to_id.items()}
    num_intents = len(intent_to_id)
    num_slot_tags = len(SLOT_LABELS)
    log.info("intents: %d  slot tags: %s", num_intents, SLOT_LABELS)

    train_rows = load_jsonl(TRAIN_PATH)
    eval_rows = load_jsonl(EVAL_PATH)
    log.info("train: %d rows  eval: %d rows", len(train_rows), len(eval_rows))
    log.info("train class balance: %s", Counter(r["intent"] for r in train_rows))
    log.info("train rows with slots: %d / %d",
             sum(1 for r in train_rows if r.get("slots")), len(train_rows))
    log.info("train chain rows: %d / %d",
             sum(1 for r in train_rows if r.get("segments")), len(train_rows))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = XLMRobertaConfig.from_pretrained(args.base_model)
    config.id2label = id_to_intent
    config.label2id = intent_to_id
    model = XLMRobertaForIntentSlot.from_pretrained(
        args.base_model,
        config=config,
        num_intents=num_intents,
        num_slot_tags=num_slot_tags,
    )

    train_ds = JointDataset(train_rows, intent_to_id, tokenizer, MAX_LEN)
    eval_ds = JointDataset(eval_rows, intent_to_id, tokenizer, MAX_LEN)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "_trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="intent_f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=SEED,
        fp16=device == "cuda",
        label_names=["intent_labels", "slot_labels", "boundary_labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    log.info("final eval metrics: %s", metrics)

    log.info("saving best model to %s", args.out_dir)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    (args.out_dir / "label_map.json").write_text(
        json.dumps(intent_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "slot_label_map.json").write_text(
        json.dumps(SLOT_TO_ID, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state_dir = args.out_dir / "_trainer_state"
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
    log.info("done.")


if __name__ == "__main__":
    main()
