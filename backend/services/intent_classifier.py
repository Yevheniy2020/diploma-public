from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        id2label: dict[int, str],
        slot_id2tag: dict[int, str] | None,
        joint: bool,
        has_boundary: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._slot_id2tag = slot_id2tag or {}
        self._joint = joint
        self._has_boundary = has_boundary
        import torch
        self._torch = torch

    @classmethod
    def load(cls, model_dir: str | Path) -> "IntentClassifier":
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"intent model dir not found: {path}")

        label_map_path = path / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json missing in {path}")
        intent_to_id = json.loads(label_map_path.read_text(encoding="utf-8"))
        id2label = {int(v): k for k, v in intent_to_id.items()}

        slot_map_path = path / "slot_label_map.json"
        slot_id2tag: dict[int, str] = {}
        joint = False
        if slot_map_path.exists():
            slot_to_id = json.loads(slot_map_path.read_text(encoding="utf-8"))
            slot_id2tag = {int(v): k for k, v in slot_to_id.items()}
            joint = True

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path))

        if joint:
            from transformers import (  # type: ignore[import-untyped]
                XLMRobertaConfig,
                XLMRobertaModel,
                XLMRobertaPreTrainedModel,
            )
            import torch.nn as nn

            saved_config = XLMRobertaConfig.from_pretrained(str(path))
            num_boundary_tags = int(getattr(saved_config, "num_boundary_tags", 0) or 0)
            has_boundary = num_boundary_tags > 0

            class XLMRobertaForIntentSlot(XLMRobertaPreTrainedModel):
                def __init__(self, config: XLMRobertaConfig) -> None:
                    super().__init__(config)
                    self.num_intents = getattr(config, "num_intents", len(id2label))
                    self.num_slot_tags = getattr(config, "num_slot_tags", len(slot_id2tag))
                    self.num_boundary_tags = int(getattr(config, "num_boundary_tags", 0) or 0)
                    self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
                    dropout_p = getattr(config, "hidden_dropout_prob", 0.1)
                    self.dropout = nn.Dropout(dropout_p)
                    self.intent_classifier = nn.Linear(config.hidden_size, self.num_intents)
                    self.slot_classifier = nn.Linear(config.hidden_size, self.num_slot_tags)
                    if self.num_boundary_tags > 0:
                        self.boundary_classifier = nn.Linear(
                            config.hidden_size, self.num_boundary_tags
                        )
                    self.post_init()

                def forward(self, input_ids, attention_mask=None, **_):
                    out = self.roberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    seq = out.last_hidden_state
                    pooled = self.dropout(seq[:, 0])
                    result = {
                        "intent_logits": self.intent_classifier(pooled),
                        "slot_logits": self.slot_classifier(self.dropout(seq)),
                    }
                    if self.num_boundary_tags > 0:
                        result["boundary_logits"] = self.boundary_classifier(self.dropout(seq))
                    return result

            model = XLMRobertaForIntentSlot.from_pretrained(str(path))
            model.eval()
            _log.info(
                "joint intent+slot%s classifier loaded from %s "
                "(%d intents, %d slot tags%s)",
                "+boundary" if has_boundary else "",
                path,
                len(id2label),
                len(slot_id2tag),
                f", {num_boundary_tags} boundary tags" if has_boundary else "",
            )
        else:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSequenceClassification,
            )
            model = AutoModelForSequenceClassification.from_pretrained(str(path))
            model.eval()
            has_boundary = False
            _log.info("intent-only classifier loaded from %s (%d labels)",
                      path, len(id2label))

        return cls(
            model, tokenizer, id2label, slot_id2tag, joint,
            has_boundary=has_boundary,
        )

    def predict(
        self, text: str, max_len: int = 64
    ) -> tuple[str, float, dict[str, list[tuple[int, int]]], list[int]]:
        text = (text or "").strip()
        if not text:
            return "UNKNOWN", 0.0, {}, []
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=self._joint,
        )
        offsets = None
        if self._joint:
            offsets = enc.pop("offset_mapping")[0].tolist()
        with self._torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
            )

        if self._joint:
            intent_logits = out["intent_logits"][0]
        else:
            intent_logits = out.logits[0]
        probs = self._torch.softmax(intent_logits, dim=-1)
        idx = int(probs.argmax().item())
        intent = self._id2label[idx]
        confidence = float(probs[idx].item())

        boundary_offsets: list[int] = []
        if self._joint and self._has_boundary and offsets is not None:
            boundary_logits = out.get("boundary_logits")
            if boundary_logits is not None:
                bd_ids = boundary_logits[0].argmax(dim=-1).tolist()
                boundary_offsets = _decode_boundary_offsets(bd_ids, offsets, text)

        spans: dict[str, list[tuple[int, int]]] = {}
        if self._joint and offsets is not None:
            slot_logits = out["slot_logits"][0]
            slot_ids = slot_logits.argmax(dim=-1).tolist()
            spans = _decode_bio_spans(slot_ids, offsets, self._slot_id2tag, text)
        return intent, confidence, spans, boundary_offsets


def _decode_boundary_offsets(
    boundary_ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
) -> list[int]:
    out: list[int] = []
    for tag_id, (start, end) in zip(boundary_ids, offsets):
        if start == 0 and end == 0:
            continue
        if tag_id == 1:
            out.append(start)
    return out


def _decode_bio_spans(
    slot_ids: list[int],
    offsets: list[tuple[int, int]],
    id2tag: dict[int, str],
    text: str,
) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None and cur_start is not None and cur_end is not None:
            spans.setdefault(cur_type, []).append((cur_start, cur_end))
        cur_type = cur_start = cur_end = None

    for tok_id, (start, end) in zip(slot_ids, offsets):
        if start == 0 and end == 0:
            flush()
            continue
        tag = id2tag.get(tok_id, "O")
        if tag == "O":
            flush()
            continue
        prefix, _, kind = tag.partition("-")
        if prefix == "B":
            flush()
            cur_type = kind
            cur_start = start
            cur_end = end
        elif prefix == "I" and cur_type == kind and cur_end is not None:
            cur_end = end
        else:
            flush()
            cur_type = kind
            cur_start = start
            cur_end = end
    flush()

    cleaned: dict[str, list[tuple[int, int]]] = {}
    for kind, items in spans.items():
        out: list[tuple[int, int]] = []
        for s, e in items:
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                out.append((s, e))
        if out:
            cleaned[kind] = out
    return cleaned


_instance: Optional[IntentClassifier] = None


def get_classifier() -> Optional[IntentClassifier]:
    return _instance


def set_classifier(clf: Optional[IntentClassifier]) -> None:
    global _instance
    _instance = clf
