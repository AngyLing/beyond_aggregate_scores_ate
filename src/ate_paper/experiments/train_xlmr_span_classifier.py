#!/usr/bin/env python3
"""Nested-aware XLM-R span classifier baseline.

This baseline treats term extraction as independent candidate-span
classification. It does not convert nested annotations to BIO labels and does
not suppress overlapping predictions.
"""

import argparse
import json
import math
import os
import shutil
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ate_paper.experiments.supervised_tokenizers import (
    add_pretrained_arguments, pretrained_revision_kwargs, tokenizer_kwargs,
)
from typing import Optional


NO_TERM = "NO_TERM"
TERM_LABEL = "TERM"  # single positive class for unlabeled datasets (e.g. ACTER v1.5)
SPAN_ANNOTATION_UNIT_TYPES = {None, "span_mention"}
NON_SPAN_TEXT_UNIT_TYPES = {
    "term_string",
    "candidate_string",
    "sentence_term_string",
    "variant_cluster_member",
}


def normalize_surface(text):  # lowercases: see train_mbert_sequence_labeler.normalize_surface (L-G1)
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_no_doc_id_overlap(named_splits):
    """Fail if any two splits share a doc_id.

    Train and dev are built by partitioning documents, so an overlap can only
    come from a data-prep error. Catch it before training instead of reporting
    leaked scores. Thresholds are tuned on dev, so dev must stay disjoint from
    train.
    """
    doc_ids = {
        name: {row.get("doc_id") for row in rows if row.get("doc_id") is not None}
        for name, rows in named_splits
    }
    names = list(doc_ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = doc_ids[left] & doc_ids[right]
            if shared:
                example = sorted(str(d) for d in shared)[:5]
                raise ValueError(
                    f"doc_id overlap between {left} and {right} splits: "
                    f"{len(shared)} shared documents (e.g. {example}). "
                    "Splits must be disjoint to avoid train/test leakage."
                )


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def span_from_label_item(item):
    start = item.get("start")
    end = item.get("end")
    if isinstance(start, int) and isinstance(end, int) and start < end:
        return {
            "start": start,
            "end": end,
            "class_label": get_class_label(item),
            "surface": item.get("surface"),
            "norm": item.get("norm"),
        }
    return None


def get_class_label(item):
    return item.get("class_label") or item.get("classterm") or item.get("label")


def group_rows_into_text_units(rows, dataset=None, include_spans=True):
    """Read either project unified JSONL rows or {"id", "text", "label": []}.

    Nested/overlapping spans are preserved as separate entries.
    """
    generic_units = []
    unified = {}

    for row in rows:
        if dataset and row.get("dataset") not in {None, dataset}:
            continue
        if "label" in row and isinstance(row.get("label"), list):
            if not row.get("text"):
                continue
            spans = []
            if include_spans:
                for item in row.get("label", []):
                    span = span_from_label_item(item)
                    if span:
                        spans.append(span)
            unit = {
                "id": row.get("id"),
                "text": row.get("text"),
                "spans": spans,
                "input_format": "label_list",
            }
            for key in ("dataset", "version", "split", "language", "domain", "doc_id", "sent_id"):
                if key in row:
                    unit[key] = row.get(key)
            generic_units.append(unit)
            continue

        if not accepts_unified_row_as_text_unit(row, include_spans=include_spans):
            continue
        if not row.get("text"):
            continue
        key = (
            row.get("dataset"),
            row.get("version"),
            row.get("split"),
            row.get("language"),
            row.get("domain"),
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("text"),
        )
        if key not in unified:
            unified[key] = {
                "dataset": row.get("dataset"),
                "version": row.get("version"),
                "split": row.get("split"),
                "language": row.get("language"),
                "domain": row.get("domain"),
                "doc_id": row.get("doc_id"),
                "sent_id": row.get("sent_id"),
                "id": row.get("id") or row.get("sent_id") or row.get("doc_id"),
                "text": row.get("text"),
                "spans": [],
                "input_format": "unified",
            }
        if include_spans and row.get("unit_type") == "span_mention":
            span = span_from_label_item(row)
            if span:
                unified[key]["spans"].append(span)

    return generic_units + list(unified.values())


def accepts_unified_row_as_text_unit(row, include_spans=True):
    unit_type = row.get("unit_type")
    if include_spans:
        return unit_type in SPAN_ANNOTATION_UNIT_TYPES
    return unit_type not in NON_SPAN_TEXT_UNIT_TYPES


def class_labels_from_units(units):
    labels = sorted({
        span.get("class_label")
        for unit in units
        for span in unit.get("spans", [])
        if span.get("class_label")
    })
    return labels or [TERM_LABEL]


def selection_metric_for_labels(class_labels):
    """Describe the effective primary metric; the selection tuple is unchanged."""
    return "dev_strict_span_f1" if class_labels == [TERM_LABEL] else "dev_class_aware_f1"


def build_label_maps(class_labels):
    labels = [NO_TERM] + list(class_labels)
    return {label: index for index, label in enumerate(labels)}, {index: label for index, label in enumerate(labels)}


def non_special_token_indices(offset_mapping):
    return [
        index
        for index, (start, end) in enumerate(offset_mapping)
        if start is not None and end is not None and end > start
    ]


def generate_candidate_spans(offset_mapping, max_span_tokens):
    candidates = []
    valid = non_special_token_indices(offset_mapping)
    valid_set = set(valid)
    for start_index in valid:
        for width in range(1, max_span_tokens + 1):
            end_index = start_index + width - 1
            if end_index not in valid_set:
                break
            char_start = offset_mapping[start_index][0]
            char_end = offset_mapping[end_index][1]
            if char_start is None or char_end is None or char_end <= char_start:
                continue
            candidates.append({
                "start_token": start_index,
                "end_token": end_index,
                "width": width,
                "start": char_start,
                "end": char_end,
            })
    return candidates


def gold_labels_by_offset(gold_spans):
    by_key = defaultdict(list)
    for span in gold_spans:
        if isinstance(span.get("start"), int) and isinstance(span.get("end"), int):
            # unlabeled datasets carry class_label=None; map to the single TERM
            # class so the span is not dropped by the falsey-filter below.
            by_key[(span["start"], span["end"])].append(span.get("class_label") or TERM_LABEL)

    labels = {}
    conflicts = 0
    for key, values in by_key.items():
        cleaned = [value for value in values if value]
        if not cleaned:
            continue
        counts = Counter(cleaned)
        if len(counts) > 1:
            conflicts += 1
        labels[key] = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]
    return labels, conflicts


def label_candidate_spans(candidates, gold_spans, label_to_id):
    gold_map, conflicts = gold_labels_by_offset(gold_spans)
    candidate_keys = {(candidate["start"], candidate["end"]) for candidate in candidates}
    labels = []
    aligned = set()
    for candidate in candidates:
        key = (candidate["start"], candidate["end"])
        gold_label = gold_map.get(key)
        if gold_label and gold_label in label_to_id:
            labels.append(label_to_id[gold_label])
            aligned.add(key)
        else:
            labels.append(label_to_id[NO_TERM])

    unaligned = [
        {"start": start, "end": end, "class_label": label}
        for (start, end), label in gold_map.items()
        if (start, end) not in candidate_keys
    ]
    return labels, {
        "aligned_gold_spans": len(aligned),
        "unaligned_gold_spans": len(unaligned),
        "duplicate_gold_span_label_conflicts": conflicts,
    }


def sample_candidate_indices(labels, no_term_id, negative_ratio, rng, max_negatives_without_positive=20):
    positives = [index for index, label in enumerate(labels) if label != no_term_id]
    negatives = [index for index, label in enumerate(labels) if label == no_term_id]
    if positives:
        negative_limit = min(len(negatives), max(1, len(positives) * negative_ratio))
    else:
        negative_limit = min(len(negatives), max_negatives_without_positive)
    sampled_negatives = rng.sample(negatives, negative_limit) if negative_limit else []
    return sorted(positives + sampled_negatives)


def apply_length_balanced_positive_sampling(
    candidates,
    labels,
    no_term_id,
    enabled=False,
    long_span_token_threshold=4,
    long_positive_weight=3,
):
    """Duplicate long positive candidates to weight them more in training loss.

    This is an opt-in diagnostic. It does not alter gold labels, candidate
    generation, dev inference, prediction decoding, or evaluation.
    """
    if not enabled or long_positive_weight <= 1:
        return list(candidates), list(labels), {
            "length_balanced_positive_duplicates": 0,
            "length_balanced_long_positive_spans": 0,
        }

    expanded_candidates = []
    expanded_labels = []
    duplicates = 0
    long_positives = 0
    for candidate, label in zip(candidates, labels):
        expanded_candidates.append(candidate)
        expanded_labels.append(label)
        if label == no_term_id or candidate.get("width", 0) < long_span_token_threshold:
            continue
        long_positives += 1
        for _ in range(long_positive_weight - 1):
            expanded_candidates.append(dict(candidate))
            expanded_labels.append(label)
            duplicates += 1

    return expanded_candidates, expanded_labels, {
        "length_balanced_positive_duplicates": duplicates,
        "length_balanced_long_positive_spans": long_positives,
    }


def span_token_length_stats(candidates, labels, no_term_id):
    lengths = [candidate["width"] for candidate, label in zip(candidates, labels) if label != no_term_id]
    if not lengths:
        return 0.0, 0
    return sum(lengths) / len(lengths), max(lengths)


@dataclass
class SpanFeature:
    unit: dict
    input_ids: list
    attention_mask: list
    offset_mapping: list
    candidates: list
    labels: Optional[list] = None


def tokenize_unit(tokenizer, text, max_length, stride):
    return tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )


def build_span_features(
    units,
    tokenizer,
    label_to_id,
    max_span_tokens=10,
    negative_ratio=5,
    max_length=256,
    stride=64,
    seed=42,
    training=True,
    length_balanced_positive_sampling=False,
    long_span_token_threshold=4,
    long_positive_weight=3,
):
    rng = random.Random(seed)
    features = []
    stats = Counter()
    stats["documents"] = len(units)
    class_distribution = Counter()
    no_term_id = label_to_id[NO_TERM]

    for unit in units:
        spans = unit.get("spans", [])
        stats["gold_spans"] += len(spans)
        for span in spans:
            if span.get("class_label"):
                class_distribution[span["class_label"]] += 1

        encoded = tokenize_unit(tokenizer, unit["text"], max_length=max_length, stride=stride)
        input_ids_windows = encoded["input_ids"]
        if input_ids_windows and isinstance(input_ids_windows[0], int):
            window_count = 1
        else:
            window_count = len(input_ids_windows)
        if window_count > 1:
            stats["texts_split_into_overflow_windows"] += 1

        unit_candidate_keys = set()
        # Char range actually covered by each window's real (non-special) tokens --
        # used below to tell "gold span never fit in any window" (a stride/max_length
        # artifact, unrelated to this encoder's tokenization quality) apart from
        # "gold span was inside a window but no candidate matched its exact offsets"
        # (a genuine tokenizer-boundary/candidate-generation miss). Both used to be
        # folded into one counter, which conflates a windowing config choice with an
        # actual property of the encoder being evaluated (see
        # probe_span_encoder_alignment.py, which reuses this same code path to gate
        # domain-encoder acceptance -- a window-truncation-inflated number there would
        # unfairly penalize whichever tokenizer happens to be least compact per
        # character, independent of how well it aligns term boundaries).
        unit_window_ranges = []

        for window_index in range(window_count):
            if window_count == 1 and input_ids_windows and isinstance(input_ids_windows[0], int):
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                offsets = [tuple(item) for item in encoded["offset_mapping"]]
            else:
                input_ids = encoded["input_ids"][window_index]
                attention_mask = encoded["attention_mask"][window_index]
                offsets = [tuple(item) for item in encoded["offset_mapping"][window_index]]

            valid_indices = non_special_token_indices(offsets)
            if valid_indices:
                unit_window_ranges.append((offsets[valid_indices[0]][0], offsets[valid_indices[-1]][1]))

            candidates = generate_candidate_spans(offsets, max_span_tokens)
            unit_candidate_keys.update((candidate["start"], candidate["end"]) for candidate in candidates)
            stats["candidate_spans"] += len(candidates)
            if training:
                labels, label_stats = label_candidate_spans(candidates, spans, label_to_id)
                selected = sample_candidate_indices(labels, no_term_id, negative_ratio, rng)
                selected_candidates = [candidates[index] for index in selected]
                selected_labels = [labels[index] for index in selected]
                stats["positive_training_spans_unweighted"] += sum(
                    label != no_term_id for label in selected_labels
                )
                stats["negative_training_spans_unweighted"] += sum(
                    label == no_term_id for label in selected_labels
                )
                selected_candidates, selected_labels, balance_stats = apply_length_balanced_positive_sampling(
                    selected_candidates,
                    selected_labels,
                    no_term_id,
                    enabled=length_balanced_positive_sampling,
                    long_span_token_threshold=long_span_token_threshold,
                    long_positive_weight=long_positive_weight,
                )
                stats["length_balanced_positive_duplicates"] += balance_stats[
                    "length_balanced_positive_duplicates"
                ]
                stats["length_balanced_long_positive_spans"] += balance_stats[
                    "length_balanced_long_positive_spans"
                ]
                avg_len, max_len = span_token_length_stats(selected_candidates, selected_labels, no_term_id)
                if avg_len:
                    stats["positive_span_token_length_sum"] += avg_len * sum(label != no_term_id for label in selected_labels)
                    stats["positive_span_token_length_count"] += sum(label != no_term_id for label in selected_labels)
                    stats["max_positive_span_tokens"] = max(stats["max_positive_span_tokens"], max_len)
                stats["positive_training_spans"] += sum(label != no_term_id for label in selected_labels)
                stats["negative_training_spans"] += sum(label == no_term_id for label in selected_labels)
                candidates = selected_candidates
                labels = selected_labels
            else:
                labels = None

            if not candidates:
                continue
            features.append(SpanFeature(
                unit=unit,
                input_ids=input_ids,
                attention_mask=attention_mask,
                offset_mapping=offsets,
                candidates=candidates,
                labels=labels,
            ))

        gold_map, conflicts = gold_labels_by_offset(spans)
        missing_keys = [key for key in gold_map if key not in unit_candidate_keys]
        stats["aligned_gold_spans"] += len(gold_map) - len(missing_keys)
        # Total kept exactly as before (same value, same name) so existing reports
        # citing this field are unaffected. window_truncated_gold_spans is a NEW
        # subset: missing keys whose gold char range wasn't fully inside any single
        # window at all, i.e. dropped by stride/max_length windowing rather than by
        # the tokenizer's own boundary alignment. boundary_misaligned_gold_spans is
        # the complement -- the actual tokenizer/candidate-generation miss rate.
        stats["unaligned_skipped_gold_spans"] += len(missing_keys)
        window_truncated = sum(
            1 for (g_start, g_end) in missing_keys
            if not any(w_start <= g_start and g_end <= w_end for w_start, w_end in unit_window_ranges)
        )
        stats["window_truncated_gold_spans"] += window_truncated
        stats["boundary_misaligned_gold_spans"] += len(missing_keys) - window_truncated
        stats["duplicate_gold_span_label_conflicts"] += conflicts

    stats["class_distribution"] = dict(class_distribution)
    stats["length_balanced_positive_sampling"] = bool(length_balanced_positive_sampling)
    stats["long_span_token_threshold"] = long_span_token_threshold
    stats["long_positive_weight"] = long_positive_weight if length_balanced_positive_sampling else 1
    if stats["positive_span_token_length_count"]:
        stats["avg_positive_span_tokens"] = (
            stats["positive_span_token_length_sum"] / stats["positive_span_token_length_count"]
        )
    else:
        stats["avg_positive_span_tokens"] = 0.0
    stats.pop("positive_span_token_length_sum", None)
    stats.pop("positive_span_token_length_count", None)
    return features, dict(stats)


def choose_device(device_arg="auto"):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpanFeatureDataset:
    def __init__(self, features):
        self.features = list(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def collate_span_features(features, tokenizer, include_labels=True):
    import torch

    encoded = tokenizer.pad(
        [{"input_ids": feature.input_ids, "attention_mask": feature.attention_mask} for feature in features],
        return_tensors="pt",
    )
    max_spans = max(len(feature.candidates) for feature in features)
    span_indices = torch.full((len(features), max_spans, 3), -1, dtype=torch.long)
    span_mask = torch.zeros((len(features), max_spans), dtype=torch.bool)
    labels = torch.full((len(features), max_spans), -100, dtype=torch.long)
    for batch_index, feature in enumerate(features):
        for span_index, candidate in enumerate(feature.candidates):
            span_indices[batch_index, span_index] = torch.tensor([
                candidate["start_token"],
                candidate["end_token"],
                min(candidate["width"], 512),
            ])
            span_mask[batch_index, span_index] = True
            if include_labels and feature.labels is not None:
                labels[batch_index, span_index] = feature.labels[span_index]
    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "span_indices": span_indices,
        "span_mask": span_mask,
    }
    if include_labels:
        batch["labels"] = labels
    return batch


def load_transformer_modules():
    import torch
    from transformers import AutoModel, AutoTokenizer

    return torch, AutoModel, AutoTokenizer


def vectorized_span_representations(hidden, span_indices, span_mask, width_embedding, max_span_tokens):
    import torch

    hidden_size = hidden.shape[-1]
    safe_indices = span_indices.clamp(min=0)
    start_indices = safe_indices[:, :, 0]
    end_indices = safe_indices[:, :, 1]
    widths = safe_indices[:, :, 2].clamp(min=1, max=max_span_tokens)

    gather_shape = start_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
    start_hidden = hidden.gather(1, gather_shape)
    end_hidden = hidden.gather(1, end_indices.unsqueeze(-1).expand(-1, -1, hidden_size))

    prefix = torch.cat([hidden.new_zeros(hidden.shape[0], 1, hidden_size), hidden.cumsum(dim=1)], dim=1)
    prefix_start = prefix.gather(1, start_indices.unsqueeze(-1).expand(-1, -1, hidden_size))
    prefix_end = prefix.gather(1, (end_indices + 1).unsqueeze(-1).expand(-1, -1, hidden_size))
    span_lengths = (end_indices - start_indices + 1).clamp(min=1).unsqueeze(-1).to(hidden.dtype)
    mean_hidden = (prefix_end - prefix_start) / span_lengths

    width_hidden = width_embedding(widths)
    span_repr = torch.cat([start_hidden, end_hidden, mean_hidden, width_hidden], dim=-1)
    return span_repr * span_mask.unsqueeze(-1).to(span_repr.dtype)


def make_span_classifier_class():
    torch, AutoModel, _ = load_transformer_modules()

    class XlmrSpanClassifier(torch.nn.Module):
        def __init__(
            self,
            model_name,
            num_labels,
            max_span_tokens=10,
            width_embedding_dim=32,
            hidden_dropout=0.1,
            model_revision=None,
        ):
            super().__init__()
            self.model_name = model_name
            self.num_labels = num_labels
            self.max_span_tokens = max_span_tokens
            self.width_embedding_dim = width_embedding_dim
            self.encoder = AutoModel.from_pretrained(model_name, **pretrained_revision_kwargs(model_revision))
            hidden_size = self.encoder.config.hidden_size
            self.width_embedding = torch.nn.Embedding(max_span_tokens + 1, width_embedding_dim)
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(hidden_dropout),
                torch.nn.Linear(hidden_size * 3 + width_embedding_dim, hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(hidden_dropout),
                torch.nn.Linear(hidden_size, num_labels),
            )

        def span_representations(self, hidden, span_indices, span_mask):
            return vectorized_span_representations(
                hidden,
                span_indices,
                span_mask,
                self.width_embedding,
                self.max_span_tokens,
            )

        def forward(self, input_ids, attention_mask, span_indices, span_mask, labels=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            span_repr = self.span_representations(outputs.last_hidden_state, span_indices, span_mask)
            logits = self.classifier(span_repr)
            loss = None
            if labels is not None:
                active = span_mask & labels.ne(-100)
                if active.any():
                    loss = torch.nn.functional.cross_entropy(logits[active], labels[active])
                else:
                    loss = logits.sum() * 0
            return {"loss": loss, "logits": logits}

    return XlmrSpanClassifier


def save_model_bundle(model, tokenizer, output_dir, config):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    model.encoder.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir)
    write_json(config, output_dir / "span_classifier_config.json")


def load_model_bundle(model_dir, device=None, scibert_do_lower_case=None):
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    config = read_json(model_dir / "span_classifier_config.json")
    # Deliberately plain (no fix_mistral_regex, unlike the BIO entry points):
    # the flag is a no-op for the tokenizers this pipeline loads (see
    # scripts/validate/check_tokenizer_flag_equivalence.py), and the published
    # span pipeline is left byte-stable rather than unified cosmetically.
    stored_casing = config.get("scibert_do_lower_case")
    if scibert_do_lower_case is not None and stored_casing is not None and scibert_do_lower_case != stored_casing:
        raise ValueError("SciBERT casing differs from the trained model bundle.")
    casing = scibert_do_lower_case if scibert_do_lower_case is not None else stored_casing
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs(
        model_dir, casing, model_name=config["model_name"],
    ))
    ModelClass = make_span_classifier_class()
    encoder_source = str(model_dir / "encoder") if (model_dir / "encoder").exists() else config["model_name"]
    model = ModelClass(
        model_name=encoder_source,
        num_labels=len(config["label2id"]),
        max_span_tokens=config["max_span_tokens"],
        width_embedding_dim=config.get("width_embedding_dim", 32),
        model_revision=None if (model_dir / "encoder").exists() else config.get("model_revision"),
    )
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model, tokenizer, config


def evaluate_loss(model, features, tokenizer, batch_size, device):
    import torch
    from torch.utils.data import DataLoader

    if not features:
        return math.inf
    loader_kwargs = {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(
        SpanFeatureDataset(features),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_span_features(batch, tokenizer, include_labels=True),
        **loader_kwargs,
    )
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(**batch)
            if output["loss"] is not None:
                losses.append(float(output["loss"].detach().cpu()))
    return sum(losses) / len(losses) if losses else math.inf


def scored_candidate_spans(model, features, tokenizer, batch_size, device, id2label):
    import torch

    # Drives the loop directly off _batched(features, ...) instead of pairing it
    # with a separately-constructed DataLoader via zip() -- the two iterators are
    # index-aligned today only because the loader happens to use shuffle=False and
    # no batch_sampler, with nothing enforcing that invariant. A future change to
    # this loader (sampler, drop_last, batch_sampler) would silently desync
    # predictions from features with no error, which is a worse failure mode for a
    # paper's reported numbers than losing DataLoader's worker prefetch on this
    # inference-only pass.
    model.eval()
    by_unit = defaultdict(list)
    with torch.no_grad():
        for batch_features in _batched(features, batch_size):
            batch = collate_span_features(batch_features, tokenizer, include_labels=False)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            probs = torch.softmax(model(**batch)["logits"], dim=-1).detach().cpu()
            for local_index, feature in enumerate(batch_features):
                for span_index, candidate in enumerate(feature.candidates):
                    if not batch["span_mask"][local_index, span_index]:
                        continue
                    label_id = int(torch.argmax(probs[local_index, span_index]).item())
                    label = id2label[label_id]
                    if label == NO_TERM:
                        continue
                    by_unit[id(feature.unit)].append({
                        "start": candidate["start"],
                        "end": candidate["end"],
                        "class_label": label,
                        "score": float(probs[local_index, span_index, label_id].item()),
                    })
    return by_unit


def gold_span_sets(units):
    unlabeled = set()
    class_aware = set()
    for unit in units:
        unit_key = id(unit)
        for span in unit.get("spans", []):
            start = span.get("start")
            end = span.get("end")
            if isinstance(start, int) and isinstance(end, int) and start < end:
                unlabeled.add((unit_key, start, end))
                class_aware.add((unit_key, start, end, span.get("class_label")))
    return unlabeled, class_aware


def prediction_span_sets(scored_by_unit, threshold):
    unlabeled = set()
    class_aware = set()
    for unit_key, spans in scored_by_unit.items():
        kept = [span for span in spans if span.get("score", 0.0) >= threshold]
        for span in deduplicate_scored_spans(kept, class_aware=True):
            unlabeled.add((unit_key, span["start"], span["end"]))
            class_aware.add((unit_key, span["start"], span["end"], span.get("class_label")))
    return unlabeled, class_aware


def precision_recall_f1(predicted, gold):
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _dev_collapse_fn(name):
    """Class-label collapse applied ONLY to the dev-selection metric (E1 arm A:
    train on 36 classes but pick threshold/best-epoch on the scored 6-class
    objective). genia_coarse6 imports without heavy deps."""
    if not name:
        return None
    if name == "genia_coarse6":
        from ate_paper.experiments.genia_label_maps import genia_coarse6
        return genia_coarse6
    raise ValueError(f"Unknown dev-eval-collapse: {name!r}")


def evaluate_scored_spans(scored_by_unit, units, threshold, collapse_fn=None):
    gold_unlabeled, gold_class = gold_span_sets(units)
    pred_unlabeled, pred_class = prediction_span_sets(scored_by_unit, threshold)
    if collapse_fn is not None:
        gold_class = {(u, s, e, collapse_fn(c)) for (u, s, e, c) in gold_class}
        pred_class = {(u, s, e, collapse_fn(c)) for (u, s, e, c) in pred_class}
    unlabeled_scores = precision_recall_f1(pred_unlabeled, gold_unlabeled)
    class_scores = precision_recall_f1(pred_class, gold_class)
    return {
        "threshold": threshold,
        "strict_span_precision": unlabeled_scores["precision"],
        "strict_span_recall": unlabeled_scores["recall"],
        "strict_span_f1": unlabeled_scores["f1"],
        "class_aware_precision": class_scores["precision"],
        "class_aware_recall": class_scores["recall"],
        "class_aware_f1": class_scores["f1"],
        "predicted_spans": len(pred_unlabeled),
        "predicted_class_spans": len(pred_class),
        "gold_spans": len(gold_unlabeled),
        "gold_class_spans": len(gold_class),
    }


def parse_threshold_grid(value):
    if value is None:
        return [round(index / 10, 1) for index in range(1, 10)]
    thresholds = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not thresholds:
        raise ValueError("Threshold grid is empty.")
    return thresholds


def epoch_selection_key(metrics):
    """Sort key deciding which epoch's weights are kept. Higher is better.

    The `strict_span_f1` component is not decoration. On an UNLABELED gold this
    module's dev metric is structurally 0.0 at every epoch -- even for perfect
    predictions -- because of an ASYMMETRY inside this file: `gold_span_sets` keys
    gold spans on the raw `span.get("class_label")` (None here) with no fallback,
    while the label vocabulary and the prediction path both go through
    `class_labels_from_units`'s `or [TERM_LABEL]`, so predictions key on "TERM".
    The two sets can never intersect. Before 2026-07-28 the selection was a bare
    `class_aware_f1 > best` scalar starting at -1.0, so `0.0 > -1.0` fired on epoch 1
    and `0.0 > 0.0` never fired again: every such run trained N epochs and shipped
    EPOCH-1 weights -- and, via `build_config()`'s `best_epoch_metrics`, epoch-1's
    decoding threshold too. On a labeled gold `class_aware_f1` varies normally and
    still decides, so this changes nothing there.

    Confirmed against production stdout logs of the affected runs: 25/25 seed-runs
    logged `dev_class_f1=0.000000` at every epoch while `dev_span_f1` climbed
    (e.g. CL `xlmr_span_unlabeled` 0.6438 -> 0.7486), and 35/35 labeled control runs
    were unaffected. Affected cells: `xlmr_span_unlabeled` on CL-RuTerm3, GENIA and
    ACL RD-TEC 2.0; `rubert_span_unlabeled` on CL-RuTerm3; `mbert_span` and
    `xlmr_span_classifier` on ACTER v1.5 (base panel + E4).

    NOTE on why BIO models were never affected -- it is NOT that
    train_mbert_sequence_labeler.py uses this tuple (it does, but that is not the
    reason). Its `predictions_from_logits` nulls `class_label` on BOTH sides when
    `label_mode != "class"` (`:366`), so its dev class-aware F1 simply EQUALS its
    strict-span F1 on an unlabeled gold (measured: 0.8064 -> 0.8649 across epochs,
    best epoch 5) and never degenerates. A BIO run would have survived the scalar
    rule unchanged. The tuple is the correct general guard; the asymmetry above is
    the actual defect this file had.
    """
    return (metrics["class_aware_f1"], metrics["strict_span_f1"], -metrics["threshold"])


def select_best_dev_threshold(scored_by_unit, units, thresholds, collapse_fn=None):
    results = [evaluate_scored_spans(scored_by_unit, units, threshold, collapse_fn=collapse_fn) for threshold in thresholds]
    best = sorted(
        results,
        key=lambda row: (-row["class_aware_f1"], -row["strict_span_f1"], row["threshold"]),
    )[0]
    return best, results


def snapshot_horizon_bundle(output_dir, cap):
    """Copy the on-disk best-so-far bundle to <output_dir>__h<cap>, replacing a stale
    copy. train_model calls it at the END of epoch `cap`, so the copy is best<=cap and
    main() can predict from it without re-training the shared early epochs.

    A plain module-level function (not inline in train_model) so it is unit-tested:
    the inline version shipped 2026-09-10 without `import os` and every 15-epoch
    Pilot A span job died with NameError at the epoch-5 snapshot (40 jobs, ~5
    trained epochs each), a path no test and no CPU smoke had ever executed.
    """
    snap = f"{output_dir}__h{cap}"
    if os.path.exists(snap):
        shutil.rmtree(snap)
    shutil.copytree(output_dir, snap)
    return snap


def train_model(args):
    tokenizer_load_kwargs = tokenizer_kwargs(
        args.model_name, getattr(args, "scibert_do_lower_case", None),
        revision=getattr(args, "model_revision", None),
    )
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    set_seed(args.seed)
    # Resolve the None argparse defaults to the historical training values here,
    # once, so build_span_features calls and build_config() below see concrete
    # numbers (and the saved config records what training actually used).
    if args.max_length is None:
        args.max_length = 256
    if args.stride is None:
        args.stride = 64
    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    assert_no_doc_id_overlap([("train", train_rows), ("dev", dev_rows)])
    train_units = group_rows_into_text_units(train_rows, dataset=args.dataset, include_spans=True)
    dev_units = group_rows_into_text_units(dev_rows, dataset=args.dataset, include_spans=True)
    if getattr(args, "label_list", None):
        class_labels = [c.strip() for c in args.label_list.split(",") if c.strip()]
    else:
        class_labels = class_labels_from_units(train_units)
    label2id, id2label = build_label_maps(class_labels)

    # Deliberately plain (no fix_mistral_regex) -- same note as predict_model's
    # loader above.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_load_kwargs)
    train_features, train_stats = build_span_features(
        train_units,
        tokenizer,
        label2id,
        max_span_tokens=args.max_span_tokens,
        negative_ratio=args.negative_ratio,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed,
        training=True,
        length_balanced_positive_sampling=args.length_balanced_positive_sampling,
        long_span_token_threshold=args.long_span_token_threshold,
        long_positive_weight=args.long_positive_weight,
    )
    dev_features, dev_stats = build_span_features(
        dev_units,
        tokenizer,
        label2id,
        max_span_tokens=args.max_span_tokens,
        negative_ratio=args.negative_ratio,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed + 1,
        training=True,
        length_balanced_positive_sampling=False,
        long_span_token_threshold=args.long_span_token_threshold,
        long_positive_weight=args.long_positive_weight,
    )
    dev_inference_features, dev_inference_stats = build_span_features(
        dev_units,
        tokenizer,
        label2id,
        max_span_tokens=args.max_span_tokens,
        negative_ratio=args.negative_ratio,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed + 2,
        training=False,
    )
    validate_training_preprocessing(train_features, train_stats, split_name="train")
    validate_training_preprocessing(dev_features, dev_stats, split_name="dev")
    warn_on_high_unaligned_spans(
        train_stats,
        split_name="train",
        threshold=args.unaligned_warning_threshold,
    )
    warn_on_high_unaligned_spans(
        dev_stats,
        split_name="dev",
        threshold=args.unaligned_warning_threshold,
    )
    warn_on_low_alignment(train_stats, split_name="train", threshold=args.alignment_warning_threshold)
    warn_on_low_alignment(dev_stats, split_name="dev", threshold=args.alignment_warning_threshold)
    device = choose_device(args.device)
    ModelClass = make_span_classifier_class()
    model = ModelClass(
        model_name=args.model_name,
        num_labels=len(label2id),
        max_span_tokens=args.max_span_tokens,
        width_embedding_dim=args.width_embedding_dim,
        model_revision=getattr(args, "model_revision", None),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    # torch.amp.GradScaler("cuda", ...) needs torch>=2.4; torch.cuda.amp.GradScaler
    # is a thin deprecated-but-functionally-identical wrapper around it (verified by
    # reading torch's own source), so this falls back cleanly on whatever torch
    # Kaggle actually has instead of assuming a version we haven't checked there.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        # AttributeError: torch.amp.GradScaler itself doesn't exist yet (torch<2.3,
        # where torch.amp only exports autocast) -- accessing the missing attribute
        # raises AttributeError, not TypeError, so both must be caught for the
        # fallback below to actually run instead of crashing before any training.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # file_system sharing defends against small /dev/shm on Kaggle-style
    # containers, where the default (file_descriptor) strategy can crash
    # multi-worker DataLoaders. persistent_workers only pays off here because
    # this loader is iterated across many epochs from this one object -- the
    # eval-loaders in evaluate_loss/scored_candidate_spans are freshly
    # constructed on every call, so persistent_workers would do nothing there.
    if device.type == "cuda":
        torch.multiprocessing.set_sharing_strategy("file_system")
    loader_kwargs = {"num_workers": 2, "pin_memory": True, "persistent_workers": True} \
        if device.type == "cuda" else {}
    loader = DataLoader(
        SpanFeatureDataset(train_features),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_span_features(batch, tokenizer, include_labels=True),
        **loader_kwargs,
    )
    # Opt-in (default 0.0 = current behavior, a flat LR): linear warmup for the
    # first warmup_ratio fraction of steps, then linear decay to 0 -- the
    # standard BERT fine-tuning schedule. See train_mbert_sequence_labeler.py's
    # train_model for the same mechanism and rationale.
    warmup_ratio = float(getattr(args, "warmup_ratio", 0.0) or 0.0)
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(
            f"--warmup-ratio must be in [0, 1], got {warmup_ratio}: a negative value would "
            f"silently train at a flat LR while the run summary still reported warmup as "
            f"configured, and a value above 1 never reaches the peak LR nor decays."
        )
    scheduler = None
    if warmup_ratio > 0:
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(loader) * args.epochs
        num_warmup_steps = round(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

    history = []
    threshold_grid = parse_threshold_grid(args.dev_thresholds)
    best_epoch_key = None  # see epoch_selection_key() for why this is a tuple, not a scalar
    best_state = None
    best_epoch_metrics = None
    _horizons = parse_select_horizons(getattr(args, "select_horizons", None), args.epochs)

    def build_config():
        return {
            "model_name": args.model_name,
            "model_revision": getattr(args, "model_revision", None),
            "scibert_do_lower_case": getattr(args, "scibert_do_lower_case", None),
            "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
            "label2id": label2id,
            "id2label": {str(key): value for key, value in id2label.items()},
            "class_labels": class_labels,
            "max_span_tokens": args.max_span_tokens,
            "width_embedding_dim": args.width_embedding_dim,
            "negative_ratio": args.negative_ratio,
            "length_balanced_positive_sampling": args.length_balanced_positive_sampling,
            "long_span_token_threshold": args.long_span_token_threshold,
            "long_positive_weight": args.long_positive_weight,
            "max_length": args.max_length,
            "stride": args.stride,
            "seed": args.seed,
            "log_every_n_batches": args.log_every_n_batches,
            "training_args": vars(args),
            "training_history": history,
            "selection_metric": selection_metric_for_labels(class_labels),
            "best_epoch": best_epoch_metrics,
            "best_threshold": best_epoch_metrics["best_threshold"] if best_epoch_metrics else 0.5,
            "dev_thresholds": threshold_grid,
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "dev_inference_stats": dev_inference_stats,
            # Full ACTUAL runtime config (not just requested CLI args) -- so a run is
            # reproducible from its own report and the "requested vs effective" gap
            # (the E1 warmup incident) can never recur silently.
            "runtime": {
                "optimizer": type(optimizer).__name__,
                "adam_betas": list(optimizer.defaults.get("betas", [])),
                "adam_eps": optimizer.defaults.get("eps"),
                "weight_decay_default": optimizer.defaults.get("weight_decay"),
                "scheduler": "linear_warmup_then_decay" if scheduler is not None else "constant_flat",
                "warmup_ratio_effective": float(getattr(args, "warmup_ratio", 0.0)),
                "num_training_steps": len(loader) * args.epochs,
                "num_warmup_steps": int(round(len(loader) * args.epochs * float(getattr(args, "warmup_ratio", 0.0)))),
                "amp": bool(getattr(args, "amp", False)),
                "max_grad_norm": float(getattr(args, "max_grad_norm", 0.0)),
                "batch_size": args.batch_size,
                "select_horizons": _horizons,
                "torch_version": __import__("torch").__version__,
                "transformers_version": __import__("transformers").__version__,
            },
        }

    log_every = max(0, getattr(args, "log_every_n_batches", 0) or 0)
    total_batches = len(loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Accumulate loss on-device and only sync to host at log points / epoch
        # end (was: float(...detach().cpu()) every batch -- a full device sync
        # per step that serializes the training loop).
        epoch_loss_sum = torch.zeros((), device=device)
        batch_count = 0
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(**batch)
                loss = output["loss"]
            loss_detached = loss.detach()
            epoch_loss_sum += loss_detached
            batch_count += 1
            if use_amp:
                scaler.scale(loss).backward()
                # A GradScaler-skipped step (inf/nan grads) lowers the scale and takes
                # no optimizer step -- the LR schedule must not advance for it. Same
                # guard as train_mbert_sequence_labeler.py's train_model.
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            if log_every and (batch_index % log_every == 0 or batch_index == total_batches):
                batch_loss = float(loss_detached)
                running_loss = float(epoch_loss_sum) / batch_count if batch_count else float("nan")
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_index}/{total_batches} "
                    f"batch_loss={batch_loss:.6f} running_loss={running_loss:.6f}",
                    flush=True,
                )
        train_loss = float(epoch_loss_sum) / batch_count if batch_count else math.inf
        dev_loss = evaluate_loss(model, dev_features, tokenizer, args.batch_size, device)
        dev_scored = scored_candidate_spans(model, dev_inference_features, tokenizer, args.batch_size, device, id2label)
        best_threshold_metrics, threshold_results = select_best_dev_threshold(
            dev_scored, dev_units, threshold_grid,
            collapse_fn=_dev_collapse_fn(getattr(args, "dev_eval_collapse", None)))
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "best_threshold": best_threshold_metrics["threshold"],
            "dev_strict_span_f1": best_threshold_metrics["strict_span_f1"],
            "dev_class_aware_f1": best_threshold_metrics["class_aware_f1"],
            "dev_threshold_results": threshold_results,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} dev_loss={dev_loss:.6f} "
            f"dev_class_f1={best_threshold_metrics['class_aware_f1']:.6f} "
            f"dev_span_f1={best_threshold_metrics['strict_span_f1']:.6f} "
            f"threshold={best_threshold_metrics['threshold']:.3f}",
            flush=True,
        )
        epoch_key = epoch_selection_key(best_threshold_metrics)
        if best_epoch_key is None or epoch_key > best_epoch_key:
            best_epoch_key = epoch_key
            best_epoch_metrics = dict(epoch_record)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            # Persist the best-so-far checkpoint to disk on every improvement, same
            # as train_mbert_sequence_labeler.py's train_model. Kaggle sessions have
            # a 12h cap and long GPU runs are occasionally killed outright; without
            # this the in-memory best_state is lost and the whole seed must be
            # retrained. The model's current weights ARE best_state at this point
            # (we just captured them), so no reload is needed before saving. This
            # does not change training results: weights, optimizer, and seed are
            # untouched.
            save_model_bundle(model, tokenizer, args.output_dir, build_config())

        # Horizon snapshot: at the END of an internal cap, copy the best-so-far
        # (best<=this epoch) checkpoint that is now on disk in args.output_dir, so
        # main() can predict best<=cap later without re-training the shared early
        # epochs. Training continues from the in-memory model (this epoch's weights),
        # unaffected by the copy. cap == args.epochs needs no snapshot (that IS the
        # final args.output_dir). Deterministic: no weights/optimizer/seed touched.
        for cap in _horizons:
            if epoch == cap and cap < args.epochs:
                snap = snapshot_horizon_bundle(args.output_dir, cap)
                print(f"horizon snapshot: best<=epoch{cap} -> {snap}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    config = build_config()
    save_model_bundle(model, tokenizer, args.output_dir, config)
    write_training_report(config, args.report)
    return args.output_dir


def prediction_output_mode(unit, output_format):
    if output_format != "auto":
        return output_format
    return "unified" if unit.get("input_format") == "unified" or unit.get("dataset") else "label_list"


def prediction_rows_from_scored_spans(unit, spans, output_format="auto"):
    mode = prediction_output_mode(unit, output_format)
    text = unit["text"]
    spans = sorted(spans, key=lambda span: (span["start"], span["end"], str(span.get("class_label"))))
    if mode == "label_list":
        return [{
            "id": unit.get("id"),
            "text": text,
            "label": [
                {
                    "start": span["start"],
                    "end": span["end"],
                    "surface": text[span["start"]:span["end"]],
                    "class_label": span.get("class_label"),
                    "score": span.get("score"),
                }
                for span in spans
            ],
        }]

    rows = []
    for span in spans:
        surface = text[span["start"]:span["end"]]
        row = {
            "dataset": unit.get("dataset"),
            "version": unit.get("version"),
            "split": unit.get("split"),
            "language": unit.get("language"),
            "domain": unit.get("domain"),
            "doc_id": unit.get("doc_id"),
            "sent_id": unit.get("sent_id"),
            "text": text,
            "unit_type": "span_mention",
            "surface": surface,
            "norm": normalize_surface(surface),
            "start": span["start"],
            "end": span["end"],
            "class_label": span.get("class_label"),
            "score": span.get("score"),
        }
        if row.get("dataset") == "CL-RuTerm3":
            row["is_nomen"] = row.get("class_label") == "Nomen"
        rows.append(row)
    return rows


def deduplicate_prediction_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        if "label" in row:
            key = ("label_list", row.get("id"), json.dumps(row.get("label", []), sort_keys=True, ensure_ascii=False))
        else:
            key = (
                row.get("dataset"),
                row.get("doc_id"),
                row.get("sent_id"),
                row.get("start"),
                row.get("end"),
                row.get("class_label"),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def predict_model(args, append_report=False, report_label=None):
    """Predict with the bundle in args.model_dir and write args.output.

    append_report/report_label: when main() trains AND predicts in one invocation,
    the prediction summary is APPENDED to the training report as a
    "## Prediction (<label>)" section instead of overwriting it -- before
    2026-09-11 the second write_prediction_report call replaced the whole training
    report (best epoch, per-epoch dev history, effective warmup steps, versions),
    so every committed span `_report.md` is a 9-line prediction report and the
    training provenance survived only in the unpersisted model dir. Predict-only
    invocations (no --train) keep writing the standalone prediction report.
    """
    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer, config = load_model_bundle(
        args.model_dir, device=device,
        scibert_do_lower_case=getattr(args, "scibert_do_lower_case", None),
    )
    id2label = {int(key): value for key, value in config["id2label"].items()}
    label2id = {label: int(index) for label, index in config["label2id"].items()}
    rows = read_jsonl(args.predict)
    units = group_rows_into_text_units(rows, dataset=args.dataset, include_spans=False)
    if not units:
        raise ValueError(
            "No prediction text units found. Check --dataset and the input unit_type values; "
            "prediction accepts rows with text, but skips non-span task rows such as "
            "term_string, sentence_term_string, and variant_cluster_member."
        )
    # CLI wins only when EXPLICITLY given (argparse default is None); otherwise
    # the model's own training values apply, so prediction windows match the
    # windows the model was trained on. An explicit mismatch is allowed but
    # loudly flagged -- mismatched windowing silently degrades long-document
    # predictions with no error.
    max_length = args.max_length if args.max_length is not None else config.get("max_length", 256)
    stride = args.stride if args.stride is not None else config.get("stride", 64)
    for name, cli_value, trained_value in (
        ("max_length", args.max_length, config.get("max_length")),
        ("stride", args.stride, config.get("stride")),
    ):
        if cli_value is not None and trained_value is not None and cli_value != trained_value:
            print(
                f"WARNING: --{name} {cli_value} differs from the model's training "
                f"{name} {trained_value}; prediction windows will not match training.",
                flush=True,
            )
    features, predict_stats = build_span_features(
        units,
        tokenizer,
        label2id,
        max_span_tokens=config.get("max_span_tokens", args.max_span_tokens),
        negative_ratio=args.negative_ratio,
        max_length=max_length,
        stride=stride,
        seed=args.seed,
        training=False,
    )
    threshold = args.threshold if args.threshold is not None else config.get("best_threshold", 0.5)
    by_unit = scored_candidate_spans(model, features, tokenizer, args.batch_size, device, id2label)

    output_rows = []
    for unit in units:
        spans = deduplicate_scored_spans(by_unit.get(id(unit), []), class_aware=True)
        spans = [span for span in spans if span.get("score", 0.0) >= threshold]
        if args.top_k_per_doc is not None:
            spans = sorted(spans, key=lambda span: (-span["score"], span["start"], span["end"]))[:args.top_k_per_doc]
        output_rows.extend(prediction_rows_from_scored_spans(unit, spans, output_format=args.output_format))
    output_rows = deduplicate_prediction_rows(output_rows)
    write_jsonl(output_rows, args.output)
    # same marker the BIO trainers emit, so per-seed progress is greppable
    print(f"Saved predictions: {args.output}", flush=True)
    if args.report:
        report = {
            "model_dir": str(args.model_dir),
            "predict_file": str(args.predict),
            "output": str(args.output),
            "threshold": threshold,
            "top_k_per_doc": args.top_k_per_doc,
            "prediction_rows_written": len(output_rows),
            "predict_stats": predict_stats,
        }
        write_prediction_report(report, args.report, append=append_report, label=report_label)
    return output_rows


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def deduplicate_scored_spans(spans, class_aware=True):
    best = {}
    for span in spans:
        key = (span["start"], span["end"], span.get("class_label") if class_aware else None)
        if key not in best or span.get("score", 0) > best[key].get("score", 0):
            best[key] = span
    return sorted(best.values(), key=lambda item: (item["start"], item["end"], str(item.get("class_label"))))


def write_training_report(config, report_path):
    if not report_path:
        return
    training_args = config.get("training_args") or {}
    lines = [
        "# XLM-R Span Classifier Report",
        "",
        f"- model_name: `{config['model_name']}`",
        f"- model_revision: `{config.get('model_revision')}`",
        f"- scibert_do_lower_case: `{config.get('scibert_do_lower_case')}`",
        f"- tokenizer_is_fast: `{config.get('tokenizer_is_fast')}`",
        f"- tokenizer_do_lower_case: `{config.get('tokenizer_do_lower_case')}`",
        f"- labels: `{', '.join(config['class_labels'])}`",
        f"- max_span_tokens: `{config['max_span_tokens']}`",
        f"- negative_ratio: `{config['negative_ratio']}`",
        f"- length_balanced_positive_sampling: `{config.get('length_balanced_positive_sampling', False)}`",
        f"- long_span_token_threshold: `{config.get('long_span_token_threshold', 4)}`",
        f"- long_positive_weight: `{config.get('long_positive_weight', 1)}`",
        f"- max_length: `{config['max_length']}`",
        f"- stride: `{config['stride']}`",
        # Optimization hyperparameters the report never carried before 2026-09-11:
        # without them a span report could not tell a flat-LR run from a warmup
        # run (the E1 incident), nor which seed / epoch budget produced it.
        f"- seed: `{config.get('seed', training_args.get('seed'))}`",
        f"- epochs: `{training_args.get('epochs')}`",
        f"- batch_size: `{training_args.get('batch_size')}`",
        f"- learning_rate: `{training_args.get('learning_rate')}`",
        f"- warmup_ratio: `{training_args.get('warmup_ratio', 0.0)}`",
        f"- selection_metric: `{config.get('selection_metric', 'dev_class_aware_f1')}`",
        f"- best_threshold: `{config.get('best_threshold', 0.5)}`",
        f"- best_epoch: `{config.get('best_epoch', {}).get('epoch') if config.get('best_epoch') else 'unknown'}`",
        f"- log_every_n_batches: `{config.get('log_every_n_batches', 0)}`",
        "",
        "## Train Stats",
        "",
    ]
    for key, value in config.get("train_stats", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Dev Stats", ""])
    for key, value in config.get("dev_stats", {}).items():
        lines.append(f"- {key}: `{value}`")
    # ACTUAL runtime config (optimizer betas/eps, scheduler, effective warmup
    # steps, torch/transformers versions) -- what build_config() records, so the
    # "requested vs effective" gap can be read off the report itself.
    lines.extend(["", "## Runtime", ""])
    runtime = config.get("runtime") or {}
    if runtime:
        for key, value in runtime.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- none")
    warnings = []
    warnings.extend(config.get("train_stats", {}).get("warnings", []))
    warnings.extend(config.get("dev_stats", {}).get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.extend(["", "## Loss", ""])
    for row in config.get("training_history", []):
        lines.append(
            f"- epoch {row['epoch']}: train_loss={row['train_loss']:.6f}, "
            f"dev_loss={row['dev_loss']:.6f}, "
            f"dev_class_aware_f1={row.get('dev_class_aware_f1', 0.0):.6f}, "
            f"dev_strict_span_f1={row.get('dev_strict_span_f1', 0.0):.6f}, "
            f"threshold={row.get('best_threshold', 0.5):.3f}"
        )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_training_preprocessing(features, stats, split_name="train"):
    if not features:
        raise ValueError(f"No {split_name} features after preprocessing. Check dataset filter and input text units.")
    if stats.get("positive_training_spans", 0) == 0:
        raise ValueError(
            "No positive training spans after preprocessing. Check class labels, offsets, "
            "tokenizer alignment, and dataset filter."
        )


def alignment_ratio(stats):
    gold_spans = stats.get("gold_spans", 0)
    if not gold_spans:
        return 1.0
    return stats.get("aligned_gold_spans", 0) / gold_spans


def unaligned_gold_span_rate(stats):
    gold_spans = stats.get("gold_spans", 0)
    if not gold_spans:
        return 0.0
    return stats.get("unaligned_skipped_gold_spans", 0) / gold_spans


def boundary_misaligned_gold_span_rate(stats):
    """The genuine tokenizer-boundary-miss rate, with window-truncation misses
    (a stride/max_length artifact, not a property of the encoder) excluded --
    this is the number that should drive an encoder accept/reject decision,
    not the raw unaligned_gold_span_rate above."""
    gold_spans = stats.get("gold_spans", 0)
    if not gold_spans:
        return 0.0
    return stats.get("boundary_misaligned_gold_spans", 0) / gold_spans


def warn_on_high_unaligned_spans(stats, split_name="train", threshold=0.05):
    rate = unaligned_gold_span_rate(stats)
    stats["unaligned_gold_span_rate"] = rate
    if stats.get("gold_spans", 0) and rate > threshold:
        window_truncated = stats.get("window_truncated_gold_spans", 0)
        boundary_misaligned = stats.get("boundary_misaligned_gold_spans", 0)
        warning = (
            f"WARNING: high unaligned gold span rate for {split_name}: {rate:.3f} "
            f"({stats.get('unaligned_skipped_gold_spans', 0)}/{stats.get('gold_spans', 0)} gold spans). "
            f"Of these, {window_truncated} never fit inside any single stride/max_length "
            f"window (a windowing-config artifact, not a property of this encoder's tokenizer) and "
            f"{boundary_misaligned} were inside a window but did not match a generated candidate's exact "
            "boundaries (a genuine tokenizer-alignment miss). CL-RuTerm3 and other nested span datasets "
            "may be underestimated by the boundary-misaligned portion."
        )
        stats.setdefault("warnings", []).append(warning)
        print(warning, flush=True)


def warn_on_low_alignment(stats, split_name="train", threshold=0.9):
    ratio = alignment_ratio(stats)
    stats["alignment_ratio"] = ratio
    if ratio < threshold:
        warning = (
            f"WARNING: {split_name} aligned_gold_spans/gold_spans = {ratio:.3f}, "
            f"below {threshold:.3f}. Check tokenizer boundary alignment."
        )
        stats.setdefault("warnings", []).append(warning)
        print(warning, flush=True)


def write_prediction_report(report, report_path, append=False, label=None):
    """Standalone prediction report (default), or -- with append=True and an
    existing report at report_path -- a "## Prediction (<label>)" section appended
    to it. main() uses the appending form when one invocation both trains and
    predicts, so the training report (and, with --select-horizons, one section per
    horizon) is never overwritten. See predict_model's docstring."""
    path = Path(report_path)
    if append and path.exists():
        title = f"## Prediction ({label})" if label else "## Prediction"
        lines = ["", title, ""]
        for key, value in report.items():
            lines.append(f"- {key}: `{value}`")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return
    lines = ["# XLM-R Span Classifier Prediction Report", ""]
    for key, value in report.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train or run a nested-aware XLM-R span classifier baseline.")
    parser.add_argument("--train", help="Training gold JSONL with span annotations.")
    parser.add_argument("--dev", help="Development gold JSONL with span annotations.")
    parser.add_argument("--predict", help="Text-unit or gold JSONL to predict over.")
    parser.add_argument("--output_dir", "--output-dir", help="Directory for trained model bundle.")
    parser.add_argument("--model_dir", "--model-dir", help="Directory containing a trained model bundle.")
    parser.add_argument("--output", help="Prediction JSONL path.")
    parser.add_argument("--report", default="build/experiments/supervised/reports/xlmr_span_classifier_report.md")
    parser.add_argument("--dataset", help="Optional dataset filter, e.g. CL-RuTerm3 or ACL RD-TEC 2.0.")
    parser.add_argument("--label-list", default=None,
                        help="Comma-separated FIXED class label vocabulary, overriding "
                             "class_labels_from_units's default (observed from --train "
                             "only). Required whenever a training arm can legitimately "
                             "remove every example of one class (e.g. an NE-inclusion "
                             "contrast): without this, that class silently vanishes from "
                             "THAT arm's softmax and other arms keep it, handing different "
                             "arms different label-space widths -- the E1/E6 confound. "
                             "Every other call site omits this and keeps prior behavior.")
    parser.add_argument("--model_name", "--model-name", default="xlm-roberta-base")
    parser.add_argument("--max_span_tokens", "--max-span-tokens", type=int, default=10)
    parser.add_argument("--negative_ratio", "--negative-ratio", type=int, default=5)
    parser.add_argument("--length-balanced-positive-sampling", action="store_true")
    parser.add_argument("--long-span-token-threshold", type=int, default=4)
    parser.add_argument("--long-positive-weight", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    # Default None, NOT a number: predict_model falls back to the trained
    # model's own config precisely when the flag is absent. A numeric default
    # here made "not given" indistinguishable from "given 256/64", so the
    # config fallback was dead code and a model trained at a different length
    # would silently predict under mismatched windowing. Training resolves
    # None to the historical 256/64 in train_model, so every existing training
    # command behaves identically.
    parser.add_argument("--max_length", "--max-length", type=int, default=None,
                        help="Tokenizer window length. Training default: 256. "
                             "Prediction default: the model's own training value.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Overflow-window stride. Training default: 64. "
                             "Prediction default: the model's own training value.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--dev-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dev-eval-collapse", default=None,
                        help="collapse dev class labels for threshold/best-epoch "
                             "selection only (e.g. genia_coarse6 for E1 arm A)")
    parser.add_argument("--top_k_per_doc", "--top-k-per-doc", type=int)
    parser.add_argument("--width_embedding_dim", "--width-embedding-dim", type=int, default=32)
    parser.add_argument("--alignment-warning-threshold", type=float, default=0.9)
    parser.add_argument("--unaligned-warning-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true",
                         help="Mixed-precision training (torch.cuda.amp). No-op off CUDA.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                         help="Fraction of total steps for linear LR warmup, then linear "
                              "decay to 0 (standard BERT fine-tuning schedule). Default 0.0 (flat LR, no warmup). Was 0.1 (10 percent, the standard BERT fine-tuning ratio) from 2026-07-27 to 2026-07-28, reverted after concluding the per-entry-point bookkeeping this required (tracking which runs/dates used which default) cost more than a literature value never validated on this project's own data was worth; every already-published result was flat-LR anyway.")
    parser.add_argument("--output-format", choices=["auto", "unified", "label_list"], default="auto")
    parser.add_argument("--log-every-n-batches", type=int, default=50)
    parser.add_argument("--select-horizons", default=None,
                        help="Comma-separated epoch caps (e.g. '5,10,15'). When set, the "
                             "best checkpoint WITHIN each cap is snapshotted during training "
                             "and used to write an additional prediction file suffixed _h<cap>. "
                             "Lets one long run yield best<=5 and best<=15 without re-training "
                             "the shared early epochs. The largest cap should equal --epochs; "
                             "the un-suffixed --output stays best<=epochs. Default off (no change).")
    add_pretrained_arguments(parser)
    return parser


def parse_select_horizons(value, epochs):
    if not value:
        return []
    caps = sorted({int(x) for x in str(value).split(",") if x.strip()})
    bad = [c for c in caps if c < 1 or c > epochs]
    if bad:
        raise ValueError(f"--select-horizons values must be in [1, {epochs}]: got {bad}")
    return caps


def _with_h_suffix(path, cap):
    p = str(path)
    if p.endswith(".jsonl"):
        return p[: -len(".jsonl")] + f"_h{cap}.jsonl"
    return p + f"_h{cap}"


def validate_args(args):
    if args.train or args.dev:
        if not args.train or not args.dev or not args.output_dir:
            raise ValueError("Training requires --train, --dev, and --output_dir.")
        if args.long_span_token_threshold < 1:
            raise ValueError("--long-span-token-threshold must be >= 1.")
        if args.long_positive_weight < 1:
            raise ValueError("--long-positive-weight must be >= 1.")
    if args.predict:
        if not args.model_dir:
            model_dir = args.output_dir
            if not model_dir:
                raise ValueError("Prediction requires --model_dir or a just-trained --output_dir.")
            args.model_dir = model_dir
        if not args.output:
            raise ValueError("Prediction requires --output.")
    if not args.train and not args.predict:
        raise ValueError("Provide --train/--dev for training and/or --predict for inference.")
    return args


def main(argv=None):
    parser = build_arg_parser()
    args = validate_args(parser.parse_args(argv))
    if args.train:
        train_model(args)
        if not args.model_dir:
            args.model_dir = args.output_dir
    if args.predict:
        # Trained in this same invocation -> the training report already sits at
        # args.report; append the prediction summaries to it instead of overwriting
        # (see predict_model). Predict-only keeps the standalone prediction report.
        trained_here = bool(args.train)
        predict_model(  # best<=epochs -> args.output (un-suffixed; unchanged default)
            args, append_report=trained_here,
            report_label=f"best<={args.epochs}, {args.output}" if trained_here else None)
        # Horizon reuse: predict the best<=cap snapshots too, so one long run yields
        # best<=5 and best<=15 without re-training the shared early epochs. cap ==
        # epochs reuses the final bundle; smaller caps use the __h<cap> snapshots
        # train_model saved. best<=epochs is also written to _h<epochs> for a
        # symmetric analysis file set (equals the un-suffixed output).
        _hz = parse_select_horizons(getattr(args, "select_horizons", None), args.epochs)
        if _hz:
            _base_out, _base_mdir = args.output, args.model_dir
            for cap in _hz:
                args.model_dir = _base_mdir if cap >= args.epochs else f"{args.output_dir}__h{cap}"
                args.output = _with_h_suffix(_base_out, cap)
                predict_model(args, append_report=trained_here,
                              report_label=f"best<={cap}, {args.output}" if trained_here else None)
            args.output, args.model_dir = _base_out, _base_mdir


if __name__ == "__main__":
    main()
