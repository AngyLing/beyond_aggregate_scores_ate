#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from ate_paper.experiments.supervised_tokenizers import (
    add_pretrained_arguments, pretrained_revision_kwargs, tokenizer_kwargs,
)


ACL_CLASSES = [
    "Technology and Method",
    "Tool and Library",
    "Language Resource",
    "Language Resource Product",
    "Model",
    "Measures and Measurements",
    "Other",
]
CL_RUTERM3_CLASSES = ["Specific", "Common", "Nomen"]
SPAN_DATASETS = {"ACL RD-TEC 2.0", "CL-RuTerm3", "GENIA"}
NON_SPAN_TEXT_UNIT_TYPES = {
    "term_string",
    "candidate_string",
    "sentence_term_string",
    "variant_cluster_member",
}


def normalize_surface(text):
    # Lowercase like every span gold's norm and the LLM writer's normalize_lower:
    # a case-preserving norm can never match the lowercased gold in norm-keyed
    # metrics (type_f1, MentionTypeGap). Sole call site is the
    # prediction row's "norm" field.
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

    The annotation-shift study compares baselines across evaluation contracts,
    so train/dev/test must stay disjoint at the document level. The splits are
    built by partitioning documents, so overlap can only come from a data-prep
    error (a copied file, a wrong path). Catch it here instead of silently
    reporting leaked scores.
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


def group_annotations_into_text_units(rows, dataset=None, include_spans=True):
    grouped = {}

    for row in rows:
        if dataset and row.get("dataset") != dataset:
            continue
        unit_type = row.get("unit_type")
        if include_spans and unit_type not in {None, "span_mention"}:
            continue
        if not include_spans and unit_type in NON_SPAN_TEXT_UNIT_TYPES:
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
        if key not in grouped:
            grouped[key] = {
                "dataset": row.get("dataset"),
                "version": row.get("version"),
                "split": row.get("split"),
                "language": row.get("language"),
                "domain": row.get("domain"),
                "doc_id": row.get("doc_id"),
                "sent_id": row.get("sent_id"),
                "text": row.get("text"),
                "spans": [],
            }

        if include_spans and row.get("unit_type") == "span_mention":
            start = row.get("start")
            end = row.get("end")
            if isinstance(start, int) and isinstance(end, int) and start < end:
                grouped[key]["spans"].append({
                    "start": start,
                    "end": end,
                    "class_label": row.get("class_label"),
                    "surface": row.get("surface"),
                    "norm": row.get("norm"),
                })

    return list(grouped.values())


def test_text_units_from_rows(rows, dataset):
    annotation_like = any(
        row.get("dataset") == dataset
        and row.get("unit_type") == "span_mention"
        and (
            row.get("start") is not None
            or row.get("end") is not None
            or row.get("class_label") not in {None, ""}
            or row.get("surface") not in {None, ""}
            or row.get("norm") not in {None, ""}
        )
        for row in rows
    )
    units = group_annotations_into_text_units(rows, dataset=dataset, include_spans=False)
    warnings = []
    if annotation_like and units:
        warnings.append(
            "WARNING: test text units were reconstructed from gold annotations; "
            "text units without gold terms may be missing, so false positives in "
            "empty-gold units cannot be produced and precision may be inflated. "
            "Use complete test text-unit files for fair evaluation."
        )
    return units, warnings


def labels_for_dataset(dataset, label_mode, train_units=None, dev_units=None):
    if label_mode == "unlabeled":
        classes = ["TERM"]
    elif dataset == "ACL RD-TEC 2.0":
        classes = list(ACL_CLASSES)
    elif dataset == "CL-RuTerm3":
        classes = list(CL_RUTERM3_CLASSES)
    else:
        observed = set()
        for unit in (train_units or []) + (dev_units or []):
            for span in unit.get("spans", []):
                if span.get("class_label"):
                    observed.add(span.get("class_label"))
        classes = sorted(observed) or ["TERM"]

    labels = ["O"]
    for label in classes:
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")
    return labels


def sort_spans_for_flattening(spans, flatten_policy="longest"):
    if flatten_policy != "longest":
        raise ValueError(f"Unsupported flatten policy: {flatten_policy}")
    return sorted(
        spans,
        key=lambda span: (-(span["end"] - span["start"]), span["start"], -span["end"]),
    )


def span_token_indices(span, offsets):
    indices = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start is None or token_end is None:
            continue
        if token_start == token_end:
            continue
        if token_start >= span["start"] and token_end <= span["end"]:
            indices.append(index)
    return indices


def align_labels_to_offsets(
    spans,
    offsets,
    label_to_id,
    label_mode="class",
    flatten_policy="longest",
    length_balanced_positive_sampling=False,
    long_span_token_threshold=4,
    long_positive_weight=3,
):
    label_ids = [label_to_id["O"] for _ in offsets]
    label_weights = [1.0 for _ in offsets]
    skipped_due_to_overlap = 0
    # skipped_due_to_alignment is kept as the TOTAL of the four causes below (same
    # name/value as before, so existing reports citing it are unaffected). The four
    # causes used to be indistinguishable under one counter, which conflates a
    # windowing-config artifact (skipped_outside_window: this span's document was
    # split by stride/max_length and this span didn't fit any single window -- not a
    # property of the encoder being evaluated) with a genuine tokenizer-boundary miss
    # (skipped_no_token_indices: the span WAS inside a window but no token boundary
    # matches its offsets) and with two data/inventory issues that aren't "alignment"
    # at all (skipped_missing_label, skipped_label_not_in_inventory).
    skipped_outside_window = 0
    skipped_no_token_indices = 0
    skipped_missing_label = 0
    skipped_label_not_in_inventory = 0
    length_balanced_long_spans = 0
    length_balanced_weighted_tokens = 0
    real_offsets = [
        (token_start, token_end)
        for token_start, token_end in offsets
        if token_start is not None and token_end is not None and token_start != token_end
    ]
    window_start = min((start for start, _ in real_offsets), default=None)
    window_end = max((end for _, end in real_offsets), default=None)

    for span in sort_spans_for_flattening(spans, flatten_policy):
        if window_start is None or window_end is None or span["start"] < window_start or span["end"] > window_end:
            skipped_outside_window += 1
            continue

        token_indices = span_token_indices(span, offsets)
        if not token_indices:
            skipped_no_token_indices += 1
            continue
        if any(label_ids[index] != label_to_id["O"] for index in token_indices):
            skipped_due_to_overlap += 1
            continue

        label = "TERM" if label_mode == "unlabeled" else span.get("class_label")
        if not label:
            skipped_missing_label += 1
            continue

        b_label = f"B-{label}"
        i_label = f"I-{label}"
        if b_label not in label_to_id or i_label not in label_to_id:
            skipped_label_not_in_inventory += 1
            continue

        for position, token_index in enumerate(token_indices):
            label_ids[token_index] = label_to_id[b_label if position == 0 else i_label]
            if length_balanced_positive_sampling and len(token_indices) >= long_span_token_threshold:
                label_weights[token_index] = float(long_positive_weight)
        if length_balanced_positive_sampling and len(token_indices) >= long_span_token_threshold:
            length_balanced_long_spans += 1
            length_balanced_weighted_tokens += len(token_indices)

    skipped_due_to_alignment = (
        skipped_outside_window + skipped_no_token_indices
        + skipped_missing_label + skipped_label_not_in_inventory
    )
    return label_ids, {
        "label_weights": label_weights,
        "skipped_due_to_overlap": skipped_due_to_overlap,
        "skipped_due_to_alignment": skipped_due_to_alignment,
        "skipped_outside_window": skipped_outside_window,
        "skipped_no_token_indices": skipped_no_token_indices,
        "skipped_missing_label": skipped_missing_label,
        "skipped_label_not_in_inventory": skipped_label_not_in_inventory,
        "length_balanced_long_spans": length_balanced_long_spans,
        "length_balanced_weighted_tokens": length_balanced_weighted_tokens,
    }


def decode_bio_to_offsets(labels, offsets, label_mode="class"):
    spans = []
    current = None

    def close_current():
        nonlocal current
        if current is not None:
            spans.append(current)
            current = None

    for label, (token_start, token_end) in zip(labels, offsets):
        if token_start is None or token_end is None or token_start == token_end:
            close_current()
            continue

        if label == "O":
            close_current()
            continue

        if "-" not in label:
            close_current()
            continue

        prefix, value = label.split("-", 1)
        class_label = None if label_mode == "unlabeled" else value

        if prefix == "B":
            close_current()
            current = {"start": token_start, "end": token_end, "class_label": class_label, "label": value}
        elif prefix == "I":
            if current is None or current["label"] != value:
                close_current()
                current = {"start": token_start, "end": token_end, "class_label": class_label, "label": value}
            else:
                current["end"] = token_end
        else:
            close_current()

    close_current()

    for span in spans:
        span.pop("label", None)
    return spans


def deduplicate_predictions(rows, label_mode="class"):
    seen = set()
    result = []
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("start"),
            row.get("end"),
        )
        if label_mode == "class":
            key += (row.get("class_label"),)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def prediction_rows_from_decoded_spans(unit, spans, label_mode="class"):
    rows = []
    text = unit["text"]
    for span in spans:
        start = span["start"]
        end = span["end"]
        surface = text[start:end]
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
            "start": start,
            "end": end,
            "class_label": span.get("class_label") if label_mode == "class" else None,
        }
        if unit.get("dataset") == "CL-RuTerm3":
            row["is_nomen"] = row.get("class_label") == "Nomen"
        rows.append(row)
    return rows


def validate_prediction_schema(rows):
    errors = []
    for row in rows:
        required = ["dataset", "doc_id", "text", "unit_type", "surface", "norm", "start", "end"]
        missing = [field for field in required if field not in row]
        if missing:
            errors.append(f"Missing fields {missing}: {row}")
            continue
        text = row.get("text")
        start = row.get("start")
        end = row.get("end")
        if row.get("unit_type") != "span_mention":
            errors.append(f"Prediction is not span_mention: {row}")
        elif not isinstance(text, str) or not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"Invalid span offset types: {row}")
        elif start < 0 or end > len(text) or start >= end or text[start:end] != row.get("surface"):
            errors.append(f"Offset mismatch: {row}")

    if errors:
        raise ValueError("\n".join(errors))


def validate_predict_only(model_dir, predict_only, threshold=None):
    if predict_only:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise ValueError("--predict-only requires --model-dir to contain a saved HuggingFace model.")
        if threshold is None:
            raise ValueError(
                "--predict-only requires an explicit --threshold: the dev-tuned "
                "decision threshold is not stored in the model directory, so "
                "decoding would silently fall back to argmax -- a different "
                "decode than the training run used. Read the threshold from the "
                "training report and pass it via --threshold.")


def set_random_seeds(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_torch_device(requested="auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TokenClassificationDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        item = dict(self.features[index]["model_inputs"])
        if "labels" in self.features[index]:
            item["labels"] = self.features[index]["labels"]
        return item


def tokenize_units(
    tokenizer,
    units,
    label_to_id=None,
    label_mode="class",
    flatten_policy="longest",
    max_length=256,
    stride=64,
    length_balanced_positive_sampling=False,
    long_span_token_threshold=4,
    long_positive_weight=3,
):
    texts = [unit["text"] for unit in units]
    tokenized = tokenizer(
        texts,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    features = []
    skipped_due_to_overlap = 0
    skipped_due_to_alignment = 0
    skipped_outside_window = 0
    skipped_no_token_indices = 0
    skipped_missing_label = 0
    skipped_label_not_in_inventory = 0
    length_balanced_long_spans = 0
    length_balanced_weighted_tokens = 0
    overflow_texts = Counter(tokenized.get("overflow_to_sample_mapping", []))
    texts_split_into_overflow_windows = sum(1 for count in overflow_texts.values() if count > 1)

    for feature_index, offsets in enumerate(tokenized["offset_mapping"]):
        sample_index = tokenized["overflow_to_sample_mapping"][feature_index]
        unit = units[sample_index]
        model_inputs = {}
        for key, values in tokenized.items():
            if key in {"offset_mapping", "overflow_to_sample_mapping"}:
                continue
            model_inputs[key] = values[feature_index]

        feature = {
            "model_inputs": model_inputs,
            "offset_mapping": offsets,
            "unit": unit,
            "sample_index": sample_index,
        }

        if label_to_id is not None:
            labels, stats = align_labels_to_offsets(
                unit.get("spans", []),
                offsets,
                label_to_id,
                label_mode=label_mode,
                flatten_policy=flatten_policy,
                length_balanced_positive_sampling=length_balanced_positive_sampling,
                long_span_token_threshold=long_span_token_threshold,
                long_positive_weight=long_positive_weight,
            )
            label_weights = [
                0.0 if token_start == token_end else float(weight)
                for weight, (token_start, token_end) in zip(stats["label_weights"], offsets)
            ]
            labels = [
                -100 if token_start == token_end else label_id
                for label_id, (token_start, token_end) in zip(labels, offsets)
            ]
            feature["labels"] = labels
            if length_balanced_positive_sampling:
                feature["label_weights"] = label_weights
            skipped_due_to_overlap += stats["skipped_due_to_overlap"]
            skipped_due_to_alignment += stats["skipped_due_to_alignment"]
            skipped_outside_window += stats["skipped_outside_window"]
            skipped_no_token_indices += stats["skipped_no_token_indices"]
            skipped_missing_label += stats["skipped_missing_label"]
            skipped_label_not_in_inventory += stats["skipped_label_not_in_inventory"]
            length_balanced_long_spans += stats["length_balanced_long_spans"]
            length_balanced_weighted_tokens += stats["length_balanced_weighted_tokens"]

        features.append(feature)

    stats = {
        "skipped_due_to_overlap": skipped_due_to_overlap,
        "skipped_due_to_alignment": skipped_due_to_alignment,
        "skipped_outside_window": skipped_outside_window,
        "skipped_no_token_indices": skipped_no_token_indices,
        "skipped_missing_label": skipped_missing_label,
        "skipped_label_not_in_inventory": skipped_label_not_in_inventory,
        "texts_split_into_overflow_windows": texts_split_into_overflow_windows,
        "length_balanced_positive_sampling": bool(length_balanced_positive_sampling),
        "long_span_token_threshold": long_span_token_threshold,
        "long_positive_weight": long_positive_weight if length_balanced_positive_sampling else 1,
        "length_balanced_long_spans": length_balanced_long_spans,
        "length_balanced_weighted_tokens": length_balanced_weighted_tokens,
    }
    return features, stats


def decode_prediction_features(features, predictions, id_to_label, label_mode="class"):
    rows = []
    for feature, prediction_ids in zip(features, predictions):
        labels = [id_to_label[int(label_id)] for label_id in prediction_ids[:len(feature["offset_mapping"])]]
        spans = decode_bio_to_offsets(labels, feature["offset_mapping"], label_mode=label_mode)
        rows.extend(prediction_rows_from_decoded_spans(feature["unit"], spans, label_mode=label_mode))
    return deduplicate_predictions(rows, label_mode=label_mode)


def prediction_ids_from_logits(logits, id_to_label, threshold=None):
    if threshold is None:
        return logits.argmax(dim=-1).detach().cpu().tolist()

    import torch

    probs = torch.softmax(logits, dim=-1)
    o_ids = [index for index, label in id_to_label.items() if label == "O"]
    if not o_ids:
        raise ValueError("id_to_label must contain O for thresholded BIO decoding.")
    o_id = o_ids[0]
    non_o_ids = [index for index, label in sorted(id_to_label.items()) if label != "O"]
    if not non_o_ids:
        return torch.full(logits.shape[:2], o_id, dtype=torch.long).tolist()

    non_o_probs = probs[:, :, non_o_ids]
    best_probs, best_positions = torch.max(non_o_probs, dim=-1)
    non_o_tensor = torch.tensor(non_o_ids, device=logits.device)
    best_non_o_ids = non_o_tensor[best_positions]
    o_tensor = torch.full_like(best_non_o_ids, o_id)
    selected = torch.where(best_probs >= threshold, best_non_o_ids, o_tensor)
    return selected.detach().cpu().tolist()


def collate_features(features, tokenizer, include_labels=True):
    import torch

    model_inputs = [feature["model_inputs"] for feature in features]
    batch = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")

    if include_labels and "labels" in features[0]:
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        padded_weights = []
        for feature in features:
            labels = list(feature["labels"])
            labels = labels + [-100] * (max_length - len(labels))
            padded_labels.append(labels)
            if "label_weights" in feature:
                weights = list(feature["label_weights"])
                weights = weights + [0.0] * (max_length - len(weights))
                padded_weights.append(weights)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        if padded_weights:
            batch["label_weights"] = torch.tensor(padded_weights, dtype=torch.float)

    return batch


def weighted_token_classification_loss(logits, labels, label_weights):
    import torch

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    flat_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    flat_weights = label_weights.view(-1).to(flat_loss.device)
    active = labels.view(-1) != -100
    weighted_loss = flat_loss[active] * flat_weights[active]
    normalizer = flat_weights[active].sum().clamp(min=1.0)
    return weighted_loss.sum() / normalizer


def build_optimizer(model, args):
    """AdamW, optionally with the standard no-decay parameter groups.

    Defaults reproduce the historical behavior bit-for-bit: weight_decay 0.01
    (torch's own AdamW default, previously applied implicitly) on EVERY
    parameter, bias and LayerNorm included. --no-decay-bias-layernorm opts in
    to the HF-Trainer convention (bias and normalization weights excluded from
    decay) for the modern-recipe pilot; it is NOT what any published panel
    used, so leave it off when reproducing panel numbers.
    """
    import torch

    weight_decay = float(getattr(args, "weight_decay", 0.01))
    if not getattr(args, "no_decay_bias_layernorm", False):
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )


def train_model(model, tokenizer, train_features, dev_features, dev_units, id_to_label, args):
    import torch
    from torch.utils.data import DataLoader

    device = select_torch_device(args.device)
    model.to(device)
    model.train()

    # AMP is a no-op (enabled=False) off CUDA -- autocast/GradScaler are inert
    # context managers in that case, so this is safe on cpu/mps too.
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

    # num_workers/pin_memory only help when there's a GPU waiting on collation;
    # on CPU the main process is the only consumer anyway. file_system sharing
    # defends against small /dev/shm on Kaggle-style containers, where the
    # default (file_descriptor) strategy can crash multi-worker DataLoaders.
    # persistent_workers only pays off here because train_loader is iterated
    # across many epochs from this one object -- the eval-loaders below are
    # freshly constructed on every call, so persistent_workers would do nothing
    # for them (workers still die with the loader at function return).
    if device.type == "cuda":
        torch.multiprocessing.set_sharing_strategy("file_system")
    loader_kwargs = {"num_workers": 2, "pin_memory": True, "persistent_workers": True} \
        if device.type == "cuda" else {}
    train_loader = DataLoader(
        train_features,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_features(batch, tokenizer, include_labels=True),
        **loader_kwargs,
    )
    optimizer = build_optimizer(model, args)
    # Opt-in (default 0.0 = current behavior, a flat LR): linear warmup for the
    # first warmup_ratio fraction of steps, then linear decay to 0 over the rest --
    # the standard BERT fine-tuning schedule. Adam's moment estimates are noisy on
    # the first few steps; warmup keeps those steps small instead of taking a full-
    # size step on an unreliable gradient estimate. Total steps counts every
    # optimizer.step() call across all epochs, so it already accounts for dataset
    # size and batch size together.
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
        total_steps = len(train_loader) * args.epochs
        num_warmup_steps = round(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
    history = []
    best_epoch_key = None
    best_epoch = None
    best_state = None
    best_epoch_metrics = None
    threshold_grid = parse_threshold_grid(args.dev_thresholds)

    # 0.0 = off (flat behavior of every published panel); >0 clips the global L2
    # grad norm at that value, HF-Trainer-style. Same 0-means-off convention as
    # --warmup-ratio so dispatchers can always pass it explicitly.
    max_grad_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
    if max_grad_norm < 0:
        raise ValueError(f"--max-grad-norm must be >= 0 (0 disables clipping), got {max_grad_norm}")

    # ACTUAL runtime config, recorded into the report (same block the span trainer
    # writes): the requested CLI value and the effective schedule can differ (the
    # E1 warmup incident), so the report must carry what actually ran, including
    # the effective warmup STEPS -- num_warmup_steps scales with --epochs, which is
    # why a 5-epoch and a 15-epoch warmup run are different schedules.
    runtime = {
        "optimizer": type(optimizer).__name__,
        "adam_betas": list(optimizer.defaults.get("betas", [])),
        "adam_eps": optimizer.defaults.get("eps"),
        "weight_decay": float(getattr(args, "weight_decay", 0.01)),
        "no_decay_bias_layernorm": bool(getattr(args, "no_decay_bias_layernorm", False)),
        "scheduler": "linear_warmup_then_decay" if scheduler is not None else "constant_flat",
        "warmup_ratio_effective": warmup_ratio,
        "num_training_steps": len(train_loader) * args.epochs,
        "num_warmup_steps": int(round(warmup_ratio * len(train_loader) * args.epochs)),
        "amp": use_amp,
        "max_grad_norm": max_grad_norm,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }

    log_every = max(0, getattr(args, "log_every_n_batches", 0) or 0)
    total_batches = len(train_loader)
    for epoch in range(args.epochs):
        epoch_started = time.monotonic()
        # Accumulate loss as a GPU tensor and only sync to host at log points /
        # epoch end (was: float(loss.detach().cpu()) on EVERY batch, which forces
        # a full device sync each step and serializes the training loop).
        epoch_loss_sum = torch.zeros((), device=device)
        batch_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            label_weights = batch.pop("label_weights", None)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                if label_weights is not None:
                    loss = weighted_token_classification_loss(outputs.logits, batch["labels"], label_weights)
            loss_detached = loss.detach()
            # A NaN/Inf loss (rare, but can happen under AMP on exactly the batch
            # GradScaler itself detects via scale_before/scale_after below and
            # skips) must not be folded into epoch_loss_sum -- NaN + x == NaN, which
            # would silently poison every train_loss printed/saved for the rest of
            # this epoch even though only that one batch was bad and its optimizer
            # step was correctly skipped.
            if torch.isfinite(loss_detached):
                epoch_loss_sum += loss_detached
                batch_count += 1
            if use_amp:
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    # Grads must be unscaled before clipping or the norm is computed
                    # on scaled values and the clip threshold is meaningless. After
                    # unscale_, scaler.step() skips its own unscale and still performs
                    # the inf/nan check -- the documented torch AMP clipping recipe.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # GradScaler SKIPS optimizer.step() when it finds inf/nan grads, and
                # lowers its scale when it does -- so scale_after < scale_before means
                # no step was taken. The LR schedule must not advance on a step the
                # optimizer never took (it would then run ahead of training); this is
                # the same guard HuggingFace's Trainer applies.
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            if log_every and (batch_index % log_every == 0 or batch_index == total_batches):
                batch_loss = float(loss_detached)
                running_loss = float(epoch_loss_sum) / batch_count if batch_count else float("nan")
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_index}/{total_batches}: "
                    f"batch_loss={batch_loss:.4f} running_loss={running_loss:.4f}",
                    flush=True,
                )

        train_seconds = round(time.monotonic() - epoch_started, 1)
        train_loss = float(epoch_loss_sum) / batch_count if batch_count else 0.0
        dev_loss = None
        best_threshold_metrics = None
        threshold_results = []
        if dev_features:
            dev_loss = evaluate_loss(model, tokenizer, dev_features, args.batch_size, device)
            dev_logits = collect_feature_logits(model, tokenizer, dev_features, args.batch_size)
            best_threshold_metrics, threshold_results = evaluate_dev_thresholds(
                model,
                tokenizer,
                dev_features,
                dev_units,
                id_to_label,
                args.label_mode,
                args.batch_size,
                threshold_grid,
                logits_by_feature=dev_logits,
                collapse_fn=_dev_collapse_fn(getattr(args, "dev_eval_collapse", None)),
            )
            model.train()
            epoch_key = (
                best_threshold_metrics["class_aware_f1"],
                best_threshold_metrics["strict_span_f1"],
                -best_threshold_metrics["threshold"],
            )
            if best_epoch_key is None or epoch_key > best_epoch_key:
                best_epoch_key = epoch_key
                best_epoch = epoch + 1
                best_epoch_metrics = {
                    "epoch": best_epoch,
                    "dev_loss": dev_loss,
                    **best_threshold_metrics,
                }
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                # Persist the best-so-far checkpoint to disk on every improvement.
                # Long CPU runs are occasionally killed by the OS (no traceback);
                # without this the in-memory best_state is lost and the whole seed
                # must be retrained. With it, a crashed run still leaves a usable
                # best checkpoint that --predict-only can turn into predictions
                # (use the logged best threshold). This does not change training
                # results: weights, optimizer, and seed are untouched.
                model_dir = getattr(args, "model_dir", None)
                if model_dir:
                    Path(model_dir).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
        dev_text = f", dev_loss={dev_loss:.4f}" if dev_loss is not None else ""
        metric_text = ""
        if best_threshold_metrics:
            metric_text = (
                f", dev_class_f1={best_threshold_metrics['class_aware_f1']:.4f}"
                f", dev_span_f1={best_threshold_metrics['strict_span_f1']:.4f}"
                f", threshold={best_threshold_metrics['threshold']:.3f}"
            )
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}{dev_text}{metric_text}", flush=True)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "best_threshold": best_threshold_metrics["threshold"] if best_threshold_metrics else None,
            "dev_strict_span_f1": best_threshold_metrics["strict_span_f1"] if best_threshold_metrics else None,
            "dev_class_aware_f1": best_threshold_metrics["class_aware_f1"] if best_threshold_metrics else None,
            "dev_threshold_results": threshold_results,
            # Record training-batch time separately from total epoch time, which
            # also includes dev loss, the dev threshold sweep and checkpointing.
            "train_seconds": train_seconds,
            "epoch_seconds": round(time.monotonic() - epoch_started, 1),
        })

    if best_state is not None:
        model.load_state_dict(best_state)
        selection_metric = "dev_class_aware_f1"
    else:
        best_epoch = args.epochs
        selection_metric = "final_epoch"

    return {
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_dev_loss": best_epoch_metrics.get("dev_loss") if best_epoch_metrics else None,
        "best_threshold": best_epoch_metrics.get("threshold") if best_epoch_metrics else None,
        "best_dev_strict_span_f1": best_epoch_metrics.get("strict_span_f1") if best_epoch_metrics else None,
        "best_dev_class_aware_f1": best_epoch_metrics.get("class_aware_f1") if best_epoch_metrics else None,
        "dev_thresholds": threshold_grid,
        "training_history": history,
        "runtime": runtime,
    }


def evaluate_loss(model, tokenizer, features, batch_size, device):
    import torch
    from torch.utils.data import DataLoader

    loader_kwargs = {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_features(batch, tokenizer, include_labels=True),
        **loader_kwargs,
    )
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


def predict_model(model, tokenizer, features, batch_size, id_to_label=None, threshold=None):
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    loader_kwargs = {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_features(batch, tokenizer, include_labels=False),
        **loader_kwargs,
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            if threshold is not None:
                if id_to_label is None:
                    raise ValueError("id_to_label is required for thresholded prediction.")
                batch_predictions = prediction_ids_from_logits(outputs.logits, id_to_label, threshold=threshold)
            else:
                batch_predictions = prediction_ids_from_logits(outputs.logits, id_to_label or {}, threshold=None)
            predictions.extend(batch_predictions)
    return predictions


def gold_rows_from_units(units, label_mode="class"):
    rows = []
    for unit in units:
        rows.extend(prediction_rows_from_decoded_spans(unit, unit.get("spans", []), label_mode=label_mode))
    return deduplicate_predictions(rows, label_mode=label_mode)


def span_f1(gold_rows, pred_rows, class_aware=True):
    def key(row):
        base = (
            row.get("dataset"),
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("start"),
            row.get("end"),
        )
        if class_aware:
            return base + (row.get("class_label"),)
        return base

    gold = {key(row) for row in gold_rows}
    pred = {key(row) for row in pred_rows}
    true_positive = len(gold & pred)
    precision = true_positive / len(pred) if pred else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "pred": len(pred),
        "gold": len(gold),
    }


def parse_threshold_grid(value):
    thresholds = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not thresholds:
        raise ValueError("Threshold grid is empty.")
    invalid = [threshold for threshold in thresholds if threshold < 0.0 or threshold > 1.0]
    if invalid:
        raise ValueError(f"Thresholds must be in [0, 1], got: {invalid}")
    return thresholds


def collect_feature_logits(model, tokenizer, features, batch_size):
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    loader_kwargs = {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_features(batch, tokenizer, include_labels=False),
        **loader_kwargs,
    )

    logits_by_feature = []
    model.eval()
    feature_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size_actual = batch["input_ids"].shape[0]
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            batch_logits = outputs.logits.detach().cpu()
            for row_index in range(batch_size_actual):
                feature = features[feature_offset + row_index]
                length = len(feature["offset_mapping"])
                logits_by_feature.append(batch_logits[row_index, :length, :])
            feature_offset += batch_size_actual
    return logits_by_feature


def prediction_ids_from_feature_logits(logits_by_feature, id_to_label, threshold=None):
    return [
        prediction_ids_from_logits(logits.unsqueeze(0), id_to_label, threshold=threshold)[0]
        for logits in logits_by_feature
    ]


def _dev_collapse_fn(name):
    """Resolve a class-label collapse used ONLY for the dev-selection metric.
    For E1 arm A (train on 36 classes, scored on collapsed 6), the threshold and
    best epoch must be picked on the 6-class dev objective the arm is scored on,
    not the native 36-class one. genia_coarse6 imports without heavy deps."""
    if not name:
        return None
    if name == "genia_coarse6":
        from ate_paper.experiments.genia_label_maps import genia_coarse6
        return genia_coarse6
    raise ValueError(f"Unknown dev-eval-collapse: {name!r}")


def evaluate_dev_thresholds(
    model,
    tokenizer,
    dev_features,
    dev_units,
    id_to_label,
    label_mode,
    batch_size,
    thresholds,
    logits_by_feature=None,
    collapse_fn=None,
):
    gold_rows = gold_rows_from_units(dev_units, label_mode=label_mode)
    if collapse_fn is not None:
        for _r in gold_rows:
            _r["class_label"] = collapse_fn(_r.get("class_label"))
    if logits_by_feature is None:
        logits_by_feature = collect_feature_logits(model, tokenizer, dev_features, batch_size)
    results = []
    for threshold in thresholds:
        prediction_ids = prediction_ids_from_feature_logits(logits_by_feature, id_to_label, threshold=threshold)
        pred_rows = decode_prediction_features(dev_features, prediction_ids, id_to_label, label_mode=label_mode)
        if collapse_fn is not None:
            for _r in pred_rows:
                _r["class_label"] = collapse_fn(_r.get("class_label"))
        strict_span = span_f1(gold_rows, pred_rows, class_aware=False)
        class_aware = span_f1(gold_rows, pred_rows, class_aware=True)
        results.append({
            "threshold": threshold,
            "strict_span_f1": strict_span["f1"],
            "class_aware_f1": class_aware["f1"],
            "strict_span_precision": strict_span["precision"],
            "strict_span_recall": strict_span["recall"],
            "class_aware_precision": class_aware["precision"],
            "class_aware_recall": class_aware["recall"],
        })
    best = sorted(
        results,
        key=lambda row: (-row["class_aware_f1"], -row["strict_span_f1"], row["threshold"]),
    )[0]
    return best, results


def train_and_predict(args):
    tokenizer_source = args.model_dir if args.predict_only else args.model_name
    tokenizer_load_kwargs = tokenizer_kwargs(
        tokenizer_source,
        getattr(args, "scibert_do_lower_case", None),
        revision=None if args.predict_only else getattr(args, "model_revision", None),
        # Predict-only loads a saved tokenizer; the parser's default mBERT
        # identity must not override that checkpoint's tokenizer metadata.
        model_name=None if args.predict_only else args.model_name,
    )
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "train_mbert_sequence_labeler.py requires transformers, torch, and numpy. "
            "Install them in the project environment before training."
        ) from exc

    if args.dataset not in SPAN_DATASETS:
        raise ValueError(
            f"BIO sequence labeling baseline supports only: {', '.join(sorted(SPAN_DATASETS))}."
        )

    validate_predict_only(args.model_dir, args.predict_only, threshold=args.threshold)
    set_random_seeds(args.seed)

    train_rows = read_jsonl(args.train_gold)
    dev_rows = read_jsonl(args.dev_gold)
    test_rows = read_jsonl(args.test_text)

    assert_no_doc_id_overlap([
        ("train", train_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ])

    train_units = group_annotations_into_text_units(train_rows, dataset=args.dataset, include_spans=True)
    dev_units = group_annotations_into_text_units(dev_rows, dataset=args.dataset, include_spans=True)
    test_units, warnings = test_text_units_from_rows(test_rows, args.dataset)

    label_list = labels_for_dataset(args.dataset, args.label_mode, train_units, dev_units)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    # fix_mistral_regex targets the buggy Mistral Split-regex pretokenizer only.
    # None of the panel tokenizers carries one (mBERT/WordPiece use
    # BertPreTokenizer, XLM-R uses WhitespaceSplit+Metaspace -- verified
    # statically on the local checkpoints, 2026-08-22), so the flag is a no-op
    # here; it is kept for uniformity across entry points and transformers
    # versions, with a fallback where the kwarg is rejected.
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, fix_mistral_regex=True, **tokenizer_load_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_load_kwargs)

    train_features, train_stats = tokenize_units(
        tokenizer,
        train_units,
        label_to_id,
        args.label_mode,
        args.flatten_policy,
        args.max_length,
        args.stride,
        length_balanced_positive_sampling=args.length_balanced_positive_sampling,
        long_span_token_threshold=args.long_span_token_threshold,
        long_positive_weight=args.long_positive_weight,
    )
    dev_features, dev_stats = tokenize_units(
        tokenizer,
        dev_units,
        label_to_id,
        args.label_mode,
        args.flatten_policy,
        args.max_length,
        args.stride,
    )
    test_features, test_stats = tokenize_units(
        tokenizer,
        test_units,
        None,
        args.label_mode,
        args.flatten_policy,
        args.max_length,
        args.stride,
    )

    if args.predict_only:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
            **pretrained_revision_kwargs(getattr(args, "model_revision", None)),
        )

    if not args.predict_only:
        training_info = train_model(model, tokenizer, train_features, dev_features, dev_units, id_to_label, args)
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    else:
        training_info = {
            "selection_metric": "predict_only",
            "best_epoch": None,
            "best_dev_loss": None,
            "best_threshold": None,
            "best_dev_strict_span_f1": None,
            "best_dev_class_aware_f1": None,
            "dev_thresholds": [],
            "training_history": [],
        }

    prediction_threshold = args.threshold
    if prediction_threshold is None:
        prediction_threshold = training_info.get("best_threshold")
    prediction_ids = predict_model(
        model,
        tokenizer,
        test_features,
        args.batch_size,
        id_to_label=id_to_label,
        threshold=prediction_threshold,
    )
    rows = decode_prediction_features(test_features, prediction_ids, id_to_label, label_mode=args.label_mode)
    validate_prediction_schema(rows)
    write_jsonl(rows, args.output)

    report = {
        "model_name": tokenizer_source if args.predict_only else args.model_name,
        "dataset": args.dataset,
        "model_revision": getattr(args, "model_revision", None),
        "scibert_do_lower_case": getattr(args, "scibert_do_lower_case", None),
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "label_mode": args.label_mode,
        "train_examples": len(train_units),
        "dev_examples": len(dev_units),
        "test_examples": len(test_units),
        "number_of_labels": len(label_list),
        "label_list": label_list,
        "max_length": args.max_length,
        "stride": args.stride,
        "flatten_policy": args.flatten_policy,
        "skipped_due_to_overlap": train_stats["skipped_due_to_overlap"] + dev_stats["skipped_due_to_overlap"],
        "skipped_due_to_alignment": train_stats["skipped_due_to_alignment"] + dev_stats["skipped_due_to_alignment"],
        "skipped_outside_window": train_stats["skipped_outside_window"] + dev_stats["skipped_outside_window"],
        "skipped_no_token_indices": train_stats["skipped_no_token_indices"] + dev_stats["skipped_no_token_indices"],
        "skipped_missing_label": train_stats["skipped_missing_label"] + dev_stats["skipped_missing_label"],
        "skipped_label_not_in_inventory": (
            train_stats["skipped_label_not_in_inventory"] + dev_stats["skipped_label_not_in_inventory"]
        ),
        "texts_split_into_overflow_windows": (
            train_stats["texts_split_into_overflow_windows"]
            + dev_stats["texts_split_into_overflow_windows"]
            + test_stats["texts_split_into_overflow_windows"]
        ),
        "prediction_rows_written": len(rows),
        "prediction_class_distribution": Counter(row.get("class_label") for row in rows if row.get("class_label")),
        "warnings": warnings,
        "device": str(next(model.parameters()).device),
        # Optimization hyperparameters, so a run's own report says how it was
        # trained (the span classifier already captures these via vars(args)).
        # Without these a report cannot distinguish a flat-LR run from a warmup
        # run, or fp32 from mixed precision.
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "amp": bool(getattr(args, "amp", False)),
        "warmup_ratio": float(getattr(args, "warmup_ratio", 0.0) or 0.0),
        "max_grad_norm": float(getattr(args, "max_grad_norm", 0.0) or 0.0),
        "weight_decay": float(getattr(args, "weight_decay", 0.01)),
        "no_decay_bias_layernorm": bool(getattr(args, "no_decay_bias_layernorm", False)),
        "log_every_n_batches": getattr(args, "log_every_n_batches", 0),
        "report_title": getattr(args, "report_title", "mBERT Sequence Labeler Report"),
        "length_balanced_positive_sampling": args.length_balanced_positive_sampling,
        "long_span_token_threshold": args.long_span_token_threshold,
        "long_positive_weight": args.long_positive_weight if args.length_balanced_positive_sampling else 1,
        "length_balanced_long_spans": train_stats["length_balanced_long_spans"],
        "length_balanced_weighted_tokens": train_stats["length_balanced_weighted_tokens"],
        "selection_metric": training_info["selection_metric"],
        "best_epoch": training_info["best_epoch"],
        "best_dev_loss": training_info["best_dev_loss"],
        "best_threshold": training_info["best_threshold"],
        "prediction_threshold": prediction_threshold,
        "best_dev_strict_span_f1": training_info["best_dev_strict_span_f1"],
        "best_dev_class_aware_f1": training_info["best_dev_class_aware_f1"],
        "dev_thresholds": training_info["dev_thresholds"],
        "training_history": training_info["training_history"],
        # None on --predict-only (nothing was trained in this invocation).
        "runtime": training_info.get("runtime"),
    }
    write_report(report, args.report)
    return report


def write_report(report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {report.get('report_title', 'mBERT Sequence Labeler Report')}",
        "",
        f"- model name: {report['model_name']}",
        f"- model revision: {report.get('model_revision')}",
        f"- scibert do lower case choice: {report.get('scibert_do_lower_case')}",
        f"- tokenizer is fast: {report.get('tokenizer_is_fast')}",
        f"- tokenizer do lower case: {report.get('tokenizer_do_lower_case')}",
        f"- dataset: {report['dataset']}",
        f"- label mode: {report['label_mode']}",
        f"- device: {report['device']}",
        f"- epochs: {report.get('epochs')}",
        f"- batch size: {report.get('batch_size')}",
        f"- learning rate: {report.get('learning_rate')}",
        f"- amp: {report.get('amp')}",
        f"- warmup ratio: {report.get('warmup_ratio')}",
        f"- max grad norm: {report.get('max_grad_norm', 0.0)}",
        f"- weight decay: {report.get('weight_decay', 0.01)}",
        f"- no-decay bias/LayerNorm: {report.get('no_decay_bias_layernorm', False)}",
        f"- log every n batches: {report.get('log_every_n_batches', 0)}",
        f"- length-balanced positive sampling: {report.get('length_balanced_positive_sampling')}",
        f"- long span token threshold: {report.get('long_span_token_threshold')}",
        f"- long positive weight: {report.get('long_positive_weight')}",
        f"- length-balanced long spans: {report.get('length_balanced_long_spans')}",
        f"- length-balanced weighted tokens: {report.get('length_balanced_weighted_tokens')}",
        f"- selection metric: {report.get('selection_metric')}",
        f"- best epoch: {report.get('best_epoch')}",
        f"- best dev loss: {report.get('best_dev_loss')}",
        f"- best threshold: {report.get('best_threshold')}",
        f"- prediction threshold: {report.get('prediction_threshold')}",
        f"- best dev strict span F1: {report.get('best_dev_strict_span_f1')}",
        f"- best dev class-aware F1: {report.get('best_dev_class_aware_f1')}",
        f"- train examples: {report['train_examples']}",
        f"- dev examples: {report['dev_examples']}",
        f"- test examples: {report['test_examples']}",
        f"- number of labels: {report['number_of_labels']}",
        f"- max length: {report['max_length']}",
        f"- stride: {report['stride']}",
        f"- flatten policy: {report['flatten_policy']}",
        f"- skipped due to overlap: {report['skipped_due_to_overlap']}",
        f"- skipped due to alignment (total): {report['skipped_due_to_alignment']}",
        f"  - outside window (stride/max_length artifact, not a tokenizer property): "
        f"{report.get('skipped_outside_window', 'n/a')}",
        f"  - no matching token indices (genuine tokenizer-boundary miss): "
        f"{report.get('skipped_no_token_indices', 'n/a')}",
        f"  - missing class label: {report.get('skipped_missing_label', 'n/a')}",
        f"  - class not in label inventory: {report.get('skipped_label_not_in_inventory', 'n/a')}",
        f"- texts split into overflow windows: {report['texts_split_into_overflow_windows']}",
        f"- prediction rows written: {report['prediction_rows_written']}",
        "",
        "## Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in report["label_list"])

    lines.extend(["", "## Prediction Class Distribution", ""])
    if report["prediction_class_distribution"]:
        for label, count in sorted(report["prediction_class_distribution"].items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"]) if report["warnings"] else lines.append("- none")

    # Effective optimizer/schedule/version facts (see train_model's `runtime`).
    lines.extend(["", "## Runtime", ""])
    runtime = report.get("runtime") or {}
    if runtime:
        lines.extend(f"- {key}: `{value}`" for key, value in runtime.items())
    else:
        lines.append("- none")

    lines.extend(["", "## Training History", ""])
    if report.get("training_history"):
        for row in report["training_history"]:
            dev_loss = row["dev_loss"]
            dev_text = f"{dev_loss:.6f}" if dev_loss is not None else "n/a"
            timing_text = ""
            if row.get("epoch_seconds") is not None:
                timing_text = (
                    f", train_seconds={row.get('train_seconds')}"
                    f", epoch_seconds={row.get('epoch_seconds')}"
                )
            lines.append(
                f"- epoch {row['epoch']}: train_loss={row['train_loss']:.6f}, dev_loss={dev_text}"
                f", dev_class_aware_f1={row.get('dev_class_aware_f1')}"
                f", dev_strict_span_f1={row.get('dev_strict_span_f1')}"
                f", threshold={row.get('best_threshold')}"
                + timing_text
            )
    else:
        lines.append("- none")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gold", required=True)
    parser.add_argument("--dev-gold", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(SPAN_DATASETS))
    parser.add_argument("--label-mode", choices=["class", "unlabeled"], default="class")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--dev-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dev-eval-collapse", default=None,
                        help="collapse dev class labels for threshold/best-epoch "
                             "selection only (e.g. genia_coarse6 for E1 arm A)")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--amp", action="store_true",
                         help="Mixed-precision training (torch.cuda.amp). No-op off CUDA.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                         help="Fraction of total steps for linear LR warmup, then linear "
                              "decay to 0 (standard BERT fine-tuning schedule). Default 0.0 (flat LR, no warmup). Was 0.1 (10 percent, the standard BERT fine-tuning ratio) from 2026-07-27 to 2026-07-28, reverted after concluding the per-entry-point bookkeeping this required (tracking which runs/dates used which default) cost more than a literature value never validated on this project's own data was worth; every already-published result was flat-LR anyway.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                         help="Clip the global L2 gradient norm at this value before each "
                              "optimizer step (HF Trainer's default recipe uses 1.0). "
                              "Default 0.0 = no clipping, the behavior of every published "
                              "panel. Same 0-means-off convention as --warmup-ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                         help="AdamW weight decay. Default 0.01 -- torch's own AdamW "
                              "default, which every published panel already trained under "
                              "implicitly; this flag only makes it explicit and overridable.")
    parser.add_argument("--no-decay-bias-layernorm", action="store_true",
                         help="Exclude bias and LayerNorm parameters from weight decay "
                              "(the HF-Trainer parameter-group convention). OFF by default: "
                              "published panels decayed every parameter uniformly.")
    parser.add_argument("--flatten-policy", choices=["longest"], default="longest")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--report", default="build/experiments/supervised/reports/mbert_sequence_labeler_report.md")
    parser.add_argument("--report-title", default="mBERT Sequence Labeler Report")
    parser.add_argument("--log-every-n-batches", type=int, default=50)
    parser.add_argument("--length-balanced-positive-sampling", action="store_true")
    parser.add_argument("--long-span-token-threshold", type=int, default=4)
    parser.add_argument("--long-positive-weight", type=int, default=3)
    add_pretrained_arguments(parser)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.long_span_token_threshold < 1:
        raise ValueError("--long-span-token-threshold must be >= 1.")
    if args.long_positive_weight < 1:
        raise ValueError("--long-positive-weight must be >= 1.")

    report = train_and_predict(args)
    print(f"Saved predictions: {args.output}")
    print(f"Saved report: {args.report}")
    print(f"Prediction rows written: {report['prediction_rows_written']}")


if __name__ == "__main__":
    main()
