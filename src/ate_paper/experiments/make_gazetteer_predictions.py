#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


TRAIN_UNIT_TYPES = {"term_string", "span_mention", "sentence_term_string"}
SPAN_DATASETS = {"ACL RD-TEC 2.0", "CL-RuTerm3", "GENIA", "ACTER v1.5"}
STOPWORD_TERMS = {
    "English": {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    },
    "Russian": {
        "а", "без", "в", "во", "для", "и", "из", "к", "ко", "на", "не", "о",
        "об", "от", "по", "при", "с", "со", "у",
    },
    "French": {
        "a", "au", "aux", "avec", "ce", "ces", "de", "des", "du", "en", "et",
        "la", "le", "les", "ou", "par", "pour", "un", "une",
    },
    "Dutch": {
        "aan", "als", "bij", "de", "een", "en", "het", "in", "is", "met",
        "of", "op", "te", "van", "voor",
    },
    "Italian": {
        "a", "al", "alla", "con", "da", "del", "della", "di", "e", "gli",
        "i", "il", "in", "la", "le", "o", "per", "un", "una",
    },
}
ACTER_DOMAIN_MAP = {
    "corp": "corruption",
    "equi": "dressage/equitation",
    "htfl": "heart failure",
    "wind": "wind energy",
}
ACTER_LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "nl": "Dutch",
}
NORMALIZATION_POLICY = {
    "ACTER v1.2": "lowercased whitespace-normalized norm strings; no offsets",
    "ATE-IT": "lowercased whitespace-normalized norm strings",
    "span_datasets": (
        "strict span metrics use offsets; derived type-level diagnostics use "
        "the converted lowercased norm strings, matching the current ACL RD-TEC "
        "2.0 and CL-RuTerm3 gold converters"
    ),
}
CLASS_TIE_PRIORITIES = {
    "ACTER v1.2": {"Specific": 0, "Common": 1, "OOD": 2, "NE": 3},
    "CL-RuTerm3": {"Specific": 0, "Common": 1, "Nomen": 2},
}


def normalize_text(text):
    return re.sub(r"\s+", " ", ("" if text is None else str(text)).strip().lower())


def alnum_count(text):
    return sum(1 for char in str(text) if char.isalnum())


def token_count(text):
    return len(re.findall(r"\w+", str(text), flags=re.UNICODE))


def load_excluded_terms(path):
    if not path:
        return set()
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            term = normalize_text(line.partition("#")[0])
            if term:
                terms.add(term)
    return terms


def default_filter_options():
    return {
        "min_term_chars": 1,
        "min_alnum_chars": 1,
        "min_term_tokens": 1,
        "drop_stopword_terms": False,
        "drop_single_letter_terms": False,
        "excluded_terms": set(),
        "exclude_term_file": None,
    }


def is_single_letter_term(norm):
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё]\.?", norm))


def should_filter_inventory_term(norm, row, filter_options):
    if len(norm) < filter_options["min_term_chars"]:
        return "below_min_term_chars"
    if alnum_count(norm) < filter_options["min_alnum_chars"]:
        return "below_min_alnum_chars"
    if token_count(norm) < filter_options["min_term_tokens"]:
        return "below_min_term_tokens"
    if norm in filter_options["excluded_terms"]:
        return "explicit_exclude_term"
    if filter_options["drop_stopword_terms"]:
        language = row.get("language")
        if norm in STOPWORD_TERMS.get(language, set()):
            return "stopword_term"
    if filter_options["drop_single_letter_terms"] and is_single_letter_term(norm):
        return "single_letter_term"
    return None


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scope_key(row, inventory_scope):
    key = (
        row.get("dataset"),
        row.get("language"),
    )
    if inventory_scope == "dataset_language_domain":
        key += (row.get("domain"),)
    return key


def majority(counter):
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def majority_class(dataset, counter):
    if not counter:
        return None
    priority = CLASS_TIE_PRIORITIES.get(dataset, {})
    return sorted(
        counter.items(),
        key=lambda item: (-item[1], priority.get(item[0], 99), str(item[0])),
    )[0][0]


def build_inventory(rows, inventory_scope="dataset_language", filter_options=None):
    filter_options = filter_options or default_filter_options()
    inventory = defaultdict(dict)
    class_counts = defaultdict(Counter)
    surface_counts = defaultdict(Counter)
    metadata = {}
    filtered = Counter()
    filtered_examples = defaultdict(list)

    for row in rows:
        if row.get("unit_type") not in TRAIN_UNIT_TYPES:
            continue

        norm = normalize_text(row.get("norm", ""))
        if not norm:
            continue

        filter_reason = should_filter_inventory_term(norm, row, filter_options)
        if filter_reason:
            key = scope_key(row, inventory_scope)
            filtered[(key, filter_reason)] += 1
            if len(filtered_examples[(key, filter_reason)]) < 10:
                filtered_examples[(key, filter_reason)].append(norm)
            continue

        key = scope_key(row, inventory_scope)
        item_key = (key, norm)
        surface = row.get("surface") or norm

        surface_counts[item_key][surface] += 1
        if row.get("class_label") not in {None, ""}:
            class_counts[item_key][row.get("class_label")] += 1

        metadata[item_key] = {
            "dataset": row.get("dataset"),
            "language": row.get("language"),
            "domain": row.get("domain"),
        }

    ambiguous = defaultdict(int)
    ambiguous_examples = defaultdict(list)

    for (key, norm), surfaces in surface_counts.items():
        class_counter = class_counts[(key, norm)]
        dataset = metadata[(key, norm)].get("dataset")
        class_label = majority_class(dataset, class_counter)
        if len(class_counter) > 1:
            ambiguous[key] += 1
            if len(ambiguous_examples[key]) < 10:
                ambiguous_examples[key].append({
                    "norm": norm,
                    "classes": dict(sorted(class_counter.items())),
                    "chosen_class": class_label,
                })

        inventory[key][norm] = {
            "norm": norm,
            "surface": majority(surfaces) or norm,
            "class_label": class_label,
            **metadata[(key, norm)],
        }

    return inventory, ambiguous, ambiguous_examples, filtered, filtered_examples


def is_token_char(char):
    return char.isalnum() or char in {"-", "_"}


def is_boundary_ok(text, start, end):
    left_ok = start == 0 or not is_token_char(text[start - 1])
    right_ok = end == len(text) or not is_token_char(text[end])
    return left_ok and right_ok


@lru_cache(maxsize=50000)
def ignorecase_literal_pattern(term):
    return re.compile(f"(?=({re.escape(term)}))", flags=re.IGNORECASE)


def exact_matches(text, term):
    if not text or not term:
        return []
    matches = []
    pattern = ignorecase_literal_pattern(str(term))
    for match in pattern.finditer(str(text)):
        index = match.start(1)
        end = match.end(1)
        if index >= 0 and end >= 0 and is_boundary_ok(text, index, end):
            matches.append((index, end))

    return matches


def text_units_from_rows(rows, unit_type, warnings):
    units = {}
    annotation_like = False

    for row in rows:
        if row.get("unit_type") != unit_type:
            continue
        if row.get("text") in {None, ""}:
            continue

        if row.get("norm") or row.get("surface") or row.get("start") is not None or row.get("end") is not None:
            annotation_like = True

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

        if key not in units:
            units[key] = {
                "dataset": row.get("dataset"),
                "version": row.get("version"),
                "split": row.get("split"),
                "language": row.get("language"),
                "domain": row.get("domain"),
                "doc_id": row.get("doc_id"),
                "sent_id": row.get("sent_id"),
                "text": row.get("text"),
            }

    if annotation_like and units:
        warnings.append(
            "WARNING: test text units were reconstructed from gold annotations; "
            "text units without gold terms may be missing, so false positives in "
            "empty-gold units cannot be produced and precision may be inflated. "
            "Use complete test text-unit files for fair evaluation."
        )

    return list(units.values())


def acter_text_dir(language_root, domain_code):
    candidates = [
        language_root / domain_code / "texts" / "annotated" / "texts",
        language_root / domain_code / "annotated" / "texts",
        language_root / domain_code / "unannotated_texts",
    ]
    if domain_code == "corp":
        candidates.extend([
            language_root / "cor" / "texts" / "annotated" / "texts",
            language_root / "cor" / "annotated" / "texts",
            language_root / "cor" / "unannotated_texts",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_acter_raw_text_units(root, languages=None, domains=None):
    languages = languages or ["en", "fr", "nl"]
    domains = domains or ["htfl"]
    rows = []
    root = Path(root)
    for language_code in languages:
        language = ACTER_LANGUAGE_MAP[language_code]
        language_root = root / language_code
        for domain_code in domains:
            domain = ACTER_DOMAIN_MAP[domain_code]
            text_dir = acter_text_dir(language_root, domain_code)
            if not text_dir:
                continue
            for path in sorted(text_dir.glob("*.txt")):
                rows.append({
                    "dataset": "ACTER v1.2",
                    "version": "v1.2",
                    "split": "test" if domain_code == "htfl" else "train",
                    "language": language,
                    "language_code": language_code,
                    "domain": domain,
                    "domain_code": domain_code,
                    "doc_id": path.stem,
                    "text": path.read_text(encoding="utf-8"),
                    "unit_type": "acter_text_unit",
                })
    return rows


def make_acter_predictions(test_rows, inventory, inventory_scope, warnings):
    text_units = [
        row for row in test_rows
        if row.get("dataset") == "ACTER v1.2"
        and row.get("unit_type") == "acter_text_unit"
        and row.get("text") not in {None, ""}
    ]
    if not text_units and any(row.get("dataset") == "ACTER v1.2" for row in test_rows):
        warnings.append(
            "WARNING: ACTER gazetteer requires raw text units; no ACTER predictions "
            "were produced from term-list-only rows."
        )

    matched = {}
    for row in test_rows:
        if row.get("dataset") != "ACTER v1.2" or row.get("unit_type") != "acter_text_unit":
            continue
        terms = inventory.get(scope_key(row, inventory_scope), {})
        text = row.get("text") or ""
        for norm, term in sorted(terms.items()):
            if term.get("dataset") != "ACTER v1.2":
                continue
            if not exact_matches(text, norm):
                continue
            prediction_key = (
                row.get("language"),
                row.get("domain"),
                norm,
            )
            matched.setdefault(prediction_key, {
                "dataset": "ACTER v1.2",
                "version": row.get("version"),
                "split": row.get("split"),
                "language": row.get("language"),
                "domain": row.get("domain"),
                "unit_type": "term_string",
                "surface": norm,
                "norm": norm,
                "class_label": term.get("class_label"),
                "start": None,
                "end": None,
            })
    return [matched[key] for key in sorted(matched, key=lambda item: tuple("" if part is None else str(part) for part in item))]


def make_span_predictions(test_rows, inventory, inventory_scope, warnings):
    units = text_units_from_rows(test_rows, "span_mention", warnings)
    predictions = []
    repeated_surface_matches = Counter()

    for unit in sorted(units, key=lambda row: (str(row.get("dataset")), str(row.get("doc_id")), str(row.get("sent_id")), str(row.get("text")))):
        dataset = unit.get("dataset")
        if dataset not in SPAN_DATASETS:
            continue

        terms = inventory.get(scope_key(unit, inventory_scope), {})
        text = unit.get("text") or ""

        for norm, term in sorted(terms.items(), key=lambda item: (-len(item[0]), item[0])):
            if term.get("dataset") != dataset:
                continue

            matches = exact_matches(text, norm)
            if len(matches) > 1:
                repeated_surface_matches[(dataset, norm)] += 1
            for start, end in matches:
                surface = text[start:end]
                class_label = term.get("class_label")
                row = {
                    "dataset": dataset,
                    "version": unit.get("version"),
                    "split": unit.get("split"),
                    "language": unit.get("language"),
                    "domain": unit.get("domain"),
                    "doc_id": unit.get("doc_id"),
                    "sent_id": unit.get("sent_id"),
                    "text": text,
                    "unit_type": "span_mention",
                    "surface": surface,
                    "norm": normalize_text(surface),
                    "start": start,
                    "end": end,
                    "class_label": class_label,
                }
                if dataset == "CL-RuTerm3":
                    row["is_nomen"] = class_label == "Nomen"
                predictions.append(row)

    return predictions, repeated_surface_matches


def longest_ate_it_a_matches(matches):
    kept = []
    for match in sorted(matches, key=lambda row: (-(row["end"] - row["start"]), row["start"], row["norm"])):
        contained = any(
            other["start"] <= match["start"]
            and match["end"] <= other["end"]
            and (other["end"] - other["start"]) > (match["end"] - match["start"])
            for other in kept
        )
        if not contained:
            kept.append(match)
    return sorted(kept, key=lambda row: (row["start"], row["end"], row["norm"]))


def make_ate_it_a_predictions(test_rows, inventory, inventory_scope, warnings):
    units = [
        unit for unit in text_units_from_rows(test_rows, "sentence_term_string", warnings)
        if unit.get("dataset") == "ATE-IT"
    ]
    predictions = []

    for unit in sorted(units, key=lambda row: (str(row.get("doc_id")), str(row.get("sent_id")))):
        terms = inventory.get(scope_key(unit, inventory_scope), {})
        text = unit.get("text") or ""
        matches = []

        for norm, term in sorted(terms.items(), key=lambda item: (-len(item[0]), item[0])):
            if term.get("dataset") != "ATE-IT":
                continue
            for start, end in exact_matches(text, norm):
                surface_norm = normalize_text(text[start:end])
                matches.append({
                    "start": start,
                    "end": end,
                    "norm": surface_norm,
                })

        for match in longest_ate_it_a_matches(matches):
            predictions.append({
                "dataset": "ATE-IT",
                "version": unit.get("version"),
                "split": unit.get("split"),
                "language": unit.get("language"),
                "domain": unit.get("domain"),
                "doc_id": unit.get("doc_id"),
                "sent_id": unit.get("sent_id"),
                "text": text,
                "unit_type": "sentence_term_string",
                "surface": match["norm"],
                "norm": match["norm"],
                "start": None,
                "end": None,
            })

    return predictions


def make_ate_it_b_predictions(test_rows):
    predictions = []
    item_universe = set()
    duplicate_member_ids = set()
    norms = Counter()

    for row in test_rows:
        if row.get("dataset") != "ATE-IT" or row.get("unit_type") != "variant_cluster_member":
            continue

        member_id = row.get("member_id")
        if not member_id:
            raise ValueError("ATE-IT B item lacks member_id; predictions must copy gold member_id values unchanged.")
        if member_id in item_universe:
            duplicate_member_ids.add(member_id)
            continue

        norm = normalize_text(row.get("norm", ""))
        if not norm:
            raise ValueError(f"ATE-IT B item {member_id!r} lacks norm.")

        item_universe.add(member_id)
        norms[norm] += 1
        predictions.append({
            "dataset": "ATE-IT",
            "version": row.get("version"),
            "split": row.get("split"),
            "language": row.get("language"),
            "domain": row.get("domain"),
            "unit_type": "variant_cluster_member",
            "member_id": member_id,
            "surface": row.get("surface") or norm,
            "norm": norm,
            "cluster_id": f"norm:{norm}",
        })

    if duplicate_member_ids:
        sample = ", ".join(sorted(str(member_id) for member_id in duplicate_member_ids)[:5])
        raise ValueError(f"ATE-IT B input item universe contains duplicate member_id values: {sample}")

    ambiguous_norms = sum(1 for count in norms.values() if count > 1)
    return predictions, item_universe, ambiguous_norms


def prediction_key(row):
    dataset = row.get("dataset")
    unit_type = row.get("unit_type")

    if dataset == "ACTER v1.2" and unit_type == "term_string":
        return (
            dataset,
            row.get("split"),
            row.get("language"),
            row.get("domain"),
            unit_type,
            row.get("norm"),
            row.get("class_label"),
        )

    if unit_type == "span_mention":
        return (
            dataset,
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("start"),
            row.get("end"),
            row.get("class_label"),
        )

    if dataset == "ATE-IT" and unit_type == "sentence_term_string":
        return (
            dataset,
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("norm"),
        )

    if dataset == "ATE-IT" and unit_type == "variant_cluster_member":
        return (
            dataset,
            row.get("member_id"),
        )

    return tuple(sorted(row.items()))


def deduplicate_predictions(rows):
    seen = set()
    result = []
    for row in rows:
        key = prediction_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def validate_prediction_format(predictions, test_rows):
    errors = []
    b_input_items = {
        row.get("member_id")
        for row in test_rows
        if row.get("dataset") == "ATE-IT" and row.get("unit_type") == "variant_cluster_member"
    }
    b_input_items.discard(None)
    b_pred_items = set()

    for row in predictions:
        dataset = row.get("dataset")
        unit_type = row.get("unit_type")

        if dataset == "ACTER v1.2":
            if unit_type != "term_string" or row.get("start") is not None or row.get("end") is not None or not row.get("norm"):
                errors.append(f"Bad ACTER prediction: {row}")

        elif unit_type == "span_mention":
            required = ["dataset", "doc_id", "sent_id", "text", "start", "end", "surface", "norm"]
            missing = [field for field in required if field not in row]
            if missing:
                errors.append(f"Span prediction missing fields {missing}: {row}")
                continue
            text = row.get("text")
            start = row.get("start")
            end = row.get("end")
            if not isinstance(text, str) or not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"Span prediction has invalid offset types: {row}")
            elif start < 0 or end > len(text) or start >= end or text[start:end] != row.get("surface"):
                errors.append(f"Span prediction offset mismatch: {row}")

        elif dataset == "ATE-IT" and unit_type == "sentence_term_string":
            if not row.get("doc_id") or not row.get("sent_id") or not row.get("norm"):
                errors.append(f"ATE-IT A prediction missing ids/norm: {row}")
            if row.get("start") is not None or row.get("end") is not None:
                errors.append(f"ATE-IT A prediction must not have offsets: {row}")

        elif dataset == "ATE-IT" and unit_type == "variant_cluster_member":
            if not row.get("member_id") or not row.get("cluster_id"):
                errors.append(f"ATE-IT B prediction missing member_id/cluster_id: {row}")
            b_pred_items.add(row.get("member_id"))

    if b_input_items and b_pred_items != b_input_items:
        errors.append(
            "ATE-IT B prediction member_id set differs from input item universe: "
            f"missing={len(b_input_items - b_pred_items)}, extra={len(b_pred_items - b_input_items)}"
        )

    if errors:
        raise ValueError("\n".join(errors))


def inventory_size_by_scope(inventory):
    result = {}
    for key, terms in inventory.items():
        result[key] = len(terms)
    return result


def split_distribution(rows):
    return Counter(row.get("split") or "missing" for row in rows)


def ate_it_a_text_unit_stats(test_rows, reference_rows=None):
    text_units = {
        (row.get("doc_id"), row.get("sent_id"))
        for row in test_rows
        if row.get("dataset") == "ATE-IT"
        and row.get("unit_type") == "sentence_term_string"
        and row.get("text") not in {None, ""}
    }
    annotation_positive_units = {
        (row.get("doc_id"), row.get("sent_id"))
        for row in test_rows
        if row.get("dataset") == "ATE-IT"
        and row.get("unit_type") == "sentence_term_string"
        and row.get("norm") not in {None, ""}
    }
    reference_positive_units = set()
    if reference_rows is not None:
        reference_positive_units = {
            (row.get("doc_id"), row.get("sent_id"))
            for row in reference_rows
            if row.get("dataset") == "ATE-IT"
            and row.get("unit_type") == "sentence_term_string"
            and row.get("split") == "test"
            and row.get("norm") not in {None, ""}
        }
    positive_units = reference_positive_units or annotation_positive_units
    empty_units = text_units - positive_units
    return {
        "text_units_read": len(text_units),
        "positive_sentence_units": len(text_units & positive_units),
        "empty_sentence_units": len(empty_units),
        "missing_positive_sentence_units": len(positive_units - text_units),
        "positive_source": "reference_gold" if reference_positive_units else "test_rows",
    }


def predict_rows(train_rows, test_rows, inventory_scope="dataset_language", filter_options=None, ate_it_a_reference_rows=None):
    filter_options = filter_options or default_filter_options()
    warnings = []
    skipped = []
    train_split_distribution = split_distribution(train_rows)
    test_split_distribution = split_distribution(test_rows)
    if train_split_distribution.get("test", 0):
        warnings.append(
            "WARNING: train rows contain split=test; this may leak held-out gold "
            "terms into the gazetteer inventory."
        )
    inventory, ambiguous, ambiguous_examples, filtered_terms, filtered_examples = build_inventory(
        train_rows,
        inventory_scope,
        filter_options,
    )

    predictions = []
    predictions.extend(make_acter_predictions(test_rows, inventory, inventory_scope, warnings))
    span_predictions, repeated_surface_matches = make_span_predictions(test_rows, inventory, inventory_scope, warnings)
    predictions.extend(span_predictions)
    predictions.extend(make_ate_it_a_predictions(test_rows, inventory, inventory_scope, warnings))
    ate_it_b_predictions, ate_it_b_items, ate_it_b_ambiguous_norms = make_ate_it_b_predictions(test_rows)
    predictions.extend(ate_it_b_predictions)

    for unit_type in sorted({row.get("unit_type") for row in test_rows}):
        if unit_type in {None, "acter_text_unit", "term_string", "span_mention", "sentence_term_string", "variant_cluster_member"}:
            continue
        if unit_type == "candidate_string":
            skipped.append("ACL RD-TEC first release candidate_string rows skipped; no candidate-level baseline input was provided.")
        else:
            skipped.append(f"Unsupported unit_type skipped: {unit_type}")

    predictions = deduplicate_predictions(predictions)
    validate_prediction_format(predictions, test_rows)

    report = {
        "train_rows_read": len(train_rows),
        "test_rows_read": len(test_rows),
        "train_split_distribution": dict(train_split_distribution),
        "test_split_distribution": dict(test_split_distribution),
        "normalization_policy": dict(NORMALIZATION_POLICY),
        "inventory": inventory_size_by_scope(inventory),
        "inventory_filter_policy": {
            "min_term_chars": filter_options["min_term_chars"],
            "min_alnum_chars": filter_options["min_alnum_chars"],
            "min_term_tokens": filter_options["min_term_tokens"],
            "drop_stopword_terms": filter_options["drop_stopword_terms"],
            "drop_single_letter_terms": filter_options["drop_single_letter_terms"],
            "exclude_term_file": filter_options["exclude_term_file"],
        },
        "filtered_inventory_terms_total": sum(filtered_terms.values()),
        "filtered_inventory_terms": {
            (key, reason): count for (key, reason), count in filtered_terms.items()
        },
        "filtered_inventory_examples": {
            (key, reason): examples for (key, reason), examples in filtered_examples.items()
        },
        "ambiguous": dict(ambiguous),
        "ambiguous_total": sum(ambiguous.values()),
        "ambiguous_examples": {
            key: examples for key, examples in ambiguous_examples.items()
        },
        "predictions_by_dataset_unit": Counter((row.get("dataset"), row.get("unit_type")) for row in predictions),
        "repeated_surface_match_count": sum(repeated_surface_matches.values()),
        "repeated_surface_matches": {
            f"{dataset} / {norm}": count
            for (dataset, norm), count in repeated_surface_matches.items()
        },
        "ate_it_a_text_unit_stats": ate_it_a_text_unit_stats(test_rows, ate_it_a_reference_rows),
        "ate_it_b_member_id_count": len(ate_it_b_items),
        "ate_it_b_ambiguous_norm_count": ate_it_b_ambiguous_norms,
        "acter_text_units_read": sum(
            1
            for row in test_rows
            if row.get("dataset") == "ACTER v1.2" and row.get("unit_type") == "acter_text_unit"
        ),
        "skipped": skipped,
        "warnings": warnings,
    }
    if report["ate_it_a_text_unit_stats"]["text_units_read"] and not report["ate_it_a_text_unit_stats"]["empty_sentence_units"]:
        report["warnings"].append(
            "WARNING: ATE-IT A has zero empty/no-term sentence units in the test input; "
            "the test file may be positive-only and precision may be inflated."
        )
    return predictions, report


def format_scope(key):
    return " / ".join(str(part) for part in key)


def write_report(report, path, inventory_scope):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gazetteer Baseline Report",
        "",
        f"- train rows read: {report['train_rows_read']}",
        f"- test rows read: {report['test_rows_read']}",
        f"- inventory scope: `{inventory_scope}`",
        "",
        "## Split Distribution",
        "",
        "### Train",
        "",
    ]

    for split, count in sorted(report["train_split_distribution"].items(), key=lambda item: str(item[0])):
        lines.append(f"- {split}: {count}")

    lines.extend([
        "",
        "### Test",
        "",
    ])

    for split, count in sorted(report["test_split_distribution"].items(), key=lambda item: str(item[0])):
        lines.append(f"- {split}: {count}")

    lines.extend([
        "",
        "## Normalization Policy",
        "",
    ])

    for dataset, policy in sorted(report["normalization_policy"].items()):
        lines.append(f"- {dataset}: {policy}")

    lines.extend([
        "",
        "## Inventory Filtering",
        "",
    ])
    for name, value in report["inventory_filter_policy"].items():
        lines.append(f"- {name}: `{value}`")
    lines.append(f"- filtered inventory term rows: {report['filtered_inventory_terms_total']}")
    if report["filtered_inventory_terms"]:
        lines.extend(["", "### Filtered Counts", ""])
        for (key, reason), count in sorted(report["filtered_inventory_terms"].items(), key=lambda item: (str(item[0][0]), item[0][1])):
            lines.append(f"- {format_scope(key)} / {reason}: {count}")
        lines.extend(["", "### Filtered Examples", ""])
        for (key, reason), examples in sorted(report["filtered_inventory_examples"].items(), key=lambda item: (str(item[0][0]), item[0][1])):
            lines.append(f"- {format_scope(key)} / {reason}: {', '.join(examples)}")
    else:
        lines.append("- no inventory terms filtered")

    lines.extend([
        "",
        "## Inventory Size",
        "",
    ])

    for key, size in sorted(report["inventory"].items(), key=lambda item: str(item[0])):
        lines.append(f"- {format_scope(key)}: {size}")

    lines.extend([
        "",
        "## Ambiguous Train Terms by Class",
        "",
        f"- total ambiguous train terms: {report['ambiguous_total']}",
        "",
        "### Counts",
        "",
    ])
    if report["ambiguous"]:
        for key, count in sorted(report["ambiguous"].items(), key=lambda item: str(item[0])):
            lines.append(f"- {format_scope(key)}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "### Examples", ""])
    if report["ambiguous_examples"]:
        for key, examples in sorted(report["ambiguous_examples"].items(), key=lambda item: str(item[0])):
            lines.append(f"- {format_scope(key)}:")
            for example in examples:
                classes = ", ".join(f"{label}={count}" for label, count in sorted(example["classes"].items()))
                lines.append(f"  - {example['norm']}: {classes}; chosen={example['chosen_class']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Predictions Written", ""])
    if report["predictions_by_dataset_unit"]:
        for (dataset, unit_type), count in sorted(report["predictions_by_dataset_unit"].items()):
            lines.append(f"- {dataset} / {unit_type}: {count}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Repeated Surface Matches",
        "",
        f"- repeated surface match cases: {report['repeated_surface_match_count']}",
    ])
    if report["repeated_surface_matches"]:
        for label, count in sorted(report["repeated_surface_matches"].items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## ACTER",
        "",
        f"- raw text units read: {report['acter_text_units_read']}",
        "",
        "## ATE-IT A",
        "",
        f"- text units read: {report['ate_it_a_text_unit_stats']['text_units_read']}",
        f"- positive sentence units: {report['ate_it_a_text_unit_stats']['positive_sentence_units']}",
        f"- empty/no-term sentence units: {report['ate_it_a_text_unit_stats']['empty_sentence_units']}",
        f"- missing positive sentence units: {report['ate_it_a_text_unit_stats']['missing_positive_sentence_units']}",
        f"- positive source: `{report['ate_it_a_text_unit_stats']['positive_source']}`",
        "",
        "## ATE-IT B",
        "",
        f"- member_id count: {report['ate_it_b_member_id_count']}",
        f"- ambiguous norm count in input universe: {report['ate_it_b_ambiguous_norm_count']}",
        "- identity clustering uses `cluster_id = \"norm:\" + norm`; this can over-cluster distinct items that share the same normalized term. This is an expected limitation of the simple baseline, not a data error.",
        "- If this test input has zero ambiguous normalized strings, identity clustering can obtain BCubed precision 1.0 on the test subset. This does not make `member_id` unnecessary: the full ATE-IT B resource contains ambiguous normalized strings, so `member_id` remains the correct evaluation item identity.",
        "",
        "## Skipped",
        "",
    ])
    lines.extend(f"- {item}" for item in report["skipped"]) if report["skipped"] else lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"]) if report["warnings"] else lines.append("- none")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="build/experiment_runs/gazetteer/report.md")
    parser.add_argument("--acter-raw-root")
    parser.add_argument("--acter-languages", nargs="+", default=["en", "fr", "nl"], choices=sorted(ACTER_LANGUAGE_MAP))
    parser.add_argument("--acter-domains", nargs="+", default=["htfl"], choices=sorted(ACTER_DOMAIN_MAP))
    parser.add_argument(
        "--inventory-scope",
        choices=["dataset_language", "dataset_language_domain"],
        default="dataset_language",
    )
    parser.add_argument("--min-term-chars", type=int, default=1)
    parser.add_argument("--min-alnum-chars", type=int, default=1)
    parser.add_argument("--min-term-tokens", type=int, default=1)
    parser.add_argument("--drop-stopword-terms", action="store_true")
    parser.add_argument(
        "--drop-single-letter-terms",
        action="store_true",
        help="Drop one-letter inventory terms such as 'n' or 'п.' without dropping terms like 'c++'.",
    )
    parser.add_argument(
        "--exclude-term-file",
        help="Optional UTF-8 file with one normalized term per line to exclude from the gazetteer inventory.",
    )
    parser.add_argument(
        "--ate-it-a-gold",
        default=None,
        help="Converted ATE-IT A gold JSONL used only to report positive/empty test sentence-unit coverage.",
    )
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    ate_it_a_reference_rows = read_jsonl(args.ate_it_a_gold) if args.ate_it_a_gold and Path(args.ate_it_a_gold).exists() else None
    if args.acter_raw_root:
        test_rows = test_rows + load_acter_raw_text_units(args.acter_raw_root, args.acter_languages, args.acter_domains)
    filter_options = default_filter_options()
    filter_options.update({
        "min_term_chars": args.min_term_chars,
        "min_alnum_chars": args.min_alnum_chars,
        "min_term_tokens": args.min_term_tokens,
        "drop_stopword_terms": args.drop_stopword_terms,
        "drop_single_letter_terms": args.drop_single_letter_terms,
        "exclude_term_file": args.exclude_term_file,
        "excluded_terms": load_excluded_terms(args.exclude_term_file),
    })
    predictions, report = predict_rows(
        train_rows,
        test_rows,
        args.inventory_scope,
        filter_options,
        ate_it_a_reference_rows=ate_it_a_reference_rows,
    )
    write_jsonl(predictions, args.output)
    write_report(report, args.report, args.inventory_scope)

    print(f"Train rows read: {len(train_rows)}")
    print(f"Test rows read: {len(test_rows)}")
    print(f"Predictions written: {len(predictions)}")
    print(f"Saved predictions: {args.output}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
