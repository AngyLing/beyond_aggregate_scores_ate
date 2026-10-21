"""Convert the official CL-RuTerm3 1.1 train/dev/test files without repartitioning.

ROW_VERSION preserves the original normalized-row identifier, not corpus release.
"""
import json
import re


CLASS_MAP = {
    "specific": "Specific",
    "common": "Common",
    "nomen": "Nomen",
}


ROW_VERSION = "v2"


def norm(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def convert_row(row, split):
    doc_id = row["id"]
    text = row["text"]
    rows = []
    domainnow = row.get("domain", "comp_ling")
    labels = row.get("label", [])

    base = {
        "dataset": "CL-RuTerm3",
        "version": ROW_VERSION,
        "split": split,
        "language": "Russian",
        "domain": domainnow,
        "doc_id": doc_id,
        "text": text,
        "unit_type": "span_mention",
    }

    for start, end, label in labels:
        surface = text[start:end]
        class_label = CLASS_MAP.get(label, label)

        rows.append({
            **base,
            "surface": surface,
            "norm": norm(surface),
            "start": start,
            "end": end,
            "class_label": class_label,
            "is_nomen": class_label == "Nomen",
        })

    if not labels:
        # Zero-term document: emit one document-level negative placeholder row
        # so the document survives conversion instead of contributing nothing.
        # Currently inert on this corpus vintage (every document in
        # train.jsonl / dev_ref.jsonl / test_ref_v4.jsonl carries at least
        # one label), but the same convention as the other converters guards
        # against a future corpus update introducing a term-free document.
        rows.append({
            **base,
            "surface": None,
            "norm": None,
            "start": None,
            "end": None,
            "class_label": None,
            "is_nomen": False,
        })

    return rows


def read_documents(path):
    documents = []
    ids = set()
    with open(path, encoding="utf-8") as inp:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row:
                continue
            if row["id"] in ids:
                raise ValueError(f"Duplicate document id {row['id']!r} in {path}")
            ids.add(row["id"])
            documents.append(row)
    return documents


def read_dev_documents(texts_path, reference_path):
    """Join the official 1.1 dev texts and reference by id, not row order.

    Release dev comes from the shared task's initial test of Tracks 1 and 2.
    Keep source labels verbatim, including duplicates and conflicting classes.
    Annotation adjudication is a separate decision from changing the split.
    """
    texts = read_documents(texts_path)
    references = {row["id"]: row for row in read_documents(reference_path)}
    text_ids = {row["id"] for row in texts}
    if text_ids != set(references):
        raise ValueError(
            f"Dev id mismatch: texts without reference={sorted(text_ids - set(references))}; "
            f"reference without texts={sorted(set(references) - text_ids)}"
        )
    documents = []
    for row in texts:
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid dev text: {row['id']}")
        labels = references[row["id"]]["label"]
        for start, end, label in labels:
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text) or label not in CLASS_MAP):
                raise ValueError(f"Invalid dev span in {row['id']}: {(start, end, label)}")
        documents.append({**row, "domain": row.get("domain", "comp_ling"), "label": labels})
    return documents


def read_test_documents(reference_path, texts_path=None):
    documents = read_documents(reference_path)
    if texts_path is None:
        return documents
    texts = {row["id"]: row["text"] for row in read_documents(texts_path)}
    references = {row["id"]: row["text"] for row in documents}
    if set(texts) != set(references):
        raise ValueError(
            f"Test id mismatch: texts without reference={sorted(set(texts) - set(references))}; "
            f"reference without texts={sorted(set(references) - set(texts))}"
        )
    different = [doc_id for doc_id in references if references[doc_id] != texts[doc_id]]
    if different:
        raise ValueError(f"Test texts differ from reference: {different}")
    return documents


def convert(train_path, test_path, dev_texts_path=None, dev_reference_path=None, test_texts_path=None):
    if (dev_texts_path is None) != (dev_reference_path is None):
        raise ValueError("Both dev texts and dev reference are required")
    sources = [("train", read_documents(train_path))]
    if dev_texts_path is not None:
        sources.append(("dev", read_dev_documents(dev_texts_path, dev_reference_path)))
    sources.append(("test", read_test_documents(test_path, test_texts_path)))
    rows = []
    seen_ids = {}
    seen_texts = {}
    for split, documents in sources:
        for row in documents:
            doc_id, text = row["id"], row["text"]
            if doc_id in seen_ids and seen_ids[doc_id] != split:
                raise ValueError(f"Document {doc_id!r} overlaps {seen_ids[doc_id]} and {split}")
            if text in seen_texts and seen_texts[text] != split:
                raise ValueError(f"Text of {doc_id!r} overlaps {seen_texts[text]} and {split}")
            seen_ids[doc_id] = split
            seen_texts[text] = split
            rows.extend(convert_row(row, split))
    return rows
