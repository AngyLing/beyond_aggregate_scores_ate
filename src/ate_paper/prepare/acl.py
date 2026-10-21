"""ACL RD-TEC 2.0 sentence conversion; retain term-free sentence placeholders.

Only annotator1 and annotator2 inputs are read; duplicated derived views are ignored.
"""
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_MAP = {
    "tech": "Technology and Method",
    "tool": "Tool and Library",
    "lr": "Language Resource",
    "lr-prod": "Language Resource Product",
    "model": "Model",
    "measure(ment)": "Measures and Measurements",
    "other": "Other",
}


def norm(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def read_sentence(s):
    parts = []
    terms = []
    pos = 0

    def add(text):
        nonlocal pos
        if text:
            parts.append(text)
            pos += len(text)

    add(s.text)

    for elem in list(s):
        if elem.tag == "term":
            start = pos
            term_text = "".join(elem.itertext())
            add(term_text)
            end = pos

            terms.append({
                "surface": term_text,
                "start": start,
                "end": end,
                "class_label": CLASS_MAP.get(elem.get("class"), elem.get("class")),
            })

            add(elem.tail)
        else:
            add("".join(elem.itertext()))
            add(elem.tail)

    sentence = "".join(parts)
    shift = len(sentence) - len(sentence.lstrip())
    sentence = sentence.strip()

    for term in terms:
        term["start"] -= shift
        term["end"] -= shift
        term["surface"] = sentence[term["start"]:term["end"]]

    return sentence, terms


def convert_xml(xml_file, split, annotator_id):
    root = ET.parse(xml_file).getroot()
    doc_id = root.get("acl-id", Path(xml_file).stem)

    rows = []

    for sent_num, s in enumerate(root.iter("S"), start=1):
        sent_id = f"{doc_id}_s{sent_num:04d}"
        sentence, terms = read_sentence(s)

        base = {
            "dataset": "ACL RD-TEC 2.0",
            "version": "2.0",
            "split": split,
            "language": "English",
            "domain": "Computational Linguistics",
            "doc_id": doc_id,
            "sent_id": sent_id,
            "text": sentence,
            "unit_type": "span_mention",
            "annotator_id": annotator_id,
            "gold_variant": "unknown",
        }

        for term in terms:
            rows.append({
                **base,
                "surface": term["surface"],
                "norm": norm(term["surface"]),
                "start": term["start"],
                "end": term["end"],
                "class_label": term["class_label"],
            })

        if not terms:
            # Zero-term sentence: emit one sentence-level negative placeholder row
            # so the sentence survives conversion instead of contributing nothing.
            rows.append({
                **base,
                "surface": None,
                "norm": None,
                "start": None,
                "end": None,
                "class_label": None,
            })

    return rows


def get_acl_id(xml_file):
    root = ET.parse(xml_file).getroot()
    return root.get("acl-id", Path(xml_file).stem)


def collect_files(annotations_root):
    """Read each input layer in sorted order for deterministic output ordering."""
    annotations_root = Path(annotations_root)
    file_annotator = {}
    by_doc = {}
    for number in (1, 2):
        annotator_id = f"ann{number}"
        directory = annotations_root / f"annotator{number}"
        for path in sorted(directory.glob("*/*.xml")):
            acl_id = get_acl_id(path)
            file_annotator[path] = annotator_id
            by_doc.setdefault(acl_id, []).append(path)
    return file_annotator, by_doc



def convert(annotations_root, train_ratio, seed):
    file_annotator, by_doc = collect_files(annotations_root)

    # Deterministic document-level split so annotator layers of the same doc
    # stay on the same side of the train/test boundary.
    doc_ids = sorted(by_doc)
    random.Random(seed).shuffle(doc_ids)
    n_train = int(len(doc_ids) * train_ratio)
    train_ids = set(doc_ids[:n_train])

    rows = []
    for acl_id, group_files in by_doc.items():
        split = "train" if acl_id in train_ids else "test"
        for path in group_files:
            rows.extend(convert_xml(path, split, file_annotator[path]))
    return rows
