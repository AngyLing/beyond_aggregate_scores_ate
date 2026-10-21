"""GENIA 3.02 conversion for the paper: span coordination, drop fragments, 36 labels.

Adapted from the study's scripts/convert/genia_converter.py. The XML walker
recurses through unclassified wrappers to retain classified descendants.
"""
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter


def norm(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def clean_class(sem):
    """Strip the ontology prefix: 'G#protein_molecule' -> 'protein_molecule'."""
    return sem[2:] if sem.startswith("G#") else sem


def coordination_op(sem):
    m = re.match(r"\(\s*([A-Z_/]+)", sem.strip())
    return m.group(1) if m else "?"


def inner_classes(sem):
    """Classes inside a coordinated sem, cleaned and de-duplicated in order."""
    out = []
    for raw in re.findall(r"G#\S+?(?=[\s)])", sem):
        c = clean_class(raw)
        if c not in out:
            out.append(c)
    return out


def coordinated_class_label(sem):
    classes = inner_classes(sem)
    if len(classes) == 1:
        return classes[0]
    if classes:
        return "|".join(classes)  # never happens in v3.02 (classes are homogeneous)
    return clean_class(sem)


def is_coordinated(sem):
    return sem is not None and sem.strip().startswith("(")


def medline_id(article):
    info = article.find("articleinfo")
    if info is not None:
        bib = info.findtext("bibliomisc")
        if bib:
            return bib.strip()
    return None


def sentence_spans(sentence):
    """Return (text, spans) where each span is (cons_elem, start, end, depth)."""
    parts = []
    pos = [0]
    spans = []

    def add(text):
        if text:
            parts.append(text)
            pos[0] += len(text)

    def walk(elem, depth):
        add(elem.text)
        for child in elem:
            if child.tag == "cons":
                start = pos[0]
                walk(child, depth + 1)
                end = pos[0]
                spans.append((child, start, end, depth))
            else:
                walk(child, depth)  # defensively consume any other element's text
            add(child.tail)

    walk(sentence, 0)
    return "".join(parts), spans


def convert(xml_path, train_ratio=0.8, seed=42):
    root = ET.parse(xml_path).getroot()
    articles = root.findall("article")
    seen = Counter()
    doc_ids = []
    for index, article in enumerate(articles):
        source_id = medline_id(article) or f"article{index}"
        seen[source_id] += 1
        doc_ids.append(source_id if seen[source_id] == 1
                       else f"{source_id}#{seen[source_id]}")

    # Repeated source IDs identify separate XML articles. Disambiguate their
    # identities before splitting; identical text can still occur across splits.
    shuffled = sorted(doc_ids)
    random.Random(seed).shuffle(shuffled)
    train_ids = set(shuffled[:int(len(shuffled) * train_ratio)])
    rows = []
    for article, doc_id in zip(articles, doc_ids):
        split = "train" if doc_id in train_ids else "test"
        for section in ("title", "abstract"):
            element = article.find(section)
            if element is None:
                continue
            for index, sentence in enumerate(element.findall("sentence"), start=1):
                sent_id = f"{doc_id}_{section[0]}_s{index:04d}"
                text, spans = sentence_spans(sentence)
                unit = {
                    "dataset": "GENIA", "version": "term_v3.02", "split": split,
                    "language": "English", "domain": "biomedical",
                    "doc_id": doc_id, "sent_id": sent_id, "text": text,
                    "unit_type": "span_mention",
                }
                before = len(rows)
                for constituent, start, end, depth in spans:
                    sem = constituent.get("sem")
                    if sem is None:
                        continue
                    surface = text[start:end]
                    row = {**unit, "surface": surface, "norm": norm(surface),
                           "start": start, "end": end,
                           "lex": constituent.get("lex")}
                    if is_coordinated(sem):
                        rows.append({**row, "class_label": coordinated_class_label(sem),
                                     "is_coordination": True,
                                     "coordination_op": coordination_op(sem)})
                    else:
                        rows.append({**row, "class_label": clean_class(sem)})
                if len(rows) == before:
                    rows.append({**unit, "surface": None, "norm": None,
                                 "start": None, "end": None, "lex": None,
                                 "class_label": None})
    return rows
