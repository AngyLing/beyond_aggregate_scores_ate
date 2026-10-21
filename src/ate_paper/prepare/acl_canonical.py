"""Select one original annotation layer per ACL document, preferring ann1.
"""
from collections import Counter, defaultdict


def get_annotator(row):
    return row.get("annotator_id") or "unknown"


def make_canonical(rows, preferred_annotator):
    by_doc = defaultdict(list)

    for row in rows:
        if row.get("dataset") == "ACL RD-TEC 2.0":
            by_doc[row["doc_id"]].append(row)

    canonical = []
    chosen_stats = Counter()

    for doc_id, doc_rows in by_doc.items():
        by_annotator = defaultdict(list)

        for row in doc_rows:
            by_annotator[get_annotator(row)].append(row)

        if preferred_annotator in by_annotator:
            chosen = preferred_annotator
        else:
            chosen = max(by_annotator, key=lambda a: len(by_annotator[a]))

        chosen_stats[chosen] += 1

        for row in by_annotator[chosen]:
            row = dict(row)
            row["gold_variant"] = "canonical"
            row["canonical_annotator_id"] = chosen
            canonical.append(row)

    return canonical, chosen_stats
