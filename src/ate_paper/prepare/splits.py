"""The paper's document holdout: seed 42, rounded 10% annotation-row target.

Negative placeholders count as rows. Whole documents are added until the target
is reached or exceeded. Adapted from prepare_mbert_train_dev_splits.py.
"""
import random
from collections import defaultdict


def choose_dev_docs(rows, dev_frac, seed):
    by_doc = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(docs)
    target_rows = max(1, round(len(rows) * dev_frac))
    dev_docs = set()
    dev_rows = 0
    for doc_id in docs:
        if dev_rows >= target_rows and dev_docs:
            break
        dev_docs.add(doc_id)
        dev_rows += len(by_doc[doc_id])
    return dev_docs


