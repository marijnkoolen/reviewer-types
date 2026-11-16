"""Export the gt_v2 training table as one self-contained file for training on another machine (the A10 VMs, UA-025).

usage: python3.11 src/classifier/gt_v2_export.py
output: data/gt_v2/train_input.tsv.gz: one row per sentence, in review order, with the columns gt_v2_data.load()
        returns (review_id, sent_id, sent_text, labels, set, stratum, split, header, pos, n_sent, gold). With this
        file present and data/gt_v2/labels.tsv absent, gt_v2_data.load() reads it instead, so the classifier scripts
        run without metadata/, the gold file or the split file. Review text: keep it off public places.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_v2_data import EXPORT, load  # noqa: E402

d = load()
cols = ['review_id', 'sent_id', 'sent_text', 'labels', 'set', 'stratum', 'split', 'header', 'pos', 'n_sent', 'gold']
d[cols].to_csv(EXPORT, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print(EXPORT, len(d), 'sentences,', d.review_id.nunique(), 'reviews;', d.split.value_counts().to_dict())
