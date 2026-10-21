#!/usr/bin/env python3
"""SciBERT BIO sequence-labeling baseline.

Second domain-adapted encoder control for GENIA (broad scientific vocabulary).
Identical BIO pipeline to the mBERT/XLM-R/PubMedBERT BIO baselines, but defaults to
SciBERT. Paired with ``scibert_span`` it provides a second domain encoder for the
GENIA outer<->class decoder-reversal test, to check whether the dissolution seen at
PubMedBERT replicates or is graded by domain-match strength. It is intentionally not
the nested-aware span classifier.
"""

from ate_paper.experiments import train_mbert_sequence_labeler as bio

DEFAULT_SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"


def build_arg_parser():
    parser = bio.build_arg_parser()
    parser.set_defaults(
        model_name=DEFAULT_SCIBERT_MODEL,
        report="build/experiments/supervised/reports/scibert_bio_sequence_labeler_report.md",
        report_title="SciBERT BIO Sequence Labeler Report",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.long_span_token_threshold < 1:
        raise ValueError("--long-span-token-threshold must be >= 1.")
    if args.long_positive_weight < 1:
        raise ValueError("--long-positive-weight must be >= 1.")

    report = bio.train_and_predict(args)
    print(f"Saved predictions: {args.output}")
    print(f"Saved report: {args.report}")
    print(f"Prediction rows written: {report['prediction_rows_written']}")


if __name__ == "__main__":
    main()
