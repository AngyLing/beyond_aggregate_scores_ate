#!/usr/bin/env python3
"""RuBERT BIO sequence-labeling baseline.

Domain/language-adapted encoder control for CL-RuTerm3. It reuses the same BIO
token-classification pipeline, flattening policy, threshold tuning, and output
schema as the mBERT and XLM-R BIO baselines, but defaults to a Russian-pretrained
encoder. Its purpose is to separate the encoder's language/domain fit from the
output format: whether CL-RuTerm3 results (in particular the LLM-vs-supervised
strict<->class reversal) are sensitive to the general-domain multilingual encoder
rather than only to the flat-BIO decoder. It is intentionally not the
nested-aware span classifier.
"""

from ate_paper.experiments import train_mbert_sequence_labeler as bio

DEFAULT_RUBERT_MODEL = "DeepPavlov/rubert-base-cased"


def build_arg_parser():
    parser = bio.build_arg_parser()
    parser.set_defaults(
        model_name=DEFAULT_RUBERT_MODEL,
        report="build/experiments/supervised/reports/rubert_bio_sequence_labeler_report.md",
        report_title="RuBERT BIO Sequence Labeler Report",
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
