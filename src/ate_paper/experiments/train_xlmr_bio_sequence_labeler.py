#!/usr/bin/env python3
"""XLM-R BIO sequence-labeling baseline.

This is the encoder-control counterpart to the mBERT BIO baseline. It reuses
the same BIO token-classification pipeline, flattening policy, threshold tuning,
and output schema, but defaults to an XLM-R encoder. It is intentionally not the
nested-aware span classifier.
"""

from ate_paper.experiments import train_mbert_sequence_labeler as bio

DEFAULT_XLMR_MODEL = "xlm-roberta-base"


def build_arg_parser():
    parser = bio.build_arg_parser()
    parser.set_defaults(
        model_name=DEFAULT_XLMR_MODEL,
        report="build/experiments/supervised/reports/xlmr_bio_sequence_labeler_report.md",
        report_title="XLM-R BIO Sequence Labeler Report",
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
