#!/usr/bin/env python3
"""PubMedBERT BIO sequence-labeling baseline.

Domain-adapted encoder control for GENIA. Identical BIO pipeline to the mBERT and
XLM-R BIO baselines, but defaults to a biomedical-pretrained encoder
(microsoft/BiomedNLP-BiomedBERT, a.k.a. PubMedBERT). Paired with
``pubmedbert_span`` it completes the GENIA encoder x decoder 2x2: it tests whether
the GENIA outer<->class decoder reversal (flat-BIO leads outer recall, nested-span
leads class-aware) holds at a domain-adapted encoder, i.e. whether that reversal is
structural to the flat-BIO decoder rather than an encoder artifact. It is
intentionally not the nested-aware span classifier.
"""

from ate_paper.experiments import train_mbert_sequence_labeler as bio

DEFAULT_PUBMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def build_arg_parser():
    parser = bio.build_arg_parser()
    parser.set_defaults(
        model_name=DEFAULT_PUBMEDBERT_MODEL,
        report="build/experiments/supervised/reports/pubmedbert_bio_sequence_labeler_report.md",
        report_title="PubMedBERT BIO Sequence Labeler Report",
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
