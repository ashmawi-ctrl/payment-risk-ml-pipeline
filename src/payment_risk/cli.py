import argparse
import json
from pathlib import Path

from .data import DatasetError, load_dataset
from .training import save_artifacts, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="payment-risk",
        description="Train and inspect a baseline payment-risk classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("csv", type=Path)
    train.add_argument("--target", default="risky")
    train.add_argument(
        "--drop-column",
        action="append",
        default=[],
        dest="drop_columns",
        help="Exclude an identifier or leakage-prone column. Repeat as needed.",
    )
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        try:
            dataset = load_dataset(
                args.csv,
                target_column=args.target,
                drop_columns=tuple(args.drop_columns),
            )
            model, result = train_baseline(
                dataset,
                test_size=args.test_size,
                random_state=args.random_state,
                threshold=args.threshold,
            )
        except (DatasetError, ValueError) as exc:
            print(f"Training input error: {exc}")
            return 2

        save_artifacts(model, result, args.output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "train_rows": result.train_rows,
                    "test_rows": result.test_rows,
                    "positive_rate_train": result.positive_rate_train,
                    "positive_rate_test": result.positive_rate_test,
                    "metrics": result.metrics,
                },
                indent=2,
            )
        )
        return 0

    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
