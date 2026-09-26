import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)

    amount = np.round(rng.lognormal(mean=5.2, sigma=1.0, size=rows), 2)
    attempts_10m = rng.poisson(lam=1.3, size=rows) + 1
    account_age_days = rng.integers(1, 2200, size=rows)
    hour = rng.integers(0, 24, size=rows)

    channel = rng.choice(
        ["web", "mobile", "pos"],
        size=rows,
        p=[0.42, 0.43, 0.15],
    )
    card_country_match = rng.choice(
        ["yes", "no"],
        size=rows,
        p=[0.91, 0.09],
    )
    three_ds = rng.choice(
        ["authenticated", "frictionless", "not_used"],
        size=rows,
        p=[0.35, 0.38, 0.27],
    )

    logit = (
        -4.2
        + 0.0025 * np.minimum(amount, 2500)
        + 0.55 * np.maximum(attempts_10m - 2, 0)
        + 0.9 * (card_country_match == "no")
        + 0.65 * (account_age_days < 14)
        + 0.45 * ((hour <= 4) | (hour >= 23))
        + 0.35 * (three_ds == "not_used")
    )
    probability = 1 / (1 + np.exp(-logit))
    risky = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "transaction_id": [
                f"demo_txn_{index:06d}" for index in range(rows)
            ],
            "amount": amount,
            "attempts_10m": attempts_10m,
            "account_age_days": account_age_days,
            "hour": hour,
            "channel": channel,
            "card_country_match": card_country_match,
            "three_ds": three_ds,
            "risky": risky,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = generate(args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(
        f"wrote {len(frame)} rows to {args.output} "
        f"(positive rate={frame['risky'].mean():.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
