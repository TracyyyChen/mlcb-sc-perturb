#!/usr/bin/env python3

import argparse
import pandas as pd
import os

def aggregate_with_filter(csv_path: str, min_sig_count: int = 100):
    df = pd.read_csv(csv_path)

    # Apply filter
    df = df[df["de_nsig_counts_real"] >= min_sig_count]

    # Drop perturbation column (required by cell-eval)
    df = df.drop(columns=["perturbation"])

    # Same as Polars describe(): gives mean, std, min, max, median...
    agg = df.describe(include='all')

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-perturbation cell-eval metrics (with filtering)."
    )

    parser.add_argument("--input", required=True,
                        help="Path to results.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for agg_results_new.csv")
    parser.add_argument("--min-sig-count", type=int, default=100,
                        help="Minimum de_nsig_counts_real threshold (default: 100)")

    args = parser.parse_args()

    agg = aggregate_with_filter(args.input, args.min_sig_count)
    agg.to_csv(args.output, index=True)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
