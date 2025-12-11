#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np

# Metric registry mapping (ONE, ZERO, NONE)

BEST_ZERO = "ZERO"
BEST_ONE = "ONE"
BEST_NONE = "NONE"

metric_best_value = {}


best_one_list = [
    "pearson_delta", "precision_at_N", "precision_at_50", "precision_at_100",
    "precision_at_200", "precision_at_500", "overlap_at_N", "overlap_at_50",
    "overlap_at_100", "overlap_at_200", "overlap_at_500", "de_spearman_sig",
    "de_direction_match", "de_spearman_lfc_sig", "de_sig_genes_recall",
    "pr_auc", "roc_auc", "discrimination_score_l1", "discrimination_score_l2",
    "discrimination_score_cosine", "pearson_edistance", "clustering_agreement"
]

best_zero_list = ["mse", "mae", "mse_delta", "mae_delta"]

best_none_list = ["de_nsig_counts_real", "de_nsig_counts_pred"]

for m in best_one_list:
    metric_best_value[m] = BEST_ONE
for m in best_zero_list:
    metric_best_value[m] = BEST_ZERO
for m in best_none_list:
    metric_best_value[m] = BEST_NONE


# Filtering
def load_and_filter(csv_path: str, min_sig_count: int):
    df = pd.read_csv(csv_path)
    df_filtered = df[df["de_nsig_counts_real"] >= min_sig_count]
    return df_filtered


# Aggregation
def aggregate_metrics(df: pd.DataFrame):
    numeric_df = df.drop(columns=["perturbation"])
    means = numeric_df.mean(axis=0)
    return means


# Normalization
def norm_by_zero(user, base):
    """Lower is better. Improvement = (base - user) / base."""
    out = (base - user) / base
    return np.maximum(out, 0)


def norm_by_one(user, base):
    """Higher is better. Improvement = (user - base) / (1 - base)."""
    out = (user - base) / (1 - base)
    return np.maximum(out, 0)


# Scoring routine
def score_agg_metrics(agg_user: pd.Series, agg_base: pd.Series):
    metrics = agg_user.index.values
    scores = []

    for m in metrics:
        if m not in metric_best_value:
            continue  # unknown metric

        best = metric_best_value[m]

        if best == BEST_NONE:
            continue  # ignored metric

        u = agg_user[m]
        b = agg_base[m]

        if best == BEST_ZERO:
            s = norm_by_zero(u, b)
        elif best == BEST_ONE:
            s = norm_by_one(u, b)
        else:
            continue

        if np.isnan(s):
            s = 0.0

        scores.append((m, s))

    df_scores = pd.DataFrame(scores, columns=["metric", "from_baseline"])

    # Add final average score
    avg = df_scores["from_baseline"].mean()
    df_scores = pd.concat([
        df_scores,
        pd.DataFrame([["avg_score", avg]], columns=["metric", "from_baseline"])
    ])

    return df_scores

def run_custom_evaluation(user_csv, base_csv, min_sig_count, output_csv=None):

    df_user = load_and_filter(user_csv, min_sig_count)
    df_base = load_and_filter(base_csv, min_sig_count)

    agg_user = aggregate_metrics(df_user)
    agg_base = aggregate_metrics(df_base)

    scores = score_agg_metrics(agg_user, agg_base)

    if output_csv:
        scores.to_csv(output_csv, index=False)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Custom CELL-EVAL scoring with perturbation filtering."
    )

    parser.add_argument(
        "--user", required=True, help="CSV file with user model per-perturbation metrics"
    )

    parser.add_argument(
        "--base", required=True, help="CSV file with baseline per-perturbation metrics"
    )

    parser.add_argument(
        "--output", required=False, default=None,
        help="Output CSV file for aggregated scores"
    )

    parser.add_argument(
        "--min-sig-count", type=int, default=100,
        help="Minimum de_nsig_counts_real required to include a perturbation"
    )

    args = parser.parse_args()

    scores = run_custom_evaluation(
        args.user,
        args.base,
        args.min_sig_count,
        args.output,
    )

    print("\n=== Final Scores ===")
    print(scores)


if __name__ == "__main__":
    main()
