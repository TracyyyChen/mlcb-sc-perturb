#!/usr/bin/env python3
"""
Run GENECI GRN inference using selectable methods
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import pandas as pd
from geneci import InferNetwork



def get_args():
    parser = argparse.ArgumentParser(description="GRN inference using GENECI")

    parser.add_argument("--input", required=True,
                        help="Path to .h5ad expression file")
    parser.add_argument("--outdir", default="geneci_grns",
                        help="Directory to store generated GRNs")

    parser.add_argument("--methods", nargs="+", required=True,
                        help="List of GRN methods to run. "
                             "Examples: genie3 clr pidc grnboost2")

    parser.add_argument("--ncores", type=int, default=4,
                        help="Number of CPU cores to use")

    parser.add_argument("--layer", default=None,
                        help="AnnData layer to use (default: X)")

    parser.add_argument("--max_genes", type=int, default=None,
                        help="Optional: subset top variable genes")

    return parser.parse_args()


# -------------------------------------------------------------------
# Load AnnData and preprocess
# -------------------------------------------------------------------

def load_expression(path, layer=None, max_genes=None):
    print(f"Reading AnnData: {path}")
    adata = ad.read_h5ad(path)

    # extract expression matrix
    X = adata.layers[layer] if layer else adata.X

    # convert to dense
    if not hasattr(X, "toarray"):
        arr = X
    else:
        arr = X.toarray()

    df = pd.DataFrame(arr, columns=adata.var_names)

    # optional: top variable genes
    if max_genes is not None:
        variances = df.var().sort_values(ascending=False)
        keep = variances.index[:max_genes]
        df = df[keep]
        print(f"Subsetting to {len(keep)} genes (top variable)")

    print(f"Final matrix: {df.shape[0]} cells × {df.shape[1]} genes")
    return df


# -------------------------------------------------------------------
# Run GENECI for a single method
# -------------------------------------------------------------------

def run_geneci_method(df, method, outdir, ncores):
    print(f"\n=== Running GRN inference method: {method.upper()} ===")

    infer = InferNetwork(
        df=df,
        method=method,
        n_cores=ncores,
        verbose=True,
    )

    edges = infer.build_grn()
    edges = infer.edges_  # GENECI attribute: DataFrame [source, target, weight]

    outfile = Path(outdir) / f"grn_{method}.csv"
    edges.to_csv(outfile, index=False)
    print(f"[DONE] Saved GRN → {outfile} with {len(edges)} edges")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = get_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # load matrix
    df = load_expression(args.input, layer=args.layer, max_genes=args.max_genes)

    # loop: run each GRN method
    for method in args.methods:
        run_geneci_method(
            df=df,
            method=method.lower(),
            outdir=outdir,
            ncores=args.ncores
        )

    print("\nGRN inference completed successfully.")


if __name__ == "__main__":
    main()
