#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import cg
import pickle
import sys


# ----------------------------------------------------------------------
# Load symbol → Ensembl dictionary
# ----------------------------------------------------------------------
def load_symbol_dict(path):
    print(f"[INFO] Loading symbol→Ensembl mapping: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    print(f"[INFO] Mapping contains {len(mapping)} entries.")
    return mapping


# ----------------------------------------------------------------------
# Build Laplacian from GRN edgelist using Ensembl IDs
# ----------------------------------------------------------------------
def load_laplacian(grn_csv, gene_order_ensembl):
    print(f"[INFO] Loading GRN CSV: {grn_csv}")
    df = pd.read_csv(grn_csv)

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("GRN CSV must contain columns: source,target[,weight]")

    if "weight" not in df:
        df["weight"] = 1.0

    # Filter edges to genes present in dataset
    df = df[df["source"].isin(gene_order_ensembl) & df["target"].isin(gene_order_ensembl)]
    used_genes = sorted(set(df["source"]) | set(df["target"]))

    print(f"[INFO] GRN edges after filtering: {len(df)}")
    print(f"[INFO] Overlapping genes: {len(used_genes)}")

    if len(used_genes) == 0:
        return None, [], []

    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    weights = df["weight"].astype(np.float32).values

    n = len(used_genes)
    A = sparse.coo_matrix((weights, (iu, iv)), shape=(n, n)).tocsr()

    # Symmetrize adjacency
    A = A.maximum(A.T)

    # Normalized Laplacian
    deg = np.asarray(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Map used genes to global index
    used_idx = [gene_order_ensembl.index(g) for g in used_genes]

    return L.tocsr(), used_idx, used_genes


# ----------------------------------------------------------------------
# Apply smoothing using conjugate gradient
# ----------------------------------------------------------------------
def apply_smoothing(vec, M, used_idx):
    """Solve (I + tau L) y = x using conjugate gradient."""
    b = vec[used_idx]

    y, info = cg(M, b, tol=1e-4, maxiter=500)

    if info != 0:
        print(f"[WARN] CG did not converge (info={info})")

    out = vec.copy()
    out[used_idx] = y
    return out


# ----------------------------------------------------------------------
# Main smoothing pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Apply GRN Laplacian smoothing (CG-based).")
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--grn-csv", required=True)
    parser.add_argument("--symbols-pkl", required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("[INFO] Loading predictions:", args.pred_h5ad)
    ad = sc.read_h5ad(args.pred_h5ad)
    print(f"[INFO] Prediction shape: {ad.shape}")
    print("=" * 80)

    # ------------------- Load symbol→Ensembl -------------------
    mapping = load_symbol_dict(args.symbols_pkl)

    var_symbols = list(ad.var_names)
    gene_order_ensembl = [mapping.get(gs, None) for gs in var_symbols]

    mapped = sum(g is not None for g in gene_order_ensembl)
    print(f"[INFO] Successfully mapped {mapped}/{len(var_symbols)} genes to Ensembl IDs.")

    if mapped == 0:
        print("[ERROR] No genes mapped. Check mapping file.")
        sys.exit(1)

    # Replace None with placeholder string (ignored later)
    gene_order_ensembl = np.array([
        g if g is not None else "___UNMAPPED___"
        for g in gene_order_ensembl
    ], dtype=object)

    # ------------------- Build Laplacian -------------------
    L, used_idx, used_genes = load_laplacian(args.grn_csv, list(gene_order_ensembl))

    if L is None or len(used_idx) == 0:
        print("[ERROR] No overlapping genes found between GRN and predictions.")
        ad.write(args.out)
        return

    print(f"[INFO] Laplacian size: {L.shape}")
    print(f"[INFO] Number of genes smoothed: {len(used_idx)}")

    # ------------------- Construct (I + tau L) -------------------
    print("[INFO] Constructing operator M = I + tau*L ...")
    I = sparse.eye(L.shape[0], format="csr")
    M = (I + args.tau * L).tocsr()

    # ------------------- Apply smoothing -------------------
    X = ad.X
    if sparse.issparse(X):
        print("[INFO] Converting sparse matrix → dense matrix")
        X = X.toarray()

    print(f"[INFO] Starting smoothing on {X.shape[0]} cells...")

    for i in range(X.shape[0]):
        X[i] = apply_smoothing(X[i], M, used_idx)
        if i % 200 == 0:
            print(f"  Smoothed {i}/{X.shape[0]}")

    print("[INFO] Smoothing complete.")
    ad.X = X

    # ------------------- Save output -------------------
    print("[INFO] Writing smoothed output to:", args.out)
    ad.write(args.out)

    print("[INFO] Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
