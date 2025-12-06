import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import splu
import pickle
import sys


# ----------------------------------------------------------------------
# Load symbol → Ensembl dictionary
# ----------------------------------------------------------------------
def load_symbol_dict(path):
    print(f"[INFO] Loading symbol→Ensembl mapping: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)

    # Basic sanity check
    if len(mapping) < 100:
        print("[WARN] Mapping file seems very small. Check that this is correct.")
    return mapping


# ----------------------------------------------------------------------
# Build Laplacian from GRN edgelist using Ensembl IDs
# ----------------------------------------------------------------------
def load_laplacian(grn_csv, gene_order_ensembl):
    print(f"[INFO] Loading GRN from: {grn_csv}")
    df = pd.read_csv(grn_csv)

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("GRN CSV must contain columns: source, target [, weight]")

    if "weight" not in df:
        df["weight"] = 1.0

    # Filter edges to genes present in dataset
    df = df[df["source"].isin(gene_order_ensembl) & df["target"].isin(gene_order_ensembl)]

    used_genes = sorted(set(df["source"]) | set(df["target"]))
    print(f"[INFO] GRN edges originally: {len(df)}")
    print(f"[INFO] GRN overlapping genes with dataset: {len(used_genes)}")

    if len(used_genes) == 0:
        print("[ERROR] No overlap between GRN genes and prediction genes.")
        return None, [], []

    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    weights = df["weight"].astype(np.float32).values

    n = len(used_genes)
    A = sparse.coo_matrix((weights, (iu, iv)), shape=(n, n)).tocsr()

    # Symmetrize adjacency
    A = A.maximum(A.T)

    # Build normalized Laplacian
    deg = np.asarray(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Map used_genes back into full gene order
    used_idx = [gene_order_ensembl.index(g) for g in used_genes]

    return L.tocsr(), used_idx, used_genes


# ----------------------------------------------------------------------
# Precompute the linear solver for (I + τ L)
# ----------------------------------------------------------------------
def precompute_solver(L, tau):
    print("[INFO] Precomputing LU factorization for (I + tau * L)...")
    I = sparse.eye(L.shape[0], format="csr")
    M = (I + tau * L).tocsc()
    solver = splu(M)
    return solver


# ----------------------------------------------------------------------
# Apply smoothing to vector using precomputed solver
# ----------------------------------------------------------------------
def apply_smoothing(vec, solver, used_idx):
    subvec = vec[used_idx]
    smoothed = solver.solve(subvec)
    out = vec.copy()
    out[used_idx] = smoothed
    return out


# ----------------------------------------------------------------------
# Main smoothing logic
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Apply GRN Laplacian smoothing to prediction h5ad.")
    parser.add_argument("--pred-h5ad", required=True, help="Prediction h5ad file")
    parser.add_argument("--grn-csv", required=True, help="GRN edgelist CSV (Ensembl IDs)")
    parser.add_argument("--symbols-pkl", required=True, help="symbol → Ensembl mapping")
    parser.add_argument("--tau", type=float, default=0.1, help="Diffusion strength τ")
    parser.add_argument("--out", required=True, help="Output .h5ad file path")
    args = parser.parse_args()

    print("=" * 60)
    print("[INFO] Loading prediction h5ad:", args.pred_h5ad)
    adata = sc.read_h5ad(args.pred_h5ad)
    print(f"[INFO] Loaded predictions with shape: {adata.shape}")
    print("=" * 60)

    # --------------------------------------------------------
    # Map var_names (gene symbols) → Ensembl IDs
    # --------------------------------------------------------
    mapping = load_symbol_dict(args.symbols_pkl)

    var_symbols = list(adata.var_names)
    gene_order_ensembl = [mapping.get(sym, None) for sym in var_symbols]

    # Check mapping coverage
    n_mapped = sum(g is not None for g in gene_order_ensembl)
    print(f"[INFO] Successfully mapped {n_mapped}/{len(var_symbols)} genes to Ensembl IDs.")

    if n_mapped == 0:
        print("[ERROR] No genes were mapped to Ensembl IDs. Smoothing cannot proceed.")
        sys.exit(1)

    # Replace unmapped genes with None (they will be ignored)
    gene_order_ensembl = np.array(gene_order_ensembl, dtype=object)

    # --------------------------------------------------------
    # Build Laplacian from GRN
    # --------------------------------------------------------
    L, used_idx, used_genes = load_laplacian(args.grn_csv, list(gene_order_ensembl))

    if L is None or len(used_idx) == 0:
        print("[ERROR] No overlapping genes after mapping. Exiting.")
        adata.write(args.out)
        return

    print(f"[INFO] Number of genes smoothed: {len(used_idx)}")
    print("=" * 60)

    # --------------------------------------------------------
    # Build solver
    # --------------------------------------------------------
    solver = precompute_solver(L, args.tau)

    # --------------------------------------------------------
    # Apply smoothing
    # --------------------------------------------------------
    X = adata.X
    if sparse.issparse(X):
        print("[INFO] Converting sparse matrix → dense")
        X = X.toarray()

    print("[INFO] Applying smoothing row-by-row...")
    n = X.shape[0]

    for i in range(n):
        X[i] = apply_smoothing(X[i], solver, used_idx)
        if i % 500 == 0:
            print(f"  Smoothed {i}/{n}")

    adata.X = X

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    print(f"[INFO] Writing output to: {args.out}")
    adata.write(args.out)
    print("[INFO] Done. Smoothing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
