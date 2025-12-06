import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import splu

def load_laplacian(csv_path, gene_order):
    df = pd.read_csv(csv_path)

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: source,target[,weight]")

    if "weight" not in df:
        df["weight"] = 1.0

    # Filter edges to genes in gene_order
    df = df[df["source"].isin(gene_order) & df["target"].isin(gene_order)]
    used_genes = sorted(set(df["source"]) | set(df["target"]))

    gene_to_idx = {g: i for i, g in enumerate(used_genes)}
    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    w = df["weight"].astype(np.float32).values

    n = len(used_genes)
    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n)).tocsr()

    # symmetrize
    A = A.maximum(A.T)

    # normalizable Laplacian
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # return laplacian + index mapping
    used_idx = [gene_order.index(g) for g in used_genes]
    return L.tocsr(), used_idx, used_genes

def precompute_solver(L, tau):
    """Compute factorization of (I + τ L)."""
    I = sparse.eye(L.shape[0], format="csr")
    M = (I + tau * L).tocsc()
    solver = splu(M)
    return solver

def apply_smoothing_to_vector(vec, solver, used_idx):
    sub = vec[used_idx]                   # extract subvector
    smoothed = solver.solve(sub)          # solve system
    out = vec.copy()
    out[used_idx] = smoothed              # reinsert smoothed portion
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--grn-csv", required=True, default="edges_all.csv")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading predictions:", args.pred_h5ad)
    ad = sc.read_h5ad(args.pred_h5ad)

    gene_order = list(ad.var_names)
    print("Number of genes in prediction:", len(gene_order))

    # Load GRN Laplacian
    print("Loading GRN:", args.grn_csv)
    L, used_idx, used_genes = load_laplacian(args.grn_csv, gene_order)
    print(f"GRN contains {len(used_genes)} overlapping genes with dataset.")

    print("Precomputing solver...")
    solver = precompute_solver(L, tau=args.tau)

    X = ad.X.copy()
    print("Applying smoothing...")

    for i in range(X.shape[0]):
        X[i] = apply_smoothing_to_vector(X[i], solver, used_idx)
        if i % 100 == 0:
            print(f"Smoothed {i}/{X.shape[0]}")

    ad.X = X

    print("Saving smoothed predictions:", args.out)
    ad.write(args.out)
    print("Done.")

if __name__ == "__main__":
    main()
