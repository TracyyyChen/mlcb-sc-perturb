#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch


# ================================================================
# GPU Conjugate Gradient Solver (for A x = b)
# ================================================================
def cg_solve_gpu(L_gpu, b, tau=0.1, tol=1e-3, max_iter=200):
    """
    Solve (I + τL) x = b using conjugate gradient on GPU.
    L_gpu: sparse CSR tensor (n, n)
    b: dense vector (n,)
    """
    device = b.device
    n = b.numel()

    # A(x) = x + tau * (L @ x)
    def matvec(x):
        return x + tau * torch.matmul(L_gpu, x)

    x = torch.zeros_like(b)
    r = b.clone()            # r0 = b - A(0)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return x


# ================================================================
# Load Laplacian from Parquet
# ================================================================
def load_laplacian_from_parquet(path, gene_order):
    print(f"[INFO] Reading GRN parquet: {path}")
    df = pd.read_parquet(path)

    required = {"source", "target", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Parquet must contain columns: {required}")

    # Keep only edges whose genes appear in the dataset
    df = df[df["source"].isin(gene_order) & df["target"].isin(gene_order)]
    print(f"[INFO] GRN filtered edges: {len(df)}")

    used_genes = sorted(set(df["source"]) | set(df["target"]))
    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    w = df["weight"].astype(np.float32).values

    n = len(used_genes)
    print(f"[INFO] GRN overlap size: {n}")

    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)

    deg = np.array(A.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / np.sqrt(deg + 1e-12))

    L = sparse.eye(n) - D_inv @ A @ D_inv
    L = L.tocsr()

    # Convert to PyTorch sparse CSR for GPU matmul
    L_gpu = torch.sparse_csr_tensor(
        torch.tensor(L.indptr, dtype=torch.int64),
        torch.tensor(L.indices, dtype=torch.int64),
        torch.tensor(L.data, dtype=torch.float32),
        size=L.shape,
    ).cuda()

    # Map used genes back to full matrix indices
    used_idx = np.array([gene_order.index(g) for g in used_genes], dtype=int)

    return L_gpu, used_idx, used_genes


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--grn-parquet", required=True)
    parser.add_argument("--symbols-dict", required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load prediction matrix
    # ---------------------------------------------------------
    print(f"[INFO] Loading predictions: {args.pred_h5ad}")
    ad = sc.read_h5ad(args.pred_h5ad)
    X_original = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    n_cells, n_genes = X_original.shape

    print(f"[INFO] Prediction shape: {X_original.shape}")

    # ---------------------------------------------------------
    # Load gene→Ensembl mapping
    # ---------------------------------------------------------
    print(f"[INFO] Loading symbol→Ensembl mapping: {args.symbols_dict}")
    import pickle
    with open(args.symbols_dict, "rb") as f:
        mapping = pickle.load(f)

    gene_order = [mapping.get(g, None) for g in ad.var_names]
    mask_mapped = np.array([g is not None for g in gene_order])
    gene_order = np.array(gene_order)[mask_mapped].tolist()

    # Create reduced matrix only for mapped genes
    X_mapped = X_original[:, mask_mapped]
    print(f"[INFO] Mapped genes: {len(gene_order)}")

    # ---------------------------------------------------------
    # Load Laplacian
    # ---------------------------------------------------------
    L_gpu, used_idx_local, used_genes = load_laplacian_from_parquet(
        args.grn_parquet, gene_order
    )

    print(f"[INFO] Laplacian shape: {L_gpu.shape}")
    print(f"[INFO] Number of GRN genes: {len(used_idx_local)}")

    # X_grn: matrix of only GRN genes
    X_grn = X_mapped[:, used_idx_local].copy()

    # ---------------------------------------------------------
    # How many rows to smooth
    # ---------------------------------------------------------
    rows = X_grn.shape[0] if args.rows is None else args.rows
    print(f"[INFO] Smoothing {rows} rows...")

    smoothed = np.zeros_like(X_grn[:rows])

    # ---------------------------------------------------------
    # Run smoothing on GPU
    # ---------------------------------------------------------
    for i in range(rows):
        v = torch.tensor(X_grn[i], dtype=torch.float32, device="cuda")
        x_smooth = cg_solve_gpu(L_gpu, v, tau=args.tau)
        smoothed[i] = x_smooth.cpu().numpy()

        if i % 1000 == 0:
            print(f"  [GPU CG] Smoothed {i}/{rows}")

    # Insert smoothed rows
    X_grn[:rows] = smoothed

    # ---------------------------------------------------------
    # Reconstruct full matrix
    # ---------------------------------------------------------
    print("[INFO] Reconstructing full matrix...")

    full_X = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Insert GRN-smoothed genes → mapped gene positions → used_idx positions
    mask_full = np.where(mask_mapped)[0]  # absolute positions of mapped genes
    full_X[:, mask_full[used_idx_local]] = X_grn

    # Copy over non-GRN but mapped genes
    non_grn_local = np.setdiff1d(np.arange(len(mask_full)), used_idx_local)
    full_X[:, mask_full[non_grn_local]] = X_mapped[:, non_grn_local]

    # Copy unmapped genes
    full_X[:, ~mask_mapped] = X_original[:, ~mask_mapped]

    # Save
    ad.X = full_X
    print(f"[INFO] Writing output H5AD: {args.out}")
    ad.write(args.out)


if __name__ == "__main__":
    main()
