#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
import pickle


# ================================================================
# GPU Conjugate Gradient Solver (batched + torch.linalg.cg)
# ================================================================
def cg_batch_solve(L_gpu, B, tau=0.1, tol=1e-3, max_iter=200):
    """
    Solve (I + τL) X = B using classic Conjugate Gradient.
    L_gpu: sparse CSR (n,n)
    B: (batch,n)
    """

    device = B.device
    batch, n = B.shape

    # Matrix-vector multiply: y = (I + τL) x
    def matvec(x):
        # x: (batch,n)
        # (batch,n) + τ * (batch,n)@(n,n)
        return x + tau * torch.matmul(x, L_gpu)

    X = torch.zeros_like(B)
    R = B - matvec(X)
    P = R.clone()

    rs_old = (R * R).sum(dim=1)  # (batch,)

    for _ in range(max_iter):
        AP = matvec(P)                     # (batch,n)
        alpha = rs_old / (AP * P).sum(dim=1).clamp(min=1e-12)
        alpha = alpha.view(-1, 1)

        X = X + alpha * P
        R = R - alpha * AP

        rs_new = (R * R).sum(dim=1)

        if torch.all(rs_new < tol):
            break

        beta = (rs_new / rs_old).view(-1, 1)
        P = R + beta * P

        rs_old = rs_new

    return X


# ================================================================
# Load Laplacian from Parquet
# ================================================================
def load_laplacian_from_parquet(path, gene_order):
    print(f"[INFO] Reading GRN parquet: {path}")
    df = pd.read_parquet(path)

    required = {"source", "target", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Parquet must contain columns: {required}")

    # Restrict to genes present in prediction matrix
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

    # PyTorch CSR
    L_gpu = torch.sparse_csr_tensor(
        torch.tensor(L.indptr, dtype=torch.int64),
        torch.tensor(L.indices, dtype=torch.int64),
        torch.tensor(L.data, dtype=torch.float32),
        size=L.shape,
        device="cuda"
    )

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
    parser.add_argument("--batch", type=int, default=128)  # NEW
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load prediction matrix (NO `.toarray()` — keep sparse!)
    # ---------------------------------------------------------
    print(f"[INFO] Loading predictions: {args.pred_h5ad}")
    ad = sc.read_h5ad(args.pred_h5ad)
    X_original = ad.X  # KEEP SPARSE

    n_cells, n_genes = X_original.shape
    print(f"[INFO] Prediction shape: {X_original.shape}")

    # ---------------------------------------------------------
    # Load symbol → Ensembl mapping
    # ---------------------------------------------------------
    print(f"[INFO] Loading symbol→Ensembl mapping: {args.symbols_dict}")
    with open(args.symbols_dict, "rb") as f:
        mapping = pickle.load(f)

    gene_order = [mapping.get(g, None) for g in ad.var_names]
    mask_mapped = np.array([g is not None for g in gene_order])
    gene_order = np.array(gene_order)[mask_mapped].tolist()

    # Extract mapped gene submatrix (still sparse)
    X_mapped = X_original[:, mask_mapped]

    # ---------------------------------------------------------
    # Load Laplacian
    # ---------------------------------------------------------
    L_gpu, used_idx_local, used_genes = load_laplacian_from_parquet(
        args.grn_parquet, gene_order
    )
    print(f"[INFO] GRN genes: {len(used_idx_local)}")

    # Subset mapped genes to GRN overlap
    X_grn = X_mapped[:, used_idx_local]  # still sparse CSR

    rows = X_grn.shape[0] if args.rows is None else args.rows
    print(f"[INFO] Smoothing {rows} rows...")

    smoothed = np.zeros((rows, len(used_idx_local)), dtype=np.float32)

    # ---------------------------------------------------------
    # Batch CG smoothing
    # ---------------------------------------------------------
    batch = args.batch

    for start in range(0, rows, batch):
        end = min(start + batch, rows)

        # Extract sparse rows & densify ON GPU
        B = torch.tensor(
            X_grn[start:end],  # small dense slice only
            dtype=torch.float32,
            device="cuda"
        )

        X_smooth = cg_batch_solve(L_gpu, B, tau=args.tau)
        smoothed[start:end] = X_smooth.cpu().numpy()

        print(f"[GPU CG] {start}/{rows} rows done")

    # Insert smoothed rows
    X_grn_dense = smoothed

    # ---------------------------------------------------------
    # Reconstruct full matrix
    # ---------------------------------------------------------
    print("[INFO] Reconstructing full matrix...")
    full = np.zeros((n_cells, n_genes), dtype=np.float32)

    mask_full = np.where(mask_mapped)[0]

    # Insert GRN-smoothed genes
    full[:, mask_full[used_idx_local]] = X_grn_dense

    # Insert non-GRN mapped genes
    non_grn = np.setdiff1d(np.arange(len(mask_full)), used_idx_local)
    full[:, mask_full[non_grn]] = X_mapped[:, non_grn].toarray()

    # Insert unmapped genes
    full[:, ~mask_mapped] = X_original[:, ~mask_mapped].toarray()

    ad.X = full

    print(f"[INFO] Writing output H5AD: {args.out}")
    ad.write(args.out)


if __name__ == "__main__":
    main()
