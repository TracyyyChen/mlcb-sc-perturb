#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
import pickle

# Conjugate Gradient with (L @ X.T).T instead of (X @ L)
def cg_batch_solve(L_gpu, B, tau=0.1, tol=1e-3, max_iter=200):
    """
    Solve (I + τL) X = B
    using Conjugate Gradient.

    L_gpu: sparse CSR (n,n)
    B: (batch, n)
    """

    device = B.device
    batch, n = B.shape

    # ------------------------------
    # NEW matvec: (L @ x.T).T
    # ------------------------------
    def matvec(x):
        """
        x: (batch, n)
        returns: x + τ * (L @ xᵀ)ᵀ
        """
        # x.T → (n, batch)
        # L @ x.T → (n, batch)
        # .T → (batch, n)
        Lx = torch.matmul(L_gpu, x.T).T
        return x + tau * Lx

    # CG initialization
    X = torch.zeros_like(B)
    R = B - matvec(X)
    P = R.clone()
    rs_old = (R * R).sum(dim=1)

    for _ in range(max_iter):
        AP = matvec(P)
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


# Load Laplacian from GRN parquet
def load_laplacian_from_parquet(path, gene_order):
    print(f"[INFO] Reading GRN parquet: {path}")
    df = pd.read_parquet(path)

    required = {"source", "target", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Parquet must contain columns: {required}")

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

    # Convert to torch CSR
    L_gpu = torch.sparse_csr_tensor(
        torch.tensor(L.indptr, dtype=torch.int64),
        torch.tensor(L.indices, dtype=torch.int64),
        torch.tensor(L.data, dtype=torch.float32),
        size=L.shape,
        device="cuda"
    )

    used_idx = np.array([gene_order.index(g) for g in used_genes], dtype=int)

    return L_gpu, used_idx, used_genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--grn-parquet", required=True)
    parser.add_argument("--symbols-dict", required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading predictions: {args.pred_h5ad}")
    ad = sc.read_h5ad(args.pred_h5ad)
    X_original = ad.X

    n_cells, n_genes = X_original.shape
    print(f"[INFO] Prediction shape: {X_original.shape}")

    # Load mapping
    print(f"[INFO] Loading symbol→Ensembl mapping: {args.symbols_dict}")
    with open(args.symbols_dict, "rb") as f:
        mapping = pickle.load(f)

    gene_order = [mapping.get(g, None) for g in ad.var_names]
    mask_mapped = np.array([g is not None for g in gene_order])
    gene_order = np.array(gene_order)[mask_mapped].tolist()

    X_mapped = X_original[:, mask_mapped]

    # Load Laplacian
    L_gpu, used_idx_local, used_genes = load_laplacian_from_parquet(
        args.grn_parquet, gene_order
    )
    print(f"[INFO] GRN genes: {len(used_idx_local)}")

    X_grn = X_mapped[:, used_idx_local]

    rows = X_grn.shape[0] if args.rows is None else args.rows
    print(f"[INFO] Smoothing {rows} rows...")

    smoothed = np.zeros((rows, len(used_idx_local)), dtype=np.float32)

    batch = args.batch

    for start in range(0, rows, batch):
        end = min(start + batch, rows)

        B = tor
