#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

# ============================================================
# Load GRN Laplacian from Parquet (FAST)
# ============================================================
def load_laplacian_from_parquet(parquet_path, gene_order):
    print(f"[INFO] Loading GRN parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if "weight" not in df.columns:
        df["weight"] = 1.0

    # Build index for fast JOIN filtering
    valid = pd.Index(gene_order)

    print("[INFO] Filtering GRN to overlapping genes...")
    df = df.join(valid.to_frame("k1"), on="source", how="inner")
    df = df.join(valid.to_frame("k2"), on="target", how="inner")
    df = df[["source", "target", "weight"]]

    used_genes = sorted(set(df["source"]) | set(df["target"]))
    print(f"[INFO] Overlapping genes: {len(used_genes)}")

    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).to_numpy()
    iv = df["target"].map(gene_to_idx).to_numpy()
    w = df["weight"].astype(np.float32).to_numpy()

    n = len(used_genes)

    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)

    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = deg + 1e-12
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg_safe))

    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    used_idx = [gene_order.index(g) for g in used_genes]
    return L, used_idx, used_genes

# ============================================================
# GPU CG solver (correct CSR matvec)
# ============================================================
def cg_solve_gpu(A, b, tau=0.1, tol=1e-4, maxiter=500):
    """
    Solve (I + tau*A)x = b using Conjugate Gradient on GPU.
    A is sparse CSR (PyTorch GPU).
    """
    device = b.device
    n = b.shape[0]

    def matvec(v):
        return v + tau * (A @ v)     # << Correct CSR matvec

    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rsold = torch.dot(r, r)

    for i in range(maxiter):
        Ap = matvec(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

# ============================================================
# Main smoothing pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--grn-parquet", required=True)
    parser.add_argument("--symbols-pkl", required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=None,
                        help="Smooth only first N rows (testing)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("============================================================")
    print(f"[INFO] Loading predictions: {args.pred_h5ad}")
    ad = sc.read_h5ad(args.pred_h5ad)
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    X = np.asarray(X)

    gene_order = list(ad.var_names)
    print(f"[INFO] Prediction shape: {X.shape}")
    print("============================================================")

    # Load mapping
    import pickle
    print(f"[INFO] Loading symbol→Ensembl mapping from: {args.symbols_pkl}")
    with open(args.symbols_pkl, "rb") as f:
        mapping = pickle.load(f)

    ensembl_order = [mapping.get(g, None) for g in gene_order]

    print(f"[INFO] Mapped {sum(e is not None for e in ensembl_order)}/{len(ensembl_order)} genes")

    # Filter unmapped
    mask = np.array([e is not None for e in ensembl_order])
    ensembl_order = [e for e in ensembl_order if e is not None]

    # Load Laplacian
    L, used_idx, used_genes = load_laplacian_from_parquet(
        args.grn_parquet, ensembl_order
    )

    print(f"[INFO] Laplacian shape: {L.shape}")

    # Convert to PyTorch CSR
    print("[INFO] Moving Laplacian to GPU...")
    L_gpu = torch.sparse_csr_tensor(
        L.indptr.astype(np.int64),
        L.indices.astype(np.int64),
        L.data.astype(np.float32),
        size=L.shape,
        device=device,
        dtype=torch.float32
    )

    # Subset data for mapped genes only
    X = X[:, mask]  # reduces dim from 18080 → 17554 mapped genes

    # Smooth subset only for genes in used_idx
    rows = X.shape[0] if args.rows is None else args.rows
    print(f"[INFO] Smoothing {rows} rows...")

    smoothed = np.zeros_like(X[:rows])

    for i in range(rows):
        v = torch.tensor(X[i, used_idx], device=device, dtype=torch.float32)
        x_smooth = cg_solve_gpu(L_gpu, v, tau=args.tau)
        out = X[i].copy()
        out[used_idx] = x_smooth.cpu().numpy()
        smoothed[i] = out

        if i % 100 == 0:
            print(f"  [GPU CG] Smoothed {i}/{rows}")

    # Restore into full matrix
    X[:rows] = smoothed

    print("[INFO] Writing output H5AD:", args.out)
    ad.X = X
    ad.write(args.out)


if __name__ == "__main__":
    main()
