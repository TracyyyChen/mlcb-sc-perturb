#!/usr/bin/env python3
import argparse
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import pickle
import torch


# -----------------------------------------------------------
# 1. Load symbol → Ensembl mapping
# -----------------------------------------------------------
def load_symbol_dict(path):
    print(f"[INFO] Loading symbol→Ensembl mapping from: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    print(f"[INFO] Mapping contains {len(mapping)} entries.")
    return mapping


# -----------------------------------------------------------
# 2. Build Laplacian from GRN with Ensembl IDs
# -----------------------------------------------------------
def load_laplacian(grn_csv, gene_order_ensembl):
    """
    grn_csv: CSV with columns [source, target, weight?]
    gene_order_ensembl: list of Ensembl IDs, one per var (with placeholders)
    """
    print(f"[INFO] Loading GRN from: {grn_csv}")
    df = pd.read_csv(grn_csv)

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("GRN CSV must contain columns: source, target [,weight]")

    if "weight" not in df:
        df["weight"] = 1.0

    # Filter to genes present in gene_order_ensembl
    df = df[
        df["source"].isin(gene_order_ensembl)
        & df["target"].isin(gene_order_ensembl)
    ]

    used_genes = sorted(set(df["source"]) | set(df["target"]))
    print(f"[INFO] Filtered GRN edges: {len(df)}")
    print(f"[INFO] Overlapping genes: {len(used_genes)}")

    if len(used_genes) == 0:
        return None, [], []

    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    w = df["weight"].astype(np.float32).values

    n = len(used_genes)
    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)  # symmetrize

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).flatten()
    Dinv_sqrt = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.eye(n, format="csr") - Dinv_sqrt @ A @ Dinv_sqrt

    # Map used_genes (Ensembl) back to indices in gene_order_ensembl
    used_idx = [gene_order_ensembl.index(g) for g in used_genes]

    return L, used_idx, used_genes


# -----------------------------------------------------------
# 3. Custom CG solver on GPU with Jacobi preconditioning
# -----------------------------------------------------------
def cg_solve_gpu(b, L_gpu, tau, diag_M_inv, maxiter=200, rtol=1e-4):
    """
    Solve (I + tau * L) x = b on GPU using CG.
    - b:  (n,) torch.float32 (on device)
    - L_gpu: sparse_csr_tensor of shape (n, n) on device
    - diag_M_inv: (n,) diagonal preconditioner = 1 / diag(I + tau * L)
    """
    device = b.device
    n = b.shape[0]

    # Matvec: M v = v + tau * L v
    def matvec(v):
        # v: (n,)
        Lv = torch.sparse.mv(L_gpu, v)      # (n,)
        return v + tau * Lv                 # (n,)

    # Initial guess x0 = 0
    x = torch.zeros_like(b, device=device)

    r = b - matvec(x)             # r0 = b - A x0 = b
    z = diag_M_inv * r            # preconditioned residual
    p = z.clone()

    rz_old = torch.dot(r, z)

    b_norm = torch.norm(b)
    if b_norm.item() == 0.0:
        return x

    for k in range(maxiter):
        Ap = matvec(p)
        alpha = rz_old / torch.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.norm(r)
        if r_norm <= rtol * b_norm:
            # Converged
            break

        z = diag_M_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old

        p = z + beta * p
        rz_old = rz_new

    return x


# -----------------------------------------------------------
# 4. Main smoothing pipeline (GPU)
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRN Laplacian smoothing on GPU")
    parser.add_argument("--pred-h5ad", required=True,
                        help="Prediction h5ad file (e.g. prediction_infer.h5ad)")
    parser.add_argument("--grn-csv", required=True,
                        help="GRN edge list CSV with Ensembl IDs")
    parser.add_argument("--symbols-pkl", required=True,
                        help="Pickle with symbol→Ensembl dict")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Diffusion strength τ")
    parser.add_argument("--out", required=True,
                        help="Output h5ad path")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print(f"[INFO] Loading predictions: {args.pred_h5ad}")
    adata = sc.read_h5ad(args.pred_h5ad)
    print(f"[INFO] Prediction shape: {adata.shape}")
    print("=" * 80)

    # ----------------------------------------------------
    # Map var_names (symbols) → Ensembl IDs
    # ----------------------------------------------------
    mapping = load_symbol_dict(args.symbols_pkl)

    var_symbols = list(adata.var_names)
    gene_order_ensembl = [mapping.get(sym, None) for sym in var_symbols]

    n_mapped = sum(g is not None for g in gene_order_ensembl)
    print(f"[INFO] Mapped {n_mapped}/{len(var_symbols)} genes to Ensembl IDs.")
    if n_mapped == 0:
        print("[ERROR] No genes could be mapped to Ensembl IDs.")
        sys.exit(1)

    # Replace None with placeholder (ignored by GRN filter)
    gene_order_ensembl = [
        g if g is not None else "__UNMAPPED__"
        for g in gene_order_ensembl
    ]

    # ----------------------------------------------------
    # Build Laplacian on CPU, then move to GPU
    # ----------------------------------------------------
    L_csr, used_idx, used_genes = load_laplacian(args.grn_csv, gene_order_ensembl)

    if L_csr is None or len(used_idx) == 0:
        print("[ERROR] No overlapping genes between GRN and prediction after mapping.")
        adata.write(args.out)
        return

    print(f"[INFO] Laplacian shape: {L_csr.shape}")
    print(f"[INFO] Number of genes smoothed: {len(used_idx)}")

    # Extract CSR components for PyTorch
    L_csr.sort_indices()
    indptr = torch.from_numpy(L_csr.indptr.astype(np.int64))
    indices = torch.from_numpy(L_csr.indices.astype(np.int64))
    data = torch.from_numpy(L_csr.data.astype(np.float32))

    n = L_csr.shape[0]
    L_gpu = torch.sparse_csr_tensor(
        indptr, indices, data,
        size=(n, n),
        device=device,
        dtype=torch.float32,
    )
    L_gpu = L_gpu.coalesce()

    # Diagonal of L (CPU) -> diag of M = 1 + tau * diag(L)
    diag_L = L_csr.diagonal().astype(np.float32)  # shape (n,)
    diag_M = 1.0 + args.tau * diag_L
    diag_M_inv = torch.from_numpy(1.0 / (diag_M + 1e-8)).to(device)

    # ----------------------------------------------------
    # Prepare X (predictions) and smooth only used_idx genes
    # ----------------------------------------------------
    X = adata.X
    if sparse.issparse(X):
        print("[INFO] Converting sparse X → dense")
        X = X.toarray()
    else:
        X = np.asarray(X)

    n_cells, n_genes = X.shape
    print(f"[INFO] Smoothing {n_cells} rows, {n_genes} genes (subset {len(used_idx)}).")

    # For numerical stability / speed
    used_idx_np = np.array(used_idx, dtype=np.int64)

    # We'll update X in-place row-by-row
    for i in range(n_cells):
        if i % 200 == 0:
            print(f"  [GPU CG] Smoothed {i}/{n_cells}")

        row = X[i]  # shape (n_genes,)

        # Subvector for genes in GRN (length n_used)
        b_np = row[used_idx_np].astype(np.float32)  # (n_used,)
        b = torch.from_numpy(b_np).to(device)

        # Solve (I + tau L) x = b
        x_smooth = cg_solve_gpu(
            b,
            L_gpu=L_gpu,
            tau=args.tau,
            diag_M_inv=diag_M_inv,
            maxiter=200,
            rtol=1e-4,
        )

        # Put back smoothed values
        row[used_idx_np] = x_smooth.cpu().numpy()
        X[i] = row

    print("[INFO] All rows smoothed. Updating AnnData and saving...")

    adata.X = X
    adata.write(args.out)

    print(f"[INFO] Saved smoothed predictions to: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
