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
    A = sparse.coo_matrix_
