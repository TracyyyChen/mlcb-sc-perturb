import os
import argparse
import pytorch_lightning as pl
import torch
import yaml
import numpy as np
import pandas as pd
from scipy import sparse

from omegaconf import OmegaConf, DictConfig

# STATE imports (same entrypoints as train.py)
from cell_load.utils.modules import get_datamodule
from state.tx.utils import get_lightning_module
from finetune import freeze_all_but_last_n_layers, GRNReg   # reuse utilities


# -------------------------------------------------------------------------
# Load YAML config
# -------------------------------------------------------------------------
def load_cfg(path: str) -> DictConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found: {path}")
    return OmegaConf.load(path)


# -------------------------------------------------------------------------
# Build sparse Laplacian from edgelist CSV
# -------------------------------------------------------------------------
def load_laplacian(csv_path, gene_order, symmetric="max"):
    df = pd.read_csv(csv_path)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("CSV must have columns: source,target[,weight]")

    if "weight" not in df:
        df["weight"] = 1.0

    mask = df["source"].isin(gene_order) & df["target"].isin(gene_order)
    df = df[mask]

    if len(df) == 0:
        print("[WARN] No edges match gene_order")
        return None, []

    used_genes = sorted(set(df["source"]) | set(df["target"]))
    gene_to_idx = {g: i for i, g in enumerate(used_genes)}

    iu = df["source"].map(gene_to_idx).values
    iv = df["target"].map(gene_to_idx).values
    w = df["weight"].astype(np.float32).values
    n = len(used_genes)

    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n)).tocsr()

    if symmetric == "max":
        A = A.maximum(A.T)

    deg = np.array(A.sum(axis=1)).flatten()
    Dinv = sparse.diags(1.0 / np.sqrt(deg))

    L = sparse.eye(n) - Dinv @ A @ Dinv
    return L.tocsr(), used_genes


# -------------------------------------------------------------------------
# Main finetuning logic
# -------------------------------------------------------------------------
def run(cfg: DictConfig):

    # ----------------------------
    # 1. Build Dataset
    # ----------------------------
    dm = get_datamodule(
        cfg.data.name,
        cfg.data.kwargs,
        batch_size=cfg.training.batch_size,
        cell_sentence_len=cfg.model.kwargs.get("cell_set_len", 128),
    )
    dm.setup("fit")
    var_dims = dm.get_var_dims()
    gene_order = dm.get_var_names()   # list of gene symbols in order

    if cfg.data.kwargs["output_space"] == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)
    else:
        gene_dim = var_dims.get("gene_dim", 2000)
    
    decoder_cfg = {
        "latent_dim": int(var_dims["output_dim"]),
        "gene_dim": int(gene_dim),
        "hidden_dims": [int(x) for x in cfg.model.kwargs.get("decoder_hidden_dims", [1024, 1024, 512])],
        "dropout": float(cfg.model.kwargs.get("decoder_dropout", 0.1)),
        "residual_decoder": bool(cfg.model.kwargs.get("residual_decoder", False)),
    }

    #cfg.model.kwargs["decoder_cfg"] = decoder_cfg
    cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg
   
    # ----------------------------
    # 2. Build Model
    # ----------------------------
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        var_dims,)
    print("Returned model type:", type(model))

    if hasattr(model, "_build_decoder"):
        model.decoder_cfg = decoder_cfg
        model._build_decoder()
        model._decoder_externally_configured = True

    # Load pretrained checkpoint
    ckpt_path = cfg.model.kwargs.get("init_from", None)
    if ckpt_path is not None:
        print(f"Loading pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # ----------------------------
    # 3. Freeze everything except last N layers
    # ----------------------------
    n_layers = cfg.model.kwargs.get("finetune_last_layers", 2)
    trainable = freeze_all_but_last_n_layers(model, n_layers=n_layers)
    print(f"Trainable parameter tensors: {len(trainable)}")

    # ----------------------------
    # 4. Build GRN regularizer (optional)
    # ----------------------------
    grn_reg = None
    if "grn" in cfg and cfg.grn.get("path") is not None:
        print("Loading GRN edges:", cfg.grn.path)
        L, used_genes = load_laplacian(cfg.grn.path, gene_order)
        if L is not None:
            print("Constructing GRN Regularizer")
            ordered_indices = [gene_order.index(g) for g in used_genes]
            grn_reg = GRNReg(model, L, ordered_indices, grn_lambda=cfg.grn.get("lambda", 1e-3))

    # Wrap training_step to insert GRN loss
    orig_ts = model.training_step
    def ts_with_grn(batch, batch_idx):
        loss = orig_ts(batch, batch_idx)
        if grn_reg is not None:
            device = next(model.parameters()).device
            reg = grn_reg.penalty(device)
            model.log("grn_penalty", reg, on_step=True, prog_bar=True)
            loss = loss + reg
        return loss
    model.training_step = ts_with_grn

    # ----------------------------
    # 5. Optimizer
    # ----------------------------
    lr = cfg.training.get("lr", 2e-5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=cfg.training.get("weight_decay", 0.01))
    model.configure_optimizers = lambda: optimizer

    # ----------------------------
    # 6. Trainer
    # ----------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.training.get("devices", 1),
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.val_freq,
        log_every_n_steps=cfg.training.log_every_n_steps,
        strategy=cfg.training.strategy,
        callbacks=[],
    )

    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Finetuning v2")
    parser.add_argument("--hparams", type=str, default="finetune_v2.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.hparams)
    run(cfg)


if __name__ == "__main__":
    main()
