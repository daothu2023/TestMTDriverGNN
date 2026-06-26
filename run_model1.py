"""
MTDriverGNN training pipeline.

Two-head residual GCN (cancer-driver head + telomere-association auxiliary
head) with cross-disease supervised pretraining, nested 5-fold stratified
cross-validation, and grid search over the hyperparameter space reported in
the manuscript.

Grid search
-----------
For each outer cross-validation fold, every hyperparameter combination is
trained with early stopping based on validation loss. Among all
combinations, the configuration achieving the highest validation AUPRC is
selected and retrained to obtain the final test-set score for that fold.

Usage
-----
Full manuscript protocol (10 runs x 5-fold nested CV):
    python run_model.py BRCA

Run fewer repeated runs:
    python run_model.py LUAD --num_runs 3 --data_dir ./Data

Reviewer/software quick test:
    python run_model.py BRCA --quick_test

Force CPU execution:
    python run_model.py BRCA --quick_test --device cpu

Model selection (1 outer split + 3-fold inner grid search, then predict):
    python run_model.py BRCA --select_model
    python run_model.py KIRC --select_model --features my_features.csv --labels my_labels.csv
    python run_model.py BRCA --predict
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

import utils1 as utils_module
from utils1 import (
    DEVICE,
    set_seed,
    build_cancer_data_paths,
    build_dataset_for_cancer,
    build_dataset_directly,
    build_cross_disease_pretrain_labels,
    pretrain_on_cross_disease,
    train_single_configuration,
    auprc_on_mask,
    make_masks,
    model_selection,
)
from model import MultiTaskGCN

NUM_EPOCHS = 300
PATIENCE = 30
KFOLD = 5
WARMUP_EPOCHS = 10

PRETRAIN_LR = 1e-2
PRETRAIN_WEIGHT_DECAY = 5e-4
PRETRAIN_EPOCHS = 300
PRETRAIN_PATIENCE = 30

# Hyperparameter search space, as reported in the manuscript.
DEPTH_OPTIONS   = (1, 2)
HIDDEN_OPTIONS  = ((64, 64), (64, 128), (128, 64), (128, 128))
DROPOUT_OPTIONS = (0.3, 0.4, 0.5)
LR_OPTIONS      = (1e-2, 3e-3, 1e-3)
WEIGHT_DECAY_OPTIONS = (1e-4, 5e-4, 1e-3)


def format_configuration(depth, hidden_dims, dropout, lr, weight_decay):
    """Return one-line text describing the model/training configuration."""
    return (
        f"GCN layers={depth}, hidden dimensions={hidden_dims}, "
        f"dropout={dropout}, learning rate={lr}, weight decay={weight_decay}"
    )


def configure_device(device_arg):
    """
    Configure the execution device.

    Parameters
    ----------
    device_arg : {"auto", "cpu", "gpu"}
        auto: use GPU if available, otherwise CPU.
        cpu: force CPU execution.
        gpu: force GPU execution; raise an error if no GPU is available.
    """
    global DEVICE

    if device_arg == "cpu":
        selected_device = torch.device("cpu")
    elif device_arg == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but torch.cuda.is_available() is False.")
        selected_device = torch.device("cuda")
    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEVICE = selected_device
    utils_module.DEVICE = selected_device
    return selected_device


def apply_quick_test_settings(args):
    """
    Apply reduced settings for reviewer/software testing.

    This mode keeps the full CPDB/PPI graph and real processed cancer data,
    but reduces runs, folds, epochs, and hyperparameter combinations.
    It is not intended to reproduce manuscript results.
    """
    global NUM_EPOCHS, PATIENCE, KFOLD, WARMUP_EPOCHS
    global PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE
    global DEPTH_OPTIONS, HIDDEN_OPTIONS, DROPOUT_OPTIONS, LR_OPTIONS, WEIGHT_DECAY_OPTIONS

    print("Quick-test mode enabled")
    print("  Full CPDB/PPI graph is used.")
    print("  Reduced runs, folds, epochs, and hyperparameter search are used for software testing only.")

    args.num_runs = 1

    NUM_EPOCHS = 5
    PATIENCE = 3
    KFOLD = 2
    WARMUP_EPOCHS = 1

    PRETRAIN_LR = 1e-2
    PRETRAIN_WEIGHT_DECAY = 5e-4
    PRETRAIN_EPOCHS = 3
    PRETRAIN_PATIENCE = 2

    DEPTH_OPTIONS   = (1,)
    HIDDEN_OPTIONS  = ((64, 64),)
    DROPOUT_OPTIONS = (0.3,)
    LR_OPTIONS      = (1e-3,)
    WEIGHT_DECAY_OPTIONS = (5e-4,)


# --------------------------------------------------------------------- #
# Predict mode
# --------------------------------------------------------------------- #

def run_predict(args, selected_device):
    """
    Load a saved model and:
    1. Compute AUPRC on the held-out test set (saved in the checkpoint).
    2. Rank all unlabeled genes (driver label == -1) by predicted driver score.

    The PPI graph is fixed; only the feature matrix differs per user/cancer.
    """
    ppi_path = os.path.join(args.data_dir, "PPI_CPDB.csv")
    prefix = "quick_test_model" if args.quick_test else "best_model"
    if args.model_path is None:
        args.model_path = os.path.join(args.results_dir, f"{prefix}_{args.disease}.pt")
    if args.output is None:
        os.makedirs(args.results_dir, exist_ok=True)
        args.output = os.path.join(args.results_dir, f"predictions_{args.disease}.csv")

    for path, label in [
        (ppi_path,        "PPI_CPDB.csv"),
        (args.model_path, "trained model (.pt)"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found — {label}: {path}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=selected_device)
    hidden_dims  = checkpoint["hidden_dims"]
    dropout      = checkpoint.get("dropout", 0.5)
    test_indices = checkpoint.get("test_indices", None)

    # Build graph + features
    # Use user-supplied features if provided, else fall back to built-in data
    cancer_data_paths = build_cancer_data_paths(args.data_dir)
    telomere_labels_path = os.path.join(args.data_dir, "labels_telomere.csv")

    if args.features is not None:
        # User-supplied feature file (new cancer type).
        # --labels is optional:
        #   - provided → load labels, compute AUPRC on test set + rank unlabeled genes
        #   - not provided → rank all genes, skip AUPRC
        data, node_to_idx, _ = build_dataset_directly(
            ppi_path, args.features,
            driver_labels_path=args.labels if args.labels else None,
            telomere_labels_path=telomere_labels_path if os.path.exists(telomere_labels_path) else None,
        )
    elif args.disease in cancer_data_paths:
        data, node_to_idx, _ = build_dataset_for_cancer(
            args.disease, ppi_path, telomere_labels_path, cancer_data_paths
        )
    else:
        raise ValueError(
            f"Cancer type '{args.disease}' is not in the 12 built-in types "
            "and no --features file was provided."
        )

    node_names = np.array(sorted(node_to_idx, key=node_to_idx.get))

    # Load model
    model = MultiTaskGCN(in_dim=data.num_features, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()
    print(f"[Predict] Model loaded: {args.model_path}")
    print(f"[Predict] hidden_dims={hidden_dims}, dropout={dropout}")

    # Forward pass
    with torch.no_grad():
        driver_logit, _ = model(data.x, data.edge_index)
        scores = torch.sigmoid(driver_logit).cpu().numpy()

    # ── AUPRC on test set ──────────────────────────────────────────────────
    if test_indices is not None and len(test_indices) > 0:
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[test_indices] = True
        test_mask = test_mask.to(selected_device)
        test_auprc = auprc_on_mask(driver_logit, data.y, test_mask)
        print(f"\n[Predict] Test set AUPRC = {test_auprc:.4f} ({test_mask.sum().item()} genes)")
    else:
        test_auprc = None
        print("\n[Predict] No test indices found in checkpoint; skipping AUPRC evaluation.")

    # ── Ranking ───────────────────────────────────────────────────────────
    labels_cpu = data.y.cpu().numpy()
    has_labels = (labels_cpu != -1).any()

    if has_labels:
        # Labels available → rank only unlabeled genes
        unlabeled_mask   = (labels_cpu == -1)
        rank_genes       = node_names[unlabeled_mask]
        rank_scores      = scores[unlabeled_mask]
        rank_description = "unlabeled genes"
    else:
        # No labels provided → rank all genes
        print("[Predict] No labels provided; ranking all genes.")
        rank_genes       = node_names
        rank_scores      = scores
        rank_description = "all genes"

    rank_order = np.argsort(rank_scores)[::-1]
    results_df = pd.DataFrame({
        "gene":         rank_genes[rank_order],
        "driver_score": rank_scores[rank_order],
        "rank":         np.arange(1, len(rank_order) + 1),
    })

    if args.top_k is not None:
        results_df = results_df.head(args.top_k)
        print(f"[Predict] Keeping top {args.top_k} {rank_description}.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n[Done] Ranking saved to: {args.output}")
    print(f"       Genes ranked: {len(results_df)} ({rank_description})")
    print(f"\nTop 10 predicted driver gene candidates:")
    print(results_df.head(10).to_string(index=False))

    if test_auprc is not None:
        return test_auprc


# --------------------------------------------------------------------- #
# Full nested CV (original mode)
# --------------------------------------------------------------------- #

def split_inner_train_val(outer_trainval_idx, labels, seed):
    labels_np = labels[outer_trainval_idx].detach().cpu().numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_sub_idx, val_sub_idx = next(splitter.split(outer_trainval_idx.cpu(), labels_np))
    inner_train_idx = outer_trainval_idx[train_sub_idx].to(DEVICE)
    inner_val_idx   = outer_trainval_idx[val_sub_idx].to(DEVICE)
    return inner_train_idx, inner_val_idx


def grid_search_for_outer_fold(data, outer_trainval_idx, outer_test_idx,
                                pretrain_labels, pretrain_mask, seed):
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    best_hp = None
    best_val_auprc = -1.0

    total_candidates = (
        len(DEPTH_OPTIONS) * len(HIDDEN_OPTIONS) * len(DROPOUT_OPTIONS)
        * len(LR_OPTIONS) * len(WEIGHT_DECAY_OPTIONS)
    )
    candidate_id = 0

    for depth in DEPTH_OPTIONS:
        for hidden_pair in HIDDEN_OPTIONS:
            hidden_dims = [hidden_pair[0]] if depth == 1 else list(hidden_pair)
            for dropout in DROPOUT_OPTIONS:
                for lr in LR_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        candidate_id += 1
                        config_text = format_configuration(depth, hidden_dims, dropout, lr, weight_decay)
                        if total_candidates == 1:
                            print(f"  Training configuration: {config_text}")
                        else:
                            print(f"  Candidate configuration {candidate_id}/{total_candidates}: {config_text}")

                        pretrained_state = pretrain_on_cross_disease(
                            data, pretrain_labels, pretrain_mask, hidden_dims, dropout,
                            PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
                        )
                        val_loss, val_auprc, _ = train_single_configuration(
                            data, train_mask, val_mask, telomere_train_mask,
                            hidden_dims, dropout, lr, weight_decay, pretrained_state,
                            NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                        )
                        print(f"  Validation: loss={val_loss:.4f}, AUPRC={val_auprc:.4f}")

                        if val_auprc > best_val_auprc:
                            best_val_auprc = val_auprc
                            best_hp = {
                                "depth": depth, "hidden_dims": hidden_dims,
                                "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
                            }

    if total_candidates > 1:
        selected_text = format_configuration(
            best_hp["depth"], best_hp["hidden_dims"], best_hp["dropout"],
            best_hp["lr"], best_hp["weight_decay"]
        )
        print(f"  Selected configuration: {selected_text}, validation AUPRC={best_val_auprc:.4f}")

    return best_hp


def train_and_test_outer_fold(data, outer_trainval_idx, outer_test_idx,
                               pretrain_labels, pretrain_mask, best_hp, seed):
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    pretrained_state = pretrain_on_cross_disease(
        data, pretrain_labels, pretrain_mask, best_hp["hidden_dims"], best_hp["dropout"],
        PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
    )
    _, _, model = train_single_configuration(
        data, train_mask, val_mask, telomere_train_mask,
        best_hp["hidden_dims"], best_hp["dropout"], best_hp["lr"], best_hp["weight_decay"],
        pretrained_state, NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
    )

    driver_logit, _ = model(data.x, data.edge_index)
    test_auprc = auprc_on_mask(driver_logit, data.y, test_mask)
    print(f"  Test: AUPRC={test_auprc:.4f}")
    return test_auprc


def run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset):
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=41 + seed_offset)
    labels_np = data.y[labeled_idx].detach().cpu().numpy()
    fold_test_auprc = []

    for fold, (train_val_pos, test_pos) in enumerate(skf.split(labeled_idx.cpu(), labels_np), start=1):
        print("\n" + "-" * 40)
        print(f"Fold {fold}/{KFOLD}")
        outer_trainval_idx = labeled_idx[train_val_pos]
        outer_test_idx     = labeled_idx[test_pos]
        assert not set(outer_trainval_idx.tolist()) & set(outer_test_idx.tolist()), \
            "Train/test split overlap detected."

        fold_seed = 43 + seed_offset + fold
        best_hp = grid_search_for_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, fold_seed
        )
        test_auprc = train_and_test_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, best_hp, fold_seed
        )
        fold_test_auprc.append(test_auprc)
        print("-" * 40)

    print(f"\nFold summary: mean test AUPRC = {np.mean(fold_test_auprc):.4f} +/- {np.std(fold_test_auprc):.4f}")
    return fold_test_auprc


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", type=str, help="Target cancer type, e.g. BRCA, LUAD, UCEC")
    parser.add_argument("--data_dir",    type=str, default="./Data",    help="Path to the data folder")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--num_runs",    type=int, default=10,          help="Number of repeated runs (full mode only)")
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run a compact software test (reduced runs, folds, epochs, HP search).",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "gpu"],
        help="Execution device: auto / cpu / gpu.",
    )

    # ── Select model mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--select_model", action="store_true",
        help=(
            "Model selection mode: split data once (5-fold outer, 3-fold inner grid search), "
            "retrain with best HP, save model. Use --predict afterwards to evaluate and rank genes."
        ),
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help=(
            "[--select_model / --predict] Path to feature matrix CSV. "
            "Rows = genes (index), columns = omics features. "
            "Required for cancer types not in the 12 built-in types."
        ),
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help=(
            "[--select_model] Path to driver label CSV (columns: Gene, Labels). "
            "Required for cancer types not in the 12 built-in types."
        ),
    )

    # ── Predict mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--predict", action="store_true",
        help=(
            "Prediction mode: load saved model, compute AUPRC on held-out test set, "
            "and rank all unlabeled genes by predicted driver score."
        ),
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="[--predict] Path to .pt model file. Default: results/best_model_<CANCER>.pt",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="[--predict] Output CSV path. Default: results/predictions_<CANCER>.csv",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="[--predict] Save only top-K unlabeled genes. Default: save all.",
    )

    args = parser.parse_args()

    if args.quick_test:
        apply_quick_test_settings(args)

    selected_device = configure_device(args.device)

    ppi_cpdb_path        = os.path.join(args.data_dir, "PPI_CPDB.csv")
    telomere_labels_path = os.path.join(args.data_dir, "labels_telomere.csv")
    cancer_data_paths    = build_cancer_data_paths(args.data_dir)
    target_cancer        = args.disease

    # ── Select model mode ──────────────────────────────────────────────────
    if args.select_model:
        print("=" * 60)
        print(f"Model selection mode | cancer={target_cancer} | device={selected_device}")
        print("=" * 60)

        # Load data: user-supplied files or built-in 12 cancer types
        if args.features is not None and args.labels is not None:
            print("[Data] Using user-supplied feature and label files.")
            data, node_to_idx, labeled_idx = build_dataset_directly(
                ppi_cpdb_path, args.features, args.labels,
                telomere_labels_path if os.path.exists(telomere_labels_path) else None,
            )
        elif target_cancer in cancer_data_paths:
            print(f"[Data] Using built-in data for {target_cancer}.")
            data, node_to_idx, labeled_idx = build_dataset_for_cancer(
                target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
            )
        else:
            raise ValueError(
                f"Cancer type '{target_cancer}' is not in the 12 built-in types. "
                "Please provide --features and --labels."
            )

        pretrain_labels, pretrain_mask = build_cross_disease_pretrain_labels(
            node_to_idx, target_cancer, data.y, cancer_data_paths
        )

        hp_grid = {
            "depth_options":        DEPTH_OPTIONS,
            "hidden_options":       HIDDEN_OPTIONS,
            "dropout_options":      DROPOUT_OPTIONS,
            "lr_options":           LR_OPTIONS,
            "weight_decay_options": WEIGHT_DECAY_OPTIONS,
        }
        train_cfg = {
            "num_epochs":            NUM_EPOCHS,
            "patience":              PATIENCE,
            "warmup_epochs":         WARMUP_EPOCHS,
            "pretrain_lr":           PRETRAIN_LR,
            "pretrain_weight_decay": PRETRAIN_WEIGHT_DECAY,
            "pretrain_epochs":       PRETRAIN_EPOCHS,
            "pretrain_patience":     PRETRAIN_PATIENCE,
        }
        final_model, best_hp, test_idx, best_mean_val_auprc = model_selection(
            data=data,
            labeled_idx=labeled_idx,
            pretrain_labels=pretrain_labels,
            pretrain_mask=pretrain_mask,
            results_dir=args.results_dir,
            cancer=target_cancer,
            hp_grid=hp_grid,
            train_cfg=train_cfg,
            quick_test=args.quick_test,
        )

        # Save model + test indices to .pt file
        os.makedirs(args.results_dir, exist_ok=True)
        prefix = "quick_test_model" if args.quick_test else "best_model"
        save_path = os.path.join(args.results_dir, f"{prefix}_{target_cancer}.pt")
        torch.save({
            "model_state_dict":    final_model.state_dict(),
            "hidden_dims":         best_hp["hidden_dims"],
            "dropout":             best_hp["dropout"],
            "cancer":              target_cancer,
            "hyperparams":         best_hp,
            "test_indices":        test_idx.cpu(),
            "best_mean_val_auprc": best_mean_val_auprc,
        }, save_path)
        print(f"\n[Model selection] Model saved to: {save_path}")
        return

    # ── Predict mode ───────────────────────────────────────────────────────
    if args.predict:
        run_predict(args, selected_device)
        return

    # ── Full nested CV (original mode) ─────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    print("Using device:", selected_device)

    run_mean_auprc    = []
    all_fold_test_auprc = []

    for run in range(1, args.num_runs + 1):
        seed = 42 + run - 1
        set_seed(seed)

        print("\n" + "=" * 60)
        print(f"Run {run}/{args.num_runs} | seed={seed} | target cancer={target_cancer}")
        print("=" * 60)

        data, node_to_idx, labeled_idx = build_dataset_for_cancer(
            target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
        )
        pretrain_labels, pretrain_mask = build_cross_disease_pretrain_labels(
            node_to_idx, target_cancer, data.y, cancer_data_paths
        )

        fold_test_auprc = run_nested_cv_one_seed(
            data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset=run
        )
        all_fold_test_auprc.append([float(x) for x in fold_test_auprc])
        run_mean_auprc.append(float(np.mean(fold_test_auprc)))
        print(f"\n[Run {run}/{args.num_runs}] Mean test AUPRC ({KFOLD} folds) = {run_mean_auprc[-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"Summary over {args.num_runs} runs")
    print("=" * 60)
    for i, mean_auprc in enumerate(run_mean_auprc, start=1):
        print(f"  Run {i} (seed={42 + i - 1}): mean test AUPRC = {mean_auprc:.4f}")

    overall_mean = float(np.mean(run_mean_auprc))
    overall_std  = float(np.std(run_mean_auprc))
    print(f"\nOverall mean test AUPRC over {len(run_mean_auprc)} runs: {overall_mean:.4f} +/- {overall_std:.4f}")

    results = {
        "target_cancer": target_cancer,
        "mode": "quick_test" if args.quick_test else "full",
        "device": str(selected_device),
        "num_runs": args.num_runs,
        "num_folds": KFOLD,
        "num_epochs": NUM_EPOCHS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "quick_test_note": (
            "Quick-test mode uses the full CPDB/PPI graph and real processed data, "
            "but reduced runs/folds/epochs/grid search. Not for manuscript result reproduction."
        ) if args.quick_test else None,
        "fold_test_auprc":    all_fold_test_auprc,
        "run_mean_auprc":     run_mean_auprc,
        "overall_mean_auprc": overall_mean,
        "overall_std_auprc":  overall_std,
    }

    prefix   = "quick_test_results" if args.quick_test else "results"
    out_path = os.path.join(args.results_dir, f"{prefix}_{target_cancer}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
