# MTDriverGNN: Multitask Learning Graph Neural Network for Cancer Driver Gene Prioritization

MTDriverGNN is a multi-task graph neural network for cancer driver gene
prioritization. A residual GCN encoder is shared between two prediction
heads, the cancer-driver head (primary task) and the telomere-association
head (auxiliary task), combined with cross-disease supervised pretraining
and a nested cross-validation protocol with grid search over the
hyperparameter space.

## Key Features

- **Residual GCN Encoder:** stacks GCN layers with a residual connection
from the input node features to the final node embedding.
- **Multi-Task Heads:** a shared MLP layer feeds two linear output heads,
the cancer-driver head and the telomere-association head, combined via
a learnable weighting coefficient (alpha).
- **Cross-Disease Supervised Pretraining:** pretrains the encoder on
driver-gene labels aggregated from other cancer types before
fine-tuning on the target cancer.
- **Nested Cross-Validation with Grid Search:** 5-fold outer
cross-validation, each fold preceded by a grid search over GCN depth,
hidden dimensions, dropout rate, learning rate, and weight decay.
Each configuration is trained with early stopping on validation loss;
the configuration with the highest validation AUPRC is retrained to
obtain the held-out test score for that fold.
- **Model Selection + Prediction:** single outer split (5-fold) with
3-fold inner grid search, producing a saved model that can be used to
rank unlabeled genes and evaluate AUPRC on a held-out test set.

## Requirements

- Python 3.9+
- torch >= 1.9.1
- torch-geometric >= 2.0.4
- numpy >= 1.21.5
- pandas >= 1.3.5
- scikit-learn >= 1.0.2

Install with:

```
pip install -r requirements.txt
```

## Data

All required data files are included in the `Data/` folder of this
repository, so no manual download is needed; just clone the repository
and run.

- `PPI_CPDB.csv`: protein-protein interaction edge list (two gene-name columns)
- `features_for_<CANCER>.csv`: node feature matrix for each cancer type (genes as index)
- `<CANCER>_labels(0_1).csv`: driver-gene labels for each cancer type, columns `Gene,Labels`
- `labels_telomere.csv`: auxiliary task labels, columns `Gene,Labels`

`<CANCER>` must match one of: `BRCA, LUAD, CESC, BLCA, LIHC, THCA, ESCA, PRAD, STAD, COAD, UCEC, LUSC`.

---

## Usage

MTDriverGNN supports three running modes:

| Mode | Purpose |
|------|---------|
| **Full nested CV** | Reproduce manuscript results with full evaluation protocol |
| **Model selection + Predict** | Apply the model to new data and rank candidate driver genes |
| **Quick test** | Verify the pipeline runs correctly end-to-end |

---

### Mode 1: Full Nested Cross-Validation

Runs 10 repeated seeds × 5-fold nested CV × grid search, following the
protocol described in the manuscript.

```bash
# Run for BRCA with default settings (10 runs x 5-fold nested CV with grid search)
python run_model.py BRCA

# Run with fewer repeated runs
python run_model.py LUAD --num_runs 3

# Specify custom data and results folders
python run_model.py BRCA --data_dir ./Data --results_dir ./results

# Force CPU execution
python run_model.py BRCA --device cpu
```

Per-run mean test AUPRC scores are saved to `results/results_<CANCER>.json`.

---

### Mode 2: Model Selection + Predict

Use this mode to **predict and rank candidate driver genes** on new data.
The workflow consists of two steps.

**Step 1 — Model selection:** splits the labeled data once into 5 outer
folds, uses one fold as a held-out test set, and runs a 3-fold inner grid
search on the remaining 4 folds to select the best hyperparameter
configuration. The model is then retrained on all 4 folds with the best
configuration and saved to `results/best_model_<CANCER>.pt`. The held-out
test indices are also saved inside the checkpoint for use in Step 2.

**Step 2 — Predict:** loads the saved model, computes AUPRC on the
held-out test set (using test indices from the checkpoint), and ranks all
unlabeled genes by their predicted driver score.

#### 2a. Built-in cancer types (12 cancer types included in the repository)

Features and labels are loaded automatically from the `Data/` folder.
No additional files needed.

```bash
# Step 1: model selection
python run_model.py BRCA --select_model

# Step 2: predict (no extra arguments needed)
python run_model.py BRCA --predict
```

#### 2b. New cancer type (not in the 12 built-in types)

For **Step 1**, prepare two CSV files (see **Input Data Format** below)
and pass them via `--features` and `--labels`.

For **Step 2**, `--features` is required. `--labels` is optional:
- Provided → compute AUPRC on held-out test set + rank unlabeled genes
- Not provided → rank all genes, AUPRC not computed

```bash
# Step 1: model selection with user-supplied data
python run_model.py KIRC --select_model \
    --features my_features.csv \
    --labels my_labels.csv

# Step 2: predict with labels (AUPRC + ranking unlabeled genes)
python run_model.py KIRC --predict \
    --features my_features.csv \
    --labels my_labels.csv

# Step 2: predict without labels (ranking all genes only)
python run_model.py KIRC --predict --features my_features.csv
```

#### Additional options for --predict

```bash
# Keep only the top 100 highest-scoring unlabeled genes
python run_model.py BRCA --predict --top_k 100

# Specify a custom model file and output path
python run_model.py BRCA --predict \
    --model_path ./results/best_model_BRCA.pt \
    --output ./my_predictions.csv
```

The output is a CSV file with three columns:

| gene | driver_score | rank |
|------|-------------|------|
| CDK2 | 0.991 | 1 |
| TP53 | 0.985 | 2 |
| ... | ... | ... |

Only **unlabeled genes** (genes without a known driver/non-driver label)
appear in the ranking. The AUPRC on the held-out test set is printed to
the console.

---

### Mode 3: Quick Test

Verifies that the full pipeline runs correctly by using a single
hyperparameter configuration, reduced epochs, and fewer folds.
Results from this mode are not representative of full model performance
and should not be used for evaluation.

```bash
# Quick test — full nested CV
python run_model.py BRCA --quick_test --device cpu

# Quick test — model selection (built-in cancer type)
python run_model.py BRCA --select_model --quick_test --device cpu

# Quick test — predict (must run --select_model --quick_test first)
python run_model.py BRCA --predict --quick_test --device cpu
```

---

## Input Data Format

When using `--select_model` with a new cancer type, prepare two CSV files
in the following format.

### Feature matrix (`my_features.csv`)

- Rows: genes (used as the index column)
- Columns: omics features (e.g. CNV, methylation, expression, mutation)
- Gene names must match those in `PPI_CPDB.csv`

```
,CNV,methylation,expression,mutation
TP53,0.12,0.45,1.23,0
BRCA1,0.34,0.67,0.89,1
MYC,0.56,0.23,2.10,0
...
```

### Label file (`my_labels.csv`)

- Column `Gene`: gene name
- Column `Labels`: 1 = known driver gene, 0 = known non-driver gene
- Only required for `--select_model`; not needed for `--predict`

```
Gene,Labels
TP53,1
BRCA1,1
MYC,0
...
```

> **Note:** Genes not present in `PPI_CPDB.csv` are ignored. Genes present
> in the PPI graph but absent from the feature file are imputed by
> averaging the feature vectors of their direct neighbors in the graph.

---

## All Arguments

| Argument | Default | Applies to | Description |
|----------|---------|------------|-------------|
| `--data_dir` | `./Data` | All modes | Path to the data folder |
| `--results_dir` | `./results` | All modes | Path to save results and model checkpoints |
| `--device` | `auto` | All modes | Execution device: `auto`, `cpu`, or `gpu` |
| `--quick_test` | `False` | All modes | Enable quick-test mode (reduced epochs and folds) |
| `--num_runs` | `10` | Full nested CV | Number of repeated runs with different random seeds |
| `--select_model` | `False` | Model selection | Enable model selection mode |
| `--features` | `None` | `--select_model` (new cancer type), `--predict` (new cancer type) | Path to feature matrix CSV |
| `--labels` | `None` | `--select_model` (new cancer type, required), `--predict` (new cancer type, optional) | Path to driver label CSV. Required for `--select_model`; optional for `--predict` (if provided, computes AUPRC; if not, ranks all genes) |
| `--predict` | `False` | Predict | Enable predict mode |
| `--model_path` | `None` | `--predict` | Path to saved model `.pt` file. Default: `results/best_model_<CANCER>.pt` |
| `--output` | `None` | `--predict` | Output CSV path. Default: `results/predictions_<CANCER>.csv` |
| `--top_k` | `None` | `--predict` | Save only the top-K unlabeled genes by driver score. Default: save all |

---

## Project Structure

```
MTDriverGNN/
├── README.md
├── requirements.txt
├── model.py        # model definitions (residual GCN encoder, multi-task heads, learnable alpha)
├── utils.py        # data loading, cross-disease pretraining, training/evaluation, model_selection()
├── run_model.py    # main entry point (full nested CV, model selection, predict)
├── Data/           # input data (included in this repository)
└── results/        # output files (model checkpoints, evaluation results, gene rankings)
```
