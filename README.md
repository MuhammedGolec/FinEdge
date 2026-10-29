# FinEdge

**A Resource-Aware Stacking Ensemble for Financial Fraud Detection on Edge Devices**

Research code accompanying the paper by **Muhammed Golec, Emine Tasci, and Mustafa Metin Basbay**.

FinEdge combines the fraud probabilities of three heterogeneous base learners using a class-balanced Logistic Regression meta-classifier. The notebook evaluates predictive performance, model size, training time, and amortized batch inference latency on PaySim. It also includes a feature ablation experiment.

## Framework

| Component | Method |
| --- | --- |
| Base learner 1 | Random Under-Sampling (RUS) + XGBoost |
| Base learner 2 | SMOTE + optimized Random Forest |
| Base learner 3 | Optuna-optimized LightGBM |
| Meta-classifier | Logistic Regression with balanced class weights |
| Decision thresholds | Independently selected for each model by maximizing validation F1 |

The main experiment uses a stratified 70% training / 15% validation / 15% test split. The validation partition is divided equally into meta-training and threshold-selection subsets. The default random seed is 42. Test labels are not used to fit the meta-classifier or select decision thresholds.

## Repository contents

- `main.ipynb`: main training and evaluation experiment, supplementary plots, and feature ablation experiment.
- `README.md`: setup, execution instructions, and reproducibility notes.

The dataset and trained models are not bundled with this README.

## Dataset

Download the PaySim dataset from:

[PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

Extract the CSV and place it alongside `main.ipynb`. You can rename it to `paysim dataset.csv` or use its original name and update the path settings accordingly.

The main experiment uses these input columns:

```text
type
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

The target column is `isFraud`. The transaction type is one-hot encoded.

**Feature availability:** `newbalanceOrig` and `newbalanceDest` are post-transaction balances. The original feature configuration assumes these values are available. For pre-transaction screening, use the corresponding ablation configuration and assess which features are available at the actual decision point.

## Installation

Create and activate a Python environment, then install the packages imported by the notebook:

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost lightgbm optuna joblib ipython jupyterlab ipykernel
```

These dependencies were identified from the notebook imports. This is not a pinned reproduction environment; the supplied notebook does not record a complete package-version manifest. Its kernel metadata records Python 3.13.5, which does not by itself establish compatibility with every package release.

## Before running the uploaded notebook

1. In the main experiment, replace the existing Lightning AI path:

   ```python
   FILE_PATH = "/teamspace/studios/this_studio/paysim dataset.csv"
   ```

   with the location of your downloaded CSV, for example:

   ```python
   FILE_PATH = "paysim dataset.csv"
   ```

2. In the feature ablation experiment, update the separate setting if necessary:

   ```python
   DATA_PATH = "paysim dataset.csv"
   ```

3. Install dependencies **before** running any training cells. The uploaded notebook contains `!pip install lightgbm` near its end; that cell is too late to satisfy the earlier imports. Move it before those imports or use the installation command above.

4. Delete or skip the standalone cell containing only `elapsed`. This variable is local to the benchmark function, so that standalone cell raises `NameError`.

## Run the main experiment

Start JupyterLab from the repository directory:

```bash
jupyter lab main.ipynb
```

After completing the setup above, execute the large code cell beginning with the FinEdge title and ending with the output summary. It performs preprocessing, model tuning, training, threshold selection, test evaluation, inference benchmarking, model serialization, and figure generation.

The optional ROC plotting cell depends on variables created by this main experiment and must be run afterwards, before the ablation cell changes shared variables.

The full experiment includes cross-validation, SMOTE, and 30 LightGBM optimization trials. Runtime and peak memory use depend on the machine and dataset size; the notebook does not establish a universal minimum RAM requirement. Train on a machine with sufficient resources before evaluating deployment on an edge device.

### Main outputs

Generated files are written under `fraud_experiment_results/`:

| Location | Contents |
| --- | --- |
| `final_results.csv` | Combined evaluation results |
| `predictive_performance_table.csv` | Predictive performance comparison |
| `computational_performance_table.csv` | Computational performance comparison |
| `hyperparameters.joblib` | Saved hyperparameter information |
| `models/` | Serialized XGBoost, Random Forest, LightGBM, and FinEdge models |
| `figures/` | Precision-recall, ROC, computational, and predictive comparison plots |

The supplementary ROC plotting cell additionally writes `roc_curves_low_fpr.png` into the main figures directory.

## Feature ablation

The code cell headed `FinEdge - FEATURE ABLATION EXPERIMENT` evaluates:

| Configuration | Input columns before one-hot encoding |
| --- | --- |
| Original | `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` |
| Pre-transaction | `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest` |
| Balance-free | `type`, `amount` |

All configurations use the same observation partitions. Each configuration trains its own base learners and meta-classifier and selects a FinEdge threshold on a separate validation subset.

Outputs are saved in `finedge_feature_ablation_results/`:

- `feature_ablation_results.csv`
- `feature_ablation_300dpi.png`

This cell has its own imports and loads the dataset independently. It can be run in a fresh kernel after installing dependencies and setting `DATA_PATH`.

## Benchmark interpretation and scope

- **Latency:** the benchmark predicts a batch of up to 5,000 test observations, repeats this five times after warm-up, and divides each batch duration by its sample count. These are amortized batch timings, not isolated single-request response times. The reported P95 is calculated across the repeated per-batch averages, not across individual transaction latencies.
- **Hardware:** running this notebook measures the current host. It does not automatically reproduce the paper's Raspberry Pi 4 measurements.
- **Energy plot:** the supplementary energy plotting cell contains manually entered PC and edge energy values. It generates `energy_consumption_pc_edge.png` in the working directory; it does not measure power or energy on the host.
- **Repeated runs and statistics:** the main experiment defaults to one seed (42). The supplied notebook does not include an automated five-run evaluation or the paper's paired bootstrap analysis.
- **Saved models:** the FinEdge object uses the notebook-defined `ProbabilityStackingModel` class. Define or import that class before loading the object in a separate Python process, and preserve feature encoding and column order. Retrieve the selected threshold from the saved evaluation results rather than assuming 0.5.
- **Validation status:** this README was prepared through static inspection of the notebook and paper. Full training and hardware experiments were not rerun when preparing these instructions.

## Citation

If you use this work, please cite:

> Golec, M.; Tasci, E.; Basbay, M. *FinEdge: A Resource-Aware Stacking Ensemble for Financial Fraud Detection on Edge Devices.* Electronics, 2026.

Final volume, article number, and DOI should be added when the publisher provides the definitive bibliographic record. The proof's placeholder DOI is intentionally omitted.

## Contact

**Muhammed Golec**  
Department of Computer Engineering, Bogazici University  
Institute for Data Science & Artificial Intelligence, Bogazici University  
Email: [muhammed.golec@bogazici.edu.tr](mailto:muhammed.golec@bogazici.edu.tr)
