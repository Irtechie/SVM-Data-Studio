# SVM Data Studio

This project is a hands-on SVM learning workspace that moves from intuition to implementation and then broadens into itemset and episode mining. The goal is to help you understand SVMs like a professional practitioner, not just run a single classifier.

## What is included

- `simple` data: Iris with two petal features, ideal for seeing margins and decision boundaries.
- `medium` data: Breast Cancer Wisconsin, a realistic binary classification problem with 30 features.
- `complex` data: Digits, a multi-class image classification benchmark with 64 features.
- Itemset mining: frequent patterns mined from discretized versions of the SVM datasets.
- Episode mining: serial pattern mining on common event-sequence examples.
- Visualization: all final outputs are rendered with `matplotlib` and `seaborn`.

## SVM concepts you should know

### 1. Maximum margin

An SVM does not just separate classes. It tries to separate them with the widest possible margin.

- Hard-margin idea: fit a separating hyperplane when classes are perfectly separable.
- Soft-margin idea: allow some violations so the model can handle noisy real data.
- Margin size in the linear case is proportional to `2 / ||w||`.

### 2. Support vectors

Only a subset of training points determines the boundary. Those points are the support vectors. If they move, the boundary moves.

### 3. Kernel trick

Some datasets are not linearly separable in the original space. Kernels let the SVM act as if the data were mapped into a richer feature space.

Common kernels in this project:

- `linear`
- `rbf`
- `poly`

### 4. Hyperparameters that matter

- `C`: penalty for classification errors. Large `C` fits harder; small `C` regularizes more.
- `gamma` for RBF/poly: controls how local each training point's influence is.
- `degree` for polynomial kernels: controls curvature complexity.

### 5. Why scaling is mandatory

SVMs are distance-sensitive. Feature scales can distort the margin, so the project uses `StandardScaler` before `SVC`.

## Why itemset and episode mining are here

These are not SVM algorithms. They are adjacent data-mining techniques:

- Itemset mining finds frequent co-occurring attribute patterns.
- Episode mining finds frequent ordered event patterns in sequences.

Including them gives you a broader data-mining study project instead of a single classifier demo.

## Project layout

- `main.py`: runs the full study and writes all outputs.
- `streamlit_app.py`: interactive UI for uploaded or demo datasets.
- `src/svm_studio/custom_analysis.py`: reusable analysis path for arbitrary user tables.
- `src/svm_studio/datasets.py`: dataset loading.
- `src/svm_studio/svm_analysis.py`: training, tuning, evaluation.
- `src/svm_studio/itemset_mining.py`: Apriori-based mining on discretized attributes.
- `src/svm_studio/episode_mining.py`: serial episode mining on event sequences.
- `src/svm_studio/visualization.py`: all charts.
- `tests/`: lightweight unit tests for the custom mining logic.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Run the UI

```powershell
streamlit run streamlit_app.py
```

The UI lets you:

- upload your own CSV or start from a built-in demo dataset
- use built-in mining demos including `Online Retail Baskets` for itemsets and `MSNBC Journey Sequences` for episodes
- choose the target column and the feature columns
- see which columns matter most through permutation importance
- visualize the separator as a 2D decision boundary, or as a true 3D plane when the model is linear and binary
- work inside a redesigned card-based interface with a styled data atlas, model lab, and mining views
- read the core math and methods in boxed UI panels, including support formulas, margin logic, and scientific-notation examples
- run the `Mining Checker` in `Data Atlas` to see whether a spreadsheet is ready for itemset mining, episode mining, or needs reshaping
- run itemset mining on selected columns
- run episode mining from either a delimited sequence column or ordered event columns
- download CSV outputs directly from the page

## Included external demo data

- `data/external/itemset_uci_onlineretail.csv`: prepared invoice-level itemset spreadsheet derived from the UCI `Online Retail` dataset.
- `data/external/episode_uci_msnbc.csv`: prepared sequence spreadsheet derived from the UCI `MSNBC.com Anonymous Web Data` dataset.
- `data/external/cancer_uci.csv`: cancer classification spreadsheet.
- `data/external/fraud_openml.csv`: repo-safe stratified sample of the fraud classification spreadsheet.
- `data/external/dataset_manifest.csv`: local manifest with source, common name, row count, and notes.

## Outputs

Running the project creates:

- `outputs/svm_kernel_results.csv`
- `outputs/svm_selected_results.csv`
- `outputs/itemset_results.csv`
- `outputs/episode_results.csv`
- `outputs/project_summary.md`
- Several `.png` charts for decision boundaries, confusion matrices, support vectors, itemsets, and episodes

## How to read the results like a pro

- Start with `svm_kernel_results.csv` to compare kernels by cross-validation and test accuracy.
- Use the simple Iris decision-boundary plot to see how kernels change geometry.
- Use the Breast Cancer and Digits PCA plots to see where support vectors concentrate.
- Check confusion matrices to find class-specific failure modes.
- Read `project_summary.md` last to connect the SVM results with the mined patterns.
