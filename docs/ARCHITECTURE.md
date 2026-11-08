# Architecture

```text
Streamlit UI
    |
    +-- Locked experiment configuration (TOML)
    |
    +-- Dataset loader / stratified split
    |
    +-- Model factory
    |      +-- Logistic Regression
    |      +-- Decision Tree
    |      +-- Random Forest
    |      +-- Support Vector Machine
    |      +-- K-Nearest Neighbors
    |
    +-- Training service
    |      +-- Standard mode
    |      +-- Tuned mode / GridSearchCV
    |      +-- 5-fold stratified CV
    |
    +-- Evaluation service
    |      +-- Accuracy
    |      +-- Precision
    |      +-- Recall (malignant)
    |      +-- F1
    |      +-- ROC-AUC
    |
    +-- Reporting / visualizations
    |
    +-- Portable result package
           +-- local team-results collection
           +-- ZIP export/import
           +-- experiment fingerprint validation
```

## Design principles

- **Separation of concerns:** configuration, data, models, training, evaluation, reporting, storage, and UI are separate modules.
- **DRY:** all models use one shared experiment pipeline and evaluation contract.
- **Open/closed model factory:** model choices are centralized and can be extended without rewriting the application workflow.
- **Reproducibility:** the experiment contract is hashed into a fingerprint stored with every result.
- **No test leakage:** preprocessing is contained inside scikit-learn pipelines and fitted only on training folds.
- **Portable collaboration:** result ZIPs contain only data/reports; imported packages never execute code.
