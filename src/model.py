"""
Train and tune predictive models (Logistic Regression, XGBoost).

- Standard scaling for LR, tree-based for XGBoost.
- Time-series cross-validation for tuning.
- Generates SHAP plots for interpretability.
- Input: Feature tables (data/processed/)
- Output: Model artifacts, SHAP plots
- Reference: arXiv:2505.16136v1
"""
