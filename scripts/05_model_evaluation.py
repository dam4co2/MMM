# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/05_model_evaluation.txt")

import os

import numpy as np
import pandas as pd

from src.models import load_model
from src.plots import (
    plot_cv_scatterplots,
    plot_overfitting_summary,
    plot_train_test_scatterplots,
)

# %% Load best models
model_dir = "../data/best_models"
saved_models = {}
for fname in os.listdir(model_dir):
    if fname.endswith(".pkl"):
        pkg = load_model(os.path.join(model_dir, fname))
        saved_models[pkg["target_name"]] = pkg

print("Loaded models:", list(saved_models.keys()))

# %% Figure 8 — CV scatter plots
plot_cv_scatterplots(saved_models, "../images/figure_08_cv_scatterplots.png", cv_folds=5)

scatter_rows = []
for target, pkg in saved_models.items():
    for obs, pred, split in [
        (pkg["data"]["y_train"].values, pkg["data"]["y_pred_train"], "Train"),
        (pkg["data"]["y_test"].values, pkg["data"]["y_pred_test"], "Test"),
    ]:
        for o, p in zip(obs, pred):
            scatter_rows.append({"Target": target, "Split": split,
                                  "Observed": o, "Predicted": p, "Residual": o - p})

scatter_df = pd.DataFrame(scatter_rows)
scatter_df.to_csv("../plot_data/figure_08_cv_scatterplots.csv", index=False)

# %% Figure 9 — Train/test scatter plots
plot_train_test_scatterplots(saved_models, "../images/figure_09_train_test_scatterplots.png")
scatter_df.to_csv("../plot_data/figure_09_train_test_scatterplots.csv", index=False)

# %% Figure 10 — Overfitting gap summary
plot_overfitting_summary(saved_models, "../images/figure_10_overfitting_summary.png", cv_folds=5)

overfit_rows = []
for target, pkg in saved_models.items():
    overfit_rows.append({
        "Target": target,
        "Train_R2": pkg["metrics"]["train_r2"],
        "Test_R2": pkg["metrics"]["test_r2"],
        "CV_R2_Mean": pkg["metrics"]["cv_r2_mean"],
        "CV_R2_Std": pkg["metrics"]["cv_r2_std"],
        "Overfit_Gap": pkg["metrics"]["overfit_gap"],
        "Test_RMSE": pkg["metrics"]["test_rmse"],
        "Test_MAE": pkg["metrics"]["test_mae"],
        "N_Features": pkg["training_info"]["n_features"],
        "N_Train": pkg["training_info"]["n_train"],
        "N_Test": pkg["training_info"]["n_test"],
    })

overfit_df = pd.DataFrame(overfit_rows)
overfit_df.to_csv("../plot_data/figure_10_overfitting_summary.csv", index=False)
print(overfit_df.to_string(index=False))

# %% Diagonal reference data for figure reconstruction
diag_rows = []
for target, pkg in saved_models.items():
    all_obs = np.concatenate([pkg["data"]["y_train"].values, pkg["data"]["y_test"].values])
    lo, hi = all_obs.min() - 0.5, all_obs.max() + 0.5
    diag_rows.append({"Target": target, "x_lo": lo, "x_hi": hi,
                       "band_lo": lo - 1, "band_hi": hi + 1})

pd.DataFrame(diag_rows).to_csv("../plot_data/diagonal_reference.csv", index=False)

_log.close()
