# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/06_selectivity_analysis.txt")

import os

import numpy as np
import pandas as pd

from src.analysis import (
    categorize_features,
    compare_top_vs_bottom,
    compute_selectivity_correlations,
)
from src.models import load_model
from src.plots import plot_radial_correlations, plot_top_vs_bottom_selectivity

# %% Load best MMM models
model_dir = "../data/best_models"
saved_models = {}
for fname in os.listdir(model_dir):
    if fname.endswith(".pkl"):
        pkg = load_model(os.path.join(model_dir, fname))
        saved_models[pkg["target_name"]] = pkg

# %% Build analysis dataframe (test set — avoids data leakage)
pkg_co2 = saved_models.get("PCO2_MMM")
pkg_n2 = saved_models.get("PN2_MMM")

if pkg_co2 is None or pkg_n2 is None:
    raise RuntimeError("PCO2_MMM and PN2_MMM models are required for selectivity analysis.")

X_test_co2 = pkg_co2["data"]["X_test"]
y_co2_obs = pkg_co2["data"]["y_test"]
y_n2_obs = pkg_n2["data"]["y_test"]

common_idx = y_co2_obs.index.intersection(y_n2_obs.index)
X = X_test_co2.loc[common_idx]
y_co2 = y_co2_obs.loc[common_idx]
y_n2 = y_n2_obs.loc[common_idx]

analysis_df = X.copy()
analysis_df["PCO2_MMM"] = y_co2
analysis_df["PN2_MMM"] = y_n2
analysis_df["log10_alpha"] = y_co2 - y_n2
analysis_df["alpha"] = 10 ** (y_co2 - y_n2)

print(f"Analysis dataset: {analysis_df.shape}")
print(f"log10(alpha): mean={analysis_df['log10_alpha'].mean():.2f}, "
      f"median={analysis_df['log10_alpha'].median():.2f}, "
      f"std={analysis_df['log10_alpha'].std():.2f}")

# %% Selectivity correlations — all features
feature_cols = [c for c in X.columns if c not in ["PCO2_MMM", "PN2_MMM", "log10_alpha", "alpha"]]
corr_df = compute_selectivity_correlations(analysis_df, feature_cols, "log10_alpha")
corr_df.to_csv("../plot_data/selectivity_correlations_all.csv", index=False)
print(f"\nFeatures with |r| > 0.36 and p < 0.001:")
sig = corr_df[(corr_df["abs_pearson_r"] > 0.36) & (corr_df["pearson_p"] < 0.001)]
print(sig[["feature", "pearson_r", "pearson_p"]].to_string(index=False))
sig.to_csv("../plot_data/selectivity_correlations_significant.csv", index=False)

# %% Feature categorisation of top correlates
top25 = corr_df.head(25)
feat_cats = categorize_features(top25["feature"].tolist())
print("\nCategory breakdown of top-25 features:")
for cat, feats in feat_cats.items():
    print(f"  {cat}: {len(feats)}")

top25.to_csv("../plot_data/selectivity_correlations_top25.csv", index=False)

# %% Figure 11 — Radial correlation plot (top 19 features)
plot_radial_correlations(corr_df, "../images/figure_11_radial_correlations.png", n_features=19)
corr_df.head(19).to_csv("../plot_data/figure_11_radial_correlations.csv", index=False)

# %% Top 25% vs Bottom 25% selectivity analysis
top_vs_bottom = compare_top_vs_bottom(
    analysis_df, feature_cols, "log10_alpha", quantile=0.25
)
top_vs_bottom.to_csv("../plot_data/top_vs_bottom_selectivity_all.csv", index=False)

sig_tb = top_vs_bottom[top_vs_bottom["p_ttest"] < 0.0001]
sig_tb.to_csv("../plot_data/top_vs_bottom_selectivity_significant.csv", index=False)
print(f"\nFeatures significant at p < 0.0001: {len(sig_tb)}")
print(sig_tb.head(10)[["feature", "cohens_d", "p_ttest"]].to_string(index=False))

# %% Figure 12 — Top vs Bottom selectivity
plot_top_vs_bottom_selectivity(
    top_vs_bottom, "../images/figure_12_top_vs_bottom_selectivity.png", n_features=20
)
top_vs_bottom.head(20).to_csv(
    "../plot_data/figure_12_top_vs_bottom_selectivity.csv", index=False
)

# %% Summary statistics table
q25 = analysis_df["log10_alpha"].quantile(0.25)
q75 = analysis_df["log10_alpha"].quantile(0.75)
print(f"\nSelectivity thresholds:")
print(f"  Bottom 25% (low selectivity): log10(alpha) <= {q25:.3f}  (alpha <= {10**q25:.0f})")
print(f"  Top 25% (high selectivity):   log10(alpha) >= {q75:.3f}  (alpha >= {10**q75:.0f})")

_log.close()
