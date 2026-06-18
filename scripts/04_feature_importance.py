# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/04_feature_importance.txt")

import os
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis import (
    analyze_importance_by_category,
    categorize_features,
    group_permutation_importance,
    residual_analysis,
)
from src.models import load_model

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed — SHAP analysis will be skipped.")

# %% Load best models
model_dir = "../data/best_models"
saved_models = {}
for fname in os.listdir(model_dir):
    if fname.endswith(".pkl"):
        pkg = load_model(os.path.join(model_dir, fname))
        saved_models[pkg["target_name"]] = pkg

print("Loaded models:", list(saved_models.keys()))

# %% MOF contribution analysis — PCO2_MMM and PN2_MMM
mmm_targets = ["PCO2_MMM", "PN2_MMM"]
all_residual_results = {}

for target in mmm_targets:
    if target not in saved_models:
        print(f"Model for {target} not found, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"MOF Contribution Analysis: {target}")
    print(f"{'='*60}")

    pkg = saved_models[target]
    X_train = pkg["data"]["X_train"]
    X_test = pkg["data"]["X_test"]
    y_train = pkg["data"]["y_train"]
    y_test = pkg["data"]["y_test"]
    model = pkg["model"]

    feature_cats = categorize_features(X_train.columns.tolist())
    print("\nFeature categories:")
    for cat, feats in feature_cats.items():
        print(f"  {cat}: {len(feats)}")

    print("\n--- Predictive power by category (isolated) ---")
    cat_importance = analyze_importance_by_category(X_train, y_train, model, feature_cats)
    for cat, res in cat_importance.items():
        print(f"  {cat}: CV R²={res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")
        print(f"    Top 3: {res['importance_df']['feature'].head(3).tolist()}")

    print("\n--- Group permutation importance ---")
    perm_imp = group_permutation_importance(X_train, y_train, X_test, y_test, model, feature_cats)
    for cat, res in perm_imp.items():
        print(f"  {cat}: importance={res['importance']:.4f} ± {res['importance_std']:.4f}")

    print("\n--- Residual analysis (polymer vs MOF) ---")
    resid = residual_analysis(X_train, y_train, X_test, y_test, model, feature_cats)
    poly_r2 = resid.get("polymer_only", {}).get("r2_test", 0)
    mof_contrib = resid.get("mof_residual", {}).get("mof_contribution", 0)
    combined_r2 = resid.get("combined", {}).get("r2_test", 0)
    interaction = max(0, combined_r2 - poly_r2 - mof_contrib)
    print(f"  Polymer alone:  {poly_r2 * 100:.1f}% variance")
    print(f"  MOF adds:       {mof_contrib * 100:.1f}% variance")
    print(f"  Combined:       {combined_r2 * 100:.1f}% variance")
    print(f"  Interaction:    {interaction * 100:.1f}% variance")

    all_residual_results[target] = {
        "feature_cats": feature_cats,
        "cat_importance": cat_importance,
        "perm_importance": perm_imp,
        "residual": resid,
    }

    if "mof_importance" in resid:
        mof_imp_df = resid["mof_importance"]
        mof_imp_df.to_csv(f"../plot_data/{target}_mof_feature_importance.csv", index=False)
        print(f"  Saved: ../plot_data/{target}_mof_feature_importance.csv")

# %% Figure: variance decomposition (pie charts)
fig, axes = plt.subplots(1, len(mmm_targets), figsize=(14, 6))
for ax, target in zip(axes, mmm_targets):
    if target not in all_residual_results:
        ax.set_visible(False)
        continue
    r = all_residual_results[target]["residual"]
    poly_r2 = r.get("polymer_only", {}).get("r2_test", 0)
    mof_contrib = r.get("mof_residual", {}).get("mof_contribution", 0)
    combined_r2 = r.get("combined", {}).get("r2_test", 0)
    interaction = max(0, combined_r2 - poly_r2 - mof_contrib)
    unexplained = max(0, 1 - combined_r2)

    sizes = [poly_r2, mof_contrib, interaction, unexplained]
    labels = [f"Polymer\n({poly_r2*100:.1f}%)",
              f"MOF\n({mof_contrib*100:.1f}%)",
              f"Interaction\n({interaction*100:.1f}%)",
              f"Unexplained\n({unexplained*100:.1f}%)"]
    ax.pie(sizes, labels=labels, colors=["#66b3ff", "#ff9999", "#99ff99", "#ffcc99"],
           startangle=90)
    ax.set_title(f"{target} — Variance Decomposition", fontweight="bold")

plt.suptitle("Polymer vs MOF Contribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../images/figure_07_variance_decomposition.png", dpi=300, bbox_inches="tight")
plt.close()

decomp_rows = []
for target in mmm_targets:
    if target not in all_residual_results:
        continue
    r = all_residual_results[target]["residual"]
    poly_r2 = r.get("polymer_only", {}).get("r2_test", 0)
    mof_contrib = r.get("mof_residual", {}).get("mof_contribution", 0)
    combined_r2 = r.get("combined", {}).get("r2_test", 0)
    decomp_rows.append({
        "Target": target,
        "Polymer_R2": poly_r2,
        "MOF_contribution": mof_contrib,
        "Combined_R2": combined_r2,
        "Interaction": max(0, combined_r2 - poly_r2 - mof_contrib),
        "Unexplained": max(0, 1 - combined_r2),
    })

pd.DataFrame(decomp_rows).to_csv(
    "../plot_data/figure_07_variance_decomposition.csv", index=False
)

# %% SHAP analysis by category (optional)
if SHAP_AVAILABLE:
    for target in mmm_targets:
        if target not in saved_models:
            continue
        pkg = saved_models[target]
        X_train = pkg["data"]["X_train"]
        X_test = pkg["data"]["X_test"]
        model = pkg["model"]
        feature_cats = all_residual_results[target]["feature_cats"]

        X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_sample).values

        mean_shap = pd.Series(
            abs(shap_values).mean(axis=0), index=X_train.columns
        )
        shap_by_cat = {}
        for cat, feats in feature_cats.items():
            cat_feats = [f for f in feats if f in mean_shap.index]
            if cat_feats:
                shap_by_cat[cat] = mean_shap[cat_feats].sum()

        shap_df = pd.DataFrame(list(shap_by_cat.items()),
                               columns=["Category", "Total_SHAP"])
        shap_df["SHAP_pct"] = shap_df["Total_SHAP"] / shap_df["Total_SHAP"].sum() * 100
        shap_df = shap_df.sort_values("Total_SHAP", ascending=False)
        print(f"\nSHAP by category — {target}:")
        print(shap_df.to_string(index=False))
        shap_df.to_csv(f"../plot_data/{target}_shap_by_category.csv", index=False)

_log.close()
