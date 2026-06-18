# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/01_data_preparation.txt")

import pandas as pd

from src.data import (
    TARGETS,
    compute_permeability_targets,
    descriptive_stats,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_modified_zscore,
    fix_smiles,
    load_dataset,
    normality_tests,
    remove_outliers,
)
from src.plots import (
    plot_correlation_heatmap,
    plot_distributions_overview,
    plot_outliers_identification,
    plot_scatter_matrix,
    plot_violin_comparison,
)

# %% Load raw dataset
df_raw = load_dataset("../data/dataset.csv")
print(f"Raw dataset: {df_raw.shape}")
print(df_raw.columns.tolist())

# %% Compute log-permeability targets and selectivity
df = compute_permeability_targets(df_raw)
df = df.drop(columns=["polymer_type"], errors="ignore")
print(f"After target computation: {df.shape}")
print(df[TARGETS].describe())

# %% Fix known broken SMILES
df = fix_smiles(df)
print(f"Rows after SMILES fix: {len(df)}")

# %% Descriptive statistics for targets
stats_df = descriptive_stats(df, TARGETS)
print(stats_df.to_string())
stats_df.to_csv("../plot_data/descriptive_stats_raw.csv")

# %% Outlier detection — IQR, Z-score, Modified Z-score
outlier_rows = []
for target in TARGETS:
    data = df[target].dropna()
    mask_iqr, lower, upper = detect_outliers_iqr(data)
    mask_z = detect_outliers_zscore(data)
    mask_mz = detect_outliers_modified_zscore(data)
    outlier_rows.append({
        "Target": target,
        "N": len(data),
        "Outliers_IQR": mask_iqr.sum(),
        "Outliers_IQR_pct": f"{mask_iqr.sum() / len(data) * 100:.2f}%",
        "Outliers_Zscore": mask_z.sum(),
        "Outliers_ModZ": mask_mz.sum(),
        "IQR_Lower": round(lower, 3),
        "IQR_Upper": round(upper, 3),
    })

outlier_summary = pd.DataFrame(outlier_rows)
print(outlier_summary.to_string(index=False))
outlier_summary.to_csv("../plot_data/outlier_summary.csv", index=False)

# %% Normality tests
norm_df = normality_tests(df, TARGETS)
print(norm_df.to_string(index=False))
norm_df.to_csv("../plot_data/normality_tests.csv", index=False)

# %% Figure 1 — Distribution overview (histogram + boxplot + Q-Q)
plot_distributions_overview(df, TARGETS, "../images/figure_01_distributions_overview.png")
df[TARGETS].to_csv("../plot_data/figure_01_distributions_overview.csv", index=False)

# %% Figure 2 — Violin plot comparison
plot_violin_comparison(df, TARGETS, "../images/figure_02_violin_plots.png")
df[TARGETS].melt(var_name="Target", value_name="Value").to_csv(
    "../plot_data/figure_02_violin_plots.csv", index=False
)

# %% Figure 3 — Correlation heatmap
plot_correlation_heatmap(df, TARGETS, "../images/figure_03_correlation_heatmap.png")
df[TARGETS].corr().to_csv("../plot_data/figure_03_correlation_heatmap.csv")

# %% Figure 4 — Scatter matrix
plot_scatter_matrix(df, TARGETS, "../images/figure_04_scatter_matrix.png")
df[TARGETS].to_csv("../plot_data/figure_04_scatter_matrix.csv", index=False)

# %% Figure 5 — Outlier identification
plot_outliers_identification(df, TARGETS, "../images/figure_05_outliers_identification.png")
flags = pd.DataFrame(
    {t: detect_outliers_iqr(df[t])[0] for t in TARGETS},
    index=df.index,
)
flags.to_csv("../plot_data/figure_05_outliers_identification.csv")

# %% Remove outliers — strategy 'any' (recommended for this dataset)
df_clean = remove_outliers(df, TARGETS, strategy="any")
print(f"Rows removed: {len(df) - len(df_clean)} ({(len(df) - len(df_clean)) / len(df) * 100:.1f}%)")
print(f"Cleaned dataset: {df_clean.shape}")

# %% Save cleaned dataset
df_clean.to_csv("../data/dataset_cleaned.csv", index=False)
print("Saved: ../data/dataset_cleaned.csv")

# %% Missing value summary on cleaned dataset
missing_info = {
    "no_nan": [],
    "with_nan": [],
}
for col in df_clean.columns:
    n = df_clean[col].isna().sum()
    if n == 0:
        missing_info["no_nan"].append(col)
    else:
        missing_info["with_nan"].append((col, n, f"{n / len(df_clean) * 100:.1f}%"))

print(f"\nColumns without NaN ({len(missing_info['no_nan'])}):")
for c in missing_info["no_nan"]:
    print(f"  {c}")
print(f"\nColumns with NaN ({len(missing_info['with_nan'])}):")
for col, n, pct in missing_info["with_nan"]:
    print(f"  {col}: {n} NaN ({pct})")

_log.close()
