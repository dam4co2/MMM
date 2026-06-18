# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/03_ml_benchmark.txt")

import pandas as pd

from src.features import prepare_combined_datasets
from src.models import run_benchmark, save_best_models
from src.plots import plot_benchmark_analysis

# %% Load cleaned dataset and build feature sets
df = pd.read_csv("../data/dataset_cleaned.csv")
print(f"Loaded: {df.shape}")

enhanced_datasets = prepare_combined_datasets(df, include_molecular=True, include_structural=True)

# %% Run benchmark — all models × datasets × targets
results = run_benchmark(enhanced_datasets, cv_folds=5)
print(f"\nTotal experiments: {len(results)}")

# %% Best results per target
all_targets = ["PCO2_Control", "PCO2_MMM", "PN2_Control", "PN2_MMM"]
print("\nBest Test R² per target:")
for target in all_targets:
    sub = results[results["Target"] == target]
    if sub.empty:
        continue
    best = sub.loc[sub["Test_R2"].idxmax()]
    gap_flag = "" if best["Overfit_Gap"] <= 0.1 else " (moderate overfit)" if best["Overfit_Gap"] <= 0.2 else " (severe overfit)"
    print(f"  {target:15s}: {best['Model']:15s}  Test R²={best['Test_R2']:.3f}  "
          f"CV R²={best['CV_R2_Mean']:.3f}±{best['CV_R2_Std']:.3f}  "
          f"({best['Dataset']}){gap_flag}")

# %% Overfitting analysis
avg_gap = results["Overfit_Gap"].mean()
severe = (results["Overfit_Gap"] > 0.2).sum()
print(f"\nAverage overfitting gap: {avg_gap:.3f}")
print(f"Experiments with severe overfitting (gap > 0.2): {severe}/{len(results)}")

# %% Figure 6 — Benchmark analysis (9-panel)
plot_benchmark_analysis(results, "../images/figure_06_benchmark_analysis.png")
results.to_csv("../plot_data/figure_06_benchmark_analysis.csv", index=False)

# %% Save best model per target
saved_models, comparison = save_best_models(results, enhanced_datasets, save_dir="../data/best_models")
print("\nSaved models summary:")
print(comparison.to_string(index=False))
comparison.to_csv("../plot_data/model_comparison_summary.csv", index=False)

_log.close()
