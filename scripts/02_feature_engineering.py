# %% Imports
import sys
sys.path.insert(0, "..")

from src.logging_utils import Tee
_log = Tee("../results/02_feature_engineering.txt")

import pandas as pd

from src.features import prepare_combined_datasets

# %% Load cleaned dataset
df = pd.read_csv("../data/dataset_cleaned.csv")
print(f"Loaded: {df.shape}")

# %% Build all feature-set / target combinations
enhanced_datasets = prepare_combined_datasets(
    df,
    include_molecular=True,
    include_structural=True,
)

# %% Dataset summary
for combo, targets in enhanced_datasets.items():
    print(f"\n{combo}:")
    for target, data in targets.items():
        category = "Control" if "Control" in target else "MMM"
        print(f"  {target} [{category}]: "
              f"Train={len(data['y_train'])}, "
              f"Test={len(data['y_test'])}, "
              f"Features={len(data['feature_names'])}")

# %% Verify feature count difference between Control and MMM models
if "Structural_Only" in enhanced_datasets:
    ctrl = enhanced_datasets["Structural_Only"].get("PCO2_Control", {})
    mmm = enhanced_datasets["Structural_Only"].get("PCO2_MMM", {})
    print(f"\nStructural_Only feature counts:")
    print(f"  Control: {len(ctrl.get('feature_names', []))}")
    print(f"  MMM:     {len(mmm.get('feature_names', []))}")
    excluded = set(mmm.get("feature_names", [])) - set(ctrl.get("feature_names", []))
    print(f"  MOF-specific features excluded from Control: {len(excluded)}")
    for f in sorted(excluded):
        print(f"    {f}")

_log.close()
