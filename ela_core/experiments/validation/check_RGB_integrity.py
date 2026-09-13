import pandas as pd

ela_log = pd.read_csv(r"D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/dataset_log_ela.csv")
rgb_log = pd.read_csv(r"D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_RGB_v2/dataset_log.csv")

merged = ela_log.merge(rgb_log, on="filename", suffixes=("_ela", "_rgb"), how="outer", indicator=True)

print("Row counts — ELA:", len(ela_log), "| RGB:", len(rgb_log))
print("Rows only in one side (should be 0):", (merged["_merge"] != "both").sum())
print("q_primary mismatches (should be 0):", (merged["q_primary_ela"] != merged["q_primary_rgb"]).sum())
print("split mismatches (should be 0):", (merged["split_ela"] != merged["split_rgb"]).sum())