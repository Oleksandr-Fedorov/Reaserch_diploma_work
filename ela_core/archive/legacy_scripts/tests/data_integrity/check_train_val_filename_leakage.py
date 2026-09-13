"""Check whether train and validation splits share original filenames.

This reproduces the same random_split seed used during training and reports
overlap in original_filename values. Use it before trusting validation metrics.
"""

import pandas as pd
import torch
from torch.utils.data import random_split

# Загружаем лог
df = pd.read_csv("datasets/CASIA_Unified_Randomized_v2/dataset_log.csv")

# Оставляем только train (именно его потом random_split делил на train/val)
train_df = df[df["split"] == "train"].reset_index(drop=True)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
train_size = len(train_df) - val_size

# ВОСПРОИЗВОДИМ ТО ЖЕ random_split
train_subset, val_subset = random_split(
    train_df,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_idx = train_subset.indices
val_idx = val_subset.indices

train_names = set(train_df.iloc[train_idx]["original_filename"])
val_names = set(train_df.iloc[val_idx]["original_filename"])

intersection = train_names & val_names

print(f"Уникальных original_filename в train: {len(train_names)}")
print(f"Уникальных original_filename в val:   {len(val_names)}")
print(f"Пересечений: {len(intersection)}")

if len(intersection):
    print("\nПервые 20 совпадений:")
    for x in list(intersection)[:20]:
        print(x)
else:
    print("\nLeakage нет.")
