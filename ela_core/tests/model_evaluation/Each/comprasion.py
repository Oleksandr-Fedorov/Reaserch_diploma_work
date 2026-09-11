import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

rgb_val = pd.read_csv("D:/Diplom/ela_core/models/RGB_model/rgb_val_predictions.csv").set_index("filename")
ela_val = pd.read_csv("D:/Diplom/ela_core/models/ELA_model/ela_val_predictions.csv").set_index("filename")
merged_val = rgb_val[["true_label","prob_fake"]].join(ela_val[["prob_fake"]], lsuffix="_rgb", rsuffix="_ela")

# --- Этап 1: только alpha, порог = 0.5 ---
alphas = np.linspace(0, 1, 101)
best_alpha, best_f1_stage1 = None, -1
for a in alphas:
    fused = a * merged_val["prob_fake_rgb"] + (1 - a) * merged_val["prob_fake_ela"]
    preds = (fused >= 0.5).astype(int)
    f1 = f1_score(merged_val["true_label"], preds, average="macro")
    if f1 > best_f1_stage1:
        best_alpha, best_f1_stage1 = a, f1

print(f"Этап 1: alpha={best_alpha:.2f}, val macro F1 (порог=0.5)={best_f1_stage1:.4f}")

# --- Этап 2: порог при фиксированном alpha ---
fused_val = best_alpha * merged_val["prob_fake_rgb"] + (1 - best_alpha) * merged_val["prob_fake_ela"]
fpr, tpr, thresholds = roc_curve(merged_val["true_label"], fused_val)
f1_scores = [f1_score(merged_val["true_label"], (fused_val >= t).astype(int), average="macro") for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]

print(f"Этап 2: threshold={best_thresh:.4f}, val macro F1={max(f1_scores):.4f}")

# --- Один раз на test ---
rgb_test = pd.read_csv("D:/Diplom/ela_core/models/RGB_model/rgb_test_predictions.csv").set_index("filename")
ela_test = pd.read_csv("D:/Diplom/ela_core/models/ELA_model/ela_sourceaware_test_predictions.csv").set_index("filename")
merged_test = rgb_test[["true_label","prob_fake"]].join(ela_test[["prob_fake"]], lsuffix="_rgb", rsuffix="_ela")

fused_test = best_alpha * merged_test["prob_fake_rgb"] + (1 - best_alpha) * merged_test["prob_fake_ela"]
preds_test = (fused_test >= best_thresh).astype(int)

print("Test macro F1:", f1_score(merged_test["true_label"], preds_test, average="macro"))
print("Test AUC:", roc_auc_score(merged_test["true_label"], fused_test))