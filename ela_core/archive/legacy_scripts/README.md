# Experimental Test Scripts

This folder contains exploratory scripts for ELA-based image-forgery work. They
are grouped by what they test rather than by creation order.

## `ela_methods/`

- `compare_ela_srm_residuals.py` compares a standard ELA map with an SRM-style
  high-pass residual map for one manipulated image.
- `quantization_alignment_metrics.py` measures ELA mean, standard deviation,
  and entropy across primary JPEG quality levels.

## `model_evaluation/`

- `roc_auc_baseline_model.py` computes ROC-AUC and an optimal threshold for one
  baseline model on a prepared test split.
- `threshold_pr_confusion_v1.py` evaluates the V1 model with validation-selected
  thresholds and final test confusion metrics.
- `threshold_pr_confusion_v2.py` evaluates the V2/Kaggle model with the same
  thresholding pattern.

## `robustness/`

- `single_model_recompression_sweep.py` tests one model on selected images after
  simulated JPEG recompression.
- `compare_models_recompression_sweep.py` compares two models across JPEG
  recompression levels using accuracy and FAKE recall.

## `data_integrity/`

- `check_train_val_filename_leakage.py` verifies whether train and validation
  subsets share `original_filename` values.

Run scripts from the repository root so their relative `models/...` and
`datasets/...` paths resolve correctly.
