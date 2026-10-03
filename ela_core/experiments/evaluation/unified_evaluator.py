"""Evaluate preserved RGB/ELA/fusion outputs or explicit offline RGB/ELA pairs.

No dataset discovery, checkpoint download, training, or import-time execution.
See unified_evaluator.md for provenance and validation-split limitations.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["REAL", "FAKE"]
FEATURE_THRESHOLD = 0.2893
CHECKPOINTS = {
    "rgb": "models/RGB_model/model_resnet_rgb_LATEST.pth",
    "ela": "models/ELA_model/model_resnet_ela_sourceaware_v2.pth",
    "feature_fusion_v2": "models/Fusion_v2/model_fusion_sourceaware_v2.pth",
}
TABLES = {
    "rgb": "models/RGB_model/rgb_test_predictions.csv",
    "ela": "models/ELA_model/ela_sourceaware_test_predictions.csv",
    "feature_fusion_v2": "artifacts/fusion_v2/fusion_v2_test_metadata.csv",
    "rgb_val": "models/RGB_model/rgb_val_predictions.csv",
    "ela_val": "models/ELA_model/ela_val_predictions.csv",
}
SPLIT = "models/ELA_model/ela_sourceaware_split_v2.csv"
IMAGE_CONTRACT = {
    "class_to_idx": {"REAL": 0, "FAKE": 1},
    "prob_fake": "softmax(logits, dim=1)[:, 1]",
    "rgb": "offline q_primary-recompressed RGB PNG, not an unprocessed original",
    "ela": "paired offline ELA PNG; no online recompression or second ELA transform",
    "ela_generation": "abs(simulated RGB - JPEG(q_ela)) * 10, clip [0,255], uint8",
    "pairing": "same filename, augmentation and label in RGB and ELA",
    "transform": "PIL convert RGB -> Resize((224,224), bilinear) -> ToTensor",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "fusion_order": ["rgb_512", "ela_512"],
    "fusion_head": "Linear(1024,256), ReLU, Dropout(0.5), Linear(256,2); eval()",
    "evidence": ["notebooks/Colab.ipynb cells 13,20,23 (zero-based)",
                 "experiments/preprocessing/merge_CASIA2.py",
                 "experiments/preprocessing/merga_CASIA3_RGB.py"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def validate_table(frame, name, probability=True):
    columns = {"filename", "true_label"} | ({"prob_fake"} if probability else set())
    require(columns <= set(frame), f"{name}: missing columns {columns - set(frame)}")
    require(len(frame) > 0, f"{name}: empty table")
    require(frame[list(columns)].notna().all().all(), f"{name}: null values")
    require(frame.filename.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
            f"{name}: empty/non-string filename")
    require(not frame.filename.duplicated().any(), f"{name}: duplicate filenames")
    require(frame.true_label.isin([0, 1]).all(), f"{name}: labels must be 0=REAL,1=FAKE")
    require(set(frame.true_label) == {0, 1}, f"{name}: both classes are required")
    # CASIA names encode labels. Generic external names have no inferred labels.
    encoded = frame.filename.str.extract(r"^CASIA[12]_(REAL|FAKE)_", expand=False)
    mask = encoded.notna()
    require((encoded[mask].map({"REAL": 0, "FAKE": 1}) == frame.loc[mask, "true_label"]).all(),
            f"{name}: filename class conflicts with true_label")
    if probability:
        p = frame.prob_fake.to_numpy(dtype=np.float64)
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                f"{name}: invalid probabilities")
    if "pred_label" in frame:
        require(frame.pred_label.isin([0, 1]).all(), f"{name}: invalid pred_label")
    return frame.copy()


def read_table(path):
    return validate_table(pd.read_csv(path), str(path))


def align(reference, other, name):
    require(set(reference.filename) == set(other.filename), f"{name}: filename sets differ")
    other = other.set_index("filename").loc[reference.filename].reset_index()
    require(np.array_equal(reference.true_label, other.true_label), f"{name}: labels differ")
    return other


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    require(y.ndim == p.ndim == 1 and len(y) == len(p), "labels/probabilities shape mismatch")
    require(set(y) == {0, 1}, "metrics require both binary classes")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid metric probabilities")
    require(np.isfinite(threshold), "threshold must be finite")
    pred = (p >= threshold).astype(np.int64)
    return {
        "rows": len(y), "threshold": float(threshold), "comparison": "prob_fake >= threshold",
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision_fake": float(average_precision_score(y, p)),
        "per_class": classification_report(y, pred, labels=[0, 1],
            target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


def source_stem(filename):
    match = re.fullmatch(r"(CASIA[12])_(REAL|FAKE)_(.+)_aug([1-4])\.png", filename)
    require(match is not None, f"Cannot recover source key from {filename!r}")
    return match[1], match[3]


def split_evidence(root, rgb_val, ela_val, test):
    """Reconstruct historical split from retained source manifest, not datasets.

    This is conditional evidence: state_dicts contain no split metadata.
    """
    manifest = pd.read_csv(root / SPLIT)
    required = {"source_dataset", "original_filename", "subset"}
    require(required <= set(manifest), "source manifest columns missing")
    require(manifest[list(required)].notna().all().all(), "null source manifest values")
    require(not manifest.duplicated(["source_dataset", "original_filename"]).any(),
            "duplicate sources in split manifest")
    require(set(manifest.subset) == {"train", "val"}, "unexpected source manifest subsets")
    records = {}
    for row in manifest.itertuples(index=False):
        key = (row.source_dataset, Path(row.original_filename).stem)
        require(key not in records, f"Ambiguous source stem: {key}")
        records[key] = (row.original_filename, row.subset)
    originals = sorted(set(manifest.original_filename))
    random.Random(42).shuffle(originals)
    rgb_heldout = set(originals[:round(len(originals) * 0.1)])
    composite = sorted(zip(manifest.source_dataset, manifest.original_filename))
    random.Random(42).shuffle(composite)
    expected_ela_val = set(composite[:round(len(composite) * 0.1)])
    actual_ela_val = set(zip(manifest.loc[manifest.subset == "val", "source_dataset"],
                             manifest.loc[manifest.subset == "val", "original_filename"]))
    require(expected_ela_val == actual_ela_val, "ELA manifest differs from notebook split")
    ela_val = align(rgb_val, ela_val, "validation RGB/ELA")
    keys = [source_stem(f) for f in rgb_val.filename]
    require(all(k in records and records[k][1] == "val" for k in keys),
            "cached validation rows disagree with ELA validation manifest")
    expected_keys = {k for k, (_, subset) in records.items() if subset == "val"}
    require(set(keys) == expected_keys, "cached validation does not cover ELA val sources")
    counts = pd.Series(keys).value_counts()
    require((counts == 4).all(), "validation must contain four augmentations per source")
    test_keys = {source_stem(f) for f in test.filename}
    require(not test_keys.intersection(records), "held-out test overlaps training/validation sources")
    clean = np.array([records[k][0] in rgb_heldout for k in keys], dtype=bool)
    return {
        "basis": "INFERENCE from notebook cells 13/20/23 and retained ELA split manifest",
        "checkpoint_contains_split_metadata": False,
        "validation_rows": len(keys), "validation_sources": len(set(keys)),
        "common_heldout_rows": int(clean.sum()),
        "rgb_training_rows_in_composite_validation": int((~clean).sum()),
        "rgb_training_sources_in_composite_validation": len({k for k, ok in zip(keys, clean) if not ok}),
        "test_source_overlap": 0,
        "validation_prediction_provenance": "Validation CSVs have no checkpoint hash or exported "
            "embeddings. Their filenames/labels agree; association to weights relies on retained scripts.",
        "warning": "Composite head split alone does not establish end-to-end leakage-free validation: "
                   "the frozen RGB backbone used a different split. Feature Fusion V2's validation "
                   "selection has the same limitation; completed test evaluation remains retained.",
    }, clean


def choose_decision_parameters(rgb_val, ela_val, clean):
    """Original two-stage search, restricted to sources held out for BOTH branches.

    Uses validation only. This is a new calibration, not the historical full-val run.
    ROC thresholds follow sklearn's default drop_intermediate=True as in comprasion.py.
    """
    ela_val = align(rgb_val, ela_val, "decision validation")
    require(clean.any(), "no common held-out validation rows")
    y = rgb_val.true_label.to_numpy()[clean]
    require(set(y) == {0, 1}, "common validation needs both classes")
    rgb = rgb_val.prob_fake.to_numpy()[clean]
    ela = ela_val.prob_fake.to_numpy()[clean]
    alphas = np.linspace(0, 1, 101)
    scores = [f1_score(y, (a * rgb + (1 - a) * ela >= .5), average="macro") for a in alphas]
    alpha = float(alphas[int(np.argmax(scores))])
    p = alpha * rgb + (1 - alpha) * ela
    candidates = roc_curve(y, p)[2]
    # Preserve the all-negative candidate without a non-JSON infinity.
    candidates = np.where(np.isfinite(candidates), candidates, np.nextafter(p.max(), np.inf))
    scores = [f1_score(y, p >= t, average="macro") for t in candidates]
    return alpha, float(candidates[int(np.argmax(scores))])


def make_model(kind):
    import torch
    from torch import nn
    from torchvision import models

    def branch():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        return model

    if kind in ("rgb", "ela"):
        return branch()

    class LateFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb_branch, self.ela_branch = branch(), branch()
            self.rgb_branch.fc = nn.Identity()
            self.ela_branch.fc = nn.Identity()
            for part in (self.rgb_branch, self.ela_branch):
                for parameter in part.parameters():
                    parameter.requires_grad_(False)
            self.mlp_head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(),
                                          nn.Dropout(.5), nn.Linear(256, 2))

        def train(self, mode=True):
            super().train(mode)
            self.rgb_branch.eval()
            self.ela_branch.eval()
            return self

        def forward(self, rgb, ela):
            with torch.no_grad():
                features = torch.cat((self.rgb_branch(rgb), self.ela_branch(ela)), dim=1)
            return self.mlp_head(features)

    require(kind == "feature_fusion_v2", f"unknown model: {kind}")
    return LateFusionModel()


def load_models(root):
    import torch
    torch.set_num_threads(2)
    states, models = {}, {}
    for kind, path in CHECKPOINTS.items():
        # Never fall back to unrestricted pickle loading.
        state = torch.load(root / path, map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and state, f"{kind}: expected raw state_dict")
        require(all(isinstance(t, torch.Tensor) for t in state.values()), f"{kind}: non-tensor payload")
        require(all(torch.isfinite(t).all().item() for t in state.values()), f"{kind}: nonfinite weights")
        model = make_model(kind)
        model.load_state_dict(state, strict=True)
        model.eval()
        states[kind], models[kind] = state, model
    for kind in ("rgb", "ela"):
        for key, value in states[kind].items():
            if not key.startswith("fc."):
                require(torch.equal(value, states["feature_fusion_v2"][kind + "_branch." + key]),
                        f"Fusion V2 {kind} backbone differs at {key}")
    return states, models


def verify_checkpoints(root, tables):
    """Check real weights against exported features; do not rerun image backbones."""
    import torch
    states, models = load_models(root)
    bundle = root / "artifacts/fusion_v2"
    fusion = tables["feature_fusion_v2"]  # Export order MUST NOT be sorted independently of NPYs.
    n = len(fusion)
    arrays = {}
    for part, width in (("rgb", 512), ("ela", 512), ("fused", 1024)):
        a = np.load(bundle / f"test_embeddings_{part}_{width}d.npy", mmap_mode="r", allow_pickle=False)
        require(a.shape == (n, width) and a.dtype == np.float32, f"{part}: embedding shape/dtype mismatch")
        arrays[part] = a
    labels = np.load(bundle / "test_labels.npy", allow_pickle=False)
    require(labels.shape == (n,) and np.issubdtype(labels.dtype, np.integer)
            and np.array_equal(labels, fusion.true_label), "NPY/CSV labels differ")
    require({"logit_0", "logit_1"} <= set(fusion), "fusion logits missing")
    require(np.isfinite(fusion[["logit_0", "logit_1"]].to_numpy()).all(), "nonfinite cached logits")
    baseline = {k: align(fusion, tables[k], k) for k in ("rgb", "ela")}
    errors = {"rgb_prob_max_abs": 0., "ela_prob_max_abs": 0.,
              "fusion_prob_max_abs": 0., "fusion_logits_max_abs": 0.}
    with torch.inference_mode():
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            batch = {k: np.array(v[start:stop], copy=True) for k, v in arrays.items()}
            require(all(np.isfinite(v).all() for v in batch.values()), "nonfinite embeddings")
            require(np.array_equal(batch["fused"], np.concatenate((batch["rgb"], batch["ela"]), axis=1)),
                    "fusion features are not exact [RGB,ELA] concatenation")
            for kind in ("rgb", "ela"):
                logits = models[kind].fc(torch.from_numpy(batch[kind]))
                p = torch.softmax(logits, dim=1)[:, 1].numpy()
                error = float(np.max(np.abs(p - baseline[kind].prob_fake.iloc[start:stop].to_numpy())))
                errors[kind + "_prob_max_abs"] = max(errors[kind + "_prob_max_abs"], error)
            logits = models["feature_fusion_v2"].mlp_head(torch.from_numpy(batch["fused"]))
            p = torch.softmax(logits, dim=1)[:, 1].numpy()
            errors["fusion_prob_max_abs"] = max(errors["fusion_prob_max_abs"], float(np.max(
                np.abs(p - fusion.prob_fake.iloc[start:stop].to_numpy()))))
            errors["fusion_logits_max_abs"] = max(errors["fusion_logits_max_abs"], float(np.max(
                np.abs(logits.numpy() - fusion[["logit_0", "logit_1"]].iloc[start:stop].to_numpy()))))
    # CPU/GPU float32 GEMM differences are expected, never accept silent large discrepancies.
    require(max(errors[k] for k in errors if "prob" in k) <= 2e-5,
            f"Checkpoint heads do not reproduce saved probabilities: {errors}")
    require(errors["fusion_logits_max_abs"] <= 2e-4, f"Fusion head/logits mismatch: {errors}")
    return {"strict_state_dict_load": True, "finite_weights": True,
            "fusion_backbones_exact_match": True, "embedding_contract": "verified",
            "head_replay_errors": errors, "image_backbones_executed": False,
            "limitation": "Head replay links weights and saved outputs given exported embeddings; "
                          "it does not independently verify the original pixels or historical split.",
            "checkpoint_fingerprints": {k: fingerprint(root / p) for k, p in CHECKPOINTS.items()}}


def comparison(tables, alpha, decision_threshold):
    reference = tables["feature_fusion_v2"]
    aligned = {k: align(reference, v, k) for k, v in tables.items()}
    probabilities = {k: frame.prob_fake.to_numpy() for k, frame in aligned.items()}
    probabilities["decision_fusion"] = alpha * probabilities["rgb"] + (1 - alpha) * probabilities["ela"]
    thresholds = {"rgb": .5, "ela": .5, "feature_fusion_v2": FEATURE_THRESHOLD,
                  "decision_fusion": decision_threshold}
    reports, predictions = {}, []
    for kind, p in probabilities.items():
        reports[kind] = metrics(reference.true_label, p, thresholds[kind])
        frame = reference[["filename", "true_label"]].copy()
        frame["model"], frame["prob_fake"] = kind, p
        frame["pred_label"] = (p >= thresholds[kind]).astype(int)
        predictions.append(frame)
        if kind in aligned and "pred_label" in aligned[kind]:
            # Stored argmax uses REAL at a tie, while the evaluator uses >= throughout.
            mismatches = aligned[kind].pred_label.to_numpy() != frame.pred_label.to_numpy()
            reports[kind]["stored_pred_label_mismatches"] = int(mismatches.sum())
            allowed = (p == .5) if kind in ("rgb", "ela") else np.zeros(len(p), dtype=bool)
            require(not (mismatches & ~allowed).any(), f"{kind}: cached predictions disagree with threshold")
    return reports, pd.concat(predictions, ignore_index=True)


def cached(root, policy="fixed", verify=False):
    require(policy in ("fixed", "common-validation"), "unknown decision policy")
    loaded = {k: read_table(root / p) for k, p in TABLES.items()}
    tables = {k: loaded[k] for k in CHECKPOINTS}
    require(len(tables["feature_fusion_v2"]) == 11432, "Expected retained 11432-row test cohort")
    evidence, clean = split_evidence(root, loaded["rgb_val"], loaded["ela_val"],
                                     tables["feature_fusion_v2"])
    alpha, threshold = .5, .5
    if policy == "common-validation":
        alpha, threshold = choose_decision_parameters(loaded["rgb_val"], loaded["ela_val"], clean)
    reports, predictions = comparison(tables, alpha, threshold)
    result = {
        "mode": "cached", "input_contract": IMAGE_CONTRACT, "models": reports,
        "decision_fusion": {"alpha_rgb": alpha, "threshold": threshold, "policy": policy,
            "description": "Prespecified equal-weight baseline, no fitting" if policy == "fixed" else
            "NEW calibration on common held-out sources inferred from notebook; not historical full-val tuning"},
        "split_evidence": evidence,
        "feature_threshold_provenance": "0.2893 from rounded validation output in notebook cell 23; "
            "full precision threshold and validation probabilities were not retained. "
            "Agreement with test pred_label is checked, never optimized on test.",
        "sources": {k: fingerprint(root / p) for k, p in TABLES.items()},
        "split_manifest": fingerprint(root / SPLIT),
        "notebook_provenance": fingerprint(root / "notebooks/Colab.ipynb"),
        "calibration_warning": "Common-validation is conditional on recovered historical split and "
            "validation CSV provenance; no independent checkpoint-to-val-prediction replay was possible.",
        "checkpoint_verification": verify_checkpoints(root, tables) if verify else
            {"status": "not_requested", "note": "Use --verify-checkpoints for safe strict loading and head replay."},
    }
    return result, predictions


def input_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), transforms.Normalize(IMAGE_CONTRACT["mean"], IMAGE_CONTRACT["std"]),
    ])


def read_pairs(manifest):
    """Explicit paths only; no folder scans, implicit class sorting, or image regeneration."""
    frame = validate_table(pd.read_csv(manifest), str(manifest), probability=False)
    require({"rgb_path", "ela_path"} <= set(frame), "manifest needs rgb_path and ela_path")
    for kind in ("rgb", "ela"):
        require(frame[kind + "_path"].map(lambda p: isinstance(p, str) and bool(p.strip())).all(),
                f"manifest: invalid {kind}_path")
        paths = []
        for filename, value in zip(frame.filename, frame[kind + "_path"]):
            path = Path(value)
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
            require(path.is_file(), f"missing {kind} input: {path}")
            require(path.name == filename, f"{kind}: filename/pair mismatch: {path}")
            require(path.suffix.lower() == ".png", f"{kind}: expected offline PNG: {path}")
            if path.parent.name in CLASS_NAMES:
                label = int(frame.loc[frame.filename == filename, "true_label"].iloc[0])
                require(CLASS_NAMES[label] == path.parent.name, f"{kind}: directory label mismatch")
            paths.append(path)
        require(len(set(paths)) == len(paths), f"{kind}: repeated input path")
        frame[kind + "_path"] = paths
    require((frame.rgb_path != frame.ela_path).all(), "RGB and ELA must be distinct files")
    return frame


def infer(root, manifest, alpha, threshold, batch_size):
    """Explicit opt-in evaluation of prepared offline pairs; never called by cached mode."""
    import torch
    from PIL import Image
    frame = read_pairs(manifest)
    _, models = load_models(root)
    transform = input_transform()
    probabilities = {kind: [] for kind in CHECKPOINTS}
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            tensors = {}
            for kind in ("rgb", "ela"):
                images = []
                for path in frame[kind + "_path"].iloc[start:start + batch_size]:
                    with Image.open(path) as image:
                        images.append(transform(image.convert("RGB")))
                tensors[kind] = torch.stack(images)
            # Fusion backbones exactly equal baselines: execute each branch once.
            fusion = models["feature_fusion_v2"]
            embeddings = {"rgb": fusion.rgb_branch(tensors["rgb"]),
                          "ela": fusion.ela_branch(tensors["ela"])}
            logits = {k: models[k].fc(embeddings[k]) for k in ("rgb", "ela")}
            logits["feature_fusion_v2"] = fusion.mlp_head(torch.cat((embeddings["rgb"], embeddings["ela"]), dim=1))
            for kind, value in logits.items():
                probabilities[kind].extend(torch.softmax(value, dim=1)[:, 1].tolist())
    tables = {}
    for kind, p in probabilities.items():
        table = frame[["filename", "true_label"]].copy()
        table["prob_fake"] = p
        tables[kind] = validate_table(table, kind)
    reports, predictions = comparison(tables, alpha, threshold)
    return {
        "mode": "infer", "input_contract": IMAGE_CONTRACT, "models": reports,
        "decision_fusion": {"alpha_rgb": alpha, "threshold": threshold, "policy": "explicit fixed parameters; no test fitting"},
        "manifest": fingerprint(manifest),
        "checkpoints": {k: fingerprint(root / p) for k, p in CHECKPOINTS.items()},
        "limitations": ["Manifest paths do not prove the historical generation parameters or split.",
            "Supply prepared offline pairs; raw-image or online ELA evaluation is a different protocol.",
            "Feature V2 threshold 0.2893 is rounded and has the documented validation-split limitation."],
    }, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cached", "infer"], nargs="?", default="cached")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--decision-policy", choices=["fixed", "common-validation"], default="fixed")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--alpha-rgb", type=float, default=.5)
    parser.add_argument("--decision-threshold", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, help="New directory only; existing files never overwritten")
    args = parser.parse_args(argv)
    require(args.output_dir is None or not args.output_dir.exists(), "output-dir already exists")
    require(args.batch_size > 0, "batch-size must be positive")
    require(np.isfinite(args.alpha_rgb) and 0 <= args.alpha_rgb <= 1, "alpha-rgb must be in [0,1]")
    require(np.isfinite(args.decision_threshold) and 0 <= args.decision_threshold <= 1,
            "decision-threshold must be in [0,1]")
    if args.mode == "cached":
        require(args.manifest is None and args.alpha_rgb == .5 and args.decision_threshold == .5,
                "cached mode uses --decision-policy, not manifest/explicit infer parameters")
        report, predictions = cached(args.root.resolve(), args.decision_policy, args.verify_checkpoints)
    else:
        require(args.manifest is not None, "infer requires an explicit --manifest of prepared offline pairs")
        require(args.decision_policy == "fixed" and not args.verify_checkpoints,
                "infer uses explicit parameters and automatically checks checkpoint compatibility")
        report, predictions = infer(args.root.resolve(), args.manifest.resolve(), args.alpha_rgb,
                                    args.decision_threshold, args.batch_size)
    report["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        predictions.to_csv(args.output_dir / "predictions.csv", index=False, mode="x")
    print(encoded)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Evaluation stopped: {error}") from error
