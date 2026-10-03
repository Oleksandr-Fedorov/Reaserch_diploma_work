"""One additional CASIA evaluation step; put beside unified_evaluator.py.

From D:\\Diplom, using the existing environment:
  python ela_core/experiments/evaluation/calibrate_common_validation.py \
    --verified-report ela_core/artifacts/casia_verified_20260930/metrics.json \
    --output-dir ela_core/artifacts/casia_calibrated_20260930

Reads existing checkpoints, prepared training-folder PNGs and saved predictions.
No training, downloads, ELA generation, dataset modification or overwrite.
Default device: CPU. Optional --device cuda. Output directory must be new.

Protocol fixed before this run:
* The common validation cohort is reconstructed by the existing evaluator.
* All three networks are run on that same cohort with their original transforms.
* Decision Fusion uses alpha_rgb=0.5, fixed; only its threshold is selected.
* Each of the four thresholds maximizes validation macro-F1 (REAL=0, FAKE=1).
* Test predictions cannot enter threshold selection. Frozen thresholds are
  written to calibration.json before saved test probabilities are loaded.

This does NOT repair historical checkpoint selection or prove training split
provenance. A reconstructed split remains conditional on the saved training code.
CASIA test has been examined before; it is not a new external holdout.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import unified_evaluator as ev
except ModuleNotFoundError as exc:
    if exc.name == "unified_evaluator":
        raise SystemExit("Place this file beside unified_evaluator.py.") from exc
    raise


MODEL_ORDER = ("rgb", "ela", "decision_fusion", "feature_fusion_v2")
ALPHA_RGB = 0.5
LIMITATIONS = [
    "Common validation is inferred from retained notebook code and the ELA source "
    "manifest; checkpoints do not contain a recorded training split.",
    "Recalibrating thresholds does not redo historical checkpoint/epoch selection "
    "or remove its documented validation-exposure limitation for Fusion V2.",
    "Four variants of one source are correlated. Row counts are not counts of "
    "independent source images; donor relationships and visual duplicates are not audited here.",
    "Prepared PNG paths do not independently prove historical JPEG/ELA generation settings.",
    "CASIA test was already examined. This is a separately recorded recalculation, "
    "not a new independent holdout. No Columbia or IMD2020 data are used.",
]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_threshold(labels, probabilities):
    """Exact search over achievable >= decisions for thresholds in [0, 1].

    Unlike pruned ROC points, all distinct score boundaries are considered.
    Candidate ties within 1e-12: closest to .5, then larger threshold.
    """
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    ev.require(y.ndim == p.ndim == 1 and len(y) == len(p), "Invalid calibration shapes")
    ev.require(set(y) == {0, 1}, "Calibration requires both classes")
    ev.require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid probabilities")
    y = y.astype(np.int64)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p])
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    prefix_positive = np.r_[0, np.cumsum(sorted_y)]
    below = np.searchsorted(sorted_p, candidates, side="left")
    fn = prefix_positive[below]
    tn = below - fn
    tp = int(y.sum()) - fn
    fp = len(y) - int(y.sum()) - tn
    denominator_fake = 2 * tp + fp + fn
    denominator_real = 2 * tn + fp + fn
    fake_f1 = np.divide(2 * tp, denominator_fake, out=np.zeros(len(candidates)),
                        where=denominator_fake != 0)
    real_f1 = np.divide(2 * tn, denominator_real, out=np.zeros(len(candidates)),
                        where=denominator_real != 0)
    scores = (real_f1 + fake_f1) / 2
    eligible = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    chosen = min(eligible, key=lambda i: (abs(float(candidates[i]) - .5), -float(candidates[i])))
    return {"threshold": float(candidates[chosen]), "validation_macro_f1": float(scores[chosen]),
            "candidate_count": len(candidates)}


def verify_previous_run(root, report_path):
    """Pin this run to the artifacts already verified by the user's cached run."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = report.get("checkpoint_verification", {})
    ev.require(report.get("mode") == "cached", "Expected the previous cached metrics.json")
    for key in ("strict_state_dict_load", "finite_weights", "fusion_backbones_exact_match"):
        ev.require(verification.get(key) is True, "Previous report lacks successful " + key)
    ev.require(verification.get("embedding_contract") == "verified", "Embedding verification missing")
    expected = [(root / path, report["sources"][name]["sha256"])
                for name, path in ev.TABLES.items()]
    expected += [(root / path, verification["checkpoint_fingerprints"][name]["sha256"])
                 for name, path in ev.CHECKPOINTS.items()]
    expected += [(root / ev.SPLIT, report["split_manifest"]["sha256"]),
                 (root / "notebooks/Colab.ipynb", report["notebook_provenance"]["sha256"])]
    for path, sha in expected:
        ev.require(ev.fingerprint(path)["sha256"] == sha,
                   f"File differs from the verified run: {path}. Reconcile versions before continuing.")
    return report


def build_cohort(root, previous, rgb_dir, ela_dir):
    rgb_val = ev.read_table(root / ev.TABLES["rgb_val"])
    ela_val = ev.align(rgb_val, ev.read_table(root / ev.TABLES["ela_val"]), "validation")
    # Only test identities are used here to reject source overlap, not test scores.
    test = ev.validate_table(pd.read_csv(root / ev.TABLES["feature_fusion_v2"],
                                       usecols=["filename", "true_label"]), "test identities", False)
    evidence, clean = ev.split_evidence(root, rgb_val, ela_val, test)
    for key in ("validation_rows", "validation_sources", "common_heldout_rows",
                "rgb_training_sources_in_composite_validation", "test_source_overlap"):
        ev.require(evidence[key] == previous["split_evidence"][key],
                   "Recovered cohort differs from previous report at " + key)
    cohort = rgb_val.loc[clean, ["filename", "true_label"]].reset_index(drop=True)
    ev.validate_table(cohort, "common validation", probability=False)
    for kind, directory in (("rgb", rgb_dir), ("ela", ela_dir)):
        cohort[kind + "_path"] = [str((directory / ev.CLASS_NAMES[int(label)] / filename).resolve())
                                   for filename, label in zip(cohort.filename, cohort.true_label)]
        missing = [p for p in cohort[kind + "_path"] if not Path(p).is_file()]
        ev.require(not missing, f"Missing {len(missing)} {kind} files; first examples: {missing[:3]}. "
                   f"If necessary pass --{kind}-train-dir with the prepared dataset's train directory.")
    evidence["common_heldout_sources"] = len({ev.source_stem(f) for f in cohort.filename})
    return cohort, evidence, {"rgb": rgb_val, "ela": ela_val}


def predict_validation(root, manifest_path, batch_size, device_name):
    """Same architecture, class mapping and pixel transform as unified_evaluator."""
    import torch
    from PIL import Image

    if device_name == "cuda":
        ev.require(torch.cuda.is_available(), "CUDA unavailable; omit --device cuda to use CPU")
    device = torch.device(device_name)
    frame = ev.read_pairs(manifest_path)
    states, models = ev.load_models(root)
    del states
    for model in models.values():
        model.to(device).eval()
    transform = ev.input_transform()
    values = {name: [] for name in ev.CHECKPOINTS}
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = {}
            for kind in ("rgb", "ela"):
                tensors = []
                for path in frame[kind + "_path"].iloc[start:start + batch_size]:
                    with Image.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                batch[kind] = torch.stack(tensors).to(device)
            fusion = models["feature_fusion_v2"]
            rgb_features = fusion.rgb_branch(batch["rgb"])
            ela_features = fusion.ela_branch(batch["ela"])
            logits = {"rgb": models["rgb"].fc(rgb_features),
                      "ela": models["ela"].fc(ela_features),
                      "feature_fusion_v2": fusion.mlp_head(torch.cat((rgb_features, ela_features), dim=1))}
            for name, outputs in logits.items():
                values[name].extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
            done = min(start + batch_size, len(frame))
            print(f"  Validation: {done}/{len(frame)} pairs", flush=True)
    tables = {}
    for name, probabilities in values.items():
        table = frame[["filename", "true_label"]].copy()
        table["prob_fake"] = probabilities
        tables[name] = ev.validate_table(table, name)
    return add_decision_fusion(tables), {"torch": torch.__version__, "device": str(device)}


def add_decision_fusion(tables):
    reference = tables["feature_fusion_v2"]
    aligned = {name: ev.align(reference, table, name) for name, table in tables.items()}
    combined = reference[["filename", "true_label"]].copy()
    combined["prob_fake"] = (ALPHA_RGB * aligned["rgb"].prob_fake.to_numpy()
                              + (1 - ALPHA_RGB) * aligned["ela"].prob_fake.to_numpy())
    aligned["decision_fusion"] = ev.validate_table(combined, "decision_fusion")
    return aligned


def evaluate_tables(tables, thresholds):
    reports, predictions = {}, []
    for name in MODEL_ORDER:
        frame = ev.validate_table(tables[name], name)
        threshold = thresholds[name]
        ev.require(np.isfinite(threshold) and 0 <= threshold <= 1, "Invalid frozen threshold")
        result = ev.metrics(frame.true_label, frame.prob_fake, threshold)
        (tn, fp), (fn, tp) = result["confusion_matrix"]
        result.update(recall_fake=tp / (tp + fn), fpr=fp / (fp + tn))
        reports[name] = result
        out = frame[["filename", "true_label", "prob_fake"]].copy()
        out.insert(0, "model", name)
        out["threshold"] = threshold
        out["pred_label"] = (out.prob_fake >= threshold).astype(int)
        predictions.append(out)
    return reports, pd.concat(predictions, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ev.ROOT)
    parser.add_argument("--verified-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-train-dir", type=Path)
    parser.add_argument("--ela-train-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output_dir.resolve()
    ev.require(args.batch_size > 0, "batch-size must be positive")
    ev.require(not output.exists(), "Output directory exists; choose a new directory (no overwrite)")
    # Fail before creating files if the inference environment is unavailable.
    import torch
    import torchvision

    if args.device == "cuda":
        ev.require(torch.cuda.is_available(), "CUDA unavailable; use --device cpu")
    print("1/4 Checking artifacts against the verified report...", flush=True)
    previous = verify_previous_run(root, args.verified_report)
    rgb_dir = args.rgb_train_dir or root / "datasets/CASIA_Unified_Randomized_RGB_v2/train"
    ela_dir = args.ela_train_dir or root / "datasets/CASIA_Unified_Randomized_v2/train"
    cohort, evidence, cached_val = build_cohort(root, previous, rgb_dir, ela_dir)
    print(f"Common validation: {len(cohort)} files / {evidence['common_heldout_sources']} sources "
          "(conditional on the retained training code).", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    cohort_path = output / "validation_manifest.csv"
    cohort.to_csv(cohort_path, index=False, mode="x")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only_threshold_selection": True,
        "selection_rule": "max macro-F1 over all distinct validation probabilities plus 0, .5, 1; >=",
        "tie_rule": "within 1e-12, candidate closest to .5, then larger candidate",
        "decision_fusion": {"alpha_rgb": ALPHA_RGB, "alpha_policy": "fixed before calibration; no alpha search"},
        "input_contract": ev.IMAGE_CONTRACT, "split_evidence": evidence,
        "verified_report": ev.fingerprint(args.verified_report),
        "manifest": ev.fingerprint(cohort_path),
        "scripts": {"calibration": ev.fingerprint(Path(__file__)),
                    "unified_evaluator": ev.fingerprint(Path(ev.__file__))},
        "checkpoints": previous["checkpoint_verification"]["checkpoint_fingerprints"],
        "limitations": LIMITATIONS,
    }
    write_json(output / "protocol.json", protocol)
    print("2/4 Running the saved models on validation (no training)...", flush=True)
    validation, runtime = predict_validation(root, cohort_path, args.batch_size, args.device)
    chosen = {name: select_threshold(validation[name].true_label, validation[name].prob_fake)
              for name in MODEL_ORDER}
    thresholds = {name: chosen[name]["threshold"] for name in MODEL_ORDER}
    val_metrics, val_predictions = evaluate_tables(validation, thresholds)
    val_predictions.to_csv(output / "validation_predictions.csv", index=False, mode="x")
    cached_agreement = {}
    for name in ("rgb", "ela"):
        old = cached_val[name].set_index("filename").loc[validation[name].filename]
        error = float(np.max(np.abs(old.prob_fake.to_numpy() - validation[name].prob_fake.to_numpy())))
        cached_agreement[name] = {"max_probability_abs_difference": error,
                                  "within_2e_minus5": error <= 2e-5,
                                  "cached_probabilities_used_for_calibration": False}
    calibration = {"protocol": protocol, "models": chosen, "validation_metrics": val_metrics,
                   "cached_validation_agreement": cached_agreement,
                   "validation_predictions": ev.fingerprint(output / "validation_predictions.csv")}
    write_json(output / "calibration.json", calibration)
    print("3/4 Thresholds frozen in calibration.json. Reading cached test predictions...", flush=True)
    # Threshold selection has finished. There is no test-to-calibration feedback path.
    frozen = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    thresholds = {name: frozen["models"][name]["threshold"] for name in MODEL_ORDER}
    # Detect file changes during a long validation pass before using cached test scores.
    for name in ev.CHECKPOINTS:
        path = root / ev.TABLES[name]
        ev.require(ev.fingerprint(path)["sha256"] == previous["sources"][name]["sha256"],
                   f"Test CSV changed during validation: {path}")
    test_tables = add_decision_fusion({name: ev.read_table(root / ev.TABLES[name])
                                      for name in ev.CHECKPOINTS})
    test_metrics, test_predictions = evaluate_tables(test_tables, thresholds)
    test_predictions.to_csv(output / "test_predictions.csv", index=False, mode="x")
    runtime.update(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                   torchvision=torchvision.__version__)
    result = {"mode": "common_validation_calibration", "models": test_metrics,
              "calibration": frozen, "historical_test_metrics": previous["models"],
              "runtime": runtime, "limitations": LIMITATIONS}
    write_json(output / "metrics.json", result)
    rows = [{"model": name, "threshold": thresholds[name],
             "val_macro_f1": val_metrics[name]["macro_f1"],
             **{key: test_metrics[name][key] for key in
                ("accuracy", "macro_f1", "roc_auc", "recall_fake", "fpr")}}
            for name in MODEL_ORDER]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False, mode="x")
    print("4/4 Completed. Test metrics (fractions, not percentages):", flush=True)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults: {output}\nSend back metrics.json and comparison.csv.")
    print("Note: common validation is reconstructed; historical checkpoint selection is unchanged.")
    if not all(x["within_2e_minus5"] for x in cached_agreement.values()):
        print("NOTE: fresh validation scores differ from cached val CSVs. See cached_validation_agreement; "
              "only fresh scores were used to choose thresholds.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        raise SystemExit(f"Calibration stopped: {exc}") from exc
