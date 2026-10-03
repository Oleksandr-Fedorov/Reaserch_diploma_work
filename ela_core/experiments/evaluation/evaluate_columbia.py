#!/usr/bin/env python3
"""First Columbia holdout: fixed weights, fixed thresholds, original TIFF/BMP.

Save beside ela_core/experiments/evaluation/unified_evaluator.py. Run with the
existing ela_core Python environment. Defaults: D:/4cam_auth and D:/4cam_splc.
No training, dataset writes, parameter fitting, or checkpoint downloads.

One evaluation per original file. ELA: native RGB -> JPEG(q=85, 4:2:0) ->
abs(original - decoded JPEG) * 10 -> clip -> uint8 -> training transform.
RGB is decoded from the original, with no preliminary JPEG recompression.

Outputs: protocol.json, manifest.csv, predictions.csv, comparison.csv,
metrics.json, confusion_matrices.csv, run_notes.txt, status.json.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from PIL import Image, __version__ as PILLOW_VERSION, features as pil_features


MODELS = ("rgb", "ela", "decision_fusion", "feature_fusion_v2")
THRESHOLDS = {
    "rgb": 0.15752553939819336,
    "ela": 0.4166847765445709,
    "decision_fusion": 0.2981437146663666,
    "feature_fusion_v2": 0.2767023742198944,
}
ALPHA_RGB = 0.5
Q_ELA = 85
EXPECTED_COUNTS = {0: 183, 1: 180}
EXTENSIONS = {".tif", ".tiff", ".bmp"}
EVALUATOR_SHA256 = "0db2bda739fc884a0005f6d52f1598851d969c4c43eeead6a87989f639cc66e7"
CHECKPOINTS = {
    "rgb": ("models/RGB_model/model_resnet_rgb_LATEST.pth",
            "f3c74c865f315dec1430fd1abce84cb3b2018f1f336378f2239d057708148175"),
    "ela": ("models/ELA_model/model_resnet_ela_sourceaware_v2.pth",
            "a3360126a9c104a43c51da173740c947901b496127e8417dcd612f8343dcf41b"),
    "feature_fusion_v2": ("models/Fusion_v2/model_fusion_sourceaware_v2.pth",
                          "833c9d2550647346eb50e33e6552a6957d264988d2199766c727ec5fe612cced"),
}
CALIBRATION_SOURCE = {
    "policy": "common_validation_class_balanced_thresholds",
    "policy_file": "balanced_calibration.json from casia_validation_audit.zip/results",
    "policy_sha256": "6ad020317660da003269bfdbd6fd9cdeed18ba569c139bad8e4ff831cec715a9",
    "validation_csv_sha256": "aeb1f9b5100b72cd26a56837e490d4e2d59a0e6d79aa4c160d3b87af7672c765",
    "selection": "macro-F1 with total REAL weight 0.5 and total FAKE weight 0.5",
    "validation_rows": 1948,
    "validation_source_groups": {"REAL": 71, "FAKE": 416},
    "casia_test_previously_examined": True,
    "calibration_is_post_hoc_protocol_refinement": True,
    "external_predictions_used_to_select_parameters": False,
}
LIMITATIONS = [
    "Original Columbia TIFF/BMP is a distribution shift from CASIA JPEG-augmented PNG inputs.",
    "q_ela=85 is a fixed protocol choice within the historical training grid, not an externally optimized setting.",
    "Thresholds were selected on prepared CASIA validation pairs, not on Columbia or on a raw-file validation set.",
    "Common CASIA validation membership is conditional on retained training code; checkpoints contain no split metadata.",
    "Historical Feature Fusion V2 checkpoint selection retains the documented RGB validation-exposure limitation.",
    "363 files are not 363 proven independent source photographs: authentic crops can share scenes and splices share donors.",
    "Exact byte/pixel duplicate counts do not establish independence; donor/scene groups and cross-dataset visual duplicates are not audited.",
    "No confidence interval or significance claim is made without a defensible independent-source grouping.",
    "If Columbia results inform later changes, Columbia becomes development evidence; IMD2020 remains unexamined here.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def write_json(path, value, replace=False):
    with Path(path).open("w" if replace else "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def decode_rgb(data):
    with Image.open(io.BytesIO(data)) as source:
        require(source.format in ("TIFF", "BMP"),
                f"Expected original TIFF/BMP bytes, got {source.format!r}")
        require(getattr(source, "n_frames", 1) == 1, "Multi-frame input is not part of this protocol")
        info = {"format": source.format, "original_mode": source.mode,
                "width": source.width, "height": source.height}
        rgb = source.convert("RGB")
        rgb.load()
    return rgb, info


def pixel_sha256(rgb):
    prefix = f"RGB:{rgb.width}x{rgb.height}\0".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def ela_from_rgb(rgb):
    """Same fixed x10 residual as merge_CASIA2.py, at native resolution.

    Explicit JPEG defaults pin the historical Pillow RGB 4:2:0 behavior.
    There is no per-image brightness normalization and no q_primary pass.
    """
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=Q_ELA, subsampling=2,
             optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        jpeg_rgb = np.array(recompressed.convert("RGB"), dtype=float)
    residual = np.abs(np.array(rgb, dtype=float) - jpeg_rgb) * 10
    return Image.fromarray(np.clip(residual, 0, 255).astype(np.uint8))


def build_manifest(real_dir, fake_dir):
    roots = {0: Path(real_dir).resolve(), 1: Path(fake_dir).resolve()}
    require(roots[0] != roots[1], "REAL and FAKE directories must differ")
    records, ignored = [], []
    for label, root in roots.items():
        require(root.is_dir(), f"Missing directory: {root}. Extract the dataset or pass --real-dir/--fake-dir.")
        files = sorted((path for path in root.rglob("*") if path.is_file()),
                       key=lambda path: path.relative_to(root).as_posix().casefold())
        candidates = []
        for path in files:
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts) or path.suffix.lower() not in EXTENSIONS:
                ignored.append({"class": "REAL" if label == 0 else "FAKE",
                                "relative_path": relative.as_posix(),
                                "reason": "hidden metadata or extension outside TIFF/BMP"})
            else:
                candidates.append(path)
        require(len(candidates) == EXPECTED_COUNTS[label],
                f"{root}: found {len(candidates)} TIFF/BMP files; expected {EXPECTED_COUNTS[label]}. "
                "Check extraction and folder choice. No images will be silently removed or sampled.")
        prefix = "4cam_auth" if label == 0 else "4cam_splc"
        for path in candidates:
            data = path.read_bytes()
            rgb, info = decode_rgb(data)
            records.append({"filename": prefix + "/" + path.relative_to(root).as_posix(),
                            "image_path": str(path.resolve()), "true_label": label,
                            "class_name": "REAL" if label == 0 else "FAKE",
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "rgb_pixels_sha256": pixel_sha256(rgb), **info})
        print(f"  {'REAL' if label == 0 else 'FAKE'}: {len(candidates)} original files checked", flush=True)
    frame = pd.DataFrame(records)
    require(not frame.filename.duplicated().any(), "Duplicate image IDs")
    require(not frame.image_path.duplicated().any(), "Same input path appears more than once")
    require(frame.groupby("rgb_pixels_sha256").true_label.nunique().max() == 1,
            "Identical decoded pixels occur in both REAL and FAKE; inspect the dataset before evaluation")
    cohort = {"files": len(frame), "class_counts": {"REAL": int((frame.true_label == 0).sum()),
                                                     "FAKE": int((frame.true_label == 1).sum())},
              "unique_file_hashes": int(frame.sha256.nunique()),
              "unique_decoded_rgb_hashes": int(frame.rgb_pixels_sha256.nunique()),
              "independent_source_count": None,
              "independence_note": "Not established: shared scenes/crops and splice donors; file counts are not an independence claim.",
              "excluded_non_dataset_files": ignored}
    return frame, cohort


def load_evaluator(root):
    path = root / "experiments/evaluation/unified_evaluator.py"
    require(path.is_file(), f"Missing evaluator: {path}. Save this script beside the existing unified_evaluator.py.")
    require(file_sha256(path) == EVALUATOR_SHA256,
            "unified_evaluator.py differs from the verified version. Send its hash/file before proceeding.")
    spec = importlib.util.spec_from_file_location("columbia_verified_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fingerprint(path)


def verify_checkpoint_files(root):
    verified = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        path = root / relative
        require(path.is_file(), f"Missing checkpoint: {path}")
        item = fingerprint(path)
        require(item["sha256"] == expected,
                f"{name}: checkpoint SHA-256 differs from the weights used for CASIA calibration: {path}")
        verified[name] = item
    return verified


def build_protocol(manifest, cohort, checkpoint_info, evaluator_info, runtime):
    return {
        "experiment": "columbia_uncompressed_originals_q85_v1",
        "frozen_utc": utc_now(), "dataset": {
            "name": "Columbia Uncompressed Image Splicing Detection Evaluation Dataset",
            "release": "official 4cam_auth / 4cam_splc; exact local file version pinned by manifest SHA-256",
            "reference": "https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/",
            "cohort": cohort, "manifest": manifest},
        "class_to_idx": {"REAL": 0, "FAKE": 1},
        "preprocessing": {
            "source_files": "original TIFF/BMP; read only; one prediction per file; no augmentation",
            "decode": "Pillow Image.open -> convert RGB; no explicit EXIF transpose or ICC conversion",
            "rgb": "decoded original RGB; no preliminary JPEG recompression",
            "ela": "at native resolution: abs(RGB - decoded JPEG(RGB,q=85))*10; clip[0,255]; uint8",
            "q_ela": Q_ELA, "q_primary": None, "jpeg_subsampling": "4:2:0 (Pillow subsampling=2)",
            "jpeg_optimize": False, "jpeg_progressive": False,
            "q_ela_selection": "fixed before external predictions, within training grid [65,70,75,80,85,90,95,100]",
            "ela_generator_reference": "ela_core/experiments/preprocessing/merge_CASIA2.py; Git blob 76a2dd752aa278c742ffcff765569112800395c1",
            "both_inputs": "PIL Resize((224,224), bilinear, antialias=True) -> ToTensor -> Normalize",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "models": {name: {"threshold": THRESHOLDS[name]} for name in MODELS},
        "score": "softmax(logits,dim=1)[:,1]; uncalibrated FAKE score",
        "decision_fusion": {"alpha_rgb": ALPHA_RGB, "formula": "0.5*p_rgb + 0.5*p_ela"},
        "feature_fusion_v2": "[RGB_512, ELA_512] -> saved MLP; frozen identical backbones; eval mode",
        "prediction_rule": "FAKE iff prob_fake >= model threshold",
        "calibration_source": CALIBRATION_SOURCE, "checkpoints": checkpoint_info,
        "evaluator": evaluator_info, "script": fingerprint(Path(__file__)), "runtime": runtime,
        "metrics": ["confusion_matrix", "accuracy", "macro_f1", "roc_auc", "recall_fake", "fpr"],
        "confusion_matrix_order": "rows=true REAL,FAKE; columns=predicted REAL,FAKE; [[TN,FP],[FN,TP]]",
        "metric_weighting": "ordinary unweighted metrics, one row per original file",
        "parameter_search": False, "training": False, "stress_test": False,
        "imd2020_accessed": False, "limitations": LIMITATIONS,
    }


def predict(manifest, models, transform, device, batch_size, torch, thresholds):
    values = {name: [] for name in CHECKPOINTS}
    fusion = models["feature_fusion_v2"]
    with torch.inference_mode():
        for start in range(0, len(manifest), batch_size):
            rgb_batch, ela_batch = [], []
            for row in manifest.iloc[start:start + batch_size].itertuples(index=False):
                data = Path(row.image_path).read_bytes()
                require(hashlib.sha256(data).hexdigest() == row.sha256,
                        f"Input file changed after protocol was frozen: {row.image_path}")
                rgb, _ = decode_rgb(data)
                require(pixel_sha256(rgb) == row.rgb_pixels_sha256, f"Decoded pixels changed: {row.filename}")
                ela = ela_from_rgb(rgb)
                rgb_batch.append(transform(rgb))
                ela_batch.append(transform(ela))
            rgb_tensor = torch.stack(rgb_batch).to(device)
            ela_tensor = torch.stack(ela_batch).to(device)
            rgb_features = fusion.rgb_branch(rgb_tensor)
            ela_features = fusion.ela_branch(ela_tensor)
            outputs = {"rgb": models["rgb"].fc(rgb_features),
                       "ela": models["ela"].fc(ela_features),
                       "feature_fusion_v2": fusion.mlp_head(torch.cat((rgb_features, ela_features), dim=1))}
            for name, logits in outputs.items():
                values[name].extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            print(f"  Columbia: {min(start+batch_size, len(manifest))}/{len(manifest)} files", flush=True)
    values["decision_fusion"] = (ALPHA_RGB * np.array(values["rgb"], dtype=np.float64)
                                  + (1-ALPHA_RGB) * np.array(values["ela"], dtype=np.float64))
    tables = []
    for name in MODELS:
        p = np.asarray(values[name], dtype=np.float64)
        require(len(p) == len(manifest) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                f"Invalid predictions from {name}")
        table = manifest[["filename", "true_label", "sha256"]].copy()
        table.insert(0, "model", name)
        table["prob_fake"] = p
        table["threshold"] = thresholds[name]
        table["pred_label"] = (p >= thresholds[name]).astype(int)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize(predictions, evaluator):
    results, comparison, matrices = {}, [], []
    for name in MODELS:
        frame = predictions[predictions.model == name]
        require(frame.threshold.nunique() == 1, f"Multiple thresholds for {name}")
        threshold = float(frame.threshold.iloc[0])
        require(np.array_equal(frame.pred_label.to_numpy(), (frame.prob_fake.to_numpy() >= threshold).astype(int)),
                f"Stored labels differ from the frozen threshold for {name}")
        result = evaluator.metrics(frame.true_label, frame.prob_fake, threshold)
        (tn, fp), (fn, tp) = result["confusion_matrix"]
        result.update(recall_fake=tp/(tp+fn), fpr=fp/(fp+tn))
        results[name] = result
        comparison.append({"model": name, **{key: result[key] for key in
                          ("threshold", "accuracy", "macro_f1", "roc_auc", "recall_fake", "fpr")}})
        matrices.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    return results, pd.DataFrame(comparison), pd.DataFrame(matrices)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="ela_core directory")
    parser.add_argument("--real-dir", type=Path, default=Path("D:/4cam_auth"))
    parser.add_argument("--fake-dir", type=Path, default=Path("D:/4cam_splc"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(args.batch_size > 0, "batch-size must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or root / "artifacts" / ("columbia_raw_q85_" + stamp)).resolve()
    require(not output.exists(), f"Output already exists; choose a new directory: {output}")
    for source_dir in (args.real_dir.resolve(), args.fake_dir.resolve()):
        require(not output.is_relative_to(source_dir), "Output must be outside the source dataset folders")

    print("1/4 Checking original files and frozen checkpoint identities...", flush=True)
    manifest, cohort = build_manifest(args.real_dir, args.fake_dir)
    evaluator, evaluator_info = load_evaluator(root)
    checkpoints = verify_checkpoint_files(root)
    # Existing, already verified loader: strict state_dicts and exact branch equality.
    import torch
    import torchvision
    import sklearn
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable; use --device cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    states, models = evaluator.load_models(root)
    del states
    require(verify_checkpoint_files(root) == checkpoints, "Checkpoint changed while it was being loaded")
    device = torch.device(args.device)
    for model in models.values():
        model.to(device).eval()
    transform = evaluator.input_transform()
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "numpy": np.__version__,
               "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
               "pillow": PILLOW_VERSION, "jpeg_codec": pil_features.version_codec("jpg"),
               "device": str(device), "batch_size": args.batch_size,
               "torch_threads": torch.get_num_threads(), "inference_dtype": "float32; no AMP; TF32 disabled"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "prepared", "time_utc": utc_now()})
    try:
        manifest_path = output / "manifest.csv"
        manifest.to_csv(manifest_path, index=False, mode="x")
        protocol = build_protocol(fingerprint(manifest_path), cohort, checkpoints, evaluator_info, runtime)
        protocol_path = output / "protocol.json"
        write_json(protocol_path, protocol)
        protocol_sha = file_sha256(protocol_path)
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
        thresholds = {name: frozen["models"][name]["threshold"] for name in MODELS}
        print(f"2/4 Protocol frozen BEFORE predictions: {protocol_path}", flush=True)
        print("    q_ela=85; original RGB; alpha_RGB=0.5; no threshold search", flush=True)
        print("3/4 Running inference on original Columbia files (no training)...", flush=True)
        predictions = predict(manifest, models, transform, device, args.batch_size, torch, thresholds)
        require(file_sha256(protocol_path) == protocol_sha, "Frozen protocol changed during inference")
        predictions.to_csv(output / "predictions.csv", index=False, mode="x")
        results, comparison, matrices = summarize(predictions, evaluator)
        write_json(output / "metrics.json", {"mode": "external_columbia_originals", "models": results,
                    "protocol": frozen, "protocol_sha256": protocol_sha,
                    "predictions": fingerprint(output / "predictions.csv"), "limitations": LIMITATIONS})
        comparison.to_csv(output / "comparison.csv", index=False, mode="x")
        matrices.to_csv(output / "confusion_matrices.csv", index=False, mode="x")
        notes = ("Columbia original-file holdout, q_ela=85. All metric values are fractions.\n"
                 "FAKE=1; classify FAKE when score >= the frozen threshold.\n"
                 "Confusion matrix: [[TN,FP],[FN,TP]], rows=true / columns=predicted.\n"
                 f"Files: {len(manifest)}. Proven independent source count: not established.\n\n"
                 + comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}")
                 + "\n\nLimitations:\n" + "\n".join("- " + line for line in LIMITATIONS)
                 + "\n\nDo not select thresholds/q_ela on these results. Preserve IMD2020 for a later independent check.\n")
        (output / "run_notes.txt").write_text(notes, encoding="utf-8")
        write_json(output / "status.json", {"state": "completed", "time_utc": utc_now(),
                                           "protocol_sha256": protocol_sha}, replace=True)
    except Exception as exc:
        write_json(output / "status.json", {"state": "failed", "time_utc": utc_now(),
                    "error_type": type(exc).__name__, "error": str(exc)}, replace=True)
        raise
    print("4/4 Completed. Metrics are fractions, not percentages:", flush=True)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults: {output}\nSend back metrics.json, comparison.csv and predictions.csv.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
