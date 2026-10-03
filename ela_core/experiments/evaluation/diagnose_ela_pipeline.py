#!/usr/bin/env python3
"""Read-only audit of the verified CASIA/Columbia image pipeline. No training.

Save beside evaluate_columbia.py and unified_evaluator.py in
ela_core/experiments/evaluation. Run with the existing ela_core environment:

  python diagnose_ela_pipeline.py --columbia-report <old-run>/metrics.json

Default CASIA input: ela_core/artifacts/casia_calibrated_20260930.
Optional: --casia-run PATH, --ela-log PATH, --root PATH, --device cuda.

Checks 32 common-validation sources per class (one fixed variant per source):
RGB/ELA pairing, native sizes, PNG round-trip, training/evaluator transforms,
and ELA regenerated from the saved paired RGB using the logged q_ela.
Missing CASIA artifacts are explicitly reported; Columbia checks can continue.

Replays all original Columbia inputs and compares with the frozen predictions.
Then runs ONE secondary diagnostic: JPEG(q_primary=95) followed by ELA(q=85),
with the same weights, thresholds, and REAL/FAKE processing. No quality sweep,
threshold fitting, dataset modification, or changes to the original result.
This is post-hoc diagnosis, not a new independent holdout or a replacement
headline score. A change in accuracy alone does not prove a forensic mechanism.

Outputs go to a new ela_core/artifacts/ela_pipeline_diagnostic_<UTC> folder.
Send diagnostic_bundle.zip: CSV/JSON only; no source images or checkpoints.
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
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, __version__ as PILLOW_VERSION, features


RUNNER_SHA = "437abd31a13f27dde78f0407ccbaa7344e1328f1a7b0fe4de55e529f0ed0d8c1"
Q_PRIMARY = 95
Q_ELA = 85
SOURCE_SAMPLE_PER_CLASS = 32
TOLERANCE = 2e-5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jpeg_rgb(rgb, quality):
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality), subsampling=2,
             optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("RGB")


def residual(rgb, quality):
    # Signed subtraction BEFORE abs, multiplication, clipping and uint8 cast.
    original = np.asarray(rgb, dtype=np.int16)
    decoded = np.asarray(jpeg_rgb(rgb, quality), dtype=np.int16)
    return Image.fromarray(np.clip(np.abs(original-decoded) * 10, 0, 255).astype(np.uint8))


def png_roundtrip(rgb):
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("RGB")


def image_stats(image, filename, label, condition, kind):
    a = np.asarray(image)
    require(image.mode == "RGB" and a.dtype == np.uint8 and a.shape[2] == 3,
            f"Unexpected image type for {filename}: {image.mode}, {a.dtype}, {a.shape}")
    small = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
    return {"filename": filename, "true_label": int(label), "condition": condition,
            "input": kind, "width": image.width, "height": image.height,
            "dtype": str(a.dtype), "mode": image.mode,
            "native_mean_0_255": float(a.mean()), "native_std": float(a.std()),
            "native_zero_fraction": float((a == 0).mean()),
            "native_255_fraction": float((a == 255).mean()),
            "resized_mean_0_255": float(small.mean()), "resized_std": float(small.std())}


def transform_check(image, transform, historical, torch):
    tensor = transform(image)
    require(tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32,
            f"Invalid tensor contract: {tensor.shape}, {tensor.dtype}")
    require(torch.isfinite(tensor).all().item(), "Nonfinite tensor")
    require(torch.equal(tensor, historical(image)), "Training/evaluator transforms differ")
    require(torch.equal(tensor, transform(png_roundtrip(image))),
            "PNG round-trip changed the tensor")
    return tensor


def resolve_log(sample, explicit):
    if explicit:
        require(explicit.is_file(), f"Missing --ela-log: {explicit}")
        return explicit, []
    parents = {Path(path).parents[2] for path in sample.ela_path}
    require(len(parents) == 1, "CASIA pairs span multiple dataset roots; pass --ela-log")
    root = next(iter(parents))
    candidates = [p for p in (root / "dataset_log.csv", root / "dataset_log_ela.csv") if p.is_file()]
    if not candidates:
        return None, [f"No ELA generation log found in {root}; cannot regenerate saved ELA pairs."]
    if len(candidates) == 2:
        columns = ["filename", "source_dataset", "original_filename", "split", "class", "q_primary", "q_ela"]
        a, b = [pd.read_csv(p)[columns].sort_values("filename").reset_index(drop=True) for p in candidates]
        require(a.equals(b), "dataset_log.csv and dataset_log_ela.csv disagree; pass the actual training --ela-log")
    return candidates[0], []


def prepare_casia(run, explicit_log, ev):
    manifest = run / "validation_manifest.csv"
    if not manifest.is_file():
        return pd.DataFrame(), None, {"status": "not_checked", "missing": str(manifest)}
    report_path = run / "metrics.json"
    require(report_path.is_file(), f"CASIA report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report["calibration"]["protocol"]["manifest"]["sha256"]
    require(sha(manifest) == expected, "CASIA validation manifest changed since calibration")
    cohort = ev.read_pairs(manifest)
    cohort["source_key"] = cohort.filename.map(lambda name: "|".join(ev.source_stem(name)))
    require(cohort.groupby("source_key").true_label.nunique().max() == 1,
            "One CASIA source key has conflicting labels")
    one = cohort.sort_values("filename").drop_duplicates("source_key").copy()
    one["sample_key"] = one.source_key.map(lambda key: hashlib.sha256(("pipeline-audit-v1|"+key).encode()).hexdigest())
    selected = []
    for label in (0, 1):
        part = one[one.true_label == label].sort_values("sample_key").head(SOURCE_SAMPLE_PER_CLASS)
        require(len(part) == SOURCE_SAMPLE_PER_CLASS, f"Need {SOURCE_SAMPLE_PER_CLASS} CASIA validation sources of class {label}")
        selected.append(part)
    sample = pd.concat(selected, ignore_index=True).drop(columns="sample_key")
    log_path, warnings = resolve_log(sample, explicit_log)
    info = {"status": "prepared", "manifest": str(manifest), "manifest_sha256": sha(manifest),
            "sample_source_count": len(sample), "variants_per_selected_source": 1,
            "sample_selection": "32 REAL/32 FAKE common-validation sources, fixed SHA-256 order, first filename per source; no prediction-based selection",
            "warnings": warnings}
    if log_path is not None:
        log = pd.read_csv(log_path)
        needed = {"filename", "source_dataset", "original_filename", "class", "split", "q_ela", "q_primary"}
        require(needed <= set(log), f"Generation log lacks columns: {sorted(needed-set(log))}")
        require(not log.filename.duplicated().any(), "Duplicate filenames in ELA generation log")
        require(log.q_primary.isin(range(50, 101, 5)).all() and log.q_ela.isin(range(65, 101, 5)).all(),
                "JPEG qualities outside the retained generator's grid")
        require(set(sample.filename) <= set(log.filename), "Selected CASIA files absent from ELA generation log")
        joined = log.set_index("filename").loc[sample.filename].reset_index()
        require((joined["split"] == "train").all(), "Common validation includes a non-training outer split")
        require(np.array_equal(joined["class"].map({"REAL": 0, "FAKE": 1}), sample.true_label), "CSV class mismatch")
        require(np.array_equal(joined.source_dataset + "|" + joined.original_filename.map(lambda x: Path(x).stem), sample.source_key),
                "Generation log source identities differ from validation filenames")
        for key in ("q_primary", "q_ela"):
            sample[key] = joined[key].to_numpy()
        source_cols = ["source_dataset", "original_filename"]
        info.update(generation_log=str(log_path), generation_log_sha256=sha(log_path),
                    log_rows=len(log), source_keys_crossing_outer_train_test=int((log.groupby(source_cols)["split"].nunique() > 1).sum()),
                    variants_per_source_counts={str(k): int(v) for k,v in log.groupby(source_cols).size().value_counts().items()},
                    quality_counts=log.groupby(["split", "class", "q_primary", "q_ela"]).size().rename("rows").reset_index().to_dict("records"),
                    original_extension_counts=log.assign(extension=log.original_filename.map(lambda x: Path(x).suffix.lower())).drop_duplicates(source_cols).groupby(["source_dataset", "class", "extension"]).size().rename("sources").reset_index().to_dict("records"))
    return sample, log_path, info


def audit_casia(sample, info, transform, historical, torch, output):
    rows, stats = [], []
    for row in sample.itertuples(index=False):
        images = {}
        entry = {"filename": row.filename, "true_label": int(row.true_label), "source_key": row.source_key}
        for kind in ("rgb", "ela"):
            path = Path(getattr(row, kind + "_path"))
            with Image.open(path) as image:
                entry[kind + "_original_mode"] = image.mode
                entry[kind + "_container_format"] = image.format
                images[kind] = image.convert("RGB")
            transform_check(images[kind], transform, historical, torch)
            entry[kind + "_sha256"] = sha(path)
            stats.append(image_stats(images[kind], row.filename, row.true_label, "casia_prepared_validation", kind))
        require(images["rgb"].size == images["ela"].size, f"RGB/ELA native sizes differ: {row.filename}")
        entry.update(width=images["ela"].width, height=images["ela"].height,
                     tensor_contract_and_png_parity=True)
        if hasattr(row, "q_ela"):
            regenerated = residual(images["rgb"], row.q_ela)
            delta = np.abs(np.asarray(regenerated, dtype=np.int16)-np.asarray(images["ela"], dtype=np.int16))
            entry.update(q_primary=int(row.q_primary), q_ela=int(row.q_ela),
                         regenerated_exact=bool((delta == 0).all()),
                         regenerated_mae=float(delta.mean()), regenerated_max_abs=int(delta.max()),
                         regenerated_changed_fraction=float((delta != 0).mean()),
                         regenerated_tensor_max_abs=float((transform(regenerated)-transform(images["ela"])).abs().max().item()))
        rows.append(entry)
    if rows:
        pd.DataFrame(rows).to_csv(output / "casia_pair_checks.csv", index=False, mode="x")
        info["status"] = "completed"
        if "regenerated_exact" in rows[0]:
            info["exact_regeneration_pairs"] = sum(row["regenerated_exact"] for row in rows)
            info["maximum_regeneration_mae"] = max(row["regenerated_mae"] for row in rows)
            info["interpretation"] = "Nonzero residual mismatch needs inspection of pairing, generation log and Pillow/JPEG versions; it is not automatically proof of corrupted training."
    return info, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--columbia-report", type=Path, required=True)
    parser.add_argument("--casia-run", type=Path)
    parser.add_argument("--ela-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    root, report_path = args.root.resolve(), args.columbia_report.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or root / "artifacts" / ("ela_pipeline_diagnostic_"+stamp)).resolve()
    require(not output.exists(), f"Output already exists: {output}")
    runner_path = root / "experiments/evaluation/evaluate_columbia.py"
    require(sha(runner_path) == RUNNER_SHA, "evaluate_columbia.py differs from the audited version")
    runner = load_module(runner_path, "pipeline_columbia_runner")
    ev, ev_info = runner.load_evaluator(root)
    old = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = old["protocol"]
    require(old["mode"] == "external_columbia_originals" and protocol["preprocessing"]["q_ela"] == Q_ELA,
            "Expected the original Columbia q85 report")
    require(protocol["script"]["sha256"] == RUNNER_SHA, "Report used a different Columbia runner")
    require(protocol["decision_fusion"]["alpha_rgb"] == .5, "Unexpected decision fusion alpha")
    manifest_path, cached_path = report_path.parent / "manifest.csv", report_path.parent / "predictions.csv"
    require(sha(manifest_path) == protocol["dataset"]["manifest"]["sha256"], "Columbia manifest changed")
    require(sha(cached_path) == old["predictions"]["sha256"], "Columbia cached predictions changed")
    original_protocol_path = report_path.parent / "protocol.json"
    require(sha(original_protocol_path) == old["protocol_sha256"], "Original frozen protocol changed")
    require(json.loads(original_protocol_path.read_text(encoding="utf-8")) == protocol, "Report/protocol contents differ")
    manifest, cached = pd.read_csv(manifest_path), pd.read_csv(cached_path)
    require(len(manifest) == 363 and manifest.true_label.value_counts().to_dict() == {0:183, 1:180}, "Unexpected Columbia cohort")
    require(not manifest.filename.duplicated().any(), "Duplicate Columbia filenames")
    thresholds = {name: protocol["models"][name]["threshold"] for name in runner.MODELS}
    require(thresholds == runner.THRESHOLDS, "Thresholds differ from the frozen policy")
    checkpoints = runner.verify_checkpoint_files(root)
    require(all(checkpoints[n]["sha256"] == protocol["checkpoints"][n]["sha256"] for n in checkpoints), "Report/checkpoint mismatch")
    casia_run = (args.casia_run or root / "artifacts/casia_calibrated_20260930").resolve()
    sample, _, casia_info = prepare_casia(casia_run, args.ela_log, ev)
    for image_path in manifest.image_path:
        require(Path(image_path).is_file(), f"Original image missing: {image_path}")
        require(not output.is_relative_to(Path(image_path).parent), "Output must be outside dataset folders")
    for kind in ("rgb_path", "ela_path"):
        if kind in sample:
            for path in sample[kind]:
                require(not output.is_relative_to(Path(path).parents[2]), "Output must be outside CASIA dataset folders")
    import torch
    import torchvision
    from torchvision import transforms
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    states, models = ev.load_models(root)
    del states
    require(runner.verify_checkpoint_files(root) == checkpoints, "Checkpoint changed while loading")
    device = torch.device(args.device)
    for model in models.values():
        model.to(device).eval()
    transform = ev.input_transform()
    historical = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    output.mkdir(parents=True, exist_ok=False)
    sample.to_csv(output / "casia_selected_pairs.csv", index=False, mode="x")
    new_protocol = {"experiment": "posthoc_ela_pipeline_diagnostic_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_report": str(report_path), "source_report_sha256": sha(report_path),
                    "conditions": {"raw_replay": "Original RGB; native ELA q85; compare to cached predictions",
                                   "jpeg95_secondary": "Original -> JPEG q95 -> RGB and native ELA q85; same processing for REAL/FAKE"},
                    "thresholds": thresholds, "alpha_rgb": .5, "q_primary_secondary": Q_PRIMARY, "q_ela": Q_ELA,
                    "training": False, "threshold_search": False, "quality_search": False,
                    "external_results_already_examined": True, "independent_holdout_claim": False,
                    "columbia_original_modes": manifest.original_mode.value_counts().to_dict(),
                    "columbia_original_formats": manifest["format"].value_counts().to_dict(),
                    "casia": casia_info, "casia_selection_sha256": sha(output / "casia_selected_pairs.csv"),
                    "checkpoints": checkpoints, "evaluator": ev_info, "script_sha256": sha(__file__),
                    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                                "numpy": np.__version__, "pillow": PILLOW_VERSION, "jpeg_codec": features.version_codec("jpg"),
                                "device": str(device), "batch_size": 16},
                    "comparison_tolerance": TOLERANCE,
                    "limits": ["One secondary quality cannot establish causality or an optimal preprocessing policy.",
                               "CASIA pixel audit is a fixed 64-source sample, not a complete dataset audit.",
                               "File hashes and filename-based source groups do not establish independent donor/scene groups.",
                               "IMD2020 remains unused; original Columbia holdout result is preserved."]}
    write_json(output / "protocol.json", new_protocol)
    protocol_sha = sha(output / "protocol.json")
    print(f"1/4 Diagnostic protocol saved: {output}", flush=True)
    try:
        casia_info, stats = audit_casia(sample, casia_info, transform, historical, torch, output)
        print(f"2/4 CASIA pixel checks: {casia_info['status']}; pairs={len(sample)}", flush=True)
        values, parity = [], {"raw_cached_max_abs": {}, "standalone_shared_max_abs": {}}
        fusion = models["feature_fusion_v2"]
        for condition in ("raw_replay", "jpeg95_secondary"):
            predicted = {name: [] for name in runner.MODELS}
            with torch.inference_mode():
                for start in range(0, len(manifest), 16):
                    rb, eb = [], []
                    for row in manifest.iloc[start:start+16].itertuples(index=False):
                        data = Path(row.image_path).read_bytes()
                        require(hashlib.sha256(data).hexdigest() == row.sha256, f"Input file changed: {row.filename}")
                        rgb, _ = runner.decode_rgb(data)
                        require(runner.pixel_sha256(rgb) == row.rgb_pixels_sha256, f"Decoded pixels differ: {row.filename}")
                        if condition == "jpeg95_secondary":
                            rgb = jpeg_rgb(rgb, Q_PRIMARY)
                        ela = runner.ela_from_rgb(rgb)
                        require(np.array_equal(np.asarray(ela), np.asarray(residual(rgb, Q_ELA))), "Independent ELA implementation disagrees")
                        for kind, image in (("rgb",rgb), ("ela",ela)):
                            stats.append(image_stats(image, row.filename, row.true_label, condition, kind))
                        rb.append(transform_check(rgb, transform, historical, torch))
                        eb.append(transform_check(ela, transform, historical, torch))
                    rb, eb = torch.stack(rb).to(device), torch.stack(eb).to(device)
                    rf, ef = fusion.rgb_branch(rb), fusion.ela_branch(eb)
                    logits = {"rgb": models["rgb"].fc(rf), "ela": models["ela"].fc(ef),
                              "feature_fusion_v2": fusion.mlp_head(torch.cat((rf,ef), dim=1))}
                    for name, output_logits in logits.items():
                        probabilities = torch.softmax(output_logits, dim=1)[:,1]
                        predicted[name].extend(probabilities.cpu().tolist())
                        if start == 0 and name in ("rgb", "ela"):
                            direct = torch.softmax(models[name](rb if name == "rgb" else eb),dim=1)[:,1]
                            error = float((probabilities-direct).abs().max().item())
                            parity["standalone_shared_max_abs"][condition+"/"+name] = error
                            require(error <= TOLERANCE, f"Standalone/shared inference differs: {name}")
                    print(f"  {condition}: {min(start+16,len(manifest))}/{len(manifest)}", flush=True)
            predicted["decision_fusion"] = .5*np.array(predicted["rgb"]) + .5*np.array(predicted["ela"])
            for name in runner.MODELS:
                table = manifest[["filename","true_label","sha256"]].copy()
                table["condition"], table["model"] = condition, name
                table["prob_fake"] = predicted[name]
                table["threshold"] = thresholds[name]
                table["pred_label"] = (table.prob_fake >= thresholds[name]).astype(int)
                if condition == "raw_replay":
                    reference = cached[cached.model == name]
                    require(not reference.filename.duplicated().any() and set(reference.filename) == set(table.filename), f"Cached cohort mismatch: {name}")
                    reference = reference.set_index("filename").loc[table.filename].reset_index()
                    require(np.array_equal(table.true_label, reference.true_label) and np.array_equal(table.sha256, reference.sha256), f"Cached labels/hashes mismatch: {name}")
                    error = float(np.abs(table.prob_fake.to_numpy()-reference.prob_fake.to_numpy()).max())
                    parity["raw_cached_max_abs"][name] = error
                    require(error <= TOLERANCE, f"Raw replay differs from cached {name} by {error}; inspect runtime/preprocessing before interpreting secondary results")
                values.append(table)
        print("3/4 Original replay and secondary compression finished; summarizing...", flush=True)
        predictions = pd.concat(values, ignore_index=True)
        predictions.to_csv(output / "predictions.csv", index=False, mode="x")
        comparison = []
        for (condition,name), part in predictions.groupby(["condition","model"],sort=False):
            result = ev.metrics(part.true_label, part.prob_fake, thresholds[name])
            (tn,fp),(fn,tp) = result["confusion_matrix"]
            comparison.append({"condition":condition,"model":name,"threshold":thresholds[name],
                               "accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"roc_auc":result["roc_auc"],
                               "recall_fake":tp/(tp+fn),"fpr":fp/(fp+tn),"TN":tn,"FP":fp,"FN":fn,"TP":tp,
                               "median_score_real":float(part.loc[part.true_label==0,"prob_fake"].median()),
                               "median_score_fake":float(part.loc[part.true_label==1,"prob_fake"].median())})
        pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False, mode="x")
        stat_frame = pd.DataFrame(stats)
        stat_frame.to_csv(output / "image_statistics.csv", index=False, mode="x")
        stat_summary = stat_frame.groupby(["condition","input","true_label"],sort=False).agg(
            images=("filename","size"), width_median=("width","median"), height_median=("height","median"),
            ela_or_rgb_mean=("native_mean_0_255","mean"), mean_clipped_fraction=("native_255_fraction","mean"),
            resized_mean=("resized_mean_0_255","mean"), resized_std_mean=("resized_std","mean")).reset_index()
        require(sha(output / "protocol.json") == protocol_sha, "Diagnostic protocol changed")
        write_json(output / "diagnostics.json", {"status":"completed", "protocol_sha256":protocol_sha,
                   "casia":casia_info,"parity":parity,"image_statistics_summary":stat_summary.to_dict("records"),
                   "comparison":comparison,"conclusion":"Inspect regeneration parity and AUC as well as thresholded metrics. Any improvement here is secondary exploratory evidence; no parameters have been fitted or adopted."})
        with zipfile.ZipFile(output / "diagnostic_bundle.zip", "x", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.iterdir()):
                if path.suffix in (".csv", ".json"):
                    bundle.write(path, path.name)
        print("4/4 Completed. Metrics are fractions:", flush=True)
        print(pd.DataFrame(comparison)[["condition","model","accuracy","roc_auc","fpr"]].to_string(index=False))
        print(f"Send this file: {output / 'diagnostic_bundle.zip'}")
    except Exception as error:
        write_json(output / "failure.json", {"status":"failed", "error_type":type(error).__name__,"error":str(error)})
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
