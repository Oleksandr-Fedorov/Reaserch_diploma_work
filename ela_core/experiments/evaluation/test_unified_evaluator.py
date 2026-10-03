"""Regression checks using synthetic tables/images only; no project datasets/models."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import unified_evaluator as evaluator


def table():
    return pd.DataFrame({"filename": ["a.png", "b.png", "c.png", "d.png"],
                         "true_label": [0, 0, 1, 1], "prob_fake": [.1, .3, .6, .9]})


class TableChecks(unittest.TestCase):
    def test_alignment_is_by_filename_not_row_order(self):
        reference = table()
        actual = evaluator.align(reference, reference.iloc[::-1], "fixture")
        pd.testing.assert_frame_equal(reference, actual)

    def test_missing_extra_duplicate_and_conflicting_labels_fail(self):
        reference = table()
        with self.assertRaisesRegex(ValueError, "sets differ"):
            evaluator.align(reference, reference.iloc[:-1], "fixture")
        altered = reference.copy()
        altered.loc[0, "true_label"] = 1
        with self.assertRaisesRegex(ValueError, "labels differ"):
            evaluator.align(reference, altered, "fixture")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluator.validate_table(pd.concat([reference, reference]), "fixture")

    def test_invalid_probabilities_and_class_values_fail(self):
        for invalid in [float("nan"), float("inf"), -.1, 1.1]:
            frame = table()
            frame.loc[0, "prob_fake"] = invalid
            with self.assertRaises(ValueError):
                evaluator.validate_table(frame, "fixture")
        for invalid in [-1, 2, .5]:
            frame = table().astype({"true_label": float})
            frame.loc[0, "true_label"] = invalid
            with self.assertRaises(ValueError):
                evaluator.validate_table(frame, "fixture")

    def test_filename_class_mapping_is_checked(self):
        frame = table()
        frame.loc[0, "filename"] = "CASIA1_FAKE_source_aug1.png"
        with self.assertRaisesRegex(ValueError, "filename class"):
            evaluator.validate_table(frame, "fixture")

    def test_metrics_use_fake_positive_and_greater_equal(self):
        result = evaluator.metrics([0, 0, 1, 1], [.1, .5, .4, .9], .5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["macro_f1"], .5)

    def test_comparison_alignment_and_fixed_fusion(self):
        frame = table()
        result, predictions = evaluator.comparison({"rgb": frame.iloc[::-1], "ela": frame,
                                                    "feature_fusion_v2": frame}, .5, .5)
        self.assertEqual(result["decision_fusion"]["accuracy"], 1.)
        self.assertEqual(len(predictions), 16)

    def test_cached_prediction_threshold_mismatch_fails(self):
        frame = table()
        frame["pred_label"] = [0, 0, 1, 1]  # Fusion threshold .2893 predicts row b FAKE.
        with self.assertRaisesRegex(ValueError, "cached predictions disagree"):
            evaluator.comparison({"rgb": frame, "ela": frame, "feature_fusion_v2": frame}, .5, .5)

    def test_common_validation_excludes_contaminated_rows(self):
        rgb, ela = table(), table()
        clean = np.array([True, False, False, True])
        expected = evaluator.choose_decision_parameters(rgb, ela, clean)
        rgb.loc[[1, 2], "prob_fake"] = [.99, .01]
        ela.loc[[1, 2], "prob_fake"] = [.98, .02]
        self.assertEqual(expected, evaluator.choose_decision_parameters(rgb, ela, clean))

    def test_unparseable_source_fails_closed(self):
        self.assertEqual(evaluator.source_stem("CASIA2_FAKE_Tp_image_aug4.png"), ("CASIA2", "Tp_image"))
        with self.assertRaises(ValueError):
            evaluator.source_stem("unrelated.png")

    def test_output_directory_conflict_stops_before_loading(self):
        with patch.object(evaluator, "cached") as cached:
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluator.main(["cached", "--output-dir", str(Path(__file__).parent)])
            cached.assert_not_called()

    def test_infer_requires_manifest_before_model_loading(self):
        with patch.object(evaluator, "infer") as infer:
            with self.assertRaisesRegex(ValueError, "explicit --manifest"):
                evaluator.main(["infer"])
            infer.assert_not_called()

    def test_transform_matches_training_on_synthetic_image(self):
        from PIL import Image
        from torchvision import transforms
        import torch
        image = Image.fromarray(np.arange(29 * 41 * 3, dtype=np.uint8).reshape(29, 41, 3))
        original = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.assertTrue(torch.equal(original(image), evaluator.input_transform()(image)))

    def test_manifest_rejects_same_file_for_both_modalities(self):
        frame = table().drop(columns="prob_fake")
        frame["rgb_path"] = frame.filename
        frame["ela_path"] = frame.filename
        with patch.object(evaluator.pd, "read_csv", return_value=frame), patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "distinct files"):
                evaluator.read_pairs(Path(__file__).parent / "synthetic.csv")

    def test_infer_pipeline_on_synthetic_images_and_mock_heads(self):
        from PIL import Image
        import torch
        from torch import nn

        class Branch(nn.Module):
            def forward(self, image):
                return image.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 512)

        class Head(nn.Module):
            def forward(self, features):
                x = features.mean(dim=1)
                return torch.stack((-x, x), dim=1)

        class Baseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = Head()

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.rgb_branch, self.ela_branch, self.mlp_head = Branch(), Branch(), Head()

        frame = table().drop(columns="prob_fake")
        frame["rgb_path"] = frame.filename
        frame["ela_path"] = frame.filename
        mock_models = {"rgb": Baseline(), "ela": Baseline(), "feature_fusion_v2": Fusion()}
        with patch.object(evaluator, "read_pairs", return_value=frame), \
             patch.object(evaluator, "load_models", return_value=({}, mock_models)), \
             patch.object(evaluator, "fingerprint", return_value={"synthetic": True}), \
             patch.object(Image, "open", side_effect=lambda _: Image.new("RGB", (20, 30), (80, 100, 120))):
            result, predictions = evaluator.infer(Path("unused"), Path("synthetic.csv"), .5, .5, 2)
        self.assertEqual(set(result["models"]), {"rgb", "ela", "decision_fusion", "feature_fusion_v2"})
        self.assertEqual(len(predictions), 16)
        self.assertTrue(np.isfinite(predictions.prob_fake).all())


if __name__ == "__main__":
    unittest.main()
