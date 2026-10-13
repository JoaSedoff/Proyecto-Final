import io
import tempfile
import unittest
from pathlib import Path
import zipfile

from PIL import Image
import yaml

from dataset_capture.colab.colab_utils import make_grouped_splits, prepare_cvat_dataset, validate_vehicle_label, write_dataset_split


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (80, 60), (100, 120, 150)).save(output, format="PNG")
    return output.getvalue()


class ColabDatasetTests(unittest.TestCase):
    def test_annotation_only_export_uses_corrected_labels_not_original_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrected, original = root / "corrected.zip", root / "original.zip"
            with zipfile.ZipFile(corrected, "w") as archive:
                archive.writestr("data.yaml", "names: {0: vehicle}\n")
                archive.writestr("train.txt", "images/train/a.png\nimages/train/b.png\n")
                archive.writestr("labels/train/a.txt", "0 0.5 0.5 0.2 0.4\n")
            with zipfile.ZipFile(original, "w") as archive:
                archive.writestr("images/train/a.png", png_bytes())
                archive.writestr("images/train/b.png", png_bytes())
                archive.writestr("labels/train/a.txt", "2 0.1 0.1 0.1 0.1\n")
                archive.writestr("labels/train/b.txt", "7 0.5 0.5 0.1 0.1\n")
            metadata = [{"output_image": name + ".png", "session": "session1", "camera": "cam_1", "simulation_timestamp": 1, "temporal_bin": 1} for name in ("a", "b")]
            rows = prepare_cvat_dataset(corrected, root / "dataset", metadata, original)
            self.assertEqual([row["boxes"] for row in rows], [1, 0])
            self.assertEqual((root / "dataset/labels/all/a.txt").read_text(), "0 0.50000000 0.50000000 0.20000000 0.40000000\n")
            self.assertEqual((root / "dataset/labels/all/b.txt").read_text(), "")
            self.assertEqual((root / "dataset/images/all/a.png").read_bytes(), png_bytes())

    def test_full_export_and_metadata_match_by_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cvat.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("wrapper/data.yaml", "names: [vehicle]\n")
                archive.writestr("wrapper/images/train/a.png", png_bytes())
                archive.writestr("wrapper/labels/train/a.txt", "")
            rows = prepare_cvat_dataset(source, root / "dataset", [{"output_image": "a.jpg", "session": "s", "camera": "cam_2", "simulation_timestamp": 2, "temporal_bin": 1}])
            self.assertEqual(rows[0]["filename"], "a.png")
            self.assertFalse(rows[0]["missing_label_in_export"])

    def test_invalid_archive_path_and_wrong_class_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cvat.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("../data.yaml", "names: [vehicle]\n")
            with self.assertRaises(ValueError):
                prepare_cvat_dataset(source, root / "dataset", [])
        for label in ("2 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.9 0.5 0.5 0.1", "0 0.5 0.5 0.1 0.1 0.9"):
            with self.assertRaises(ValueError):
                validate_vehicle_label(label, "a.txt")

    def test_missing_metadata_never_causes_random_image_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cvat.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("data.yaml", "names: [vehicle]\n")
                archive.writestr("images/train/a.png", png_bytes())
            with self.assertRaisesRegex(ValueError, "metadatos"):
                prepare_cvat_dataset(source, root / "dataset", [])


class ColabSplitTests(unittest.TestCase):
    def records(self, sessions=1, bins=10):
        return [{"filename": f"s{session}_{index}_{camera}.png", "session": f"s{session}", "camera": camera,
                 "simulation_timestamp": float(index * 100 + 50), "temporal_bin": index,
                 "boxes": 1, "image_sha256": f"unique-{session}-{index}-{camera}",
                 "label_sha256": "label", "missing_label_in_export": False}
                for session in range(sessions) for index in range(1, bins + 1) for camera in ("cam_1", "cam_2")]

    def test_single_session_keeps_camera_pairs_and_time_blocks_together(self):
        rows = self.records()
        result, assignment, mode = make_grouped_splits(rows)
        self.assertEqual(mode, "temporal")
        self.assertEqual(len(assignment), 10)
        self.assertEqual({row["split"] for row in result}, {"train", "val", "test"})
        for index in range(1, 11):
            self.assertEqual(len({row["split"] for row in result if row["temporal_bin"] == index}), 1)
        self.assertEqual((result, assignment, mode), make_grouped_splits(rows))
        self.assertTrue(all("split" not in row for row in rows))

    def test_auto_mode_preserves_entire_sessions(self):
        result, _, mode = make_grouped_splits(self.records(sessions=3))
        self.assertEqual(mode, "session")
        for session in ("s0", "s1", "s2"):
            self.assertEqual(len({row["split"] for row in result if row["session"] == session}), 1)

    def test_boundary_margin_excludes_nearby_frames(self):
        rows = self.records(bins=3)
        rows.extend([{**rows[0], "filename": "near_left.png", "simulation_timestamp": 199, "image_sha256": "near-left"},
                     {**rows[2], "filename": "near_right.png", "simulation_timestamp": 201, "image_sha256": "near-right"}])
        result, _, _ = make_grouped_splits(rows, mode="temporal", gap_seconds=10, explicit={"s0:t01": "train", "s0:t02": "val", "s0:t03": "test"})
        self.assertEqual([row["split"] for row in result if row["filename"].startswith("near_")], ["excluded", "excluded"])

    def test_identical_images_cannot_cross_splits(self):
        rows = self.records(bins=3)
        rows[0]["image_sha256"] = rows[2]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "idéntica"):
            make_grouped_splits(rows, gap_seconds=0)

    def test_split_fingerprint_and_yolo_lists_are_reproducible(self):
        records, assignment, mode = make_grouped_splits(self.records())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, summary = write_dataset_split(root, records, assignment, mode, 12345, 10)
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config["names"], {0: "vehicle"})
            for split in ("train", "val", "test"):
                self.assertEqual(len((root / f"{split}.txt").read_text().splitlines()), summary["images_by_split"][split])
            changed = [dict(row) for row in records]
            changed[0]["label_sha256"] = "corrected-again"
            _, new_summary = write_dataset_split(root, changed, assignment, mode, 12345, 10)
            self.assertNotEqual(summary["dataset_fingerprint"], new_summary["dataset_fingerprint"])


if __name__ == "__main__":
    unittest.main()
