import tempfile
import unittest
from collections import Counter
from pathlib import Path
import zipfile

from dataset_capture.prepare_cvat import package_dataset, select_frames, yolo_label


class PrepareCvatTests(unittest.TestCase):
    def test_selection_covers_both_cameras_and_entire_session(self):
        rows = [{"camera": camera, "simulation_timestamp": str(index), "image": f"{camera}/{index}.png"}
                for camera in ("cam_1", "cam_2") for index in range(889)]
        selected = select_frames(rows)
        self.assertEqual(len(selected), 500)
        self.assertEqual(len({row["image"] for row in selected}), 500)
        self.assertEqual(Counter(row["camera"] for row in selected), {"cam_1": 250, "cam_2": 250})
        for camera in ("cam_1", "cam_2"):
            camera_rows = [row for row in selected if row["camera"] == camera]
            self.assertEqual(Counter(row["temporal_bin"] for row in camera_rows), {index: 25 for index in range(1, 11)})
            self.assertEqual(min(float(row["simulation_timestamp"]) for row in camera_rows), 0)
            self.assertEqual(max(float(row["simulation_timestamp"]) for row in camera_rows), 888)
        self.assertEqual(selected, select_frames(list(reversed(rows))))
        self.assertNotEqual(selected, select_frames(rows, seed=42))
        self.assertTrue(all("temporal_bin" not in row for row in rows))

    def test_missing_temporal_coverage_fails_instead_of_biasing_selection(self):
        rows = [{"camera": "cam_1", "simulation_timestamp": "1", "image": f"{index}.png"} for index in range(500)]
        with self.assertRaises(ValueError):
            select_frames(rows)

    def test_labels_are_class_zero_clipped_and_normalized(self):
        values = yolo_label([-20, 120, 400, 720], 800, 600).split()
        self.assertEqual(values[0], "0")
        self.assertEqual(list(map(float, values[1:])), [0.25, 0.6, 0.5, 0.8])
        with self.assertRaises(ValueError):
            yolo_label([10, 10, 0, 0], 800, 600)
        with self.assertRaises(ValueError):
            yolo_label([0, 0, float("nan"), 20], 800, 600)

    def test_zip_structure_matches_images_to_labels_and_retains_empty_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "images/train").mkdir(parents=True)
            (output / "labels/train").mkdir(parents=True)
            (output / "images/train/example.png").write_bytes(b"original-image")
            (output / "labels/train/example.txt").write_text("")
            (output / "data.yaml").write_text("path: ./\ntrain: train.txt\nnames:\n  0: vehicle\n")
            (output / "train.txt").write_text("images/train/example.png\n")
            (output / "preview.jpg").write_bytes(b"preview")
            with zipfile.ZipFile(package_dataset(output)) as archive:
                self.assertEqual(set(archive.namelist()), {"data.yaml", "train.txt", "images/train/example.png", "labels/train/example.txt"})
                self.assertEqual(archive.read("images/train/example.png"), b"original-image")
                self.assertEqual(archive.read("labels/train/example.txt"), b"")


if __name__ == "__main__":
    unittest.main()
