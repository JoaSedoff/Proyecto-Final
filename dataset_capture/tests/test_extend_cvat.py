import tempfile
import unittest
from pathlib import Path
import zipfile

from dataset_capture.extend_cvat import expand_selection, load_corrections
from dataset_capture.prepare_cvat import package_dataset


class ExtendCvatTests(unittest.TestCase):
    def test_full_session_retains_previous_names_with_uneven_camera_and_time_counts(self):
        rows = [{"image": f"{camera}/frame_{t:010d}.png", "camera": camera,
                 "carla_frame": str(t), "simulation_timestamp": str(t),
                 "width": "800", "height": "600"}
                for camera, times in (("cam_1", range(13)), ("cam_2", range(0, 13, 3))) for t in times]
        previous = [dict(rows[0], output_image="0001_old_cam_1.png", temporal_bin="1"),
                    dict(rows[13], output_image="0002_old_cam_2.png", temporal_bin="1")]
        expanded = expand_selection(rows, previous, "session", bins=4)
        self.assertEqual(len(expanded), 18)
        self.assertEqual({row["image"] for row in expanded}, {row["image"] for row in rows})
        self.assertEqual(len({row["output_image"] for row in expanded}), 18)
        self.assertEqual([row["output_image"] for row in expanded[:2]], [row["output_image"] for row in previous])
        self.assertEqual(sum(row["previous_selection"] for row in expanded), 2)
        self.assertEqual({row["temporal_bin"] for row in expanded}, {1, 2, 3, 4})
        self.assertEqual(expanded, expand_selection(list(reversed(rows)), previous, "session", bins=4))
        self.assertFalse(any("output_image" in row for row in rows))
        changed = [dict(previous[0], carla_frame="999")]
        with self.assertRaises(ValueError):
            expand_selection(rows, changed, "session", bins=4)

    def test_cvat_labels_are_kept_byte_for_byte_and_missing_txt_stays_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "corrected.zip"
            raw = b"0 0.50 0.50 0.10 0.20\r\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data.yaml", "names:\n  0: vehicle\ntrain: train.txt\n")
                archive.writestr("train.txt", "data/images/train/one.png\ndata/images/train/two.png\n")
                archive.writestr("labels/train/one.txt", raw)
            previous = [{"output_image": "one.png"}, {"output_image": "two.png"}]
            loaded = load_corrections(archive_path, previous)
            self.assertEqual(loaded["one.png"], (raw, 1, False))
            self.assertEqual(loaded["two.png"], (b"", 0, True))
            with self.assertRaises(ValueError):
                load_corrections(archive_path, previous[:1])

    def test_new_image_archive_excludes_previously_corrected_images(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "images/train").mkdir(parents=True)
            (output / "labels/train").mkdir(parents=True)
            for name in ("old", "new"):
                (output / f"images/train/{name}.png").write_bytes(name.encode())
                (output / f"labels/train/{name}.txt").write_text("")
            (output / "data.yaml").write_text("names:\n  0: vehicle\ntrain: train.txt\n")
            (output / "train.txt").write_text("images/train/old.png\nimages/train/new.png\n")
            path = package_dataset(output, ["new.png"], "new_only.zip")
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {"data.yaml", "train.txt", "images/train/new.png", "labels/train/new.txt"})
                self.assertEqual(archive.read("train.txt"), b"images/train/new.png\n")
                self.assertEqual(archive.read("images/train/new.png"), b"new")


if __name__ == "__main__":
    unittest.main()
