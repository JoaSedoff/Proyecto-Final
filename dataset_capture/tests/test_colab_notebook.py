import base64
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import zlib


COLAB = Path(__file__).resolve().parents[1] / "colab"


def notebook():
    return json.loads((COLAB / "greenlight_yolo_colab.ipynb").read_text(encoding="utf-8"))


class ColabNotebookTests(unittest.TestCase):
    def test_every_code_cell_compiles_and_outputs_are_empty(self):
        document = notebook()
        self.assertEqual(document["nbformat"], 4)
        ids = [cell["id"] for cell in document["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_embedded_helpers_and_metadata_match_source_files(self):
        sources = ["".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code"]
        self.assertIn((COLAB / "colab_utils.py").read_text(encoding="utf-8").strip(), [source.strip() for source in sources])
        namespace = {"json": json}
        exec(next(source for source in sources if "DEFAULT_METADATA = " in source), namespace)
        with (COLAB / "selection_vehicle_500.csv").open(encoding="utf-8", newline="") as incoming:
            expected = list(csv.DictReader(incoming))
        self.assertEqual(namespace["DEFAULT_METADATA"], expected)
        self.assertEqual(len(expected), 500)
        self.assertTrue(all("proposed_boxes" not in row and "review_status" not in row for row in expected))

    def run_training_cell(self, root, checkpoint_text=None):
        calls = []
        model_name = "yolo11n"
        run_dir = root / "training" / model_name
        if checkpoint_text is not None:
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights/last.pt").write_text(checkpoint_text)
            if checkpoint_text == "finished":
                (run_dir / "weights/best.pt").write_text("best")

        class FakeYolo:
            def __init__(self, weights):
                self.weights = weights

            def train(self, **kwargs):
                calls.append(kwargs)
                destination = Path(self.weights).parent.parent if kwargs.get("resume") else Path(kwargs["project"]) / kwargs["name"]
                (destination / "weights").mkdir(parents=True, exist_ok=True)
                (destination / "weights/last.pt").write_text("finished")
                (destination / "weights/best.pt").write_text("best")
                self.trainer = SimpleNamespace(save_dir=destination)

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None),
                                     load=lambda path, **kwargs: {"epoch": -1 if Path(path).read_text() == "finished" else 1})
        namespace = {"RUN_TRAINING": True, "MODELS": [model_name + ".pt"], "Path": Path,
                     "EXPERIMENT_DIR": root, "RESUME_INTERRUPTED": True, "YOLO": FakeYolo,
                     "torch": fake_torch, "COMMON_TRAIN": {}, "DATA_YAML": root / "data.yaml",
                     "WORKERS": 2, "EPOCHS": 100, "PATIENCE": 20, "split_summary": {"dataset_fingerprint": "fixture"},
                     "gc": gc, "time": time, "json": json, "datetime": datetime, "timezone": timezone}
        source = next("".join(cell["source"]) for cell in notebook()["cells"]
                      if cell["cell_type"] == "code" and "if RUN_TRAINING:" in "".join(cell["source"]))
        exec(source, namespace)
        self.assertTrue((run_dir / "completed.json").is_file())
        exec(source, namespace)
        return calls

    def test_training_state_machine_starts_new_run_and_skips_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = self.run_training_cell(Path(directory))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["epochs"], 100)

    def test_training_state_machine_resumes_interrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = self.run_training_cell(Path(directory), "interrupted")
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]["resume"])
            self.assertEqual(calls[0]["data"], str(Path(directory) / "data.yaml"))

    def test_training_state_machine_recovers_finalized_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = self.run_training_cell(Path(directory), "finished")
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
