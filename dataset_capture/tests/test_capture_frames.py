"""Pruebas de muestreo, persistencia y recepción, sin depender de CARLA."""

import csv
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np
import zmq


SCRIPT = Path(__file__).resolve().parents[1] / "capture_frames.py"
SPEC = importlib.util.spec_from_file_location("capture_frames", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def camera_data(frame, timestamp, color=(7, 31, 193, 255)):
    image = np.empty((6, 8, 4), dtype=np.uint8)
    image[:] = color
    return {"image": image, "metadata": {
        "frame": frame, "timestamp": timestamp, "width": 8, "height": 6,
    }}


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.recorder = MODULE.FrameRecorder(
            self.temp.name, "test_session", 1.0, "tcp://localhost:5555"
        )

    def tearDown(self):
        self.recorder.close("stopped")
        self.temp.cleanup()

    def test_independent_timestamps_missing_images_and_duplicates(self):
        recorder = self.recorder
        recorder.record_package({"image1": camera_data(100, 10), "image2": None})
        recorder.record_package({"image1": camera_data(100, 10),
                                 "image2": camera_data(80, 6)})
        recorder.record_package({"image1": camera_data(101, 10.4),
                                 "image2": camera_data(81, 6.5)})
        recorder.record_package({"image1": camera_data(120, 11.05),
                                 "image2": camera_data(90, 7)})
        self.assertEqual(recorder.counts, {"cam_1": 2, "cam_2": 2})
        with (recorder.session_dir / "frames.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["carla_frame"] for row in rows], ["100", "80", "120", "90"])
        for row in rows:
            self.assertTrue((recorder.session_dir / row["image"]).is_file())

    def test_png_preserves_resolution_and_bgr_colors(self):
        data = camera_data(1, 0)
        self.recorder.record_package({"image1": data})
        path = self.recorder.session_dir / "cam_1" / "frame_0000000001.png"
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, data["image"][:, :, :3])

    def test_invalid_image_does_not_prevent_other_camera(self):
        bad = camera_data(1, 0)
        bad["metadata"]["width"] = 999
        self.recorder.record_package({"image1": bad, "image2": camera_data(1, 0)})
        self.assertEqual(self.recorder.counts, {"cam_1": 0, "cam_2": 1})
        self.assertEqual(self.recorder.warnings, 1)

    def test_simulation_reset_stops_capture(self):
        self.recorder.record_package({"image1": camera_data(100, 10)})
        with self.assertRaisesRegex(RuntimeError, "retrocedió"):
            self.recorder.record_package({"image1": camera_data(1, 0)})
        self.assertEqual(self.recorder.counts["cam_1"], 1)

    def test_existing_session_is_never_reused(self):
        with self.assertRaises(FileExistsError):
            MODULE.FrameRecorder(self.temp.name, "test_session", 1, "tcp://localhost:5555")

    def test_limit_applies_to_each_camera(self):
        self.recorder.max_frames = 1
        self.recorder.record_package({"image1": camera_data(1, 0)})
        self.assertFalse(self.recorder.complete)
        self.recorder.record_package({"image1": camera_data(2, 2), "image2": camera_data(3, 3)})
        self.assertTrue(self.recorder.complete)
        self.assertEqual(self.recorder.counts, {"cam_1": 1, "cam_2": 1})

    def test_cli_validation(self):
        for option in ["--interval", "--green-time", "--yellow-time", "--all-red-time"]:
            for argument in ["0", "-1", "nan", "inf"]:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    MODULE.build_parser().parse_args([option, argument])


class ControllerTests(unittest.TestCase):
    def test_boundaries_and_complete_cycle(self):
        controller = MODULE.FixedTrafficController()
        self.assertEqual(set(controller.message()["states"].values()), {"RED"})
        for elapsed, active, color in [
            (0, "A", "GREEN"), (24.99, "A", "GREEN"), (25, "A", "YELLOW"),
            (30, None, "RED"), (31.99, None, "RED"), (32, "B", "GREEN"),
            (57, "B", "YELLOW"), (62, None, "RED"), (64, "C", "GREEN"),
            (96, "D", "GREEN"), (121, "D", "YELLOW"), (126, None, "RED"),
            (128, "A", "GREEN"), (1280 + 32, "B", "GREEN"),
        ]:
            controller.update({"image1": camera_data(1, 100 + elapsed)})
            states = controller.message()["states"]
            expected = {name: color if name == active else "RED" for name in "ABCD"}
            self.assertEqual(states, expected, f"t={elapsed}")
        self.assertEqual(set(controller.message(stop=True)["states"].values()), {"RED"})

    def test_custom_times_and_nonzero_origin(self):
        controller = MODULE.FixedTrafficController(10, 3, 1)
        for elapsed, active, color in [(0, "A", "GREEN"), (10, "A", "YELLOW"),
                                       (13, None, "RED"), (14, "B", "GREEN")]:
            controller.update({"image2": camera_data(1, 500 + elapsed)})
            expected = {name: color if name == active else "RED" for name in "ABCD"}
            self.assertEqual(controller.message()["states"], expected)

    def test_missing_invalid_or_lagging_camera_keeps_clock(self):
        controller = MODULE.FixedTrafficController()
        controller.update({"image1": camera_data(1, 100), "image2": camera_data(1, 80)})
        controller.update({"image1": None, "image2": camera_data(2, 81)})
        controller.update({"image1": camera_data(2, float("nan"))})
        controller.update({"image1": None, "image2": None})
        self.assertEqual(controller.last_timestamp, 100)
        self.assertEqual(controller.message()["states"]["A"], "GREEN")
        controller.update({"image2": camera_data(3, 125)})
        self.assertEqual(controller.message()["states"]["A"], "YELLOW")

    def test_positive_finite_durations_required(self):
        for value in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                MODULE.FixedTrafficController(green_time=value)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.context = zmq.Context()
        # XPUB permite verificar la conexión del suscriptor sin sleeps arbitrarios.
        self.publisher = self.context.socket(zmq.XPUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.port = self.publisher.bind_to_random_port("tcp://127.0.0.1")
        self.controls = self.context.socket(zmq.SUB)
        self.controls.setsockopt(zmq.LINGER, 0)
        self.controls.setsockopt_string(zmq.SUBSCRIBE, "")
        self.process = None
        self.session = Path(self.temp.name) / "integration"

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.communicate()
        self.publisher.close()
        self.controls.close()
        self.context.term()
        self.temp.cleanup()

    def start_capture(self, max_frames, interval=1):
        self.process = subprocess.Popen([
            sys.executable, str(SCRIPT), "--endpoint", f"tcp://127.0.0.1:{self.port}",
            "--control-endpoint", "tcp://127.0.0.1:*",
            "--output", self.temp.name, "--session", "integration",
            "--max-frames", str(max_frames), "--interval", str(interval),
            "--notes", "Synthetic integration test",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
           encoding="utf-8", errors="replace")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail("El capturador terminó al iniciar: " + str(self.process.communicate()))
            try:
                info = json.loads((self.session / "session.json").read_text(encoding="utf-8"))
                endpoint = info["traffic_control"]["endpoint"]
                break
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                time.sleep(0.02)
        else:
            self.fail("El capturador no anunció el endpoint de control.")
        self.assertTrue(self.publisher.poll(3000, zmq.POLLIN), "Falta suscripción de imágenes.")
        self.assertEqual(self.publisher.recv(), b"\x01")
        self.controls.connect(endpoint)
        # Simula un simulador bloqueado antes de entregar su primera imagen.
        self.wait_control(None, dict.fromkeys("ABCD", "RED"))

    def wait_control(self, timestamp, expected_states):
        deadline = time.monotonic() + 5
        received = []
        while time.monotonic() < deadline:
            if self.controls.poll(100, zmq.POLLIN):
                message = self.controls.recv_pyobj()
                received.append(message)
                if (message["simulation_timestamp"] == timestamp
                        and message["states"] == expected_states):
                    return message
        self.fail(f"No llegó control t={timestamp}, estados={expected_states}: {received}")

    def finish(self, expected_code=0):
        stdout, stderr = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, expected_code, stdout + stderr)
        return json.loads((self.session / "session.json").read_text(encoding="utf-8"))

    def test_cli_receives_publisher_and_finalizes_metadata(self):
        self.start_capture(max_frames=1)
        self.publisher.send_pyobj({"image1": camera_data(20, 2), "image2": camera_data(21, 2.1)})
        self.wait_control(2.1, {"A": "GREEN", "B": "RED", "C": "RED", "D": "RED"})
        self.wait_control(2.1, dict.fromkeys("ABCD", "RED"))
        info = self.finish()
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["counts"], {"cam_1": 1, "cam_2": 1})
        self.assertIsNotNone(info["ended_utc"])
        self.assertEqual(info["traffic_control"]["start_simulation_timestamp"], 2.1)
        self.assertEqual(len(list(self.session.glob("cam_*/*.png"))), 2)

    def test_simulator_can_advance_without_saving_every_frame(self):
        self.start_capture(max_frames=2, interval=1000)
        self.publisher.send_pyobj({"image1": None, "image2": None})
        self.wait_control(None, dict.fromkeys("ABCD", "RED"))
        # Como run_simulator.py, esperar control antes de publicar el próximo tick.
        for timestamp, active, color in [
            (0, "A", "GREEN"), (10, "A", "GREEN"), (25, "A", "YELLOW"),
            (30, None, "RED"), (32, "B", "GREEN"), (64, "C", "GREEN"),
            (96, "D", "GREEN"), (121, "D", "YELLOW"), (126, None, "RED"),
            (128, "A", "GREEN"), (1000, "D", "GREEN"),
        ]:
            self.publisher.send_pyobj({
                "image1": camera_data(100 + timestamp, timestamp),
                "image2": camera_data(100 + timestamp, timestamp),
            })
            expected = {name: color if name == active else "RED" for name in "ABCD"}
            self.wait_control(timestamp, expected)
        info = self.finish()
        self.assertEqual(info["counts"], {"cam_1": 2, "cam_2": 2})
        self.assertEqual(info["traffic_control"]["last_simulation_timestamp"], 1000)
        with (self.session / "frames.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([float(row["simulation_timestamp"]) for row in rows], [0, 0, 1000, 1000])

    def test_reset_ends_session_with_red_command_and_error_metadata(self):
        self.start_capture(max_frames=0)
        self.publisher.send_pyobj({"image1": camera_data(100, 10), "image2": camera_data(100, 10)})
        self.wait_control(10, {"A": "GREEN", "B": "RED", "C": "RED", "D": "RED"})
        self.publisher.send_pyobj({"image1": camera_data(1, 0)})
        self.wait_control(10, dict.fromkeys("ABCD", "RED"))
        info = self.finish(expected_code=1)
        self.assertEqual(info["status"], "error")
        self.assertIn("retrocedió", info["error"])
        self.assertEqual(info["counts"], {"cam_1": 1, "cam_2": 1})

    def test_occupied_control_port_fails_without_creating_session(self):
        # El publicador sintético ya ocupa este puerto; el control debe rechazarlo.
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--output", self.temp.name,
            "--session", "integration", "--control-endpoint", f"tcp://127.0.0.1:{self.port}",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("run_vision.py", result.stderr)
        self.assertFalse(self.session.exists())


if __name__ == "__main__":
    unittest.main()
