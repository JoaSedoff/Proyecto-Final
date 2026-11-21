"""Costo por frame del bloque de visión de GreenLight (YOLO + DeepSORT) sobre frames reales de Town04."""
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=Path, required=True, help="Carpeta de sesión con subcarpetas cam_1 y cam_2 (PNG)")
ap.add_argument("--weights", type=Path, required=True, help="Carpeta con yolo11n.pt y yolo11s.pt")
ap.add_argument("--out", type=Path, default=Path("vision.jsonl"))
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision_processing"))
from core.inference_class import VisionTrackerBlock  # noqa: E402

FR = args.frames
W = args.weights

frames = {}
for cam in ("cam_1", "cam_2"):
    files = sorted((FR / cam).glob("*.png"))[:400]
    frames[cam] = [cv2.resize(cv2.imread(str(f)), (800, 600)) for f in files]

out = []
for model in ("yolo11n.pt", "yolo11s.pt"):
    blocks = {}
    for cam in frames:
        b = VisionTrackerBlock(confidence_threshold=0.25, imgsz=640, max_age=1, n_init=5, nms_max_overlap=1.0)
        from ultralytics import YOLO
        b.detector.model = YOLO(str(W / model))
        blocks[cam] = b
    times, det, trk = [], [], []
    n = min(len(frames["cam_1"]), len(frames["cam_2"]))
    for i in range(n):
        t0 = time.perf_counter()
        for cam, b in blocks.items():
            f = frames[cam][i]
            t1 = time.perf_counter()
            boxes, confs, _ = b.detector.infer(f)
            t2 = time.perf_counter()
            b.tracker.update(boxes, confs, f, timestamp=None)
            t3 = time.perf_counter()
            if i >= 20:
                det.append(t2 - t1)
                trk.append(t3 - t2)
        if i >= 20:
            times.append(time.perf_counter() - t0)
    r = {"model": model, "pairs": len(times),
         "ms_pair_med": round(1000 * statistics.median(times), 1),
         "ms_pair_p95": round(1000 * sorted(times)[int(0.95 * len(times))], 1),
         "ms_yolo_med": round(1000 * statistics.median(det), 1),
         "ms_deepsort_med": round(1000 * statistics.median(trk), 1)}
    print(json.dumps(r))
    out.append(r)
args.out.write_text("\n".join(json.dumps(x) for x in out))
