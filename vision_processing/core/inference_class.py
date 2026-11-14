import numpy as np
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort 

# Clases de vehículos del dataset COCO de interesan
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class YoloDetector:
    def __init__(self, model_name="yolo11s.pt", conf=0.2, imgsz=640):
        self.model = YOLO(model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo para YOLO: {self.device}")
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        # Filtra solo las clases de vehículos que nos interesan
        mask = np.array([c in COCO_VEHICLES for c in clss], dtype=bool)
        
        return boxes[mask], confs[mask], [COCO_VEHICLES[c] for c in clss[mask]]

class VehicleTracker:
    def __init__(self, max_age=3, n_init=3, nms_max_overlap=0.7):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap)

    def update(self, boxes, scores, frame, timestamp):
        detections = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            w, h = x2 - x1, y2 - y1
            detections.append([[float(x1), float(y1), float(w), float(h)], float(score)])

        tracks = self.tracker.update_tracks(detections, frame=frame, today=timestamp)
        
        out_boxes, out_ids = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = t.to_ltrb()
            out_boxes.append([l, t_, r, b])
            out_ids.append(int(t.track_id))
            
        return np.array(out_boxes), np.array(out_ids)

class VisionTrackerBlock:
    def __init__(self, confidence_threshold=0.2, imgsz=640, max_age=3, n_init=3, nms_max_overlap=0.7):
        self.detector = YoloDetector(conf=confidence_threshold, imgsz=imgsz)
        self.tracker = VehicleTracker(max_age=max_age, n_init=n_init, nms_max_overlap=nms_max_overlap)
        self.fps = None
        self.alpha = 0.1

    def process(self, frame, timestamp):
        t0 = time.time()
        
        boxes_det, confs_det, _ = self.detector.infer(frame)
        
        boxes_trk, ids_trk = self.tracker.update(boxes_det, confs_det, frame, timestamp=timestamp)

        processing_time = time.time() - t0
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        if self.fps is None:
            self.fps = current_fps
        else:
            self.fps = (1 - self.alpha) * self.fps + self.alpha * current_fps

        return boxes_trk, ids_trk, self.fps