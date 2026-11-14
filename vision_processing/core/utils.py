import os
import json
import numpy as np
import cv2

def load_camera_config(config_path):
    """Carga un archivo JSON y arma la estructura de ROIs."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No existe el archivo: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    rois = []
    for rois_cfg in ["rois_A", "rois_B"]:
        if rois_cfg not in cfg:
            raise ValueError(f"El archivo {config_path} no contiene ROIs definidos para {rois_cfg}.")

        roi_structs = []
        for roi in cfg[rois_cfg]:
            roi_structs.append({
                "name": roi.get("name", "ROI"),
                "points": np.array(roi.get("points", []), np.int32),
                "color": tuple(roi.get("color", [255, 0, 0]))
            })
        rois.append(roi_structs)

    return {
        "send_port": cfg.get("send_port", 5557),
        "receive_port": cfg.get("receive_port", 5555),
        "display": cfg.get("display", True),
        "rois_a": rois[0],
        "rois_b": rois[1]
    }


def process_frame(data_package, rois, vision_system, window_name="Camera"):
    """
    Procesa un paquete de datos de imagen para realizar detección y conteo de objetos
    dentro de ROIs, además muestra la imagen procesada en una ventana.

    Args:
        data_package (dict): Diccionario que contiene la imagen y los metadatos.
        rois (list): Lista de diccionarios, donde cada uno define una ROI.
        vision_system: Objeto del tracker de visión (ej. VisionTrackerBlock).
        window_name (str): Nombre de la ventana para mostrar la imagen procesada.

    Returns:
        dict: Un diccionario que mapea el nombre de cada ROI a su conteo de objetos.
              Ejemplo: {"A": 5, "B": 2}
    """
    metadata = data_package["metadata"]
    image_rgba = data_package["image"]
    image_bgr = image_rgba[:, :, :3].copy()

    timestamp = metadata["timestamp"]

    # Procesamiento del modelo
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes, ids, fps_val = vision_system.process(image_rgb, timestamp)

    # Dibujar ROIs + conteo
    total_counts = {}
    for roi in rois:
        cv2.polylines(image_bgr, [roi["points"]], isClosed=True, color=roi["color"], thickness=2)
        cv2.putText(image_bgr, roi["name"],
                    tuple(roi["points"][0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, roi["color"], 2, cv2.LINE_AA)

        inside_ids = set()
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if cv2.pointPolygonTest(roi["points"], center_point, False) >= 0:
                inside_ids.add(track_id)

        total_counts[roi["name"]] = len(inside_ids)

    # --- Dibujar detecciones ---
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Información en pantalla ---
    info_lines = [f"FPS: {fps_val:.1f}"]
    for name, count in total_counts.items():
        info_lines.append(f"{name}: {count}")

    y_offset = 30
    for line in info_lines:
        cv2.putText(image_bgr, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    cv2.imshow(window_name, image_bgr)

    return total_counts.values() 

def process_frame_inference_only(data_package, rois, vision_system):
    """
    Procesa un paquete de datos de imagen para realizar detección y conteo de objetos
    dentro de ROIs.

    Args:
        data_package (dict): Diccionario que contiene la imagen y los metadatos.
        rois (list): Lista de diccionarios, donde cada uno define una ROI.
        vision_system: Objeto del tracker de visión (ej. VisionTrackerBlock).

    Returns:
        enteros: Conteos de objetos dentro de cada ROI como tupla.
    """
    # Extraer imagen y metadatos
    metadata = data_package["metadata"]
    image_rgba = data_package["image"]
    image_bgr = image_rgba[:, :, :3].copy()

    timestamp = metadata["timestamp"]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    boxes, ids, _ = vision_system.process(image_rgb, timestamp)
    
    # Conteo de objetos dentro de cada ROI
    total_counts = {}
    for roi in rois:
        roi_points = np.array(roi["points"], dtype=np.int32)
        
        inside_ids = set()
        for box, track_id in zip(boxes, ids):
            center_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

            if cv2.pointPolygonTest(roi_points, center_point, False) >= 0:
                inside_ids.add(track_id)

        total_counts[roi["name"]] = len(inside_ids)

    return total_counts.values()  # Retorna conteos como tupla (sem_A, sem_B) o (sem_C, sem_D)