"""Selecciona frames de toda una sesión y propone cajas vehicle para CVAT."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import zipfile


COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def select_frames(rows, count=500, bins=10, seed=12345):
    """Balancea cámaras y tramos de tiempo; espacia la selección dentro de cada tramo."""
    if count <= 0 or bins <= 0 or count > len(rows):
        raise ValueError("Cantidad o tramos inválidos, o faltan imágenes para la selección.")
    groups = defaultdict(list)
    for row in rows:
        timestamp = float(row["simulation_timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp inválido en frames.csv.")
        groups[row["camera"]].append(dict(row))
    rng = random.Random(seed)
    selected = []
    for camera_index, camera in enumerate(sorted(groups)):
        ordered = sorted(groups[camera], key=lambda row: float(row["simulation_timestamp"]))
        quota = count // len(groups) + (camera_index < count % len(groups))
        start = float(ordered[0]["simulation_timestamp"])
        duration = float(ordered[-1]["simulation_timestamp"]) - start
        buckets = [[] for _ in range(bins)]
        for row in ordered:
            bin_index = min(bins - 1, int((float(row["simulation_timestamp"]) - start) / duration * bins)) if duration else 0
            row["temporal_bin"] = bin_index + 1
            buckets[bin_index].append(row)
        for bin_index, bucket in enumerate(buckets):
            amount = quota // bins + (bin_index < quota % bins)
            if len(bucket) < amount:
                raise ValueError(f"{camera}, tramo {bin_index + 1}: {len(bucket)} imágenes para una cuota de {amount}.")
            # Una imagen por subtramo: evita concentrar la muestra en colas consecutivas.
            for index in range(amount):
                low = index * len(bucket) // amount
                high = (index + 1) * len(bucket) // amount
                pick = rng.randrange(low, high)
                if bin_index == 0 and index == 0:
                    pick = 0
                elif bin_index == bins - 1 and index == amount - 1:
                    pick = len(bucket) - 1
                selected.append(bucket[pick])
    rng.shuffle(selected)
    return selected


def yolo_label(xyxy, width, height):
    """Recorta al límite de la imagen y convierte una caja a clase 0 xywh normalizada."""
    x1, y1, x2, y2 = map(float, xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Caja no finita.")
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caja sin área.")
    values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def package_dataset(output, image_names=None, archive_name="greenlight_vehicle_cvat.zip"):
    """ZIP de dataset completo, sin carpeta contenedora ni previews con cajas."""
    archive = output / archive_name
    names = sorted(image_names) if image_names is not None else None
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        target.write(output / "data.yaml", "data.yaml")
        if names is None:
            target.write(output / "train.txt", "train.txt")
        else:
            target.writestr("train.txt", "".join(f"images/train/{name}\n" for name in names))
        for directory in ("images/train", "labels/train"):
            paths = sorted((output / directory).iterdir()) if names is None else [
                output / directory / (name if directory.startswith("images") else Path(name).with_suffix(".txt"))
                for name in names
            ]
            for path in paths:
                target.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as target:
        if target.testzip() is not None:
            raise ValueError("El ZIP no pasó la verificación de integridad.")
    return archive


def contact_sheet(output, selected):
    """Vista de control: una imagen por cámara y tramo, con cajas propuestas."""
    import cv2
    import numpy as np

    representatives = {}
    for row in selected:
        representatives.setdefault((row["camera"], row["temporal_bin"]), row)
    tiles = []
    for (camera, temporal_bin), row in sorted(representatives.items()):
        frame = cv2.imread(str(output / "images/train" / row["output_image"]))
        height, width = frame.shape[:2]
        label_path = output / "labels/train" / Path(row["output_image"]).with_suffix(".txt")
        for line in label_path.read_text().splitlines():
            _, cx, cy, box_width, box_height = map(float, line.split())
            left, top = int((cx - box_width / 2) * width), int((cy - box_height / 2) * height)
            right, bottom = int((cx + box_width / 2) * width), int((cy + box_height / 2) * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        tile = cv2.resize(frame, (400, 300))
        cv2.rectangle(tile, (0, 0), (400, 24), (0, 0, 0), -1)
        cv2.putText(tile, f"{camera} tramo {temporal_bin} t={float(row['simulation_timestamp']):.1f}s", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    columns = 4
    canvas = np.zeros((math.ceil(len(tiles) / columns) * 300, columns * 400, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = index // columns * 300, index % columns * 400
        canvas[y:y + 300, x:x + 400] = tile
    if not cv2.imwrite(str(output / "preview.jpg"), canvas):
        raise OSError("No se pudo guardar la vista previa.")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--model", type=Path, default=root / "yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="Por defecto CUDA si está disponible, o CPU.")
    args = parser.parse_args()
    session, output, model_path = args.session.resolve(), args.output.resolve(), args.model.resolve()
    if not model_path.is_file():
        parser.error(f"No existe el modelo local: {model_path}")
    if not 0 < args.conf < 1 or args.imgsz <= 0 or args.batch <= 0:
        parser.error("Confianza, tamaño o batch inválidos.")
    if output.exists():
        parser.error("El destino ya existe; elegir uno nuevo para no sobrescribir revisiones.")
    with (session / "frames.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    selected = select_frames(rows, args.count, args.bins, args.seed)
    for row in selected:
        path = (session / row["image"]).resolve()
        if not path.is_relative_to(session) or not path.is_file():
            raise ValueError(f"Ruta original inválida: {row['image']}")
    output.mkdir(parents=True)
    images, labels = output / "images/train", output / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    # Mantener la configuración y caché de YOLO junto al resultado, fuera del perfil del usuario.
    runtime = output / ".runtime"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "detect" or any(model.names.get(key) != name for key, name in COCO_VEHICLES.items()):
        raise ValueError("Se necesita un modelo de detección con las clases COCO esperadas.")
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    for index, row in enumerate(selected, start=1):
        row["output_image"] = f"{index:04d}_{session.name}_{row['camera']}_{Path(row['image']).name}"
        original = session / row["image"]
        shutil.copyfile(original, images / row["output_image"])
        row["source_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    (output / "data.yaml").write_text("path: ./\ntrain: train.txt\nnames:\n  0: vehicle\n", encoding="utf-8")
    (output / "train.txt").write_text("".join(f"images/train/{row['output_image']}\n" for row in selected), encoding="utf-8")
    detections = []
    print(f"Seleccionadas {len(selected)} imágenes. Dispositivo: {device}", flush=True)
    for offset in range(0, len(selected), args.batch):
        batch = selected[offset:offset + args.batch]
        results = model.predict(source=[str(images / row["output_image"]) for row in batch], classes=list(COCO_VEHICLES), conf=args.conf, imgsz=args.imgsz, iou=0.7, agnostic_nms=True, device=device, batch=args.batch, verbose=False, save=False)
        if len(results) != len(batch):
            raise ValueError("YOLO no devolvió un resultado por imagen.")
        for row, result in zip(batch, results):
            if Path(result.path).name != row["output_image"]:
                raise ValueError("El orden de los resultados no coincide con las imágenes.")
            height, width = result.orig_shape
            if (width, height) != (int(row["width"]), int(row["height"])):
                raise ValueError("Las dimensiones no coinciden con frames.csv.")
            lines = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in COCO_VEHICLES:
                    raise ValueError("Predicción fuera de las clases seleccionadas.")
                line = yolo_label(box.xyxy[0].cpu().tolist(), width, height)
                lines.append(line)
                detections.append({"image": row["output_image"], "box_index": len(lines) - 1, "source_class": COCO_VEHICLES[class_id], "confidence": float(box.conf.item()), "label": line})
            (labels / Path(row["output_image"]).with_suffix(".txt")).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            row["proposed_boxes"] = len(lines)
            row["review_status"] = "pending"
        completed = min(offset + args.batch, len(selected))
        if completed == len(selected) or completed // 25 != offset // 25:
            print(f"Preanotadas {completed}/{len(selected)}", flush=True)
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as target:
        for detection in detections:
            target.write(json.dumps(detection) + "\n")
    contact_sheet(output, selected)
    archive = package_dataset(output)
    summary = {
        "status": "complete_pending_manual_review", "session": session.name,
        "created_utc": datetime.now(timezone.utc).isoformat(), "selected_images": len(selected),
        "source_images": len(rows), "cameras": dict(Counter(row["camera"] for row in selected)),
        "temporal_bins": {camera: dict(Counter(row["temporal_bin"] for row in selected if row["camera"] == camera)) for camera in sorted({row["camera"] for row in selected})},
        "sampling": f"Equal quotas by camera and {args.bins} time bins; one random frame per sub-stratum, including session endpoints; shuffled review order.",
        "weather": "Not recorded by capture; temporal coverage is a proxy, not weather classification.",
        "seed": args.seed, "bins": args.bins, "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "ultralytics_version": __import__("ultralytics").__version__,
        "conf": args.conf, "imgsz": args.imgsz, "iou": 0.7, "agnostic_nms": True, "device": device,
        "classes": {0: "vehicle"}, "source_classes": COCO_VEHICLES,
        "proposed_boxes": len(detections), "images_without_predictions": sum(row["proposed_boxes"] == 0 for row in selected),
        "predicted_source_classes": dict(Counter(item["source_class"] for item in detections)),
        "dataset_split": "Annotation pool only; train directory is an import convention, no evaluation split assigned.",
        "archive": archive.name,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    instructions = f"""# GreenLight: {len(selected)} imágenes para revisar en CVAT

1. Crear un proyecto en CVAT con una única etiqueta `vehicle` de tipo Rectangle.
2. En el proyecto: Actions → Import dataset.
3. Seleccionar `Ultralytics YOLO Detection 1.0` y subir `greenlight_vehicle_cvat.zip`.
4. Abrir la tarea y sus jobs; revisar cajas, agregar vehículos omitidos y borrar falsos positivos.
5. Guardar y exportar el dataset revisado como `Ultralytics YOLO Detection 1.0`, con imágenes.

El ZIP contiene imágenes originales sin dibujos y etiquetas YOLO (clase 0).
CVAT creará una tarea para el subset `train`. Ese nombre es una convención del
formato de importación: todavía no se dividió el conjunto para entrenamiento/evaluación.
No entrenar con estas propuestas sin revisarlas. Un TXT vacío significa que YOLO
no detectó vehículos, no que se haya confirmado una escena vacía.

`selection.csv` relaciona cada imagen con su cámara, frame, timestamp y SHA256
original. `predictions.jsonl` conserva clase COCO y confianza para auditoría.
`preview.jpg` muestra una imagen por cámara y tramo con cajas de control;
no forma parte del ZIP ni debe subirse como imagen de entrenamiento.
La selección está equilibrada por cámara y tiempo, no por clima medido.
Los PNG originales de la sesión permanecen intactos.

Propuestas: {len(detections)} cajas con conf >= {args.conf}; car/motorcycle/bus/truck
se remapean a vehicle. Revisar también vehículos pequeños, tapados o fuera de las ROI.
Datos originales, copias y ZIP están excluidos de Git y requieren respaldo separado.

Documentación: https://docs.cvat.ai/docs/dataset_management/import-datasets/
Formato: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
"""
    (output / "IMPORTAR_EN_CVAT.md").write_text(instructions, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Dataset listo: {archive}", flush=True)


if __name__ == "__main__":
    main()
