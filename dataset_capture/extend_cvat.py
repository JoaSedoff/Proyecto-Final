"""Empaqueta una sesión completa conservando el export corregido de una selección CVAT.

Ejecutar desde la raíz: python -m dataset_capture.extend_cvat --help
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import zipfile

import yaml

from dataset_capture.colab.colab_utils import archive_members, indexed_members, validate_vehicle_label
from dataset_capture.prepare_cvat import COCO_VEHICLES, contact_sheet, package_dataset, yolo_label


def load_corrections(archive_path, previous):
    """Exige el mismo conjunto de imágenes y conserva las etiquetas del export, sin inferencia."""
    expected = {Path(row["output_image"]).stem for row in previous}
    if len(expected) != len(previous):
        raise ValueError("Nombres repetidos en la selección anterior.")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive)
        configs = [item for name, item in members.items() if PurePosixPath(name).name in {"data.yaml", "data.yml"}]
        if len(configs) != 1:
            raise ValueError("Se necesita un export Ultralytics YOLO Detection con data.yaml.")
        config = yaml.safe_load(archive.read(configs[0]).decode("utf-8-sig"))
        if config.get("names") not in (["vehicle"], {0: "vehicle"}, {"0": "vehicle"}):
            raise ValueError("El export corregido debe tener exactamente la clase 0: vehicle.")
        listed = []
        for name, item in members.items():
            if PurePosixPath(name).name in {"train.txt", "val.txt", "valid.txt", "test.txt"}:
                for line in archive.read(item).decode("utf-8-sig").splitlines():
                    path = PurePosixPath(line.strip().replace("\\", "/"))
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        listed.append(path.stem)
        if len(listed) != len(set(listed)) or set(listed) != expected:
            raise ValueError("El listado de CVAT no coincide exactamente con la selección anterior.")
        labels = indexed_members({name: item for name, item in members.items()
                                  if PurePosixPath(name).suffix == ".txt" and "labels" in PurePosixPath(name).parts}, {".txt"})
        if not labels.keys() <= expected:
            raise ValueError("El ZIP corregido tiene etiquetas ajenas a la selección anterior.")
        corrections = {}
        for row in previous:
            key = Path(row["output_image"]).stem
            raw = archive.read(labels[key]) if key in labels else b""
            _, boxes = validate_vehicle_label(raw.decode("utf-8-sig"), key)
            corrections[row["output_image"]] = (raw, boxes, key not in labels)
    return corrections


def expand_selection(rows, previous, session_name, bins=10, seed=12345):
    """Incluye todo; conserva nombres anteriores y mezcla las imágenes nuevas."""
    if not rows or bins <= 0:
        raise ValueError("Faltan imágenes o los tramos son inválidos.")
    indexed = {row["image"]: dict(row) for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Rutas repetidas en frames.csv.")
    old_index = {row["image"]: row for row in previous}
    if len(old_index) != len(previous) or not old_index.keys() <= indexed.keys():
        raise ValueError("La selección anterior no pertenece íntegramente a esta sesión.")
    for row in previous:
        current = indexed[row["image"]]
        for field in ("camera", "carla_frame", "width", "height"):
            if current[field] != row[field]:
                raise ValueError(f"Cambió {field} de {row['image']}.")
        if float(current["simulation_timestamp"]) != float(row["simulation_timestamp"]):
            raise ValueError(f"Cambió el timestamp de {row['image']}.")
        name = row["output_image"]
        if Path(name).name != name or "\\" in name or ":" in name:
            raise ValueError("Nombre de salida anterior inválido.")
        current["output_image"] = name
    for camera in sorted({row["camera"] for row in rows}):
        camera_rows = [row for row in indexed.values() if row["camera"] == camera]
        timestamps = [float(row["simulation_timestamp"]) for row in camera_rows]
        if not all(math.isfinite(t) for t in timestamps):
            raise ValueError("Timestamp no finito.")
        start, end = min(timestamps), max(timestamps)
        for row in camera_rows:
            t = float(row["simulation_timestamp"])
            row["temporal_bin"] = min(bins - 1, int((t - start) / (end - start) * bins)) + 1 if end > start else 1
            if row["image"] in old_index and row["temporal_bin"] != int(old_index[row["image"]]["temporal_bin"]):
                raise ValueError("Los tramos temporales deben coincidir con la selección anterior.")
    new = [row for key, row in sorted(indexed.items()) if key not in old_index]
    random.Random(seed).shuffle(new)
    first_index = max((int(row["output_image"].split("_", 1)[0]) for row in previous), default=0) + 1
    for index, row in enumerate(new, first_index):
        row["output_image"] = f"{index:04d}_{session_name}_{row['camera']}_{Path(row['image']).name}"
    selected = [indexed[row["image"]] for row in previous] + new
    if len({row["output_image"].casefold() for row in selected}) != len(selected):
        raise ValueError("Colisión de nombres de salida.")
    for row in selected:
        row["session"] = session_name
        row["previous_selection"] = row["image"] in old_index
    return selected


def write_selection(path, selected):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--corrected-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent.parent / "yolo11m.pt")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    session, output, model_path = args.session.resolve(), args.output.resolve(), args.model.resolve()
    if output.exists() or not model_path.is_file() or not 0 < args.conf < 1 or args.batch <= 0 or args.imgsz <= 0:
        parser.error("Elegir un destino nuevo, modelo existente y parámetros válidos.")
    with (session / "frames.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    with args.previous_selection.open(encoding="utf-8-sig", newline="") as source:
        previous = list(csv.DictReader(source))
    corrections = load_corrections(args.corrected_zip, previous)
    selected = expand_selection(rows, previous, session.name, args.bins, args.seed)
    old = {row["image"]: row for row in previous}
    for row in selected:
        path = (session / row["image"]).resolve()
        if not path.is_relative_to(session) or not path.is_file():
            raise ValueError(f"Imagen original inválida: {row['image']}")
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if row["image"] in old and row["source_sha256"] != old[row["image"]]["source_sha256"]:
            raise ValueError(f"Cambió la imagen ya corregida: {row['image']}")
    images, labels = output / "images/train", output / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    runtime = output / ".runtime"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "detect" or any(model.names.get(key) != name for key, name in COCO_VEHICLES.items()):
        raise ValueError("Se necesita un detector con las clases COCO esperadas.")
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    pending = []
    for row in selected:
        shutil.copyfile(session / row["image"], images / row["output_image"])
        row["annotation_source"] = "cvat_export" if row["previous_selection"] else "yolo11m_coco"
        row["review_status"] = "imported_from_cvat" if row["previous_selection"] else "pending"
        row["missing_label_in_export"] = False
        row["label_boxes"] = 0
        if row["previous_selection"]:
            raw, boxes, missing = corrections[row["output_image"]]
            (labels / Path(row["output_image"]).with_suffix(".txt")).write_bytes(raw)
            row["label_boxes"], row["missing_label_in_export"] = boxes, missing
        else:
            pending.append(row)
    (output / "data.yaml").write_text("path: ./\ntrain: train.txt\nnames:\n  0: vehicle\n", encoding="utf-8")
    (output / "train.txt").write_text("".join(f"images/train/{row['output_image']}\n" for row in selected), encoding="utf-8")
    detections = []
    print(f"Total: {len(selected)}. Correcciones CVAT conservadas: {len(previous)}. Preanotar: {len(pending)}. GPU: {device}", flush=True)
    for offset in range(0, len(pending), args.batch):
        batch = pending[offset:offset + args.batch]
        results = model.predict(source=[str(images / row["output_image"]) for row in batch],
                                classes=list(COCO_VEHICLES), conf=args.conf, imgsz=args.imgsz,
                                iou=0.7, agnostic_nms=True, device=device, batch=args.batch, verbose=False, save=False)
        if len(results) != len(batch):
            raise ValueError("YOLO no devolvió un resultado por imagen.")
        for row, result in zip(batch, results):
            if Path(result.path).name != row["output_image"]:
                raise ValueError("Orden de predicciones incorrecto.")
            height, width = result.orig_shape
            if (width, height) != (int(row["width"]), int(row["height"])):
                raise ValueError("Dimensiones inconsistentes con frames.csv.")
            lines = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in COCO_VEHICLES:
                    raise ValueError("Predicción fuera de las clases seleccionadas.")
                line = yolo_label(box.xyxy[0].cpu().tolist(), width, height)
                lines.append(line)
                detections.append({"image": row["output_image"], "box_index": len(lines) - 1,
                                   "source_class": COCO_VEHICLES[class_id], "confidence": float(box.conf.item()), "label": line})
            row["label_boxes"] = len(lines)
            (labels / Path(row["output_image"]).with_suffix(".txt")).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        completed = min(offset + args.batch, len(pending))
        if completed == len(pending) or completed // 100 != offset // 100:
            print(f"Preanotadas {completed}/{len(pending)}", flush=True)
    write_selection(output / "selection.csv", selected)
    if pending:
        write_selection(output / "selection_pending.csv", pending)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as target:
        for detection in detections:
            target.write(json.dumps(detection) + "\n")
    contact_sheet(output, pending or selected)
    print("Empaquetando todas las imágenes y verificando integridad...", flush=True)
    archive = package_dataset(output)
    pending_archive = None
    if pending:
        print("Empaquetando las imágenes pendientes y verificando integridad...", flush=True)
        pending_archive = package_dataset(output, [row["output_image"] for row in pending], f"greenlight_vehicle_{len(pending)}_pendientes_cvat.zip")
    summary = {
        "status": "complete_with_imported_cvat_labels_and_pending_preannotations",
        "session": session.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_images": len(selected), "source_images": len(rows),
        "cameras": dict(Counter(row["camera"] for row in selected)),
        "imported_cvat_images": len(previous), "pending_images": len(pending),
        "imported_cvat_boxes": sum(row["label_boxes"] for row in selected if row["previous_selection"]),
        "missing_labels_in_cvat_export": sum(row["missing_label_in_export"] for row in selected),
        "missing_label_handling": "Listed CVAT images without a TXT are retained with empty labels; never replaced with YOLO predictions.",
        "proposed_boxes": len(detections), "pending_images_without_predictions": sum(row["label_boxes"] == 0 for row in pending),
        "corrected_zip": str(args.corrected_zip.resolve()),
        "corrected_zip_sha256": hashlib.sha256(args.corrected_zip.read_bytes()).hexdigest(),
        "previous_selection": str(args.previous_selection.resolve()),
        "seed": args.seed, "bins": args.bins,
        "sampling": "All rows from frames.csv, previous names preserved, new frames shuffled reproducibly.",
        "weather": "Not recorded; all captured weather stages retained without filtering.",
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "ultralytics_version": __import__("ultralytics").__version__, "device": device,
        "conf": args.conf, "imgsz": args.imgsz, "iou": 0.7, "agnostic_nms": True,
        "classes": {0: "vehicle"}, "source_classes": COCO_VEHICLES,
        "archive": archive.name, "archive_bytes": archive.stat().st_size,
        "pending_archive": pending_archive.name if pending_archive else None,
        "pending_archive_bytes": pending_archive.stat().st_size if pending_archive else 0,
        "dataset_split": "Annotation pool only; train is an import convention, no training/evaluation split assigned.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    instructions = f"""# GreenLight: {len(selected)} imágenes para CVAT

## Opción A: dataset completo en una tarea nueva

1. Crear un proyecto nuevo, por ejemplo `GreenLight_vehicle_{len(selected)}`, con una sola etiqueta `vehicle` de tipo Rectangle.
2. Actions → Import dataset → Ultralytics YOLO Detection 1.0.
3. Importar `{archive.name}`. Incluye {len(previous)} imágenes con sus anotaciones de tu export CVAT y {len(pending)} imágenes con propuestas nuevas de YOLO11m.
4. Los nombres anteriores se conservan. Los prefijos 0001–{len(previous):04d} corresponden a CVAT; los nuevos siguen a continuación y están mezclados entre cámaras y momentos de captura.
5. Revisar las imágenes nuevas, guardar y exportar el dataset como Ultralytics YOLO Detection 1.0, preferentemente con imágenes.

## Opción B: conservar el proyecto de 500 y agregar solo lo que falta

En el proyecto que contiene la tarea original corregida: Actions → Import dataset → Ultralytics YOLO Detection 1.0 → `{pending_archive.name if pending_archive else ''}`.
Se crea otra tarea con {len(pending)} imágenes diferentes. Revisarla y exportar el PROYECTO completo para obtener {len(selected)} imágenes sin repetir las 500 anteriores.
Elegir una opción; importar ambos ZIP en el mismo proyecto duplicaría las imágenes nuevas.

## Etiquetas y metadatos

- Las etiquetas corregidas se copian del ZIP `{args.corrected_zip.name}`, sin volver a ejecutar YOLO sobre esas imágenes.
- {summary['missing_labels_in_cvat_export']} imágenes listadas en ese export no tenían TXT: se conservan con etiquetas vacías. No se reponen propuestas borradas en CVAT.
- Las imágenes nuevas usan YOLO11m COCO con conf={args.conf}, imgsz={args.imgsz}; car/motorcycle/bus/truck → 0: vehicle. Las propuestas todavía necesitan revisión.
- `selection.csv` corresponde al conjunto COMPLETO. `selection_pending.csv` corresponde solo al ZIP adicional.
- `preview.jpg` es una muestra con cajas, excluida de los ZIP.
- `train` es una convención de importación; aún no hay particiones de entrenamiento ni evaluación.

## Usar luego en tu Colab abierto

Subir el export corregido completo y `selection.csv` a Drive. En la celda de configuración cambiar:

```python
CVAT_ZIP = "/content/drive/MyDrive/GreenLight/datasets/greenlight_{len(selected)}_corregido.zip"
METADATA_CSV = "/content/drive/MyDrive/GreenLight/datasets/selection_vehicle_{len(selected)}.csv"
EXPERIMENT_NAME = "vehicle_{len(selected)}_v1"
```

Usar esas rutas para los archivos subidos. Si el export no incluye imágenes, subir también `{archive.name}` y configurar IMAGES_ZIP con su ruta. Colab usará SOLO las etiquetas del export corregido.
Ejecutar las celdas en orden. Los metadatos incrustados en el notebook solo corresponden a las primeras 500 imágenes: para este conjunto es obligatorio configurar METADATA_CSV.

Documentación: https://docs.cvat.ai/docs/dataset_management/import-datasets/
Formato: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
"""
    (output / "IMPORTAR_EN_CVAT.md").write_text(instructions, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Dataset listo: {archive}", flush=True)


if __name__ == "__main__":
    main()
