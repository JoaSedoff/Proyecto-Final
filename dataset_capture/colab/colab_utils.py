"""Preparación de un export CVAT de una clase y particiones agrupadas para Colab.

Este módulo se incluye dentro del notebook para que no dependa del checkout.
"""

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import zipfile

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def archive_members(archive):
    members = {}
    for item in archive.infolist():
        if item.is_dir():
            continue
        path = PurePosixPath(item.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise ValueError(f"Ruta inválida en ZIP: {item.filename}")
        if path.as_posix() in members:
            raise ValueError(f"Ruta repetida en ZIP: {item.filename}")
        members[path.as_posix()] = item
    return members


def indexed_members(members, suffixes):
    result = {}
    for name, item in members.items():
        path = PurePosixPath(name)
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Nombre base repetido: {path.stem}")
        result[path.stem] = item
    return result


def validate_vehicle_label(text, name):
    lines = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"{name}:{number}: se esperan cinco columnas y clase 0.")
        cx, cy, width, height = map(float, fields[1:])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)):
            raise ValueError(f"{name}:{number}: coordenadas inválidas.")
        if width <= 0 or height <= 0 or cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"{name}:{number}: caja fuera de imagen o sin área.")
        lines.append("0 " + " ".join(f"{value:.8f}" for value in (cx, cy, width, height)))
    return "".join(line + "\n" for line in lines), len(lines)


def metadata_from_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def metadata_index(rows):
    result = {}
    for row in rows:
        filename = row.get("output_image") or row.get("filename") or row.get("image")
        if not filename:
            raise ValueError("El CSV necesita output_image, filename o image.")
        key = Path(filename).stem
        if key in result:
            raise ValueError(f"Metadatos repetidos: {filename}")
        match = re.search(r"(sesion_.+?)_(cam_[12])_frame_", key)
        session = row.get("session") or (match.group(1) if match else None)
        camera = row.get("camera") or (match.group(2) if match else None)
        if not session or not camera:
            raise ValueError(f"Faltan sesión/cámara para {filename}.")
        timestamp = float(row["simulation_timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError(f"Timestamp inválido: {filename}")
        result[key] = {
            "session": session, "camera": camera, "simulation_timestamp": timestamp,
            "temporal_bin": int(row["temporal_bin"]) if row.get("temporal_bin") else None,
        }
    return result


def prepare_cvat_dataset(corrected_zip, destination, metadata_rows, images_zip=None):
    """Las etiquetas SIEMPRE salen de corrected_zip; el ZIP auxiliar solo aporta imágenes."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("El destino ya existe; usar una carpeta nueva.")
    metadata = metadata_index(metadata_rows)
    with zipfile.ZipFile(corrected_zip) as corrected:
        members = archive_members(corrected)
        configs = [item for name, item in members.items() if PurePosixPath(name).name in {"data.yaml", "data.yml"}]
        if len(configs) != 1:
            raise ValueError("Exportar CVAT como Ultralytics YOLO Detection: se necesita un data.yaml.")
        config = yaml.safe_load(corrected.read(configs[0]).decode("utf-8-sig"))
        names = config.get("names") if isinstance(config, dict) else None
        if names not in (["vehicle"], {0: "vehicle"}, {"0": "vehicle"}):
            raise ValueError("El export debe tener exactamente 0: vehicle.")
        corrected_images = indexed_members(members, IMAGE_SUFFIXES)
        label_members = {
            name: item for name, item in members.items()
            if PurePosixPath(name).suffix == ".txt" and "labels" in PurePosixPath(name).parts
        }
        label_index = indexed_members(label_members, {".txt"})
        listed_images = set()
        for name, item in members.items():
            if PurePosixPath(name).name in {"train.txt", "val.txt", "valid.txt", "test.txt"}:
                for line in corrected.read(item).decode("utf-8-sig").splitlines():
                    path = PurePosixPath(line.strip().replace("\\", "/"))
                    if path.suffix.lower() in IMAGE_SUFFIXES:
                        listed_images.add(path.stem)
        if not corrected_images and not images_zip:
            raise ValueError("El export no contiene imágenes. Indicar el ZIP original de imágenes en IMAGES_ZIP.")
        with zipfile.ZipFile(images_zip if not corrected_images else corrected_zip) as image_archive:
            image_index = indexed_members(archive_members(image_archive), IMAGE_SUFFIXES)
            expected = listed_images or set(image_index)
            if not expected <= image_index.keys():
                raise ValueError("Faltan imágenes nombradas en el export de CVAT.")
            if not label_index.keys() <= expected:
                raise ValueError("Hay etiquetas sin imagen correspondiente.")
            if not expected <= metadata.keys():
                missing = sorted(expected - metadata.keys())[:3]
                raise ValueError(f"Faltan metadatos: {missing}. Subir selection.csv de esta selección.")
            if not expected:
                raise ValueError("No se encontraron imágenes.")
            destination.mkdir(parents=True)
            (destination / "images/all").mkdir(parents=True)
            (destination / "labels/all").mkdir(parents=True)
            records = []
            for key in sorted(expected):
                image_bytes = image_archive.read(image_index[key])
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                    image.verify()
                filename = PurePosixPath(image_index[key].filename.replace("\\", "/")).name
                text = corrected.read(label_index[key]).decode("utf-8-sig") if key in label_index else ""
                label, boxes = validate_vehicle_label(text, filename)
                (destination / "images/all" / filename).write_bytes(image_bytes)
                (destination / "labels/all" / f"{key}.txt").write_text(label, encoding="utf-8")
                records.append({
                    "filename": filename, **metadata[key], "width": width, "height": height,
                    "boxes": boxes, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "label_sha256": hashlib.sha256(label.encode()).hexdigest(),
                    "missing_label_in_export": key not in label_index,
                })
    return records


def make_grouped_splits(records, seed=12345, mode="auto", gap_seconds=10.0, explicit=None):
    """Agrupa sesiones o tramos; excluye el margen entre tramos de splits diferentes."""
    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("El margen temporal debe ser finito y no negativo.")
    sessions = {row["session"] for row in records}
    if mode == "auto":
        mode = "session" if len(sessions) >= 3 else "temporal"
    if mode not in {"session", "temporal"}:
        raise ValueError("SPLIT_MODE debe ser auto, session o temporal.")
    result = [dict(row) for row in records]
    for row in result:
        if mode == "temporal" and row["temporal_bin"] is None:
            raise ValueError("La separación temporal necesita temporal_bin en los metadatos.")
        row["group"] = row["session"] if mode == "session" else f"{row['session']}:t{row['temporal_bin']:02d}"
    groups = sorted({row["group"] for row in result})
    if len(groups) < 3:
        raise ValueError("Se necesitan al menos tres grupos para train/val/test.")
    if explicit:
        assignment = dict(explicit)
        if set(assignment) != set(groups) or any(value not in {"train", "val", "test"} for value in assignment.values()):
            raise ValueError("GROUP_SPLITS debe asignar todos los grupos a train, val o test.")
    else:
        random.Random(seed).shuffle(groups)
        test_count = max(1, round(len(groups) * 0.15))
        val_count = max(1, round(len(groups) * 0.15))
        assignment = {group: "test" if index < test_count else "val" if index < test_count + val_count else "train"
                      for index, group in enumerate(groups)}
    boundaries = []
    if mode == "temporal" and gap_seconds:
        for session in sessions:
            spans = []
            for group in {row["group"] for row in result if row["session"] == session}:
                times = [row["simulation_timestamp"] for row in result if row["group"] == group]
                spans.append((min(times), max(times), group))
            spans.sort()
            for left, right in zip(spans, spans[1:]):
                if assignment[left[2]] != assignment[right[2]]:
                    boundaries.append((session, (left[1] + right[0]) / 2))
    for row in result:
        row["split"] = assignment[row["group"]]
        if any(row["session"] == session and abs(row["simulation_timestamp"] - boundary) <= gap_seconds for session, boundary in boundaries):
            row["split"] = "excluded"
    split_names = {row["split"] for row in result}
    if not {"train", "val", "test"} <= split_names:
        raise ValueError("Un conjunto quedó vacío. Cambiar grupos o capturar más sesiones.")
    # Evitar que dos copias idénticas terminen en conjuntos distintos.
    digest_splits = {}
    for row in result:
        if row["split"] == "excluded":
            continue
        old_split = digest_splits.setdefault(row["image_sha256"], row["split"])
        if old_split != row["split"]:
            raise ValueError("Una imagen idéntica aparece en conjuntos distintos. Revisar duplicados/grupos.")
    return result, assignment, mode


def write_dataset_split(destination, records, assignment, mode, seed, gap_seconds):
    destination = Path(destination)
    for split in ("train", "val", "test"):
        rows = [row for row in records if row["split"] == split]
        (destination / f"{split}.txt").write_text("".join(f"./images/all/{row['filename']}\n" for row in rows), encoding="utf-8")
    config = {"path": str(destination.resolve()), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": {0: "vehicle"}}
    (destination / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (destination / "split_manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    canonical = {
        "seed": seed, "mode": mode, "gap_seconds": gap_seconds,
        "images": sorted((row["filename"], row["image_sha256"], row["label_sha256"], row["split"]) for row in records),
    }
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    summary = {
        "dataset_fingerprint": fingerprint, "seed": seed, "mode": mode,
        "gap_seconds": gap_seconds, "groups": assignment,
        "images_by_split": dict(Counter(row["split"] for row in records)),
        "empty_images_by_split": dict(Counter(row["split"] for row in records if row["boxes"] == 0)),
        "missing_labels_in_export": sum(row["missing_label_in_export"] for row in records),
        "scope": "Independent capture sessions" if mode == "session" else "Internal temporal holdout; nearby scenes can remain correlated",
    }
    (destination / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination / "data.yaml", summary
