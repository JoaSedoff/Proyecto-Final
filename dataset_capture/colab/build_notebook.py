"""Genera el notebook autónomo de Colab a partir de helpers y metadatos locales."""

import base64
import csv
import hashlib
import json
from pathlib import Path
import textwrap
import zlib


ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def add(kind, source):
        source = textwrap.dedent(source).strip() + "\n"
        cell = {"cell_type": kind, "id": hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
    # GreenLight: entrenamiento y evaluación de YOLO11 en Google Colab

    Este notebook usa las **anotaciones corregidas que exportaste de CVAT**, con una
    clase `vehicle`. Entrena YOLO11n/s/m y guarda checkpoints, gráficos y comparaciones
    en tu Drive. No necesita CARLA ni ejecutar código en tu GPU local.

    **Preparación:** subir el ZIP corregido a `Mi unidad/GreenLight/datasets/`, abrir
    este notebook y elegir **Entorno de ejecución → Cambiar tipo de entorno → GPU**.
    El tipo de GPU depende de la disponibilidad de Colab.

    Si exportaste solo etiquetas, subir también `greenlight_vehicle_cvat.zip`, que
    preparamos originalmente. Ese ZIP auxiliar aporta únicamente imágenes: sus
    propuestas de etiquetas no se usan.

    Los metadatos de tus 500 imágenes están incluidos. Para una selección distinta
    (por ejemplo 1.200), subir su `selection.csv` y configurar `METADATA_CSV`.

    Ejecutar las celdas en orden. La prueba final está desactivada inicialmente para
    reservar `test` mientras elegís configuraciones usando `val`.
    """)
    add("markdown", """
    ## 1. Instalar la versión del proyecto

    Se conserva el PyTorch/CUDA de Colab. No se instalan las dependencias de CARLA.
    """)
    add("code", """
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.197"])
    """)
    add("markdown", """
    ## 2. Configurar datos y experimento

    `CVAT_ZIP` debe ser la exportación **corregida**, formato Ultralytics YOLO Detection.
    `IMAGES_ZIP` se necesita solo cuando la exportación no incluye imágenes.

    Con tres o más sesiones, `SPLIT_MODE="auto"` reserva sesiones enteras. Con tu
    sesión actual, separa tramos temporales, conserva ambas cámaras de cada tramo
    juntas y excluye imágenes cercanas a los límites entre conjuntos diferentes.
    Esta última modalidad permite un piloto; no equivale a evaluar nuevas sesiones.

    Podés asignar manualmente todos los grupos con `GROUP_SPLITS` para distribuir
    condiciones conocidas entre conjuntos. El clima no figura en los metadatos y
    el muestreo automático no garantiza cuotas por clima. Revisá la tabla del paso 5.

    Usar otro `EXPERIMENT_NAME` al cambiar dataset o parámetros. Para continuar tras
    una desconexión, mantener el nombre y ejecutar de nuevo: se recupera `last.pt`
    cuando el experimento no terminó y la configuración coincide.
    """)
    add("code", """
    CVAT_ZIP = "/content/drive/MyDrive/GreenLight/datasets/greenlight_cvat_corregido.zip"
    IMAGES_ZIP = ""  # Si faltan imágenes: /content/drive/MyDrive/GreenLight/datasets/greenlight_vehicle_cvat.zip
    METADATA_CSV = ""  # Vacío: metadatos incluidos de las 500 imágenes originales

    EXPERIMENT_NAME = "vehicle_v1"
    MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
    IMGSZ = 640
    BATCH = 8  # Reducir a 4 o 2 si la GPU asignada se queda sin memoria
    EPOCHS = 100
    PATIENCE = 20
    SEED = 12345
    WORKERS = 2
    OPTIMIZER = "AdamW"
    LR0 = 0.001

    SPLIT_MODE = "auto"  # auto, session o temporal
    GAP_SECONDS = 10.0  # Margen a cada lado del límite temporal entre splits
    GROUP_SPLITS = {}  # Ejemplo: {"sesion_...:t01": "train", ...}; asignar TODOS los grupos

    RUN_PILOT = True  # Cinco épocas con el primer modelo para medir tiempo/memoria
    RUN_TRAINING = True
    RESUME_INTERRUPTED = True
    RUN_FINAL_TEST = False  # Activar cuando hayas fijado modelos y configuración
    DETECTION_CONF = 0.20  # Umbral común para conteo/latencia; ajustar usando val
    """)
    add("markdown", """
    ## 3. Conectar Drive y comprobar GPU

    El entrenamiento lee los PNG desde el disco temporal `/content`, evitando miles
    de lecturas en Drive. Los checkpoints se escriben en Drive al terminar cada época.
    """)
    add("code", """
    from google.colab import drive
    drive.mount("/content/drive")

    import gc
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import tempfile
    import time
    from datetime import datetime, timezone

    runtime_dir = Path("/content/greenlight_runtime")
    runtime_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)

    import numpy as np
    import pandas as pd
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    # Solo logs/artefactos locales y en tu Drive; no activar plataformas de tracking.
    settings.update({key: False for key in ("wandb", "mlflow", "clearml", "comet", "neptune", "hub", "dvc") if key in settings})
    if not torch.cuda.is_available():
        raise RuntimeError("Seleccionar GPU en el tipo de entorno y volver a conectar.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", EXPERIMENT_NAME):
        raise ValueError("EXPERIMENT_NAME: usar letras, números, guiones o guion bajo.")
    if not MODELS or len(MODELS) != len(set(MODELS)) or not set(MODELS) <= {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}:
        raise ValueError("Elegir modelos únicos de YOLO11n/s/m.")
    if BATCH < 1 or EPOCHS < 1 or not 0 < DETECTION_CONF < 1:
        raise ValueError("Batch, épocas o confianza inválidos.")
    gpu = torch.cuda.get_device_properties(0)
    ENVIRONMENT = {"python": sys.version, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": gpu.name, "vram_gib": round(gpu.total_memory / 1024**3, 2)}
    print(json.dumps(ENVIRONMENT, indent=2))
    EXPERIMENT_DIR = Path("/content/drive/MyDrive/GreenLight/experiments") / EXPERIMENT_NAME
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    """)
    add("markdown", """
    ## 4. Funciones de preparación y metadatos

    Este bloque está incluido dentro del notebook: no hace falta clonar el repositorio.
    Valida clase única, cajas, imágenes y correspondencia con los metadatos.
    """)
    add("code", (ROOT / "colab_utils.py").read_text(encoding="utf-8"))
    with (ROOT / "selection_vehicle_500.csv").open(encoding="utf-8", newline="") as source:
        metadata = list(csv.DictReader(source))
    packed = base64.b64encode(zlib.compress(json.dumps(metadata).encode())).decode()
    add("code", f'import base64\nimport zlib\nDEFAULT_METADATA = json.loads(zlib.decompress(base64.b64decode({packed!r})).decode())\nprint(f"Metadatos incluidos: {{len(DEFAULT_METADATA)}} imágenes")')
    add("markdown", """
    ## 5. Cargar el export corregido y separar datos

    La separación es reproducible y se guarda con hashes de imágenes y etiquetas.
    Para comparar 500 contra 1.200 imágenes de entrenamiento, conservar el mismo
    conjunto independiente de validación/prueba mediante grupos explícitos.

    Un TXT ausente se interpreta como imagen negativa, según el formato YOLO.
    Revisá el conteo de etiquetas ausentes para detectar exportaciones incompletas.
    """)
    add("code", """
    LOCAL_ROOT = Path(tempfile.mkdtemp(prefix="greenlight_", dir="/content"))
    def local_copy(path, filename):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"No existe {source}. Subir el archivo a Drive y corregir su ruta en el paso 2.")
        target = LOCAL_ROOT / filename
        shutil.copyfile(source, target)
        return target

    corrected_zip = local_copy(CVAT_ZIP, "cvat_corregido.zip")
    images_zip = local_copy(IMAGES_ZIP, "imagenes.zip") if IMAGES_ZIP else None
    metadata_rows = metadata_from_csv(local_copy(METADATA_CSV, "selection.csv")) if METADATA_CSV else DEFAULT_METADATA
    DATASET_DIR = LOCAL_ROOT / "dataset"
    records = prepare_cvat_dataset(corrected_zip, DATASET_DIR, metadata_rows, images_zip)
    records, group_assignment, used_split_mode = make_grouped_splits(records, SEED, SPLIT_MODE, GAP_SECONDS, GROUP_SPLITS)
    DATA_YAML, split_summary = write_dataset_split(DATASET_DIR, records, group_assignment, used_split_mode, SEED, GAP_SECONDS)

    record_df = pd.DataFrame(records)
    display(record_df.groupby(["split", "camera"]).agg(images=("filename", "count"), vehicles=("boxes", "sum")))
    display(record_df.groupby(["group", "split"]).size().rename("images").reset_index())
    print(json.dumps(split_summary, indent=2))
    if used_split_mode == "temporal":
        print("Evaluación interna por tramos: no demuestra generalización a sesiones nuevas ni equilibrio por clima.")

    TRAIN_CONFIG = {"models": MODELS, "imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS,
                    "patience": PATIENCE, "seed": SEED, "workers": WORKERS, "optimizer": OPTIMIZER,
                    "lr0": LR0, "ultralytics": ultralytics.__version__}
    experiment_meta = {"dataset_fingerprint": split_summary["dataset_fingerprint"], "training": TRAIN_CONFIG}
    metadata_path = EXPERIMENT_DIR / "experiment.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != experiment_meta:
        raise ValueError("Este experimento tiene otros datos o parámetros. Usar otro EXPERIMENT_NAME.")
    metadata_path.write_text(json.dumps(experiment_meta, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    (EXPERIMENT_DIR / f"environment_{stamp}.json").write_text(json.dumps(ENVIRONMENT, indent=2))
    for filename in ("split_manifest.csv", "split_summary.json", "data.yaml", "train.txt", "val.txt", "test.txt"):
        shutil.copyfile(DATASET_DIR / filename, EXPERIMENT_DIR / filename)
    # Copia compacta de TODAS las etiquetas corregidas; imágenes conservadas en tu ZIP de Drive.
    with zipfile.ZipFile(EXPERIMENT_DIR / "corrected_labels.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for label in sorted((DATASET_DIR / "labels/all").glob("*.txt")):
            archive.write(label, label.relative_to(DATASET_DIR).as_posix())
    print("Dataset local:", DATA_YAML)
    print("Resultados/checkpoints:", EXPERIMENT_DIR)
    """)
    add("markdown", """
    ## 6. Inspección visual de las etiquetas corregidas

    Las cajas mostradas aquí vienen del export CVAT. Esta celda no ejecuta YOLO.
    """)
    add("code", """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    sample = record_df[record_df.split != "excluded"].sample(n=min(6, len(record_df[record_df.split != "excluded"])), random_state=SEED)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        image = Image.open(DATASET_DIR / "images/all" / row.filename)
        ax.imshow(image)
        for line in (DATASET_DIR / "labels/all" / (Path(row.filename).stem + ".txt")).read_text().splitlines():
            _, cx, cy, width, height = map(float, line.split())
            ax.add_patch(patches.Rectangle(((cx-width/2)*image.width, (cy-height/2)*image.height), width*image.width, height*image.height, fill=False, edgecolor="lime", linewidth=1))
        ax.set_title(f"{row.camera} · {row.split} · {row.boxes} vehículos")
        ax.axis("off")
    for ax in axes.flat[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    """)
    add("markdown", """
    ## 7. Piloto: medir costo antes del entrenamiento completo

    Usa cinco épocas del primer modelo, el mismo train/val y ningún dato de test.
    La proyección usa el promedio de las épocas posteriores al arranque; no estima
    el costo de modelos de distinto tamaño ni garantiza el tiempo del entrenamiento.
    """)
    add("code", """
    COMMON_TRAIN = dict(data=str(DATA_YAML), imgsz=IMGSZ, batch=BATCH, device=0, workers=WORKERS,
                        seed=SEED, optimizer=OPTIMIZER, lr0=LR0, amp=True, plots=True)
    pilot_summary_path = EXPERIMENT_DIR / "pilot_summary.json"
    if RUN_PILOT and not pilot_summary_path.exists():
        pilot = YOLO(MODELS[0])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        pilot.train(**COMMON_TRAIN, epochs=5, patience=5, close_mosaic=0,
                    project=str(EXPERIMENT_DIR / "pilot"), name=stamp)
        elapsed = time.perf_counter() - started
        epoch_log = pd.read_csv(Path(pilot.trainer.save_dir) / "results.csv")
        epoch_log.columns = epoch_log.columns.str.strip()
        per_epoch = float(epoch_log["time"].diff().dropna().mean())
        pilot_summary = {"model": MODELS[0], "epochs": len(epoch_log), "wall_seconds": elapsed,
                         "mean_epoch_seconds_after_first": per_epoch,
                         "estimated_full_minutes_same_model": per_epoch * EPOCHS / 60,
                         "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}
        pilot_summary_path.write_text(json.dumps(pilot_summary, indent=2))
        del pilot
        gc.collect()
        torch.cuda.empty_cache()
    if pilot_summary_path.exists():
        print(pilot_summary_path.read_text())
    else:
        print("Piloto desactivado.")
    """)
    add("markdown", """
    ## 8. Entrenar o continuar YOLO11n/s/m

    Los modelos se entrenan uno por vez, con los mismos datos y parámetros. `best.pt`
    se selecciona usando validación. Al cortar Colab se conserva lo escrito hasta la
    última época guardada en Drive; lo que estaba calculándose puede perderse.
    Una GPU distinta al reconectar puede cambiar tiempos y algunos resultados.
    """)
    add("code", """
    if RUN_TRAINING:
        for weights in MODELS:
            model_name = Path(weights).stem
            run_dir = EXPERIMENT_DIR / "training" / model_name
            complete = run_dir / "completed.json"
            checkpoint = run_dir / "weights/last.pt"
            if complete.exists():
                print(model_name, "ya completado; se conserva el resultado.")
                continue
            started = time.perf_counter()
            if checkpoint.exists() and RESUME_INTERRUPTED:
                # Ultralytics finaliza/limpia last.pt al completar un entrenamiento.
                # Recuperar el marcador si Colab se cortó entre esa limpieza y nuestra escritura.
                checkpoint_info = torch.load(checkpoint, map_location="cpu", weights_only=False)
                finalized = checkpoint_info.get("epoch", 0) == -1
                del checkpoint_info
                if finalized and (run_dir / "weights/best.pt").is_file():
                    complete.write_text(json.dumps({"model": weights, "dataset_fingerprint": split_summary["dataset_fingerprint"],
                                                    "recovered_finalized_checkpoint": True}, indent=2))
                    print(model_name, "ya finalizado; se recuperó su marcador.")
                    continue
                model = YOLO(str(checkpoint))
                print("Continuando:", checkpoint)
                model.train(resume=True, data=str(DATA_YAML), device=0, workers=WORKERS)
            elif run_dir.exists():
                raise ValueError(f"{run_dir} está incompleto. Activar RESUME_INTERRUPTED o usar otro experimento.")
            else:
                model = YOLO(weights)
                model.train(**COMMON_TRAIN, epochs=EPOCHS, patience=PATIENCE,
                            project=str(EXPERIMENT_DIR / "training"), name=model_name)
            actual_dir = Path(model.trainer.save_dir)
            best_checkpoint = actual_dir / "weights" / "best.pt"
            if (actual_dir.name != model_name or actual_dir.parent.name != "training"
                    or not best_checkpoint.is_file()):
                raise RuntimeError(f"Revisar el destino del entrenamiento y best.pt: {best_checkpoint}")
            run_dir = actual_dir
            complete = run_dir / "completed.json"
            complete.write_text(json.dumps({"model": weights, "dataset_fingerprint": split_summary["dataset_fingerprint"],
                                            "wall_seconds_this_connection": time.perf_counter() - started,
                                            "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print("Entrenamiento desactivado. Activar RUN_TRAINING para generar best.pt.")
    """)
    add("markdown", """
    ## 9. Comparar en validación

    mAP usa `conf=0.001` para construir las curvas de precisión/recall. La precisión
    y recall resumidos por Ultralytics corresponden a su punto de máximo F1.
    El MAE de conteo usa, por separado, `DETECTION_CONF` fijo y cuenta toda la imagen.

    La latencia mide el detector con batch 1, calentamiento y sincronización CUDA.
    Incluye preprocesamiento/inferencia/postprocesamiento; excluye lectura de PNG,
    CARLA, ZMQ y DeepSORT. No representa los FPS del pipeline completo.
    """)
    add("code", """
    def count_and_benchmark(model, split):
        subset = [row for row in records if row["split"] == split]
        output = []
        first = np.asarray(Image.open(DATASET_DIR / "images/all" / subset[0]["filename"]).convert("RGB"))[:, :, ::-1].copy()
        predict_options = dict(imgsz=IMGSZ, conf=DETECTION_CONF, iou=0.7, classes=[0],
                               agnostic_nms=True, rect=False, device=0, verbose=False)
        for _ in range(3):
            model.predict(first, **predict_options)
        for row in subset:
            frame = np.asarray(Image.open(DATASET_DIR / "images/all" / row["filename"]).convert("RGB"))[:, :, ::-1].copy()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = model.predict(frame, **predict_options)[0]
            predicted_count = len(prediction.boxes)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            output.append({"filename": row["filename"], "camera": row["camera"], "group": row["group"],
                           "true_count": row["boxes"], "predicted_count": predicted_count,
                           "absolute_count_error": abs(row["boxes"] - predicted_count), "detector_ms": latency_ms})
        return pd.DataFrame(output)

    def evaluate_models(split):
        comparison = []
        for weights in MODELS:
            name = Path(weights).stem
            run_dir = EXPERIMENT_DIR / "training" / name
            if not (run_dir / "completed.json").is_file():
                raise FileNotFoundError(f"Completar el entrenamiento de {name} antes de evaluar.")
            model = YOLO(str(run_dir / "weights/best.pt"))
            if model.names != {0: "vehicle"}:
                raise ValueError("Esta evaluación necesita el modelo entrenado de una clase vehicle.")
            metrics = model.val(data=str(DATA_YAML), split=split, imgsz=IMGSZ, batch=BATCH,
                                device=0, workers=WORKERS, conf=0.001, iou=0.7, rect=False,
                                plots=True, project=str(EXPERIMENT_DIR / split), name=name, exist_ok=True)
            per_image = count_and_benchmark(model, split)
            per_image.to_csv(EXPERIMENT_DIR / f"{split}_{name}_per_image.csv", index=False)
            camera_report = per_image.groupby("camera").agg(images=("filename", "count"), count_MAE=("absolute_count_error", "mean"))
            camera_report.to_csv(EXPERIMENT_DIR / f"{split}_{name}_by_camera.csv")
            comparison.append({"model": name, "split": split, "images": len(per_image),
                               "precision_max_F1": float(metrics.box.mp), "recall_max_F1": float(metrics.box.mr),
                               "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                               "count_MAE": float(per_image.absolute_count_error.mean()),
                               "detector_ms_median": float(per_image.detector_ms.median()),
                               "detector_ms_p95": float(per_image.detector_ms.quantile(0.95)),
                               "detector_FPS_mean": 1000 / float(per_image.detector_ms.mean()),
                               "count_conf": DETECTION_CONF, "dataset_fingerprint": split_summary["dataset_fingerprint"]})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        table = pd.DataFrame(comparison)
        table.to_csv(EXPERIMENT_DIR / f"{split}_comparison.csv", index=False)
        return table

    if all((EXPERIMENT_DIR / "training" / Path(weights).stem / "completed.json").exists() for weights in MODELS):
        validation_comparison = evaluate_models("val")
        display(validation_comparison)
    else:
        print("Todavía faltan entrenamientos completos; la evaluación no se ejecutó.")
    """)
    add("markdown", """
    ## 10. Prueba final reservada

    Fijá resolución, modelos y umbral usando validación antes de activar
    `RUN_FINAL_TEST=True`. Evitá ajustar parámetros mirando el resultado de test.
    Se comparan los modelos entrenados sobre exactamente el mismo conjunto reservado.

    Para comparar también pesos COCO originales hay que implementar un evaluador
    que filtre car/motorcycle/bus/truck y remapee predicciones a vehicle. No se incluyen
    en esta tabla porque su numeración de clases difiere de la del dataset.
    """)
    add("code", """
    if RUN_FINAL_TEST:
        test_comparison = evaluate_models("test")
        display(test_comparison)
    else:
        print("Test reservado. Activar RUN_FINAL_TEST cuando hayas fijado la configuración y ejecutar esta celda.")
    """)
    add("markdown", """
    ## 11. Descargar modelos y reportes

    Cada modelo está en `training/<modelo>/weights/best.pt`. Los archivos también
    quedan en Drive. El ZIP de entrega incluye mejores pesos, tablas, particiones y
    parámetros; los checkpoints para continuar entrenamiento quedan en Drive.

    Al integrar `best.pt` en GreenLight, adaptar el filtro de clases a `0: vehicle`.
    La evaluación de seguimiento y conteo por ROI en CARLA es un paso posterior.
    """)
    add("code", """
    delivery_zip = EXPERIMENT_DIR / "greenlight_models_and_reports.zip"
    with zipfile.ZipFile(delivery_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for weights in MODELS:
            best = EXPERIMENT_DIR / "training" / Path(weights).stem / "weights/best.pt"
            if best.is_file():
                archive.write(best, best.relative_to(EXPERIMENT_DIR).as_posix())
        reports = list(EXPERIMENT_DIR.glob("*.csv")) + list(EXPERIMENT_DIR.glob("*.json")) + list(EXPERIMENT_DIR.glob("*.yaml")) + list(EXPERIMENT_DIR.glob("*.txt"))
        for report in sorted(reports):
            archive.write(report, report.name)
    print("Entrega:", delivery_zip)
    from google.colab import files
    files.download(str(delivery_zip))
    """)
    add("markdown", """
    Documentación: [Colab](https://research.google.com/colaboratory/faq.html),
    [Entrenamiento](https://docs.ultralytics.com/modes/train/),
    [Evaluación](https://docs.ultralytics.com/modes/val/),
    [Exportación CVAT](https://docs.cvat.ai/docs/dataset_management/export-datasets/).
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "greenlight_yolo_colab.ipynb", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "greenlight_yolo_colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)
    return notebook


if __name__ == "__main__":
    build_notebook()
