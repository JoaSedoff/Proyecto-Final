"""Captura frames de GreenLight y controla los semáforos sin ejecutar visión."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

import cv2
import numpy as np
import zmq


CAMERAS = {"image1": "cam_1", "image2": "cam_2"}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "frames"
CONTROL_REPEAT_SECONDS = 0.1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_interval(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("El intervalo debe ser positivo y finito.")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("La cantidad no puede ser negativa.")
    return number


class FixedTrafficController:
    """Ciclo A/B/C/D con reloj simulado; inicialmente todos los semáforos en rojo."""

    SEQUENCE = ("A", "B", "C", "D")

    def __init__(self, green_time=25.0, yellow_time=5.0, all_red_time=2.0):
        durations = (green_time, yellow_time, all_red_time)
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("Los tiempos de semáforo deben ser positivos y finitos.")
        self.green_time, self.yellow_time, self.all_red_time = durations
        self.slot_time = sum(durations)
        if not math.isfinite(self.slot_time * len(self.SEQUENCE)):
            raise ValueError("La duración total del ciclo es demasiado grande.")
        self.start_timestamp = None
        self.last_timestamp = None

    def update(self, package):
        timestamps = []
        if isinstance(package, dict):
            for key in CAMERAS:
                try:
                    timestamp = float(package[key]["metadata"]["timestamp"])
                    if math.isfinite(timestamp) and timestamp >= 0:
                        timestamps.append(timestamp)
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
        if timestamps:
            timestamp = max(timestamps)
            if self.start_timestamp is None:
                self.start_timestamp = timestamp
                self.last_timestamp = timestamp
            else:
                # Una cámara atrasada no hace retroceder el reloj del controlador.
                self.last_timestamp = max(self.last_timestamp, timestamp)

    def message(self, stop=False):
        states = dict.fromkeys(self.SEQUENCE, "RED")
        if self.start_timestamp is not None and not stop:
            elapsed = self.last_timestamp - self.start_timestamp
            position = elapsed % (len(self.SEQUENCE) * self.slot_time)
            active_index = int(position // self.slot_time)
            phase_time = position - active_index * self.slot_time
            active = self.SEQUENCE[active_index]
            if phase_time < self.green_time:
                states[active] = "GREEN"
            elif phase_time < self.green_time + self.yellow_time:
                states[active] = "YELLOW"
        return {"states": states, "simulation_timestamp": self.last_timestamp}

    def metadata(self, endpoint):
        return {
            "algorithm": "fixed",
            "sequence": list(self.SEQUENCE),
            "green_sim_seconds": self.green_time,
            "yellow_sim_seconds": self.yellow_time,
            "all_red_sim_seconds": self.all_red_time,
            "clock": "Maximum received camera timestamp; never moves backwards.",
            "start_simulation_timestamp": self.start_timestamp,
            "last_simulation_timestamp": self.last_timestamp,
            "endpoint": endpoint,
            "repeat_wall_seconds": CONTROL_REPEAT_SECONDS,
            "shutdown_command": "All RED (best effort).",
        }


class FrameRecorder:
    """Muestrea cada cámara de forma independiente usando su timestamp de CARLA."""

    def __init__(self, output, session_name, interval, endpoint, notes="", max_frames=0):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", session_name):
            raise ValueError("La sesión debe tener 1-80 letras ASCII, números, '_' o '-'.")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("El intervalo debe ser positivo y finito.")
        self.interval = interval
        self.max_frames = max_frames
        self.session_dir = Path(output).resolve() / session_name
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.counts = dict.fromkeys(CAMERAS.values(), 0)
        self.last_seen = {}
        self.last_saved = {}
        self.warnings = 0
        self.metadata = {
            "schema_version": 1,
            "session": session_name,
            "started_utc": utc_now(),
            "ended_utc": None,
            "status": "recording",
            "endpoint": endpoint,
            "interval_sim_seconds": interval,
            "max_frames_per_camera": max_frames,
            "notes": notes,
            "image_format": "PNG",
            "input_pixel_format": "BGRA",
            "sampling": "First received frame, then at least interval seconds since last saved frame, per camera.",
            "counts": self.counts,
        }
        for camera in CAMERAS.values():
            (self.session_dir / camera).mkdir()
        self.write_metadata()
        self.manifest = (self.session_dir / "frames.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(self.manifest, fieldnames=[
            "image", "camera", "carla_frame", "simulation_timestamp",
            "width", "height", "saved_utc",
        ])
        self.writer.writeheader()
        self.manifest.flush()

    def write_metadata(self):
        (self.session_dir / "session.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def warn(self, message):
        self.warnings += 1
        if self.warnings <= 5:
            print(f"Aviso: {message}", file=sys.stderr, flush=True)
        elif self.warnings == 6:
            print("Aviso: se omiten avisos adicionales; el total quedará en session.json.",
                  file=sys.stderr, flush=True)

    def record_package(self, package):
        if not isinstance(package, dict):
            self.warn("Paquete inválido: se esperaba un diccionario.")
            return []
        saved = []
        for key, camera in CAMERAS.items():
            if self.max_frames and self.counts[camera] >= self.max_frames:
                continue
            data = package.get(key)
            if data is None:
                continue
            try:
                image = data["image"]
                info = data["metadata"]
                frame = info["frame"]
                timestamp = float(info["timestamp"])
                width, height = info["width"], info["height"]
                if not isinstance(frame, (int, np.integer)) or frame < 0:
                    raise ValueError("Número de frame inválido.")
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError("Timestamp inválido.")
                if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                        or image.ndim != 3 or image.shape[2] != 4
                        or image.shape[:2] != (height, width)
                        or height <= 0 or width <= 0):
                    raise ValueError("La imagen debe ser BGRA uint8 y coincidir con sus dimensiones.")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self.warn(f"{camera}: se omite una imagen inválida ({exc}).")
                continue

            previous = self.last_seen.get(camera)
            if previous is not None:
                if frame < previous[0] or timestamp < previous[1]:
                    raise RuntimeError(
                        f"El frame o timestamp retrocedió en {camera}. "
                        "Se detiene la captura para no mezclar episodios. "
                        "Iniciá otra sesión si reiniciaste CARLA."
                    )
                if frame == previous[0]:
                    continue
            self.last_seen[camera] = (frame, timestamp)
            if (camera in self.last_saved
                    and timestamp - self.last_saved[camera] + 1e-9 < self.interval):
                continue

            # CARLA entrega BGRA. OpenCV codifica BGR: solo se elimina el canal alfa.
            ok, encoded = cv2.imencode(".png", np.ascontiguousarray(image[:, :, :3]))
            if not ok:
                raise OSError(f"No se pudo codificar el frame {frame} de {camera}.")
            relative_path = Path(camera) / f"frame_{frame:010d}.png"
            # 'xb' evita sobrescribir una captura existente, incluso por error.
            with (self.session_dir / relative_path).open("xb") as image_file:
                image_file.write(encoded.tobytes())
            self.writer.writerow({
                "image": relative_path.as_posix(),
                "camera": camera,
                "carla_frame": int(frame),
                "simulation_timestamp": timestamp,
                "width": width,
                "height": height,
                "saved_utc": utc_now(),
            })
            self.manifest.flush()
            self.last_saved[camera] = timestamp
            self.counts[camera] += 1
            saved.append((camera, int(frame), timestamp))
        return saved

    @property
    def complete(self):
        return bool(self.max_frames) and all(
            count >= self.max_frames for count in self.counts.values()
        )

    def close(self, status, error=None):
        self.manifest.close()
        self.metadata.update({
            "status": status,
            "ended_utc": utc_now(),
            "invalid_data_warnings": self.warnings,
            "last_saved_simulation_timestamp": self.last_saved,
        })
        if error:
            self.metadata["error"] = error
        self.write_metadata()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://localhost:5555",
                        help="Publicador de imágenes de GreenLight.")
    parser.add_argument("--control-endpoint", default="tcp://*:5557",
                        help="Dirección donde publicar estados de semáforos (tcp://*:5557).")
    parser.add_argument("--green-time", type=positive_interval, default=25.0,
                        help="Verde por semáforo en segundos simulados (25).")
    parser.add_argument("--yellow-time", type=positive_interval, default=5.0,
                        help="Amarillo en segundos simulados (5).")
    parser.add_argument("--all-red-time", type=positive_interval, default=2.0,
                        help="Todos en rojo entre semáforos, en segundos simulados (2).")
    parser.add_argument("--interval", type=positive_interval, default=1.0,
                        help="Intervalo mínimo entre PNG por cámara, en segundos simulados (1).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Carpeta de sesiones (por defecto, frames junto al script).")
    parser.add_argument("--session", help="Nombre único de sesión; se genera si se omite.")
    parser.add_argument("--notes", default="",
                        help="Descripción manual: clima, tráfico, semillas, etc.")
    parser.add_argument("--max-frames", type=nonnegative_integer, default=0,
                        help="Límite POR CÁMARA; 0 = hasta Ctrl+C. Termina al completar ambas.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    session_name = args.session or (
        datetime.now(timezone.utc).strftime("sesion_%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )
    recorder = None
    context = None
    socket = None
    control_socket = None
    controller = None
    control_endpoint = args.control_endpoint
    status, error, exit_code = "stopped", None, 0
    try:
        controller = FixedTrafficController(args.green_time, args.yellow_time, args.all_red_time)
        context = zmq.Context()
        control_socket = context.socket(zmq.PUB)
        control_socket.setsockopt(zmq.LINGER, 200)
        try:
            control_socket.bind(args.control_endpoint)
        except zmq.ZMQError as exc:
            raise RuntimeError(
                f"No se pudo abrir el control en {args.control_endpoint}: {exc}. "
                "Cerrá run_vision.py u otro controlador que use ese puerto."
            ) from exc
        control_endpoint = control_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        # Se guardan candidatos espaciados, no una grabación exhaustiva del stream.
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(args.endpoint)
        recorder = FrameRecorder(args.output, session_name, args.interval,
                                 args.endpoint, args.notes, args.max_frames)
        recorder.metadata["schema_version"] = 2
        recorder.metadata["traffic_control"] = controller.metadata(control_endpoint)
        recorder.write_metadata()
        print(f"Destino: {recorder.session_dir}", flush=True)
        print(f"Escuchando {args.endpoint}. Intervalo: {args.interval:g} s simulados por cámara.",
              flush=True)
        print(f"Control fijo en {control_endpoint}: A/B/C/D, "
              f"verde {args.green_time:g}, amarillo {args.yellow_time:g}, "
              f"todos en rojo {args.all_red_time:g} s simulados.", flush=True)
        print("Mantené run_simulator.py activo y run_vision.py cerrado. Ctrl+C para terminar.",
              flush=True)
        next_control_send = 0.0
        while True:
            if socket.poll(50, zmq.POLLIN):
                package = socket.recv_pyobj()
                saved = recorder.record_package(package)
                controller.update(package)
                # Responder a TODOS los paquetes, aunque no corresponda guardar PNG.
                control_socket.send_pyobj(controller.message())
                next_control_send = time.monotonic() + CONTROL_REPEAT_SECONDS
                for camera, frame, timestamp in saved:
                    print(f"{camera}: frame {frame}, t={timestamp:.3f} s "
                          f"(guardados: {recorder.counts[camera]})", flush=True)
            if time.monotonic() >= next_control_send:
                # PUB/SUB puede perder el primer mensaje. También desbloquea el
                # primer tick si aún no hay imágenes o el capturador arrancó tarde.
                control_socket.send_pyobj(controller.message())
                next_control_send = time.monotonic() + CONTROL_REPEAT_SECONDS
            if recorder.complete:
                status = "completed"
                break
    except KeyboardInterrupt:
        print("\nCaptura detenida.", flush=True)
    except Exception as exc:
        status, error, exit_code = "error", str(exc), 1
        print(f"Error: {exc}", file=sys.stderr, flush=True)
    finally:
        if control_socket is not None:
            try:
                if recorder is not None:
                    control_socket.send_pyobj(controller.message(stop=True))
            except zmq.ZMQError as exc:
                print(f"Aviso: no se pudo enviar el rojo final ({exc}).", file=sys.stderr)
            finally:
                control_socket.close()
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()
        if recorder is not None:
            recorder.metadata["traffic_control"] = controller.metadata(control_endpoint)
            recorder.close(status, error)
            print(f"Total: {recorder.counts}. Carpeta: {recorder.session_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
