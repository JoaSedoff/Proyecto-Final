"""Benchmark de costo por tick y fidelidad del tránsito en CARLA (GreenLight, Town04).

Reproduce la carga del prototipo (Town04, objetos lejanos ocultos, vehículos con
autopiloto cerca del cruce, peatones, 2 cámaras 800x600, semáforos A-D con un
plan fijo) y mide, para un paso dado:
  - tiempo de pared por tick (world.tick y overhead Python/cámaras)
  - factor de aceleración (segundos simulados / segundos reales)
  - fidelidad: colisiones, cruces en rojo, posición de detención respecto de la
    línea, vehículos volcados y cruces totales de la línea de detención.
"""
import argparse
import json
import math
import queue
import random
import statistics
import sys
import time
from pathlib import Path

import carla

REPO = Path(__file__).resolve().parents[2] / "carla_sim"
sys.path.insert(0, str(REPO))
from core.setup_world import SetupWorld  # noqa: E402

CENTER = carla.Location(x=190.5, y=-239.5, z=0.0)
TARGET1 = carla.Location(x=210, y=-254, z=6.0)
TARGETS = {
    "A": carla.Location(x=207.4, y=-254.7, z=0.0),
    "B": carla.Location(x=209.4, y=-242.4, z=0.0),
    "C": carla.Location(x=197.2, y=-238, z=0.0),
    "D": carla.Location(x=191.5, y=-250, z=0.0),
}
CAMS = [
    (carla.Location(x=210, y=-254, z=6.0), carla.Rotation(pitch=-7.5, yaw=137.5, roll=0.0)),
    (carla.Location(x=195, y=-238, z=6.0), carla.Rotation(pitch=-13.1, yaw=-48.2, roll=0.0)),
]
EXCLUDE = {"vehicle.micro.microlino", "vehicle.tesla.cybertruck", "vehicle.bh.crossbike",
           "vehicle.diamondback.century", "vehicle.gazelle.omafiets", "vehicle.mitsubishi.fusorosa"}
GREEN, YELLOW, ALLRED = 20.0, 3.0, 2.0
ORDER = ["A", "B", "C", "D"]


def plan_state(t):
    cyc = 4 * (GREEN + YELLOW + ALLRED)
    t %= cyc
    idx = int(t // (GREEN + YELLOW + ALLRED))
    r = t - idx * (GREEN + YELLOW + ALLRED)
    states = {k: carla.TrafficLightState.Red for k in ORDER}
    if r < GREEN:
        states[ORDER[idx]] = carla.TrafficLightState.Green
    elif r < GREEN + YELLOW:
        states[ORDER[idx]] = carla.TrafficLightState.Yellow
    # tiempo en rojo "firme" de cada acceso (para cruces en rojo)
    return states


def nearest_tl(world, loc):
    return min(world.get_actors().filter("*.traffic_light"), key=lambda l: l.get_location().distance(loc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--substep-dt", type=float, default=0.01)
    ap.add_argument("--max-substeps", type=int, default=10)
    ap.add_argument("--vehicles", type=int, default=65)
    ap.add_argument("--walkers", type=int, default=50)
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--sensor-tick", type=float, default=0.0, help="0 = una imagen por tick")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--sim-seconds", type=float, default=300.0, help="Segundos simulados medidos")
    ap.add_argument("--warmup", type=float, default=60.0, help="Segundos simulados de calentamiento")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    random.seed(a.seed)
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town04"):
        world = client.load_world("Town04")
        time.sleep(3)
    else:
        world = client.reload_world(False)  # mundo limpio, conserva ajustes
        time.sleep(2)
    SetupWorld.toggle_far_environment_objects(world, CENTER, radius=200.0)

    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = a.dt
    s.substepping = True
    s.max_substep_delta_time = a.substep_dt
    s.max_substeps = a.max_substeps
    s.no_rendering_mode = a.no_render
    world.apply_settings(s)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)
    tm.set_hybrid_physics_mode(False)

    w = world.get_weather()
    w.sun_altitude_angle, w.cloudiness, w.precipitation, w.fog_density = 70.0, 10.0, 0.0, 0.0
    world.set_weather(w)

    lights = {k: nearest_tl(world, loc) for k, loc in TARGETS.items()}
    for tl in lights.values():
        tl.freeze(False)
        for f in (tl.set_red_time, tl.set_yellow_time, tl.set_green_time):
            f(100000)
        tl.set_state(carla.TrafficLightState.Red)
    # Líneas de detención: (pos, forward, right) por carril
    stoplines = {}
    for k, tl in lights.items():
        stoplines[k] = [(wp.transform.location, wp.transform.get_forward_vector(), wp.lane_width)
                        for wp in tl.get_stop_waypoints()]

    bl = world.get_blueprint_library()
    vbps = [b for b in bl.filter("vehicle.*") if b.id not in EXCLUDE and int(b.get_attribute("number_of_wheels")) == 4]
    spawns = [sp for sp in world.get_map().get_spawn_points() if sp.location.distance(TARGET1) < 100.0]
    col_bp = bl.find("sensor.other.collision")

    vehicles, sensors = {}, []
    collisions = set()

    def on_collision(ev, vid):
        other = ev.other_actor.type_id if ev.other_actor else "none"
        oid = ev.other_actor.id if ev.other_actor else -1
        if other.split(".")[0] in ("vehicle", "walker"):
            collisions.add((min(vid, oid), max(vid, oid), other.split(".")[0]))

    def respawn():
        # eliminar lejanos, reponer hasta el objetivo
        for vid, v in list(vehicles.items()):
            if not v.is_alive or v.get_location().distance(TARGET1) > 130.0:
                try:
                    v.destroy()
                except RuntimeError:
                    pass
                vehicles.pop(vid, None)
        need = a.vehicles - len(vehicles)
        if need <= 0:
            return
        locs = [v.get_location() for v in vehicles.values()]
        random.shuffle(spawns)
        for sp in spawns:
            if need <= 0:
                break
            if any(l.distance(sp.location) < 10.0 for l in locs):
                continue
            v = world.try_spawn_actor(random.choice(vbps), sp)
            if v is None:
                continue
            v.set_autopilot(True, 8000)
            tm.vehicle_lane_offset(v, 0.3)
            cs = world.spawn_actor(col_bp, carla.Transform(), attach_to=v)
            cs.listen(lambda ev, vid=v.id: on_collision(ev, vid))
            sensors.append(cs)
            vehicles[v.id] = v
            locs.append(sp.location)
            need -= 1

    walkers = []
    if a.walkers:
        wbps = bl.filter("walker.pedestrian.*")
        batch = []
        for _ in range(a.walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                batch.append(carla.command.SpawnActor(random.choice(wbps), carla.Transform(loc)))
        res = client.apply_batch_sync(batch, True)
        ids = [r.actor_id for r in res if not r.error]
        cbp = bl.find("controller.ai.walker")
        res = client.apply_batch_sync([carla.command.SpawnActor(cbp, carla.Transform(), i) for i in ids], True)
        world.tick()
        for r in res:
            if not r.error:
                c = world.get_actor(r.actor_id)
                c.start()
                c.go_to_location(world.get_random_location_from_navigation())
                c.set_max_speed(1 + random.random())
                walkers.append(c)
        walkers += [world.get_actor(i) for i in ids]

    cams, qs = [], []
    for loc, rot in CAMS[: a.cameras]:
        cbp = bl.find("sensor.camera.rgb")
        cbp.set_attribute("image_size_x", "800")
        cbp.set_attribute("image_size_y", "600")
        cbp.set_attribute("sensor_tick", str(a.sensor_tick))
        cam = world.spawn_actor(cbp, carla.Transform(loc, rot))
        q = queue.Queue()
        cam.listen(q.put)
        cams.append(cam)
        qs.append(q)

    respawn()
    world.tick()
    t0 = world.get_snapshot().timestamp.elapsed_seconds

    tick_w, other_w, cam_w = [], [], []
    frames_got, frames_missed_nb = 0, 0
    prev_s = {}            # (vid) -> (approach, lane_idx, s)
    red_since = {k: 0.0 for k in ORDER}
    last_state = {k: None for k in ORDER}
    crossings, red_runs, stop_pos, flipped = 0, 0, [], set()
    speeds = []
    measuring = False
    wall_start = None
    sim_start = None
    ticks = 0

    while True:
        ts = time.perf_counter()
        frame = world.tick()
        tw = time.perf_counter() - ts
        snap = world.get_snapshot()
        sim_t = snap.timestamp.elapsed_seconds - t0

        if not measuring and sim_t >= a.warmup:
            measuring = True
            wall_start = time.perf_counter()
            sim_start = sim_t
            collisions.clear()
        if measuring and sim_t - sim_start >= a.sim_seconds:
            break

        to = time.perf_counter()
        # cámaras: como el prototipo, leer sin bloquear; luego completar con bloqueo para no perder
        tc = time.perf_counter()
        for q in qs:
            try:
                img = q.get(block=False)
                hit = True
            except queue.Empty:
                hit = False
                if a.sensor_tick == 0.0:
                    try:
                        img = q.get(timeout=2.0)
                    except queue.Empty:
                        img = None
                else:
                    img = None
            if img is not None:
                _ = bytes(img.raw_data)  # copia como hace prepare_camera_data
                if measuring:
                    frames_got += 1
            if measuring and not hit and a.sensor_tick == 0.0:
                frames_missed_nb += 1
        cw = time.perf_counter() - tc

        states = plan_state(sim_t)
        for k, tl in lights.items():
            st = states[k]
            if st != last_state[k]:
                tl.set_state(st)
                if st == carla.TrafficLightState.Red:
                    red_since[k] = sim_t
                last_state[k] = st

        # fidelidad
        for vid in list(vehicles):
            av = snap.find(vid)
            if av is None:
                continue
            tr = av.get_transform()
            loc = tr.location
            if abs(tr.rotation.roll) > 30 or abs(tr.rotation.pitch) > 30:
                flipped.add(vid)
            vel = av.get_velocity()
            sp = math.sqrt(vel.x ** 2 + vel.y ** 2)
            if measuring and loc.distance(CENTER) < 60:
                speeds.append(sp)
            best = None
            for k, lines in stoplines.items():
                for i, (p, f, lw) in enumerate(lines):
                    dx, dy = loc.x - p.x, loc.y - p.y
                    lon = dx * f.x + dy * f.y
                    lat = -dx * f.y + dy * f.x
                    if abs(lat) < lw / 2 and -25 < lon < 8:
                        best = (k, i, lon)
                        break
                if best:
                    break
            if best is None:
                prev_s.pop(vid, None)
                continue
            k, i, lon = best
            prv = prev_s.get(vid)
            if measuring and prv and prv[0] == k and prv[1] == i:
                if prv[2] < 0 <= lon:
                    crossings += 1
                    if last_state[k] == carla.TrafficLightState.Red and sim_t - red_since[k] > 1.0:
                        red_runs += 1
            if measuring and sp < 0.05 and last_state[k] == carla.TrafficLightState.Red and -8 < lon < 8:
                if not prv or abs(prv[2] - lon) > 1e-3:
                    stop_pos.append(lon)
            prev_s[vid] = (k, i, lon)

        respawn()
        ow = time.perf_counter() - to
        if measuring:
            tick_w.append(tw)
            other_w.append(ow)
            cam_w.append(cw)
            ticks += 1

    wall = time.perf_counter() - wall_start
    sim = sim_t - sim_start

    def pct(x, p):
        x = sorted(x)
        return x[min(len(x) - 1, int(p * len(x)))] if x else None

    veh_coll = {c for c in collisions if c[1] == "vehicle"}
    res = {
        "label": a.label, "dt": a.dt, "substep_dt": a.substep_dt, "max_substeps": a.max_substeps,
        "covers": a.substep_dt * a.max_substeps, "vehicles": a.vehicles, "walkers": a.walkers,
        "cameras": a.cameras, "sensor_tick": a.sensor_tick, "no_render": a.no_render,
        "sim_s": round(sim, 2), "wall_s": round(wall, 2), "ticks": ticks,
        "speedup": round(sim / wall, 3),
        "ms_tick_mean": round(1000 * wall / ticks, 2),
        "ms_world_tick_med": round(1000 * statistics.median(tick_w), 2),
        "ms_world_tick_p95": round(1000 * pct(tick_w, 0.95), 2),
        "ms_python_med": round(1000 * statistics.median(other_w), 2),
        "ms_cam_med": round(1000 * statistics.median(cam_w), 2),
        "frames": frames_got, "frames_late_nonblocking": frames_missed_nb,
        "collisions_any": len(collisions), "collisions_vehicle": len(veh_coll),
        "crossings": crossings, "red_runs": red_runs,
        "stop_pos_med": round(statistics.median(stop_pos), 2) if stop_pos else None,
        "stop_pos_p95": round(pct(stop_pos, 0.95), 2) if stop_pos else None,
        "stop_past_line": sum(1 for x in stop_pos if x > 0.0),
        "stop_samples": len(stop_pos),
        "flipped": len(flipped),
        "mean_speed_mps": round(statistics.mean(speeds), 2) if speeds else None,
    }
    print(json.dumps(res))
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")

    # limpieza
    s = world.get_settings()
    for c in cams + sensors:
        try:
            c.stop()
        except RuntimeError:
            pass
    client.apply_batch([carla.command.DestroyActor(x.id) for x in cams + sensors + list(vehicles.values()) + walkers if x])
    world.tick()
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    s.no_rendering_mode = False
    world.apply_settings(s)
    tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
