# Temporización y benchmarks en GreenLight

Medido el 21/11/2026 en la notebook del proyecto (RTX 4050 Laptop, i5-13420H, 16 GB) con CARLA 0.9.16 y Town04. Las cifras sirven para este equipo; en otro equipo conviene repetir las mediciones con los scripts de [`tools/benchmark_carla/`](../tools/benchmark_carla/).

## Resumen

- **Configurar CARLA así:** `fixed_delta_seconds = 0.075`, `max_substep_delta_time = 0.0125`, `max_substeps = 6`.
- **Procesar visión cada 0,525 s simulados**, no en cada tick. Las cámaras se configuran con `sensor_tick = 0.525`.
- **No simular un día seguido.** Simular bloques de demanda fija: 150 s de calentamiento y 900 s de medición cada uno.
- **Repetir con varias semillas** en lugar de alargar una sola corrida. Con 5 semillas, el ensayo completo entra en una noche (unas 5 h).

---

## 1. Cómo funciona hoy la temporización (rama `main`)

Hay **tres relojes distintos** y no están sincronizados.

| Reloj | Dónde | Cómo avanza | Problema |
|---|---|---|---|
| **Física de CARLA** | `run_simulator.py` | 1 s simulado por tick (`fixed_delta_seconds = 1`) | Los subpasos (3 × 0,025 s) cubren solo 0,075 s de cada segundo. CARLA exige `dt ≤ max_substep_delta_time × max_substeps`. Medido: el flujo cae a ~1/9 del real. |
| **Semáforos** | `run_vision.py` → `TimingAlgorithms` | `time_i += 1` en cada vuelta del bucle de visión | No usa el tiempo de CARLA. La vuelta se cuenta aunque no haya llegado ningún frame, así que la duración real del verde depende de la velocidad del bucle. |
| **Hora del día** | `dynamic_weather.py` | El sol avanza 0,002 rad por segundo simulado | **Un día dura 52 min simulados y cada "hora" dura 131 s.** Esa hora define la demanda (65/30/10 vehículos). |

Flujo de un tick:

1. El simulador ejecuta `world.tick()`, avanza el clima y repone vehículos según la "hora".
2. Envía las 2 imágenes por ZMQ 5555.
3. Espera, sin límite de tiempo, una respuesta en 5557 y aplica los estados A–D.

Del lado de visión:

- Lee sin bloquear, ejecuta YOLO + DeepSORT y cuenta vehículos por ROI.
- Llama a `TimingAlgorithms.update(conteos, time_i)`.
- La máquina de estados recorre A → B → C → D: verde, amarillo de 5 s, todo rojo de 2 s.
- El verde depende del modo: `fixed` = 25 s, `method1` = 20 o 35 s, `pressure` = de 15 a 45 s.

Errores conocidos de `main` que afectan la medición:

- `run_vision.py` pide `"method_fixed"`, pero el modo válido es `"fixed"`. Lanza un error al primer verde.
- Si falla el procesamiento, el `except: pass` mantiene los conteos anteriores sin avisar.
- `prepare_camera_data` lee las cámaras sin bloquear y puede enviar `None`. Medido: 5 frames perdidos cada 1.800 ticks.
- La respuesta de visión no indica a qué tick corresponde.

En la rama `codex/webster`, los planes fijos ya usan el tiempo simulado (`snapshot.timestamp.elapsed_seconds`) con dt = 0,05.

---

## 2. Qué paso de simulación usar

Escenario de prueba: 65 vehículos, 50 peatones y un plan fijo. La tabla muestra cuántos vehículos cruzaron la línea de detención en 10 min simulados, con 2 o 3 semillas.

| dt (s) | Subpasos | Cruces | Velocidad (sin render) | Uso |
|---|---|---|---|---|
| 0,025 | 2 × 0,0125 | 75–96 | 1,8× | Referencia, lenta |
| 0,05 | 5 × 0,01 | 94–104 | 2,6× | Correcto |
| **0,075** | **6 × 0,0125** | **101, 101** | **3,8×** | **Recomendado** |
| 0,1 | 10 × 0,01 | 81, 81, 81 | 4,0× | 19 % menos flujo en todas las semillas |
| 0,2 | 16 × 0,0125 | 47, 60 | 5,8× | Flujo deformado, vehículos volcados |
| 0,5 | 16 × 0,033 | 14, 17 | 16× | Inservible |
| 1,0 (`main`) | 3 × 0,025 | 11, 11 | 60× | Inservible |

"Velocidad" es cuántos segundos simulados se obtienen por cada segundo real.

**Regla:** dt hasta 0,075 s, subpaso de 0,0125 s o menos (CARLA pide menos de 0,0166 y, idealmente, 0,01) y `dt ≤ subpaso × max_substeps`. Si se quiere el subpaso ideal, `0.01 × 8` cuesta unos 3 ms más por tick.

La calibración de Webster (flujo de saturación) se midió con dt = 0,05. Con 0,075 el flujo es el mismo, pero conviene repetir una descarga corta para confirmarlo.

---

## 3. Dónde se va el tiempo real

| Componente | Costo |
|---|---|
| Física + Traffic Manager (dt 0,075, sin render) | ~20 ms por tick |
| Render con las 2 cámaras en **cada** tick | +20 ms por tick |
| Render con las cámaras cada 0,5 s (`sensor_tick`) | +~11 ms por tick |
| Peatones (50) | +~3 ms por tick |
| **YOLO + DeepSORT, 2 imágenes** | **~70 ms** (YOLO ~15 ms + DeepSORT ~19 ms por cámara; n y s cuestan casi lo mismo; se midió con CARLA corriendo en paralelo, así que puede estar algo inflado) |

**La visión es lo más caro.** Si procesa imágenes en cada tick, la simulación queda más lenta que el tiempo real.

Velocidad estimada con dt 0,075. Las filas con visión suman costos medidos por separado; falta confirmarlas con el lazo completo.

| Configuración | Velocidad |
|---|---|
| Visión en cada tick | ~0,8× |
| Visión cada 0,3 s | ~1,5× |
| **Visión cada 0,525 s** | **~1,8×** |
| Plan fijo o Webster, sin visión y sin render | ~3,8× |

`sensor_tick` tiene que ser múltiplo del dt: con dt 0,075, 0,525 equivale a 7 ticks.

---

## 4. ¿Cuánto tiempo simulado es "representativo"?

Lo que importa **no son las horas simuladas sino cuántos vehículos se observan**. La intersección recibe pocos: unos 380 por hora con demanda media y unos 570 con demanda alta. Además, la espera por vehículo varía mucho: en los pilotos Webster la media fue de 28–36 s con un desvío de 32–36 s.

Con esos datos, para estimar la espera media con ±20 % de error (95 % de confianza) hacen falta unos 80–160 vehículos, es decir **unos 15 min simulados**. Para ±10 % hacen falta unos 300–640 vehículos, es decir **unos 50–70 min simulados por condición**.

Por eso el "día" de 52 min de `dynamic_weather.py` **no sirve para medir**:

- Cada franja de demanda dura 131 s, menos que el calentamiento necesario (unos 150 s), así que la cola nunca se estabiliza.
- En 131 s cruzan unos 15–20 vehículos, muy pocos para obtener una media confiable.
- La demanda, la luz y la lluvia cambian a la vez, y no se puede saber qué causó cada diferencia.

**Lo que conviene:** fijar la demanda y el clima durante cada bloque, medir 900 s y juntar varias semillas. Por ejemplo, 5 semillas × 900 s = 75 min simulados por condición, lo que da ±10 %. Además, comparar fijo y adaptativo con la misma semilla reduce todavía más el error de la diferencia.

Si se quiere una cifra "por día", se calcula como **promedio ponderado de los bloques**. Con las horas que hoy asigna el código:

- Alta: 7–9, 11–13 y 16–18 → 6 h.
- Media: resto de 6 a 22 → 10 h.
- Baja: 22 a 6 → 8 h.

Se pondera por la cantidad de vehículos de cada franja.

---

## 5. Plan de ensayo recomendado

Por cada semilla (12345, 12346, …), los mismos bloques para los dos controladores:

| Bloque | Demanda | Clima | Fijo / Webster | Adaptativo |
|---|---|---|---|---|
| 1 | 65 (alta) | Mediodía despejado | ✔ | ✔ |
| 2 | 30 (media) | Mediodía despejado | ✔ | ✔ |
| 3 | 10 (baja) | Mediodía despejado | ✔ | ✔ |
| 4 | 65 | Lluvia | — | ✔ (robustez de la visión) |
| 5 | 65 | Noche | — | ✔ (robustez de la visión) |

El fijo no usa la cámara, así que el clima no lo afecta: los bloques 4 y 5 se comparan contra el bloque 1 del fijo.

Cada bloque dura 150 s de calentamiento + 900 s de medición = 1.050 s simulados.

| Por semilla | Simulado | Real (estimado) |
|---|---|---|
| Adaptativo, 5 bloques | 87,5 min | ~49 min (a 1,8×) |
| Fijo, 3 bloques | 52,5 min | ~14 min (a 3,8×, sin render) |
| **Total** | | **~65 min** |

**5 semillas ≈ 5,5 h reales**, una noche. Con 3 semillas, unas 3,3 h. La demanda baja (10 vehículos) da pocos cruces: si se quiere una precisión similar, hay que alargar ese bloque o tomarlo como indicativo.

---

## 6. Checklist antes de correr

1. **Física:** `fixed_delta_seconds=0.075`, `max_substep_delta_time=0.0125`, `max_substeps=6`, modo síncrono para CARLA y el Traffic Manager.
2. **Un solo reloj:** todo usa `world.get_snapshot().timestamp.elapsed_seconds`. Visión recibe el número de frame y el tiempo simulado, y responde con los mismos datos. Nada de `time_i += 1` ni de `time.time()`.
3. **Cámaras:** `sensor_tick = 0.525`. El simulador espera la respuesta de visión **solo en los ticks con imagen**, con un timeout. Las imágenes se leen con `queue.get(timeout=…)`.
4. **Modo de control:** corregir `"method_fixed"` a `"fixed"`, o usar los planes locales de `codex/webster` (`--fixed-green` / `--webster-plan`).
5. **Bloques fijos:** demanda constante (`--traffic-target`) y clima constante por bloque. Desactivar el sol continuo de `dynamic_weather.py` durante la medición.
6. **Mismas condiciones para los dos controladores:** semilla, calentamiento (150 s), ventana (900 s) y rutas.
7. **Métricas:** usar el recolector por acceso de `metricas` / `codex/webster`, que registra `run.json`, `events.csv` y `summary.csv`.
8. **Visión a 2 Hz:** con `n_init=5`, DeepSORT tarda 2,5 s en confirmar un vehículo, y un auto avanza ~5 m entre imágenes. Validar el conteo a esa frecuencia (bajar `n_init` o contar detecciones sin tracker) antes del ensayo final.
9. **Piloto:** antes de la corrida larga, hacer un bloque corto y verificar la velocidad real obtenida y que el verde dure lo planeado en tiempo simulado.
10. **Fijo sin render:** correr el fijo o Webster con `--no-rendering`. El adaptativo necesita render porque usa las cámaras.

---

## 7. Cómo repetir estas mediciones

Iniciar CARLA (`C:\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen`) y ejecutar desde `tools/benchmark_carla/`:

```powershell
# Costo y fidelidad de un paso (sin cámaras, sin render)
python bench_tick.py --dt 0.075 --substep-dt 0.0125 --max-substeps 6 --cameras 0 --no-render --sim-seconds 600 --out resultados/fid.jsonl
# Con cámaras a 0,525 s
python bench_tick.py --dt 0.075 --substep-dt 0.0125 --max-substeps 6 --sensor-tick 0.525 --sim-seconds 120 --out resultados/cost.jsonl
# Tabla resumen de fidelidad
python summarize.py resultados/fid.jsonl
# Costo de visión (no correr al mismo tiempo que CARLA)
python bench_vision.py --frames <carpeta_sesion_con_cam_1_y_cam_2> --weights <carpeta_con_yolo11n.pt_y_yolo11s.pt>
```

Los resultados en bruto de esta medición están en `tools/benchmark_carla/resultados/`.
