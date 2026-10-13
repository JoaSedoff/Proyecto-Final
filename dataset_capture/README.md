# Captura de frames para el dataset de GreenLight

Este capturador guarda imágenes originales de las dos cámaras de Town04 para
seleccionarlas y etiquetarlas manualmente. Escucha el publicador de GreenLight
en `tcp://localhost:5555` y publica estados de semáforos en el puerto `5557`.
Controla un ciclo fijo sin importar CARLA, cargar YOLO o DeepSORT ni abrir ventanas.
Usa `opencv-python`, `numpy` y `pyzmq`, ya incluidos en `requirements.txt`.

## Ejecución

Desde la raíz del proyecto y con el entorno Python habitual:

1. Abrir CARLA.
2. En una terminal: `python .\carla_sim\run_simulator.py`.
3. En otra terminal, iniciar el capturador:

```powershell
python .\dataset_capture\capture_frames.py
```

La captura comienza al recibir las imágenes. También se puede iniciar el
capturador antes del simulador para que quede esperando.
Mantener `run_simulator.py` activo. **No ejecutar `run_vision.py` durante este
pipeline**: el capturador ocupa su puerto de control y envía los estados que el
simulador necesita recibir para avanzar. Si el puerto está ocupado, el capturador
termina con un error antes de crear una sesión.

```text
CARLA + run_simulator.py --imágenes :5555--> capture_frames.py --> PNG
                        <--estados :5557--
```

Para terminar el capturador, presionar **Ctrl+C** en su terminal; los PNG y el
CSV se guardan a medida que llegan, y al cerrar se actualiza `session.json`.
Al terminar (también por `--max-frames`), intenta enviar todos los semáforos en
rojo y cierra el control. Esa última entrega no está garantizada por PUB/SUB.
El simulador puede avanzar hasta quedar esperando otro mensaje; detenerlo con
Ctrl+C en su terminal. Para continuar capturando, iniciar una sesión nueva.

Ejemplo con nombre de sesión y límite de 150 imágenes por cámara:

```powershell
python .\dataset_capture\capture_frames.py --session noche_trafico_medio_01 --interval 1 --max-frames 150 --notes "Noche; tráfico medio; semillas actuales 12345"
```

`--notes` es una descripción escrita por ustedes: no cambia ni consulta el
clima, la cantidad de vehículos o las semillas del simulador.

## Control fijo de semáforos

La secuencia es **A → B → C → D → A**, con un único semáforo activo cada vez.
Por defecto, cada turno tiene **25 segundos de verde, 5 de amarillo y 2 con todos
en rojo**, medidos en tiempo simulado. Los 25 segundos se aplican también al
primer verde; no se copia la excepción de 20 segundos iniciales del controlador
actual de visión, ni su uso del reloj de Windows.

Los tiempos se pueden configurar:

```powershell
python .\dataset_capture\capture_frames.py --green-time 30 --yellow-time 5 --all-red-time 2
```

El ciclo empieza con A en verde al recibir el primer timestamp válido de las
cámaras. Antes de eso, envía todos en rojo. Usa el mayor timestamp disponible de
cada paquete y conserva un reloj que nunca retrocede por una cámara atrasada.
Sin timestamps válidos mantiene la última fase. Si hay saltos de tiempo, calcula
la fase correspondiente al tiempo recibido; no reconstruye ticks ausentes.

Responde a **cada paquete recibido**, incluso cuando no se guarda ninguna imagen.
Además repite el último estado aproximadamente cada 0,1 segundos reales para
permitir la conexión inicial, desbloquear un paquete sin imágenes y recuperarse
de pérdidas de mensajes. Ese reloj real solo programa los envíos: el ciclo de
semáforos depende de los timestamps de CARLA.

Los estados se envían como `{"states": {"A": "GREEN", ...},
"simulation_timestamp": ...}`; el simulador usa el campo `states`. El protocolo
actual no confirma qué frame recibió cada comando, por lo que el timestamp del
mensaje identifica el reloj del controlador, no una confirmación de aplicación.

Por defecto se escucha `--endpoint tcp://localhost:5555` y se publica mediante
`--control-endpoint tcp://*:5557`. Cambiar el puerto de control solo sirve si el
simulador ya está configurado para escuchar ese otro puerto; el script no modifica
esa configuración. `run_simulator.py` y `run_vision.py` se mantienen sin cambios.

## Archivos generados

```text
dataset_capture/
  capture_frames.py
  README.md
  frames/
    sesion_20261010_120000_a1b2c3/
      session.json
      frames.csv
      cam_1/
        frame_0000000842.png
      cam_2/
        frame_0000000843.png
```

- Cada ejecución crea una sesión nueva. Si el nombre ya existe, termina con un
  error para evitar mezclar o sobrescribir datos.
- Cada PNG conserva la resolución original y no contiene cajas, ROI, IDs ni
  textos de las ventanas de visión. La imagen BGRA se convierte a BGR eliminando
  únicamente el canal alfa, sin intercambiar los colores.
- `frames.csv` registra ruta relativa, cámara, frame CARLA, timestamp simulado,
  dimensiones y fecha UTC de guardado por imagen.
- `session.json` registra parámetros, notas, inicio y fin en UTC, conteos y estado
  final. También registra los tiempos, secuencia, puerto y timestamps inicial y
  final del control fijo. El mapa, el clima y las semillas no se verifican
  automáticamente.
- Las sesiones dentro de `frames/` se excluyen de Git; respaldarlas por separado.
  Si se usa otro destino con `--output`, hay que gestionar su respaldo y su
  exclusión de Git si corresponde.

## Muestreo y selección

Por defecto se guarda el primer frame recibido de cada cámara y luego otro
cuando su timestamp haya avanzado **al menos un segundo simulado** desde el
último PNG guardado. No se usan los FPS de las ventanas ni el reloj de Windows.
Se puede modificar con `--interval 0.5` o `--interval 2`.

Las cámaras se muestrean por separado: no se supone que sus imágenes correspondan
al mismo instante. Una imagen ausente se omite, y los frames repetidos no se
guardan dos veces. Si retrocede el frame o el timestamp de una cámara, la captura
se detiene; iniciar otra sesión después de reiniciar el simulador.

El suscriptor prioriza el paquete más reciente (`CONFLATE`). Puede perder paquetes
si el disco o el proceso son lentos: este flujo obtiene candidatos espaciados,
no garantiza una grabación completa ni intervalos exactos. `--max-frames` es
un límite por cámara; la ejecución termina cuando ambas lo alcanzan. Si una
cámara deja de enviar datos, terminar manualmente con Ctrl+C.

Después de capturar, revisar los candidatos antes de etiquetar: reducir imágenes
de colas casi idénticas y conservar cambios de vehículos, posición, tamaño,
oclusión, iluminación y escenas vacías. El script no aplica selección basada en
detecciones de YOLO ni elimina automáticamente escenas similares.

Para obtener variedad, capturar varias sesiones con condiciones y semillas
distintas. Actualmente las semillas de Python y Traffic Manager están fijadas
en `12345`; este capturador no las modifica. Conservar ambas cámaras de una misma
sesión juntas al separar datos de ajuste y evaluación.

## Preparar una selección preanotada para CVAT

`prepare_cvat.py` copia una selección distribuida por toda una sesión y genera
propuestas de cajas con el modelo local `yolo11m.pt`. No necesita CARLA activo y
no modifica los PNG originales ni el detector usado por `run_vision.py`.

```powershell
python .\dataset_capture\prepare_cvat.py --session .\dataset_capture\frames\sesion_20261011_153055_525f31 --output .\dataset_capture\prepared\vehicle_500_sesion_20261011 --count 500
```

Con esta sesión y los valores predeterminados selecciona **500 imágenes**:
250 de cada cámara, con 25 de cada uno de diez tramos de tiempo simulado por
cámara. Dentro de cada tramo divide las imágenes en subtramos y selecciona una
por subtramo con semilla reproducible; incluye los extremos de la sesión y mezcla
el orden de revisión. No toma las primeras 500 imágenes ni selecciona en función
de las detecciones. El clima no está registrado en `frames.csv`: la cobertura
temporal conserva distintas etapas, pero no garantiza cuotas por clima.

La preanotación usa `conf=0.20`, `imgsz=800`, NMS independiente de clase y GPU si
está disponible. Las clases COCO `car`, `motorcycle`, `bus` y `truck` se convierten
en **`0: vehicle`**. No incluye peatones ni bicicletas. Los TXT contienen solo las
cinco columnas YOLO; la clase original y confianza quedan en `predictions.jsonl`.

El destino debe ser nuevo; si ya existe, el script se detiene para proteger
posibles revisiones manuales. `--count`, `--bins`, `--seed`, `--conf`, `--imgsz`,
`--batch`, `--model` y `--device` permiten ajustar otra ejecución. Si un tramo no
contiene suficientes imágenes para su cuota, se detiene en lugar de concentrar
la selección en otra parte de la sesión.

El resultado incluye:

- `greenlight_vehicle_cvat.zip`: imágenes sin dibujos, etiquetas, `data.yaml` y
  `train.txt`, listo para importar como dataset en un proyecto CVAT.
- `selection.csv`: relación con el archivo original, cámara, frame, timestamp,
  tramo, hash SHA256 y cantidad de cajas propuestas.
- `summary.json`: parámetros, conteos y estado pendiente de revisión manual.
- `preview.jpg`: una muestra por cámara y tramo con cajas, excluida del ZIP.
- `IMPORTAR_EN_CVAT.md`: pasos concretos de importación y revisión.

En CVAT, crear un proyecto con la etiqueta `vehicle` de tipo Rectangle, elegir
**Actions → Import dataset → Ultralytics YOLO Detection 1.0** y cargar el ZIP.
CVAT crea una tarea con imágenes y propuestas. Revisar vehículos omitidos,
falsos positivos y límites de cajas, guardar y exportar el dataset corregido.
Un TXT vacío significa que YOLO no propuso cajas, no una escena vacía confirmada.

`train` es el nombre del subset de importación: todavía no se separó entrenamiento
y evaluación. Hacer esa separación después de revisar, manteniendo sesiones
completas juntas. `prepared/` se excluye de Git y requiere respaldo separado.

Formato e importación:
[Ultralytics YOLO en CVAT](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
y [Importar datasets](https://docs.cvat.ai/docs/dataset_management/import-datasets/).

Pruebas de selección, normalización y estructura del ZIP:

```powershell
python -m unittest dataset_capture.tests.test_prepare_cvat -v
```

## Ampliar al total de imágenes conservando las correcciones de CVAT

`extend_cvat.py` incluye **todas las imágenes de `frames.csv`**, conserva los
nombres y las etiquetas del export corregido de la selección anterior y ejecuta
YOLO11m únicamente sobre las imágenes nuevas. No requiere CARLA activo.

```powershell
python -m dataset_capture.extend_cvat --session .\dataset_capture\frames\sesion_20261011_153055_525f31 --previous-selection .\dataset_capture\prepared\vehicle_500_sesion_20261011\selection.csv --corrected-zip "C:\ruta\CARLA_dataset.zip" --output .\dataset_capture\prepared\vehicle_1778_sesion_20261011
```

Para la sesión actual genera:

- `greenlight_vehicle_cvat.zip`: 1.778 imágenes (889 por cámara); las primeras
  500 conservan sus anotaciones de CVAT y las otras 1.278 reciben propuestas YOLO.
- `greenlight_vehicle_1278_pendientes_cvat.zip`: solamente las 1.278 imágenes nuevas,
  sin repetir ninguna imagen de la tarea de 500.
- `selection.csv`: metadatos de las 1.778 imágenes para usar luego en Colab.
- `selection_pending.csv`, `summary.json`, `predictions.jsonl`, `preview.jpg`
  y `IMPORTAR_EN_CVAT.md`: metadatos del adicional, procedencia, auditoría y pasos.

Elegir una de estas dos formas de importación:

1. Importar el ZIP completo en **un proyecto nuevo** con la clase `vehicle`.
   Revisar las imágenes nuevas; los nombres 0001–0500 corresponden al export de CVAT.
2. Importar solo el ZIP adicional en el proyecto que ya contiene las 500 corregidas.
   CVAT crea otra tarea; al terminar, exportar el **proyecto completo** para reunir
   las 1.778 imágenes. No importar ambos ZIP en ese proyecto para evitar duplicados.

Se validan la clase, las cajas, el listado de imágenes del export y los hashes
de las imágenes originales ya revisadas. Si el export lista una imagen pero
omite su TXT, se conserva con etiqueta vacía; no se repone una detección que pudo
haber sido borrada en CVAT. Las imágenes originales y la selección anterior no
se modifican. Las imágenes nuevas se mezclan con semilla reproducible; se incluyen
todas las etapas de captura, sin descartar imágenes por clima ni por detecciones.

Para usar el conjunto completo en el **Colab que ya tenés abierto**, subir su export
corregido y este `selection.csv` a Drive. En la celda de configuración, indicar
las nuevas rutas `CVAT_ZIP` y `METADATA_CSV`, y usar otro `EXPERIMENT_NAME`, por ejemplo
`vehicle_1778_v1`. Los metadatos incrustados en el notebook solo corresponden a las
500 imágenes originales. `IMPORTAR_EN_CVAT.md` incluye un ejemplo de las líneas
que hay que cambiar. Si el export contiene solo etiquetas, configurar también
`IMAGES_ZIP` con el ZIP completo de esta ampliación.

## Entrenamiento y evaluación en Google Colab

El notebook autónomo [`colab/greenlight_yolo_colab.ipynb`](colab/greenlight_yolo_colab.ipynb)
carga un export corregido de CVAT, prepara particiones agrupadas, entrena
YOLO11n/s/m y compara detección, conteo y latencia. Guarda checkpoints en Drive
y permite continuar una sesión interrumpida. Incluye los metadatos de las 500
imágenes seleccionadas; para nuevas selecciones usa su `selection.csv`.

Ver [instrucciones de Colab](colab/README.md). La prueba final queda reservada
hasta fijar la configuración mediante validación. No se entrena sobre las
preanotaciones originales cuando se proporciona el ZIP corregido.

## Verificación del capturador

```powershell
python -m unittest discover -s .\dataset_capture\tests -v
```

Las pruebas verifican las fases con tiempo simulado y usan un simulador sintético
con ZeroMQ local que espera respuestas de control antes de enviar más imágenes.
Incluyen paquetes sin imágenes y frames que no deben guardarse. No requieren
CARLA ni generan capturas en la carpeta del dataset.
