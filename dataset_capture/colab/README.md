# Entrenar GreenLight en Google Colab

Abrir `greenlight_yolo_colab.ipynb` en [Colab](https://colab.research.google.com/)
mediante **Archivo → Subir notebook** y guardar una copia en Drive. El notebook es
autónomo: no requiere subir los scripts auxiliares ni clonar este repositorio.

## Archivos de entrada

1. Exportar la tarea completa corregida de CVAT como **Ultralytics YOLO Detection 1.0**.
2. Subir ese ZIP a `Mi unidad/GreenLight/datasets/greenlight_cvat_corregido.zip`.
3. Si CVAT exporta solo etiquetas, subir además el ZIP original
   `dataset_capture/prepared/vehicle_500_sesion_20261011/greenlight_vehicle_cvat.zip`
   a esa carpeta de Drive y configurar `IMAGES_ZIP`. Sus etiquetas propuestas se
   ignoran: solo se aprovechan las imágenes.
4. Para las 500 imágenes actuales, el notebook ya incluye sus metadatos. Para otra
   selección, subir el `selection.csv` correspondiente y configurar `METADATA_CSV`.
   Para combinar sesiones, agregar una columna `session` y mantener nombres únicos;
   los nombres del capturador permiten inferir la sesión si no está explícita.

## Ejecución

Seleccionar **Entorno de ejecución → Cambiar tipo de entorno → GPU**, ajustar las
rutas de entrada y ejecutar las celdas en orden. Por defecto realiza un piloto de
cinco épocas con YOLO11n y luego entrena YOLO11n/s/m hasta 100 épocas con parada
temprana. Cambiar `MODELS` para entrenar solo uno; reducir `BATCH` si falta memoria.

La comparación de validación incluye mAP, precisión/recall, error absoluto medio
de conteo y latencia del detector con batch 1. La prueba final se ejecuta al activar
`RUN_FINAL_TEST`, después de fijar la configuración mediante validación. Las
tablas de precisión/recall usan el punto de máximo F1 de Ultralytics; el conteo usa
el umbral explícito `DETECTION_CONF`. La latencia excluye lectura de imágenes,
CARLA, ZMQ y DeepSORT.

## Separación de datos

Se agrupan sesiones completas cuando hay al menos tres. Con una sola sesión se
agrupan sus diez tramos temporales: ambas cámaras de cada tramo permanecen juntas.
Se excluye un margen de diez segundos a cada lado de los límites entre splits
distintos. Esto evita imágenes muy cercanas, pero sigue siendo una evaluación
interna: no demuestra generalización a otra ejecución del simulador.

Las proporciones se aplican a grupos, no a imágenes independientes; con diez
grupos el reparto inicial es 6/2/2 y el margen temporal reduce los conteos finales.
Se muestran esos conteos antes de entrenar. Revisar condiciones y ajustar
`GROUP_SPLITS` si hace falta; el clima no se registró automáticamente. Para evaluar
generalización, capturar más sesiones con distintas semillas y condiciones.

El notebook no tiene un fallback que separe imágenes al azar cuando faltan
metadatos. Se detiene para evitar mezclar cámaras o frames cercanos entre conjuntos.
Las imágenes sin TXT se interpretan como negativas y se informa su cantidad.

## Guardado y recuperación

El dataset se copia al disco de Colab y los resultados se guardan en
`Mi unidad/GreenLight/experiments/vehicle_v1/`. Los checkpoints se escriben al final
de cada época. Para continuar un entrenamiento interrumpido, reconectar GPU,
ejecutar nuevamente con el mismo `EXPERIMENT_NAME` y mantener
`RESUME_INTERRUPTED=True`. El notebook comprueba hashes del dataset y parámetros
antes de reutilizar el experimento. La disponibilidad y duración de la GPU de
Colab varían; el notebook registra el dispositivo de cada conexión.

Los entrenamientos completados se conservan. Para otros datos, otra resolución,
otras semillas o parámetros, usar un nombre nuevo de experimento. `best.pt` se
selecciona con validación; `last.pt` permite continuar durante el entrenamiento.
Después se pueden descargar los mejores pesos y reportes en un ZIP.

## Mantener el notebook

`colab_utils.py` contiene las funciones de preparación que se incrustan en el
notebook. `selection_vehicle_500.csv` contiene únicamente metadatos de selección,
no imágenes ni anotaciones propuestas. Después de cambiar esas fuentes, regenerar:

```powershell
python .\dataset_capture\colab\build_notebook.py
python -m unittest dataset_capture.tests.test_colab_utils -v
```

No se ejecutó un entrenamiento de Colab desde este repositorio: el ZIP corregido
de CVAT se proporciona al correr el notebook. Los tests locales verifican carga de
ZIP completos y de anotaciones, prioridad de etiquetas corregidas, validación de
cajas y separación por grupos.
