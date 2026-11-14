import cv2
import argparse
from core.inference_class import VisionTrackerBlock
from core.utils import load_camera_config, process_frame, process_frame_inference_only
from core.timing_algorithms import TimingAlgorithms
from core.zmq_utils import VisionSubscriber, DataPublisher

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Config JSON de la cámara 1"
    )
    args = parser.parse_args()

    # --- Cargar configuración de ambas cámaras ---
    conf = load_camera_config(args.config)
    print(conf)
    display = conf["display"]
    roi_a = conf["rois_a"]
    roi_b = conf["rois_b"]

    # --- ZeroMQ ---
    subscriber = VisionSubscriber(conf["receive_port"])
    
    # --- Publicador para enviar datos procesados ---
    publisher = DataPublisher(conf["send_port"])

    # Un solo modelo de visión para ambas cámaras
    vision_cam1 = VisionTrackerBlock(
        confidence_threshold=0.25,
        max_age=1,
        n_init=5,
        nms_max_overlap=1.0
    )

    vision_cam2 = VisionTrackerBlock(
        confidence_threshold=0.25,
        max_age=1,
        n_init=5,
        nms_max_overlap=1.0
    )

    sem_A, sem_B, sem_C, sem_D = 0, 0, 0, 0
    time_i = 0
    timing_algorithm = TimingAlgorithms(algorithm_type="method_fixed")

    print("Iniciando recepción de imágenes... Presione 'q' para salir.")
    try:
        while True:

            data = subscriber.receive_frame()
            time_i += 1# 0.1

            try:
                if display:
                    sem_A, sem_B = process_frame(data["image1"], roi_a, vision_cam1, "Camera 1")
                    sem_C, sem_D = process_frame(data["image2"], roi_b, vision_cam2, "Camera 2")
                else:
                    sem_A, sem_B = process_frame_inference_only(data["image1"], roi_a, vision_cam1)
                    sem_C, sem_D = process_frame_inference_only(data["image2"], roi_b, vision_cam2)
            except:
                pass

            sem_counts = {
                "A": sem_A,
                "B": sem_B,
                "C": sem_C,
                "D": sem_D
            }

            timing_output = timing_algorithm.update(sem_counts, time_i)

            publisher.send_data(timing_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        subscriber.close()
        publisher.close()
        print("Cerrado correctamente")


if __name__ == "__main__":
    main()