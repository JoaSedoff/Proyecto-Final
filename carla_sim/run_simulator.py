
import carla
import sys
import time

from core.spawn_utils import spawn_vehicles, spawn_pedestrians, delete_vehicles
from core.zmq_class import ZMQPublisher, ZMQSubscriber
from core.setup_world import SetupWorld
from core.setup_camera import add_camera, prepare_camera_data
from core.dynamic_weather import Weather
from core.traffic_metrics import TrafficMetrics
from core.traffic_light_utils import TrafficLightManager

def main():
    # --- Configuración de ZeroMQ ---
    zmq_publisher = ZMQPublisher(port=5555)

    # --- Conexión a CARLA ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)

    setup = SetupWorld(client, map_name='Town04')
    client.set_timeout(25.0)

    world = setup.load_map()

    actor_list = []
    original_settings = world.get_settings()
    
    # --- Configurar ROI y ocultar objetos lejanos ---
    central_point = carla.Location(x=190.5, y=-239.5, z=0.0)
    SetupWorld.toggle_far_environment_objects(setup.world, central_point, radius=200.0)

    # --- Configurar suscriptor ZeroMQ para recibir datos de visión ---
    subscriber = ZMQSubscriber(port=5557)

    try:
        # --- CONFIGURAR EL MUNDO EN MODO SÍNCRONO ---
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1#0.1
        settings.max_substep_delta_time = 0.025     # Asegura estabilidad del paso de física
        settings.max_substeps = 3
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # --- CONFIGURAR SEMÁFOROS ---
        targets = {
            "A": carla.Location(x=207.4, y=-254.7, z=0.0), # Posición -> x=207.37, y=-254.69, z=6.44
            "B": carla.Location(x=209.4, y=-242.4, z=0.0), # Posición -> x=209.37, y=-242.38, z=6.63
            "C": carla.Location(x=197.2, y=-238, z=0.0), # Posición -> x=197.16, y=-238.00, z=6.91
            "D": carla.Location(x=191.5, y=-250, z=0.0) # Posición -> x=191.53, y=-250.18, z=6.91
        }

        traffic_light_manager = TrafficLightManager(world, targets)

        # --- CONFIGURAR EL CLIMA DINÁMICO ---
        weather = Weather(world.get_weather())
        speed_factor = 0.25  # Velocidad de cambio climático
        update_freq = 0.1 / speed_factor

        elapsed_time = 0.0

        # --- GENERAR TRÁFICO ---
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(12345)

        # Limitar la zana de spawn de vehículos
        spawn_points = world.get_map().get_spawn_points()
        
        #target_location_1 = carla.Location(x=351, y=-180, z=0.00)
        target_location_1 = carla.Location(x=210, y=-254, z=6.0)
        target_rotation_1 = carla.Rotation(pitch=-7.5, yaw=137.5, roll=0.0)
        target_location_2 = carla.Location(x=195, y=-238, z=6.0)
        target_rotation_2 = carla.Rotation(pitch=-13.1, yaw=-48.2, roll=0.0)

        nearby_spawns = [
            sp for sp in spawn_points 
            if sp.location.distance(target_location_1) < 100.0
        ]

        # Generate vehicles
        number_of_vehicles = 20

        # No queremos que ciertos vehículos aparezcan
        vehicles_to_not_spawn = ["vehicle.micro.microlino", "vehicle.tesla.cybertruck", "vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]
        vehicles_to_not_spawn += ["vehicle.mitsubishi.fusorosa"]
        blueprints = blueprint_library.filter('vehicle.*')
        blueprints = [bp for bp in blueprints if bp.id not in vehicles_to_not_spawn]
        
        actor_list = spawn_vehicles(world, traffic_manager, blueprints, nearby_spawns, number_of_vehicles, actor_list)

        nearby_spawns = [
            sp for sp in spawn_points 
            if sp.location.distance(target_location_1) > 60.0
            and sp.location.distance(target_location_1) < 115.0
        ]

        # Generate pedestrians
        number_of_pedestrians = 50

        actor_list = spawn_pedestrians(world, client, number_of_pedestrians, actor_list)

        # --- CONFIGURAR CÁMARAS EN SEMÁFOROS ---
        camera, image_queue_1 = add_camera(world, blueprint_library, target_location_1, target_rotation_1)
        actor_list.append(camera)

        camera, image_queue_2 = add_camera(world, blueprint_library, target_location_2, target_rotation_2)
        actor_list.append(camera)

        # --- CONFIGURAR MÉTRICAS DE TRÁFICO ---
        traffic_metrics = TrafficMetrics(world, central_point, roi_radius=62.5, draw_roi=False)
        
        # --- BUCLE PRINCIPAL MAESTRO ---
        while True:
            try:
                world.tick()
                
                # --- Actualizar clima dinámico ---
                world_snapshot = world.get_snapshot()
                timestamp = world_snapshot.timestamp
                elapsed_time += timestamp.delta_seconds
                if elapsed_time > update_freq:
                    weather.tick(speed_factor * elapsed_time)
                    world.set_weather(weather.weather)
                    sys.stdout.write('\r' + str(weather) + 12 * ' ')
                    sys.stdout.flush()
                    elapsed_time = 0.0

                # --- Determinar hora simulada y tránsito dinámico ---
                current_hour = weather.current_hour()
                
                # Determinar cantidad de vehículos según hora
                if 7 <= current_hour < 9 or 11 <= current_hour < 13 or 16 <= current_hour < 18:
                    number_of_vehicles = 65  # Tránsito alto
                elif 6 <= current_hour < 22:
                    number_of_vehicles = 30  # Tránsito moderado
                else:
                    number_of_vehicles = 10  # Tránsito bajo
                
                # --- Gestionar vehículos dinámicamente ---
                # Eliminar vehículos lejanos al semáforo
                actor_list = delete_vehicles(actor_list, target_location_1)
                # Generar nuevos vehículos si es necesario
                actor_list = spawn_vehicles(world, traffic_manager, blueprints, nearby_spawns, number_of_vehicles, actor_list)
                
                # --- ACTUALIZAR MÉTRICAS DE TRÁFICO ---
                traffic_metrics.update(current_hour)

                # --- PUBLICAR DATOS A TRAVÉS DE ZEROMQ ---
                # Enviar imagen de la cámara
                camera_data = prepare_camera_data(image_queue_1, image_queue_2)
                zmq_publisher.send_frame(camera_data)
                 
                data_received = False
                while not data_received:
                    # Recibir datos procesados de visión
                    try:
                        data = subscriber.reseive()

                        # Aplicar estados de semáforos
                        if data:
                            data_received = True
                            traffic_light_manager.apply_traffic_lights_state(data['states'])
                    except:
                        # No hay datos disponibles en este momento
                        pass

                    time.sleep(0.01)
            except KeyboardInterrupt:
                break
    finally:
        try:
            if 'world' in locals() and 'original_settings' in locals():
                world.apply_settings(original_settings)
            if 'client' in locals() and 'actor_list' in locals():
                actors_to_destroy = [x for x in actor_list if x and x.is_alive]
                client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
            
            zmq_publisher.close()
            subscriber.close()
            print("\nSimulación terminada. Actores eliminados.")
        except Exception as e:
            print(f"Error durante la limpieza: {e}")

if __name__ == '__main__':
    main()