import carla

class TrafficLightManager:
    def __init__(self, world, traffic_lights):
        self.world = world
        self.traffic_lights = {}
        self.state_map = {
            "RED": carla.TrafficLightState.Red,
            "YELLOW": carla.TrafficLightState.Yellow,
            "GREEN": carla.TrafficLightState.Green
        }

        if traffic_lights:
            for key, location in traffic_lights.items():
                traffic_light = self.find_traffic_light(world, location)
                if traffic_light:
                    traffic_light.set_state(carla.TrafficLightState.Red)  # Inicialmente en rojo
                    traffic_light.set_red_time(10000)     # Tiempos realmente altos para evitar cambios automáticos
                    traffic_light.set_yellow_time(10000)
                    traffic_light.set_green_time(10000)
                    self.traffic_lights[key] = traffic_light
        """
        Tengo los blueprints de los semáforos en un diccionario:
        {
            "A": traffic_light_A,
            "B": traffic_light_B,
            "C": traffic_light_C,
            "D": traffic_light_D
        }
        """

    def find_traffic_light(self, world, target_location):
        """
        Busca el semáforo más cercano a la ubicación objetivo proporcionada.
        Retorna el actor del semáforo si se encuentra, de lo contrario None.
        """
        all_traffic_lights = world.get_actors().filter('*.traffic_light')
        target_traffic_light = None
            
        min_distance = float('inf')
        for light in all_traffic_lights:
            distance = light.get_location().distance(target_location)
            if distance < min_distance:
                min_distance = distance
                target_traffic_light = light
        
        return target_traffic_light

    def apply_traffic_lights_state(self, state_dict):
        """
        state_dict = {
            "A": "GREEN",
            "B": "RED",
            "C": "RED",
            "D": "RED" 
        }
        Aplica los estados de los semáforos según el diccionario proporcionado.
        """
        for key, state_str in state_dict.items():
            if key in self.traffic_lights.keys():
                traffic_light = self.traffic_lights[key]
                if state_str in self.state_map.keys():
                    traffic_light.set_state(self.state_map[state_str])