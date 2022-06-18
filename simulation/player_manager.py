import carla
import random
from simulation.constants import *
from simulation.cameras import AgentCamera, SpectatorCamera, TopdownCamera
from simulation.agent.agent import DriveAgent
import math

class Player:
    def __init__(self, client):
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map().get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()
        self.__spawn_player()
        self.self_driving_agent = DriveAgent()
        self.agent_camera_class = AgentCamera(client, self.vehicle)
        self.agent_camera = self.agent_camera_class.get_camera()
        self.spectator_camera_class = SpectatorCamera(client, self.vehicle)
        self.spectator_camera = self.spectator_camera_class.get_camera()
        self.topdown_camera_class = TopdownCamera(client, self.vehicle)
        self.topdown_camera = self.topdown_camera_class.get_camera()

    def __spawn_player(self):
        mustang_blueprint = self.blueprint_library.filter("vehicle.ford.mustang")[0]
        spawn_point = random.choice( self.world.get_map().get_spawn_points() )
        self.vehicle = None
        while self.vehicle == None:
            self.vehicle = self.world.try_spawn_actor(mustang_blueprint, spawn_point)

    def get_sensors(self):
        return {"agent_camera": self.agent_camera, 
                "spectator_camera": self.spectator_camera, 
                "topdown_camera": self.topdown_camera}

    def remove_all(self):
        raise NotImplementedError("Method has not been yet implemented")

    def process_camera_image(self, image):
        return self.agent_camera_class.process_image(image)

    def process_spectator_camera(self, image):
        return self.spectator_camera_class.process_image(image)

    def change_ego_vehicle_direction(self, direction):
        self.self_driving_agent.change_ego_vehicle_direction(direction)

    def apply_autopilot_step(self, image, speed, return_values=True):
        
        step = self.self_driving_agent(image, speed)
        control = carla.VehicleControl(**step
        )
        self.vehicle.apply_control(control)
        

        if return_values:
            return step

    def apply_manual_step(self, action_table):
        control = carla.VehicleControl(**action_table)
        self.vehicle.apply_control(control)
    
    def get_ego_velocity(self):
        speed_vector = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(speed_vector.x**2 + speed_vector.y**2 + speed_vector.z**2)                    
        return speed