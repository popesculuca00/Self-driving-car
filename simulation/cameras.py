import numpy as np
import carla
from simulation.constants import *
from numba import jit


@jit(nopython=True)
def global_process_image(image):
    image = image.reshape((600, 800, 4))
    image = image[:, :, :3]
    image = image.astype(np.uint8)
    return image

class AgentCamera:
    """
    Initializes agent camera used for the self-driving functionality.
    A client instance and an ego-vehicle blueprint must be provided.
    """
    def __init__(self, client, vehicle):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        camera_blueprint =  self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.width, self.height, self.fov, self.location, self.rotation = PLAYER_CAM_PARAMETERS["agent_camera"]
        camera_blueprint.set_attribute("image_size_x", f"{self.width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.height}")
        camera_blueprint.set_attribute("fov", f"{self.fov}")
        spawn_point = carla.Transform( carla.Location(**self.location),
                                       carla.Rotation(**self.rotation))
        self.agent_camera = self.world.spawn_actor(camera_blueprint, 
                                                   spawn_point, 
                                                   attach_to=self.vehicle)

    def get_camera(self):
        return self.agent_camera

    def process_image(self, image):
        return global_process_image(np.array(image.raw_data))


class SpectatorCamera:
    """
    Initializes the third player-view camera used by the spectator.
    A client instance and an ego-vehicle blueprint must be provided.
    """
    def __init__(self, client, vehicle):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        camera_blueprint =  self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.width, self.height, self.fov, self.location, self.rotation = PLAYER_CAM_PARAMETERS["spectator_camera"]
        camera_blueprint.set_attribute("image_size_x", f"{self.width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.height}")
        camera_blueprint.set_attribute("fov", f"{self.fov}")
        spawn_point = carla.Transform( carla.Location(**self.location),
                                       carla.Rotation(**self.rotation))
        self.camera = self.world.spawn_actor(camera_blueprint, 
                                             spawn_point, 
                                             attach_to=self.vehicle)

    def get_camera(self):
        return self.camera

    def process_image(self, image):
        return global_process_image(np.array(image.raw_data))


class TopdownCamera:
    """
    Initializes a topdown camera used for spectating.
    A client instance and an ego-vehicle blueprint must be provided.
    """
    def __init__(self, client, vehicle):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        camera_blueprint =  self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.width, self.height, self.fov, self.location, self.rotation = PLAYER_CAM_PARAMETERS["topdown_camera"]
        camera_blueprint.set_attribute("image_size_x", f"{self.width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.height}")
        camera_blueprint.set_attribute("fov", f"{self.fov}")
        spawn_point = carla.Transform( carla.Location(**self.location),
                                       carla.Rotation(**self.rotation))
        self.camera = self.world.spawn_actor(camera_blueprint, 
                                             spawn_point, 
                                             attach_to=self.vehicle)

    def get_camera(self):
        return self.camera

    def process_image(self, image):
        return global_process_image(np.array(image.raw_data))
