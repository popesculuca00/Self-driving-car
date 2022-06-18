import carla
import random
from simulation.constants import *


class NPCManager:
    def __init__(self, client):
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.num_vehicles = 0
        self.num_walkers = 0
        self.walkers_list = []
        self.vehicles_list = []
        self.all_actors = []

    def spawn_npc(self, num_vehicles, num_walkers):
        if num_vehicles>0:
            spawned_vehicles = self.spawn_vehicles(num_vehicles)
            self.vehicles_list+=spawned_vehicles
        if num_walkers>0:
            spawned_walkers, new_actors = self.spawn_walkers(num_walkers)    
            self.walkers_list+=spawned_walkers
            self.all_actors+= new_actors

    def spawn_walkers(self, num_walkers):
        all_id = []
        walkers_list = []
        walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        spawn_points=[]
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            location = self.world.get_random_location_from_navigation()
            if location:
                spawn_point.location = location
                spawn_points.append(spawn_point)
        batch=[]
        for spawn_point in spawn_points:
            walker_blueprint = random.choice(walker_blueprints)
            if walker_blueprint.has_attribute("is_invincible"):
                walker_blueprint.set_attribute("is_invincible", "false")
            batch.append(SpawnActor(walker_blueprint, spawn_point))
        responses = self.client.apply_batch_sync(batch, True)
        for response in responses:
            if not response.error:
                walkers_list.append({"id": response.actor_id})
        batch=[]
        controller_blueprint = self.world.get_blueprint_library().find("controller.ai.walker")
        for walker in walkers_list:
            batch.append(SpawnActor(controller_blueprint, carla.Transform(), walker["id"]))
        responses = self.client.apply_batch_sync(batch, True)
        for index, response in enumerate(responses):
            if not response.error:
                walkers_list[index]["con"] = response.actor_id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)
        self.world.tick()
        self.world.wait_for_tick()
        for i in range(0, len(all_id), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(1+ random.random()/2)
        return walkers_list, all_actors

    def spawn_vehicles(self, num_vehicles):
        vehicles_list = []
        vehicle_blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("etron")]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        num_vehicles = min( len(spawn_points), num_vehicles) 
        batch=[]
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                vehicles_list.append(response.actor_id)
        return vehicles_list

    def remove_vehicles(self, number=0, all=False):
        if all:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            self.vehicles_list = []
        else:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list[:number]])
            self.vehicles_list = self.vehicles_list[number:]

    def remove_walkers(self, number=0, all=False):
        if all:
            for i in range(0, len(self.all_actors), 2):
                self.all_actors[i].stop()
            self.all_actors = []
            self.walkers_list = []
        else:
            number = min(number*2, len(self.all_actors))
            for i in range(0, number, 2):
                self.all_actors[i].stop()
            self.all_actors = self.all_actors[number+2:]
            self.walkers_list = self.walkers_list[number+2:]

    def remove_npc(self, vehicles=0, walkers=0):
        if vehicles:
            self.remove_vehicles(vehicles)
        if walkers:
            self.remove_walkers(walkers)

    def remove_all(self):
        self.remove_vehicles(all=True)
        self.remove_walkers(all=True)