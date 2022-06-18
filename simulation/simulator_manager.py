import numpy as np
from time import sleep
from gc import collect
import psutil
import carla
import subprocess
from simulation.constants import *
from simulation.npc_manager import NPCManager
from simulation.player_manager import Player



class SimulatorManager:
    def __init__(self, town="Town01", path="D:/Carla/CARLA_0.9.13", graphics="Epic", open_sim="auto_detect"):

        if open_sim == "auto_detect":
            self.carla_process = None
            self.find_pid()
            open_sim = True if self.carla_process is None else False
        if open_sim is True:
            print("Opening simulator.. ", file=FILE)
            self.open_simulator(path, graphics)
            sleep(5)
            self.find_pid()
        else:
            print("Found open simulator", file=FILE)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        print("Successfully connected to carla", file=FILE)
        self.change_map(town)



    def open_simulator(self, path, graphics):
        cmd = path + "/" + "CarlaUE4.exe --quality-level=" + graphics       
        subprocess.Popen(cmd)

    def change_map(self, map):
        if map not in AVAILABLE_MAPS:
            return
        if hasattr(self, "npc_manager"):
            self.npc_manager.remove_all()
        if hasattr(self, "player"):
            # todo : handle player in case of map changing event -------------------
            pass
        self.world = self.client.load_world(map)
        
        self.current_map = map
        self.npc_manager = NPCManager(self.client)      ### Todo: reset npc lists
        self.player = Player(self.client)
        self.change_weather()
        collect()

    def spawn_npc(self, vehicles=0, walkers=0):
        self.npc_manager.spawn_npc(vehicles, walkers)

    def remove_all_npc(self):
        self.npc_manager.remove_all()

    def remove_npc(self, vehicles=0, walkers=0):  
        self.npc_manager.remove_npc(vehicles, walkers)

    def get_player(self):
        return self.player

    def get_player_sensors(self):
        return [self.player.agent_camera, self.player.spectator_camera, self.player.topdown_camera]
    
    def get_npc_manager(self):
        return self.npc_manager

    def change_weather(self, preset=carla.WeatherParameters.ClearNoon):
       # if preset in WEATHER_PRESETS.keys():
       #     weather = carla.WeatherParameters( *WEATHER_PRESETS[preset] )
        self.world.set_weather(preset)        

    def find_pid(self):
        for process in psutil.process_iter():
            if "CarlaUE4" in process.name():
                self.carla_process = process
                break

    def safe_close(self):
        self.npc_manager.remove_all()
        self.player.remove_all()
        self.carla_process.kill()
    
    def restart_simulator(self):
        self.safe_close()
        self.__init__()

    def change_ego_vehicle_direction(self, direction):
        self.player.change_ego_vehicle_direction(direction)

    def process_camera_image(self, data):
        return self.player.process_camera_image(data)

    def process_spectator_camera(self, data):
        return self.player.process_spectator_camera(data)

    def apply_autopilot_step(self, image, speed, return_values=True):
        return self.player.apply_autopilot_step(image, speed, return_values)

    def apply_manual_step(self, action_table):
        self.player.apply_manual_step(action_table)

    def get_ego_velocity(self):
        return self.player.get_ego_velocity()

if __name__ == "__main__":
    x = SimulatorManager()