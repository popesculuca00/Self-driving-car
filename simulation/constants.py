import sys
import carla

FILE= sys.stderr

DEFAULT_MAP = "Town_01"

TARGET_FPS = 20  #compute target frames interval with 1000ms / TARGET_FPS

WEATHER_PRESETS={
    "morning": [20.0, 90.0, 30.0, 30.0, 0.0, 30.0],
    "midday": [30.0, 0.0, 60.0, 30.0, 0.0, 80.0],
    "afternoon": [50.0, 0.0, 40.0, 30.0, 0.0, -40.0],
    "default": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "evening": [30.0, 30.0, 0.0, 30.0, 0.0, -60.0]}

AVAILABLE_MAPS=[
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town10HD"]

PLAYER_CAM_PARAMETERS={
    "agent_camera":[800, 600, 100, 
                    {"x":2.0, "y":0.0, "z":1.4}, 
                    {"pitch":-15.0, "yaw":0.0, "roll" :0.0}],
    "spectator_camera":[800, 600, 90,
                    {"x":-4.0, "y":0.0, "z":2.2},
                    {"pitch":-5.0, "yaw":0.0, "roll":0.0}],
    "topdown_camera":[800, 600, 110,
                    {"x":2.0, "y":0.0, "z":5.5},
                    {"pitch":-90, "yaw":0.0, "roll":0.0}]
}

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

DEFAULT_DRIVING_DIRECTION = 1 # follow the road when possible