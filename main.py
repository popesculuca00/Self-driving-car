import numpy as np
from flask import Flask,render_template,Response, request, jsonify
from simulation.simulator_manager import SimulatorManager
from simulation.constants import *
from utils.synchronous_mode import CarlaSyncMode
from utils.gpu_utils import get_gpu_stats
import carla
import cv2
import psutil
import torch

torch.set_num_threads(1)
torch.cuda.empty_cache()
app=Flask(__name__)

simulator_manager = SimulatorManager()
is_open = True
current_frames = 0
camera_view = 0
brake_autopilot = True
autopilot_off = False
manual_control = {"w": 0, "a": 0, "s": 0, "d":0, "q":0}
reverse_gear = 0



def generate_frames():
    with CarlaSyncMode(simulator_manager.world, *simulator_manager.get_player_sensors(), fps=10) as sync_mode:
        global current_frames
        global camera_view
        global brake_autopilot
        global autopilot_off
        global manual_control
        global reverse_gear


        tm = simulator_manager.client.get_trafficmanager() 
        carla.TrafficManager.set_synchronous_mode(tm, True)
        while True: 
            try:
                _, agent_camera, spectator_camera, topdown_camera = sync_mode.tick(3.0)
            except:
                continue
            
            if camera_view == 0:
                frame = simulator_manager.process_camera_image(agent_camera)
            elif camera_view==1:
                frame = simulator_manager.process_camera_image(spectator_camera)
            elif camera_view==2:
                frame = simulator_manager.process_camera_image(topdown_camera)
            
            ret ,frame = cv2.imencode('.jpg', frame)
            speed = simulator_manager.get_ego_velocity()

            last_speed = speed

            if not autopilot_off:
                if not brake_autopilot:
                    step = simulator_manager.apply_autopilot_step(agent_camera, speed)
                else:
                    simulator_manager.apply_manual_step({"brake": 1})

            else:
                controls={"steer":0, "throttle":0, "brake":0}
                controls["throttle"] += manual_control["w"]
                controls["steer"] -= manual_control["a"]
                controls["steer"] += manual_control["d"]
                controls["brake"] += manual_control["s"]  
                reverse_gear = reverse_gear^1 if manual_control["q"] else reverse_gear 
                controls["reverse"] = reverse_gear
                controls["throttle"] = controls["throttle"] if speed <= 30 else 0             
                simulator_manager.apply_manual_step(controls)
            frame = frame.tobytes()
            current_frames+=1
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["POST", "GET"])
def index():
    return render_template('index.html')


@app.route('/video', methods=["GET", "POST"])
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/changedirection", methods=["POST"])
def get_new_direction():
    data = request.get_json()
    simulator_manager.change_ego_vehicle_direction(data["data"])
    return jsonify(data)


@app.route("/api/spawnwalkers", methods=["POST"])
def spawn_new_walkers():
    data = request.get_json()
    print(data["data"], file=FILE)
    try:
        simulator_manager.spawn_npc(0, int(data["data"]))
    except:
        print("Error spawning npcs", file=FILE)
    return jsonify(data)

@app.route("/api/spawnvehicles", methods=["POST"])
def spawn_new_vehicles():
    data = request.get_json()
    print(data["data"], file=FILE)
    try:
        simulator_manager.spawn_npc(int(data["data"]), 0)
    except:
        print("Error spawning npcs", file=FILE)
    return jsonify(data)

@app.route("/api/changetown", methods=["POST"])
def change_map():
    data = request.get_json()
    print(data["data"], file=FILE)
    simulator_manager.change_map(data["data"])
    return jsonify(data)

@app.route("/api/serverstats", methods=["GET"])
def get_serverstats():
    global current_frames
    gpu, vram = get_gpu_stats()
    data = {"fps" : current_frames,
            "ram" : int(psutil.virtual_memory().percent),
            "cpu" : int(psutil.cpu_percent()),
            "gpu" : int(gpu),
            "vram" : int(vram)
           }
    current_frames=0
    return jsonify(data)

@app.route("/api/changeweather", methods=["POST"])
def change_weather():
    data = request.get_json()
    simulator_manager.change_weather( eval( "carla.WeatherParameters." + str(data["data"])) )
    return jsonify(data)

@app.route("/api/changecam", methods=["POST"])
def change_camera():
    global camera_view
    data = request.get_json()
    camera_view = int(data["data"])
    return jsonify(data)

@app.route("/api/brakeautopilot", methods=["POST"])
def brake_autopilot():
    global brake_autopilot
    data = request.get_json()
    brake_autopilot = bool(data["data"])
    return jsonify(data)

@app.route("/api/changepilotmode", methods=["POST"])
def change_pilot_mode():
    global autopilot_off
    data = request.get_json()
    autopilot_off = not data["data"]
    print(f"autopilot engaged: {autopilot_off}", file=FILE)
    return jsonify(data)

@app.route("/api/manualcontrol", methods=["POST"])
def get_manual_control():
    global manual_control
    data = request.get_json()
    manual_control = {**data}
    manual_control = { key : int(value) for key,value in manual_control.items() }
    return jsonify(data)


if __name__=="__main__":
    app.run(debug=True)

    

