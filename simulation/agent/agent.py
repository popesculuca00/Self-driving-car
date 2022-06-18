import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from simulation.constants import *
from simulation.agent.models.cil_model import ConditionalBranchModel
from simulation.agent.models.auto_optimizer import JunctionDetector
from simulation.agent.models.object_detector.object_detector import ObjectDetectorNetwork

torch.set_num_threads(1)

class SelfDrivingAgent:
    """
    Module encapsulating the self-driving agent's logic
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        left_path = "/simulation/agent/models/weights/cil_left_optimized.pth"
        front_path = "/simulation/agent/models/weights/cil_front_optimized.pth"
        right_path = "/simulation/agent/models/weights/cil_right_optimized.pth"
        self.object_detector = ObjectDetectorNetwork(object_thresh=0.4)
        self.auto_optimizer = JunctionDetector()
        self.auto_optimizer.load_state_dict(torch.load("/simulation/agent/models/weights/junction_detector.pth"))
        self.auto_optimizer = torch.jit.script(self.auto_optimizer.to(self.device).eval())
        self.left_branch = ConditionalBranchModel()
        self.left_branch.load_state_dict(torch.load(left_path))
        self.front_branch = ConditionalBranchModel()
        self.front_branch.load_state_dict(torch.load(front_path))
        self.right_branch = ConditionalBranchModel()
        self.right_branch.load_state_dict(torch.load(right_path))
        self.controls = {0 : torch.jit.script(self.right_branch.to(self.device).eval()),
                         1 : torch.jit.script(self.front_branch.to(self.device).eval()),
                         2 : torch.jit.script(self.left_branch.to(self.device).eval())
                        }
        self.direction = 1
        
        self.object_blocking = False
        self.stop_encounter = False
        self.stoplight_counter = 0
        self.wait_for_stoplight = False
        self.stop_confidence = 0

        self.green_encounter = False
        self.green_counter = 0 

        self.stop_vehicle = False
        self.stop_ped = False
        self.stop_trafficlight = False
       
        self.max_allowed_speed = 20 
        self.recommended_max_speed = 15

        self.recommended_direction = -1
        self.cooldown = 0
        self.inter_counter = 0
        self.inter_type = []
 
        self.transforms = nn.Sequential(
            T.Resize( (88,200) )
        )

    def preprocess_image(self, image):
        image = image.reshape((600, 800, 4)) 
        image = image[:, :, :3] 
        return image

    @torch.no_grad()
    def __call__(self, image, speed):
        image = self.preprocess_image( np.array( image.raw_data ) )
        img = torch.cuda.FloatTensor(image).unsqueeze(0) / 255.
        spd = torch.cuda.FloatTensor([speed]).unsqueeze(0) / 30.
        img = img.permute( 0, 3, 1, 2 )
        objs = self.object_detector(img)
        self.stop_encounter, self.car_back_encounter, self.stop_encounter, self.green_encounter = False, False, False, False
        if len(objs):
            for bbox in objs:
                x0 = 800* bbox["boxes"][0]
                y0 = 600* bbox["boxes"][1]
                x1 = x0 + 800 * bbox["boxes"][2]
                y1 = y0 + 600 * bbox["boxes"][3]
                area = abs(x1-x0) * abs(y1-y0)
                print(f" Object: {bbox['class']}, confidence: {bbox['confidence']}, area: {area}")
                if bbox["class"] == "stop":
                    self.stop_encounter = True  
                    self.stop_confidence = bbox["confidence"]
                elif bbox["class"] == "go":
                    self.green_encounter = True
                elif bbox["class"] == "car_back" and area > 4000 and x0 >0.2 and x0<0.8:
                    self.stop_vehicle = True
                elif bbox["class"] == "pedestrian" and area > 2500 and x0>0.2 and x0<0.8:
                    self.stop_ped = True
                elif bbox["class"] == "30":
                    self.max_allowed_speed = 30         
                elif bbox["class"] == "60":
                    self.max_allowed_speed = 30          
                elif bbox["class"] == "90":
                    self.max_allowed_speed = 30  
                    
        self.stop_confidence = self.stop_confidence*self.stop_encounter
        if self.stop_vehicle or self.stop_ped:
            self.object_blocking = True
        else:
            self.object_blocking = False
        img = img[:, :, 115:510, :]
        if self.green_encounter:
            self.green_counter+=1
        else:
            self.green_counter = 0

        if self.wait_for_stoplight and self.green_counter>4:
            self.wait_for_stoplight = False
            self.stop_encounter = 0
        
        if self.wait_for_stoplight:
            print("Waiting for green light..")
            return np.array([0.0, 0.0, 1.0])

        img = self.transforms(img.clone())

        preds_optim = self.auto_optimizer(img*255.)[0]
        preds_optim = preds_optim > 0.9

        if self.stop_encounter:
            self.stoplight_counter+=1
        else:
            self.stoplight_counter=0

        if (self.stoplight_counter>4 and self.stop_confidence>0.6) and self.inter_counter:
            self.wait_for_stoplight = True
            return np.array([0.0, 0.0, 1.0])

        if (not self.object_blocking) and speed<=0.1:
            return np.array([0.3,1.0,0.0])

        if preds_optim[:3].sum() > 1:   # validate in multiple frames intersection type
            if len(self.inter_type) and torch.equal(self.inter_type, preds_optim[:3]):
                self.inter_counter+=1
            else:
                self.inter_counter = 1
                self.inter_type = preds_optim[:3]

        if self.inter_counter >= 10:  # intersection encountered logic block
            if self.inter_type[0] and self.inter_type[2]:   # T intersection
                if self.direction == 1 or self.direction == 2:
                    self.recommended_direction = 2
                else:
                    self.recommended_direction = 0
              
            if self.inter_type[0] and self.inter_type[1]: # left-front intersection
                if self.direction == 0:
                    self.recommended_direction = 1
                else:
                    self.recommended_direction = self.direction
              
            if self.inter_type[1] and self.inter_type[2]: # front-right intersection
                if self.direction == 2:
                    self.recommended_direction = 1
                else:
                    self.recommended_direction = self.direction
            
            self.inter_type = []
            self.inter_counter = 0
            self.cooldown=40

        if self.cooldown:  # run optimized decision on encountering intersection
            self.cooldown -= 1
            preds = self.controls[self.recommended_direction](img, spd).cpu().numpy()
            return preds[0]     
        
        preds = self.controls[1](img, spd).cpu().numpy()
        return preds[0]

    def set_direction(self, direction):
        if direction in [0, 1, 2]:
            self.direction = direction


direction_mapper = {
    "right" : 0,
    "up" : 1,
    "left" : 2
}


class DriveAgent:
    """
    Encapsulates and controls all self driving components
    Call method returns a dictionary containing vehicle commands
    in order to navigate a specified direction 
    """
    def __init__(self):
        self.agent = SelfDrivingAgent()
        self.directions = {}
        
    def change_ego_vehicle_direction(self, target):
        print(target, file=FILE)
        target = direction_mapper[target]
        self.agent.set_direction(target)

    def __call__(self, image, speed):
        steer, throttle , brake = self.agent( image, speed )
        throttle = throttle if speed<self.agent.recommended_max_speed else 0
        brake = brake if throttle<brake else 0
        steer = float( "{:.3f}".format(steer) )
        throttle = float( "{:.3f}".format(throttle) )
        brake = float( "{:.3f}".format(brake) )
        return {
                "throttle": throttle,
                "steer": steer,
                "brake": brake
                }


