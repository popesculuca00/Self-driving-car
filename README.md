# Self-driving-car
Self driving car for Carla simulator
This if my final year bachelor's project.
It uses a combination of behavioural cloning, road analysis techniques and a Yolo object detector to move around a simulated city. The only sensor used for navigation is a frontal 600x800 rgb camera. The user can command the car to take a certain turn via a web application.
In order to run the project you will need a modern CPU and GPU (with at least 16Gb of RAM and 8Gb of VRAM). It was developed on a desktop with an Intel I5-11400 with 32Gb RAM and an Nvidia RTX 3060Ti 8Gb VRAM.  
Carla Simulator version 0.9.13 and Python 3.8.0 are required. 
Clone the repository and download the weights file https://drive.google.com/file/d/19Lyt3a9jw30NYh5nh8mvYInCWx6de9Ny/view?usp=sharing. 
Please unzip the weights file in /simulation/agent/models/weights.
After everything is done please run main.py from the main directory and wait a few seconds to load everything.

A video demonstration of the project running can be found at https://youtu.be/WoC4BIcjFYE
