# Self-driving car
This is the practical component developed for my bachelor`s thesis: a self-driving car algorithm that runs inside Carla simulator.<br>  
It uses a combination of behavioural cloning and road analysis techniques, as well as a Yolo-based object detector to move around a simulated traffic environment.<br>
The only sensor used for navigation is a frontal 600x800 rgb camera, alongside a speed measurement. The user can command the car to take a certain turn via a web application developed using Flask.<br>  

## Installation
In order to run the project you will need a modern CPU and GPU (with at least 16Gb of RAM and 8Gb of VRAM). It was developed on a desktop PC using:
- an Intel I5-11400 
- 32Gb RAM
- Nvidia RTX 3060Ti 8Gb VRAM.  <br>

Carla Simulator version 0.9.13 and Python 3.8.0 are required. <br>
Clone the repository and download the weights file https://drive.google.com/file/d/19Lyt3a9jw30NYh5nh8mvYInCWx6de9Ny/view?usp=sharing. <br>
Please unzip the weights file in /simulation/agent/models/weights.
After everything is done run main.py from the main directory and wait a few seconds to load everything.<br>
<br>
A video demonstration of the project running can be found at https://youtu.be/WoC4BIcjFYE <br>
The complete thesis (RO) can be found at 
https://drive.google.com/file/d/1bTJinLjAZC-N1vA6BHPDbfQSbaTYW_VH/view?usp=sharing <br>
