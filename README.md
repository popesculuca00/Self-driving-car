# Self-driving car
This is the practical component developed for my bachelor`s thesis: a self-driving car algorithm that runs inside Carla simulator.<br>  
It uses a combination of behavioural cloning and road analysis techniques, as well as a Yolo-based object detector to move around a simulated traffic environment.<br>
The only sensor used for navigation is a frontal 600x800 rgb camera, alongside a speed measurement. The user can command the car to take a certain turn via a web application developed using Flask.<br>  

## Installation
Carla Simulator version 0.9.13 and Python 3.8.0 are required. <br>
Clone the repository and drag the simulator files in the main directory.
<br>
A video demonstration of the project running can be found at https://youtu.be/WoC4BIcjFYE <br>
The complete thesis (RO) can be found at 
https://drive.google.com/file/d/1bTJinLjAZC-N1vA6BHPDbfQSbaTYW_VH/view?usp=sharing <br>

## For docker
```docker
docker build -t selfdriving_img .
docker run -p 2000-2002:2000-2002 -p 5000:5000 --gpus all selfdriving_img
```