# Moving cube robot q-learning using open ai gym in ros and gazebo

Moving cube example using open ai gym in ros gazebo.

Look communications between python scripts. And use urdf, yaml files to set model and parameters.

<br>

<h2>Model</h2>

<h4>Robot Model</h4>

`Cube Robot(The Cubli)`

<em>Youtube</em> : <https://www.youtube.com/watch?v=n_6p-1J551Y>

<br>

<h4>Training Example</h4>

<em>Youtube</em> : <https://www.youtube.com/watch?v=3_afZzjAQbc>

<Br>

<h2>Settings</h2>

<h4>Download files</h4>

```shell
cd ~/catkin_ws/src

git clone https://github.com/subinlab/rl_moving_cube
```

<br>

<h3>Setting description</h3>

<h4>Ros package in catkin_ws</h4>

- `moving_cube_description`
- `moving_cube_training_pkg`
- `moving_cube_learning`

<br>

<h4>Files in ros package</h4>

`.py` `.urdf` `.yaml` `.launch` 

<br>

<h2>Run</h2>

<h4>Run gazebo empty world</h4>

```
roslaunch gazebo_ros empty_world.launch
```

<h4>Spawn cube model in gazebo world</h4>

```
roslaunch moving_cube_description spawn_moving_cube.launch
```

<h4> Launch cube controller</h4>

```
roslaunch moving_cube_description moving_cube_control.launch
```

<h4>Q-learning traning</h4>

```
rosrun moving_cube_training_pkg cube_rl_utils.py
```

<h4>Q-learning training using open ai gym</h4>

```
roslaunch my_moving_cube_training_pkg start_training.launch
```

<br>

<h2><em>References</em></h2>

- <https://bitbucket.org/theconstructcore/moving_cube/src/master/>
- <https://bitbucket.org/theconstructcore/moving_cube_training/src/master/>
- <https://bitbucket.org/theconstructcore/moving_cube_ai/src/master/>