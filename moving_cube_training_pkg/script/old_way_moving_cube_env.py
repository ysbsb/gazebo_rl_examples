import gym
import rospy
import time
import numpy as np
import math
import copy
from gym import utils, spaces
import numpy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

reg = register(
    id='OldMovingCube-v0',
    entry_point='old_way_moving_cube_env:OldMovingCubeEnv',
    timestep_limit=1000,
    )

class OldMovingCubeEnv(gym.Env):

    def __init__(self):
        
        

        number_actions = rospy.get_param('/moving_cube/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        self._seed()
        
        #get configuration parameters
        self.init_roll_vel = rospy.get_param('/moving_cube/init_roll_vel')
        
        # Actions
        self.roll_speed_fixed_value = rospy.get_param('/moving_cube/roll_speed_fixed_value')
        self.roll_speed_increment_value = rospy.get_param('/moving_cube/roll_speed_increment_value')
        
        self.start_point = Point()
        self.start_point.x = rospy.get_param("/moving_cube/init_cube_pose/x")
        self.start_point.y = rospy.get_param("/moving_cube/init_cube_pose/y")
        self.start_point.z = rospy.get_param("/moving_cube/init_cube_pose/z")
        
        # Done
        self.max_pitch_angle = rospy.get_param('/moving_cube/max_pitch_angle')
        
        # Rewards
        self.move_distance_reward_weight = rospy.get_param("/moving_cube/move_distance_reward_weight")
        self.y_linear_speed_reward_weight = rospy.get_param("/moving_cube/y_linear_speed_reward_weight")
        self.y_axis_angle_reward_weight = rospy.get_param("/moving_cube/y_axis_angle_reward_weight")
        self.end_episode_points = rospy.get_param("/moving_cube/end_episode_points")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.controllers_list = ['joint_state_controller',
                         'inertia_wheel_roll_joint_velocity_controller'
                         ]
        self.controllers_object = ControllersConnection(namespace="moving_cube",
                                                        controllers_list=self.controllers_list)


        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()

        rospy.Subscriber("/moving_cube/joint_states", JointState, self.joints_callback)
        rospy.Subscriber("/moving_cube/odom", Odometry, self.odom_callback)
        
        
        self._roll_vel_pub = rospy.Publisher('/moving_cube/inertia_wheel_roll_joint_velocity_controller/command', Float64, queue_size=1)

        self.check_publishers_connection()
        
        self.gazebo.pauseSim()

    
    def _seed(self, seed=None): #overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):#overriden function

        self.gazebo.unpauseSim()
        self.set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self.compute_reward(obs, done)
        simplified_obs = self.convert_obs_to_state(obs)

        return simplified_obs, reward, done, info
        
    
    def _reset(self):

        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()
        self.set_init_pose()
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()
        self.gazebo.pauseSim()
        self.init_env_variables()
        obs = self._get_obs()
        simplified_obs = self.convert_obs_to_state(obs)

        return simplified_obs
        
        
    def init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.total_distance_moved = 0.0
        self.current_y_distance = self.get_y_dir_distance_from_start_point(self.start_point)
        self.roll_turn_speed = rospy.get_param('/moving_cube/init_roll_vel')
        
    def _is_done(self, observations):

        pitch_angle = observations[3]

        if abs(pitch_angle) > self.max_pitch_angle:
            rospy.logerr("WRONG Cube Pitch Orientation==>" + str(pitch_angle))
            done = True
        else:
            rospy.logdebug("Cube Pitch Orientation Ok==>" + str(pitch_angle))
            done = False

        return done

    def set_action(self, action):

        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:# Move Speed Wheel Forwards
            self.roll_turn_speed = self.roll_speed_fixed_value
        elif action == 1:# Move Speed Wheel Backwards
            self.roll_turn_speed = self.roll_speed_fixed_value
        elif action == 2:# Stop Speed Wheel
            self.roll_turn_speed = 0.0
        elif action == 3:# Increment Speed
            self.roll_turn_speed += self.roll_speed_increment_value
        elif action == 4:# Decrement Speed
            self.roll_turn_speed -= self.roll_speed_increment_value

        # We clamp Values to maximum
        rospy.logdebug("roll_turn_speed before clamp=="+str(self.roll_turn_speed))
        self.roll_turn_speed = numpy.clip(self.roll_turn_speed,
                                          -1*self.roll_speed_fixed_value,
                                          self.roll_speed_fixed_value)
        rospy.logdebug("roll_turn_speed after clamp==" + str(self.roll_turn_speed))

        # We tell the OneDiskCube to spin the RollDisk at the selected speed
        self.move_joints(self.roll_turn_speed)
        
        
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        MyCubeSingleDiskEnv API DOCS
        :return:
        """

        # We get the orientation of the cube in RPY
        roll, pitch, yaw = self.get_orientation_euler()

        # We get the distance from the origin
        y_distance = self.get_y_dir_distance_from_start_point(self.start_point)

        # We get the current speed of the Roll Disk
        current_disk_roll_vel = self.get_roll_velocity()

        # We get the linear speed in the y axis
        y_linear_speed = self.get_y_linear_speed()

        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]

        return cube_observations
        
        
    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw
    
    def get_roll_velocity(self):
        # We get the current joint roll velocity
        roll_vel = self.joints.velocity[0]
        return roll_vel
    
    def get_y_linear_speed(self):
        # We get the current joint roll velocity
        y_linear_speed = self.odom.twist.twist.linear.y
        return y_linear_speed
        
        
    def get_y_dir_distance_from_start_point(self, start_point):
        """
        Calculates the distance from the given point and the current position
        given by odometry. In this case the increase or decrease in y.
        :param start_point:
        :return:
        """
        y_dist_dir = self.odom.pose.pose.position.y - start_point.y
    
        return y_dist_dir

    def compute_reward(self, observations, done):

        if not done:

            y_distance_now = observations[1]
            delta_distance = y_distance_now - self.current_y_distance
            rospy.logdebug("y_distance_now=" + str(y_distance_now)+", current_y_distance=" + str(self.current_y_distance))
            rospy.logdebug("delta_distance=" + str(delta_distance))
            reward_distance = delta_distance * self.move_distance_reward_weight
            self.current_y_distance = y_distance_now

            y_linear_speed = observations[4]
            rospy.logdebug("y_linear_speed=" + str(y_linear_speed))
            reward_y_axis_speed = y_linear_speed * self.y_linear_speed_reward_weight

            # Negative Reward for yaw different from zero.
            yaw_angle = observations[5]
            rospy.logdebug("yaw_angle=" + str(yaw_angle))
            # Worst yaw is 90 and 270 degrees, best 0 and 180. We use sin function for giving reward.
            sin_yaw_angle = math.sin(yaw_angle)
            rospy.logdebug("sin_yaw_angle=" + str(sin_yaw_angle))
            reward_y_axis_angle = -1 * abs(sin_yaw_angle) * self.y_axis_angle_reward_weight


            # We are not intereseted in decimals of the reward, doesnt give any advatage.
            reward = round(reward_distance, 0) + round(reward_y_axis_speed, 0) + round(reward_y_axis_angle, 0)
            rospy.logdebug("reward_distance=" + str(reward_distance))
            rospy.logdebug("reward_y_axis_speed=" + str(reward_y_axis_speed))
            rospy.logdebug("reward_y_axis_angle=" + str(reward_y_axis_angle))
            rospy.logdebug("reward=" + str(reward))
        else:
            reward = -1*self.end_episode_points

        return reward


    def joints_callback(self, data):
        self.joints = data

    def odom_callback(self, data):
        self.odom = data
        
        
    def check_all_sensors_ready(self):
        self.check_joint_states_ready()
        self.check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/moving_cube/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current moving_cube/joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current moving_cube/joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/moving_cube/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /moving_cube/odom READY=>" + str(self.odom))

            except:
                rospy.logerr("Current /moving_cube/odom not ready yet, retrying for getting odom")

        return self.odom

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._roll_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _roll_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_base_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
        
    def move_joints(self, roll_speed):
        joint_speed_value = Float64()
        joint_speed_value.data = roll_speed
        rospy.logdebug("Single Disk Roll Velocity>>" + str(joint_speed_value))
        self._roll_vel_pub.publish(joint_speed_value)
        self.wait_until_roll_is_in_vel(joint_speed_value.data)
    
    def wait_until_roll_is_in_vel(self, velocity):
    
        rate = rospy.Rate(10)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.1
        v_plus = velocity + epsilon
        v_minus = velocity - epsilon
        while not rospy.is_shutdown():
            joint_data = self.check_joint_states_ready()
            roll_vel = joint_data.velocity[0]
            rospy.logdebug("VEL=" + str(roll_vel) + ", ?RANGE=[" + str(v_minus) + ","+str(v_plus)+"]")
            are_close = (roll_vel <= v_plus) and (roll_vel > v_minus)
            if are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        return delta_time


    def set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_joints(self.init_roll_vel)

        return True
        
        
    def convert_obs_to_state(self,observations):
        """
        Converts the observations used for reward and so on to the essentials for the robot state
        In this case we only need the orientation of the cube and the speed of the disc.
        The distance doesnt condition at all the actions
        """
        disk_roll_vel = observations[0]
        y_linear_speed = observations[4]
        yaw_angle = observations[5]
    
        state_converted = [disk_roll_vel, y_linear_speed, yaw_angle]
    
        return state_converted