import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

NUM_OBS_STATES = 3  # Line is 0: left, 1: center, 2: right
NUM_ACTIONS = 3  # Drive 0: left, 1: forward, 2: right
FORWARD_SPEED = 0.25
TURN_SPEED = 0.8


class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)


        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)


        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms


        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)

        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}


    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)

        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}


    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right
        @retval vel_cmd: Twist message equivalent of the action
        '''
        vel_cmd = RosTwist()  # Create clean twist command

        if action == 0:
            # Line on the left, turn + / left / CCW
            vel_cmd.linear.x = FORWARD_SPEED
            vel_cmd.angular.z = TURN_SPEED
        elif action == 1:
            vel_cmd.linear.x = FORWARD_SPEED
            vel_cmd.angular.z = 0.0
        elif action == 2:
            vel_cmd.linear.x = FORWARD_SPEED
            vel_cmd.angular.z = -TURN_SPEED
        else:
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0  # TODO: is this the best default option?

        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign a reward
        @retval reward: the reward for the current step
        '''
        # Don't like off line
        if terminated == True or state == -1: 
            reward = -100.0

        # Forward + on line good
        if state == 1 and action == 1: 
            reward = 5.0

        # Turn + off line ok, otherwise not ok
        elif state == 0 and action == 0: 
            reward = 3.0
        elif state == 2 and action == 2:
            reward = 3.0
        else: 
            reward = -1.0

        return reward


    def process_obs(self, obs: RosImage):
        '''
        @brief Identifies the line in the image and populates the state array
                based on the line position (i.e. state=0 line is left,
                state=1 line is in the center)
        @param obs raw image from Gazebo sim
        @retval state: the line position as a value:
            -1 (no line), 0 (left), 1 (center), 2 (right)
        @retval terminated: the termination condition occured - the line is no
                    longer in the field of the camera
        '''
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')  # Turn msg to cv2 format

        # Slice region of interest
        h, w, _ = cv_image.shape
        roi_start = int(h * 0.85)  # Focus on bottom 15%
        roi = cv_image[roi_start:h, :]

        # Grayscale and threshold
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)

        # Check if line is lost
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return -1, True  # Line out of frame, terminate

        # Calc horizontal centroid and determine state
        cx = int(M['m10']/M['m00'])
        if cx < (w / 3.0):
            state = 0
        elif cx < (w * 2.0 / 3.0):
            state = 1
        else:
            state = 2

        terminated = False  # TODO: figure out where terminated leads to?
        return state, terminated


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()