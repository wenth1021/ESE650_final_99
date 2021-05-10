import math
import os
import random
import sys
import cv2
import time

import git
import imageio
import magnum as mn
import numpy as np
import quaternion

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim


# Helper function to render observations from the stereo agent
def _render(sim, display, depth=False):
    for _ in range(100):
        # Just spin in a circle
        obs = sim.step("turn_right")
        # Put the two stereo observations next to eachother
        stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)

        # If it is a depth pair, manually normalize into [0, 1]
        # so that images are always consistent
        if depth:
            stereo_pair = np.clip(stereo_pair, 0, 10)
            stereo_pair /= 10.0

        # If in RGB/RGBA format, change first to RGB and change to BGR
        if len(stereo_pair.shape) > 2:
            stereo_pair = stereo_pair[..., 0:3][..., ::-1]

        # display=False is used for the smoke test
        if display:
            cv2.imshow("stereo_pair", stereo_pair)
            k = cv2.waitKey()
            if k == ord("q"):
                break


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

# frame rate
frame_rate = 30.0

# save directory
output_directory = "examples/stereo_apartment_0"  # @param {type:"string"}
output_path = os.path.join(output_directory)
left_path = os.path.join(output_path, "image_0")
right_path = os.path.join(output_path, "image_1")
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(left_path)
    os.mkdir(right_path)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [settings["height"], settings["width"]]
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    sensor_specs.append(left_rgb_sensor)

    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [settings["height"], settings["width"]]
    # The right RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the right of the center of the agent
    right_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    sensor_specs.append(right_rgb_sensor)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.03)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=0.5)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0.5)
        ),
        "do_nothing": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0.0)
        )
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

#scene_filepath = "../data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
scene_filepath = "../data/scene_datasets/habitat-test-scenes/apartment_1.glb"

sim_settings = {
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "scene": scene_filepath,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}


cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # world space
#agent_state.position = np.array([-1.79, 0.11, 19.25])  # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

# open files to write
file_gt = open(os.path.join(output_path, "groundtruth.txt"), "w")
file_gt.truncate(0)
file_gt.write('# ground truth trajectory\n')
file_gt.write('# file: \'rgbd_dataset_freiburg1_xyz.bag\'\n')
file_gt.write('# timestamp tx ty tz qx qy qz qw\n')

file_time = open(os.path.join(output_path, "times.txt"), "w")
file_time.truncate(0)

# save initial frame
frame_id = 0
observations = sim.step("do_nothing")
stereo_pair = np.concatenate([observations["left_sensor"], observations["right_sensor"]], axis=1)

# If in RGB/RGBA format, change first to RGB and change to BGR
if len(stereo_pair.shape) > 2:
    stereo_pair = stereo_pair[..., 0:3][..., ::-1]
cv2.imshow("stereo_pair", stereo_pair)

left_obs = observations["left_sensor"]
left_img = Image.fromarray(left_obs, mode="RGBA")
filename = os.path.join(left_path, str(frame_id).zfill(6) + ".png")
left_img.save(filename)

right_obs = observations["right_sensor"]
right_img = Image.fromarray(right_obs, mode="RGBA")
filename = os.path.join(right_path, str(frame_id).zfill(6) + ".png")
right_img.save(filename)

curr_time_float = 0.0
curr_time = "{:e}".format(curr_time_float)
agent_state = agent.get_state()
initial_height = -agent_state.position[1]
file_gt.write("{} {} {} {} {} {} {} {}\n".format(curr_time,
                                                   format(agent_state.position[0], '.4f'),
                                                   format(-agent_state.position[1] - initial_height, '.4f'),
                                                   format(-agent_state.position[2], '.4f'),
                                                   format(-quaternion.as_float_array(agent_state.rotation)[1], '.4f'),
                                                   format(-quaternion.as_float_array(agent_state.rotation)[2], '.4f'),
                                                   format(-quaternion.as_float_array(agent_state.rotation)[3], '.4f'),
                                                   format(quaternion.as_float_array(agent_state.rotation)[0], '.4f'),
                                                   ))
file_time.write("{}\n".format(curr_time))

frame_id += 1


# get user control
keystroke = ord(FORWARD_KEY)
while keystroke != ord(FINISH):
    keystroke = cv2.waitKey(0)

    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        #print("action", action)
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        #print("action", action)
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        #print("action", action)
    elif keystroke == ord(FINISH):
        sim.close()
        #print("action", action)
        continue
    else:
        print("INVALID KEY")
        continue

    observations = sim.step(action)
    stereo_pair = np.concatenate([observations["left_sensor"], observations["right_sensor"]], axis=1)

    # If in RGB/RGBA format, change first to RGB and change to BGR
    if len(stereo_pair.shape) > 2:
        stereo_pair = stereo_pair[..., 0:3][..., ::-1]
    cv2.imshow("stereo_pair", stereo_pair)

    left_obs = observations["left_sensor"]
    left_img = Image.fromarray(left_obs, mode="RGBA")
    filename = os.path.join(left_path, str(frame_id).zfill(6) + ".png")
    left_img.save(filename)

    right_obs = observations["right_sensor"]
    right_img = Image.fromarray(right_obs, mode="RGBA")
    filename = os.path.join(right_path, str(frame_id).zfill(6) + ".png")
    right_img.save(filename)

    curr_time_float = 0.0
    curr_time = "{:e}".format(curr_time_float)
    agent_state = agent.get_state()
    initial_height = -agent_state.position[1]
    file_gt.write("{} {} {} {} {} {} {} {}\n".format(curr_time,
                                                     format(agent_state.position[0], '.4f'),
                                                     format(-agent_state.position[1] - initial_height, '.4f'),
                                                     format(-agent_state.position[2], '.4f'),
                                                     format(-quaternion.as_float_array(agent_state.rotation)[1], '.4f'),
                                                     format(-quaternion.as_float_array(agent_state.rotation)[2], '.4f'),
                                                     format(-quaternion.as_float_array(agent_state.rotation)[3], '.4f'),
                                                     format(quaternion.as_float_array(agent_state.rotation)[0], '.4f'),
                                                     ))
    file_time.write("{}\n".format(curr_time))

    frame_id += 1

# close files
file_gt.close()
file_time.close()