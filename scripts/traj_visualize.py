
import numpy as np
import argparse

from matplotlib import pyplot as plt
# function to display the topdown map
from PIL import Image
import habitat_sim
from habitat.utils.visualizations import maps
from tqdm import tqdm
from associate import read_file_list


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(1, 1, 1)
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show()


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.05)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plot the agent trajectory on the top-down map
    ''')
    parser.add_argument(
        'first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument(
        'second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('scene_filepath', help='path to the sence file(.glb)')
    args = parser.parse_args()
    test_scene = args.scene_filepath
    first_file = args.first_file
    second_file = args.second_file

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": test_scene,  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    height = sim.pathfinder.get_bounds()[0][1]
    meters_per_pixel = 0.01

    hablab_topdown_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    matches = first_list.keys()
    first_xyz = [[float(value) for value in first_list[a][0:3]]
                 for a in matches]
    second_xyz = [[float(value)*float(1.0)
                   for value in second_list[b][0:3]] for b in matches]
    first_rot = [[float(value) for value in first_list[a][3:]]
                 for a in matches]
    grid_dimensions = (
        hablab_topdown_map.shape[0], hablab_topdown_map.shape[1])
    trajectory = [
        maps.to_grid(
            -path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in second_xyz
    ]

    ground_truth = [
        maps.to_grid(
            -path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in first_xyz
    ]

    colored_map = maps.colorize_topdown_map(hablab_topdown_map)
    trajectory = np.array(trajectory)
    ground_truth = np.array(ground_truth)

    #plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    ax.imshow(colored_map)
    ax.plot(trajectory[:, 1], trajectory[:, 0],
            linewidth=1, color='b', label='estimated')
    ax.plot(ground_truth[:, 1], ground_truth[:, 0],
            linewidth=1, color='r', label='ground truth')
    ax.legend()

    plt.show()

    # curr_traj_x = []
    # curr_traj_y = []
    # curr_gt_x = []
    # curr_gt_y = []
    # frames = []

    # for i in tqdm(range(trajectory.shape[0])):
    #     curr_traj_x.append(trajectory[i, 1])
    #     curr_traj_y.append(trajectory[i, 0])
    #     curr_gt_x.append(ground_truth[i, 1])
    #     curr_gt_y.append(ground_truth[i, 0])
    #     im = ax.imshow(colored_map)
    #     lines, = ax.plot(curr_traj_x, curr_traj_y,
    #                      linewidth=1, color='b', animated=True)
    #     gtlines, = ax.plot(curr_gt_x, curr_gt_y,
    #                        linewidth=1, color='r', animated=True)
    #     frames.append([im, lines, gtlines])

    # import matplotlib.animation as animation

    # ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True,
    #                                 repeat=False)

    # f = r"animation.mp4"
    # writervideo = animation.FFMpegWriter(fps=30)
    # ani.save(f, writer=writervideo)
