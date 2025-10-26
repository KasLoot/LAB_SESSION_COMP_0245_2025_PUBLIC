import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys

import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.part_1.model import Part_1_Model
from final.rollout_loader import load_rollouts

from pathlib import Path




def initialize_simulation():
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    # print init joint
    print(f"Initial joint angles: {init_joint_angles}")
    
    # check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")


    joint_vel_limits = sim.GetBotJointsVelLimit()
    # increase the joint vel limits to not trigger warning in the simulation
    #joint_vel_limits = [vel * 100 for vel in joint_vel_limits]
    
    print(f"joint vel limits: {joint_vel_limits}")
    
    # desired value for regulation
    q_des =  init_joint_angles
    qd_des_clip = np.zeros(num_joints)

    return sim, dyn_model, init_joint_angles, controlled_frame_name, joint_vel_limits


def generate_pose(num_poses, init_joint_angles):
    list_of_desired_cartesian_positions = []
    for _ in range(num_poses):
        # First element: [-0.5:-0.2] or [0.2:0.5]
        x = np.random.uniform(0.3, 0.5)
        # Second element: [-0.5:-0.2] or [0.2:0.5]
        y = np.random.uniform(-0.4, 0.5)
        # Third element: [0.1:0.6]
        z = np.random.uniform(0.1, 0.6)
        list_of_desired_cartesian_positions.append([x, y, z])
    
    # desired cartesian orientation in quaternion (XYZW)
    # Generate random unit quaternions
    list_of_desired_cartesian_orientations = []
    for _ in range(num_poses):
        list_of_desired_cartesian_orientations.append([0.0, 0.0, 0.0, 1.0])
    print(f"list_of_desired_cartesian_orientations: {list_of_desired_cartesian_orientations}")

    list_of_type_of_control = ["pos"] * num_poses # "pos",  "ori" or "both"
    list_of_duration_per_desired_cartesian_positions = [5.0] * num_poses # in seconds
    list_of_initialjoint_positions = [init_joint_angles] * num_poses

    return (list_of_desired_cartesian_positions, list_of_desired_cartesian_orientations,
            list_of_type_of_control, list_of_duration_per_desired_cartesian_positions, list_of_initialjoint_positions)


def load_checkpoint(model: torch.nn.Module, checkpoint_path, device):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model checkpoint from {checkpoint_path}")
    return model


def collect_data(num_poses=5, save_path="q_diff.pt"):

    cmd = MotorCommands()  # Initialize command structure for motors


    # P controller high level
    kp_pos = 100  # position
    kp_ori = 0    # orientation

    k_p = 1000
    k_d = 100

    sim, dyn_model, init_joint_angles, controlled_frame_name, joint_vel_limits = initialize_simulation()

    (list_of_desired_cartesian_positions,
     list_of_desired_cartesian_orientations,
     list_of_type_of_control,
     list_of_duration_per_desired_cartesian_positions,
     list_of_initialjoint_positions) = generate_pose(num_poses, init_joint_angles)

    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()

    model = Part_1_Model(input_size=7, hidden_size=64, output_size=7, dropout_rate=0.3)
    model = load_checkpoint(model, "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints/best_part_1_model.pth", device=torch.device('cuda'))

    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        initial_joint_positions = list_of_initialjoint_positions[i]

        steps = int(duration_per_desired_cartesian_pos/time_step)

        sim.ResetPose()
        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            # tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity

            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, k_p, k_d)
            print(f"feedback lin ctrl tau_cmd: {tau_cmd}")
            
            q_diff_input = torch.tensor(q_des - q_mes, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                tau_cmd_tensor = model(q_diff_input.to(torch.float32).to(torch.device('cuda')))
            tau_cmd = tau_cmd_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
            print(f"ML model tau_cmd: {tau_cmd}\n\n")

            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command
            current_time += time_step

            cart_distance = sum((np.array(cart_pos) - desired_cartesian_pos)**2)**0.5
            print(f"Step {t+1}/{steps}, Cart Pos: {cart_pos}, Desired Pos: {desired_cartesian_pos}, Distance: {cart_distance}")



collect_data(num_poses=5, save_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/")