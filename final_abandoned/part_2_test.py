import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys

import torch

from part_2.model import Part_2_Model
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from part_1.model import Part_1_Model
from rollout_loader import load_rollouts

from pathlib import Path

np.random.seed(42)



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
        x = np.random.choice([np.random.uniform(0.3, 0.5), np.random.uniform(-0.5, -0.3)])
        y = np.random.choice([np.random.uniform(0.4, 0.5), np.random.uniform(-0.5, -0.4)])
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

    # Convergence threshold for end-effector position (meters)
    cartesian_pos_tolerance = 0.0001  # Adjust this value as needed


    q_mes_all = []
    q_des_cartdiffk_all = []
    qd_mes_all = []
    qd_des_clip_cartdiffk_all = []
    tau_cmd_all = []
    cart_pos_all = []
    cart_des_pos_all = []

    q_des_model_all = []
    qd_des_clip_model_all = []


    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        initial_joint_positions = list_of_initialjoint_positions[i]

        steps = int(duration_per_desired_cartesian_pos/time_step)
        print(f"steps: {steps}")

        sim.ResetPose()
        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            # tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity

            q_des_cartdiffk, qd_des_clip_cartdiffk = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))

            def predict_with_model(q_mes, cart_pos):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Part_2_Model(input_size=10, hidden_size=64, output_size=14)
                checkpoint_path = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints/best_part_2_model.pth"
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict'])
                model.to(device).eval()
                model_input = torch.tensor(np.concatenate((q_mes, cart_pos), axis=0), dtype=torch.float32)
                with torch.no_grad():
                    model_output = model(model_input.to(device))
                q_des_model = model_output[:7].cpu().numpy()
                qd_des_clip_model = model_output[7:].cpu().numpy()
                return q_des_model, qd_des_clip_model
            q_des_model, qd_des_clip_model = predict_with_model(q_mes, cart_pos)

            q_des = q_des_model
            qd_des_clip = qd_des_clip_model

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, k_p, k_d)

            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command
            current_time += time_step

            cart_distance = np.linalg.norm(np.array(cart_pos) - desired_cartesian_pos)


            # Store data
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)
            q_des_cartdiffk_all.append(q_des_cartdiffk)
            qd_des_clip_cartdiffk_all.append(qd_des_clip_cartdiffk)
            q_des_model_all.append(q_des_model) 
            qd_des_clip_model_all.append(qd_des_clip_model)

            tau_cmd_all.append(tau_cmd)
            cart_pos_all.append(cart_pos)
            cart_des_pos_all.append(desired_cartesian_pos)

            if cart_distance < cartesian_pos_tolerance:
                print(f"Not collecting data at step {t}, cart_distance: {cart_distance}")
                break

    
    # Save collected data
    q_mes_all = torch.tensor(q_mes_all)
    qd_mes_all = torch.tensor(qd_mes_all)
    q_des_cartdiffk_all = torch.tensor(q_des_cartdiffk_all)
    qd_des_clip_cartdiffk_all = torch.tensor(qd_des_clip_cartdiffk_all)
    q_des_model_all = torch.tensor(q_des_model_all)
    qd_des_clip_model_all = torch.tensor(qd_des_clip_model_all)

    tau_cmd_all = torch.tensor(tau_cmd_all)
    cart_pos_all = torch.tensor(cart_pos_all)
    cart_des_pos_all = torch.tensor(cart_des_pos_all)


    print(f"tau_cmd_all shape: {tau_cmd_all.shape}")
    print(f"cart_pos_all shape: {cart_pos_all.shape}")
    print(f"cart_des_pos_all shape: {cart_des_pos_all.shape}")
    print(f"q_des_cartdiffk_all shape: {q_des_cartdiffk_all.shape}")
    print(f"qd_des_clip_cartdiffk_all shape: {qd_des_clip_cartdiffk_all.shape}")
    print(f"q_des_model_all shape: {q_des_model_all.shape}")
    print(f"qd_des_clip_model_all shape: {qd_des_clip_model_all.shape}")

    plot_comparison(q_des_cartdiffk_all, q_des_model_all, qd_des_clip_cartdiffk_all, qd_des_clip_model_all, cart_pos_all, cart_des_pos_all)


def plot_comparison(q_des_all, q_des_model_all, qd_des_clip_all, qd_des_clip_model_all, cart_pos_all, cart_des_pos_all):
    """
    Plots the comparison between ground truth and model predictions for joint positions and velocities.
    """
    q_des_all = np.array(q_des_all)
    q_des_model_all = np.array(q_des_model_all)
    qd_des_clip_all = np.array(qd_des_clip_all)
    qd_des_clip_model_all = np.array(qd_des_clip_model_all)
    cart_pos_all = np.array(cart_pos_all)
    cart_des_pos_all = np.array(cart_des_pos_all)

    num_joints = q_des_all.shape[1]
    timesteps = np.arange(q_des_all.shape[0])

    # Plot for q_des
    plt.figure(figsize=(15, 10))
    plt.suptitle('Desired Joint Positions (q_des): Ground Truth vs. Model Prediction')
    for i in range(num_joints):
        plt.subplot(4, 2, i + 1)
        plt.plot(timesteps, q_des_all[:, i], label='Ground Truth')
        plt.plot(timesteps, q_des_model_all[:, i], label='Model Prediction', linestyle='--')
        plt.title(f'Joint {i+1}')
        plt.xlabel('Timestep')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot for qd_des_clip
    plt.figure(figsize=(15, 10))
    plt.suptitle('Desired Joint Velocities (qd_des_clip): Ground Truth vs. Model Prediction')
    for i in range(num_joints):
        plt.subplot(4, 2, i + 1)
        plt.plot(timesteps, qd_des_clip_all[:, i], label='Ground Truth')
        plt.plot(timesteps, qd_des_clip_model_all[:, i], label='Model Prediction', linestyle='--')
        plt.title(f'Joint {i+1}')
        plt.xlabel('Timestep')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot for cartesian position
    plt.figure(figsize=(15, 5))
    plt.suptitle('Cartesian Position: Current vs. Target')
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(timesteps, cart_pos_all[:, i], label='Current Position')
        plt.plot(timesteps, cart_des_pos_all[:, i], label='Target Position', linestyle='--')
        plt.title(f'Position {pos_labels[i]}')
        plt.xlabel('Timestep')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


collect_data(num_poses=5)
