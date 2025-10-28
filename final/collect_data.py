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
    cartesian_pos_tolerance = 0.001  # Adjust this value as needed

    q_diff_all = []
    qd_diff_all = []
    tau_cmd_all = []
    cart_pos_all = []
    cart_des_pos_all = []
    q_des_all = []
    qd_des_clip_all = []

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

            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, k_p, k_d)

            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command
            current_time += time_step

            cart_distance = np.linalg.norm(np.array(cart_pos) - desired_cartesian_pos)

            # print(t > 50 and cart_distance > cartesian_pos_tolerance)

            # Store data
            if t > 50:  # Skip initial transient
                q_diff = q_des - q_mes
                qd_diff = qd_des_clip - qd_mes
                q_diff_all.append(q_diff)
                qd_diff_all.append(qd_diff)
                tau_cmd_all.append(tau_cmd)
                cart_pos_all.append(cart_pos)
                cart_des_pos_all.append(desired_cartesian_pos)
                q_des_all.append(q_des)
                qd_des_clip_all.append(qd_des_clip)

            if cart_distance < cartesian_pos_tolerance:
                print(f"Not collecting data at step {t}, cart_distance: {cart_distance}")
                break

    
    # Save collected data
    q_diff_all = torch.tensor(q_diff_all)
    qd_diff_all = torch.tensor(qd_diff_all)
    tau_cmd_all = torch.tensor(tau_cmd_all)
    cart_pos_all = torch.tensor(cart_pos_all)
    cart_des_pos_all = torch.tensor(cart_des_pos_all)
    q_des_all = torch.tensor(q_des_all)
    qd_des_clip_all = torch.tensor(qd_des_clip_all)

    print(f"q_diff_all shape: {q_diff_all.shape}")
    print(f"qd_diff_all shape: {qd_diff_all.shape}")
    print(f"tau_cmd_all shape: {tau_cmd_all.shape}")
    print(f"cart_pos_all shape: {cart_pos_all.shape}")
    print(f"cart_des_pos_all shape: {cart_des_pos_all.shape}")
    print(f"q_des_all shape: {q_des_all.shape}")
    print(f"qd_des_clip_all shape: {qd_des_clip_all.shape}")
    torch.save(q_diff_all, os.path.join(save_path, "q_diff.pt"))
    torch.save(qd_diff_all, os.path.join(save_path, "qd_diff.pt"))
    torch.save(tau_cmd_all, os.path.join(save_path, "tau_cmd.pt"))
    torch.save(cart_pos_all, os.path.join(save_path, "cart_pos.pt"))
    torch.save(cart_des_pos_all, os.path.join(save_path, "cart_des_pos.pt"))
    torch.save(q_des_all, os.path.join(save_path, "q_des.pt"))
    torch.save(qd_des_clip_all, os.path.join(save_path, "qd_des_clip.pt"))


collect_data(num_poses=10, save_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/")