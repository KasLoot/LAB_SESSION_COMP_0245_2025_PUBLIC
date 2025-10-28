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
from part_2.model import Part_2_Model
from rollout_loader import load_rollouts

from pathlib import Path

np.random.seed(42)


def initialize_simulation():
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

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


def plot_comparison(q_des_diffkin, qd_des_diffkin, q_des_model, qd_des_model, timestamps):
    """
    Plot comparison between CartesianDiffKin and Neural Network predictions
    """
    q_des_diffkin = np.array(q_des_diffkin)
    qd_des_diffkin = np.array(qd_des_diffkin)
    q_des_model = np.array(q_des_model)
    qd_des_model = np.array(qd_des_model)
    timestamps = np.array(timestamps)
    
    num_joints = q_des_diffkin.shape[1]
    
    # Plot q_des comparison
    fig1, axes1 = plt.subplots(num_joints, 1, figsize=(12, 2*num_joints))
    fig1.suptitle('Joint Position Desired (q_des): CartesianDiffKin vs Neural Network', fontsize=14, fontweight='bold')
    
    for i in range(num_joints):
        axes1[i].plot(timestamps, q_des_diffkin[:, i], label='CartesianDiffKin', linewidth=2, alpha=0.7)
        axes1[i].plot(timestamps, q_des_model[:, i], label='Neural Network', linewidth=2, alpha=0.7, linestyle='--')
        axes1[i].set_ylabel(f'Joint {i+1} (rad)', fontsize=10)
        axes1[i].legend(loc='best')
        axes1[i].grid(True, alpha=0.3)
    
    axes1[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig('q_des_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot: q_des_comparison.png")
    
    # Plot qd_des comparison
    fig2, axes2 = plt.subplots(num_joints, 1, figsize=(12, 2*num_joints))
    fig2.suptitle('Joint Velocity Desired (qd_des): CartesianDiffKin vs Neural Network', fontsize=14, fontweight='bold')
    
    for i in range(num_joints):
        axes2[i].plot(timestamps, qd_des_diffkin[:, i], label='CartesianDiffKin', linewidth=2, alpha=0.7)
        axes2[i].plot(timestamps, qd_des_model[:, i], label='Neural Network', linewidth=2, alpha=0.7, linestyle='--')
        axes2[i].set_ylabel(f'Joint {i+1} (rad/s)', fontsize=10)
        axes2[i].legend(loc='best')
        axes2[i].grid(True, alpha=0.3)
    
    axes2[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig('qd_des_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot: qd_des_comparison.png")
    
    # Plot difference/error
    q_diff = np.abs(q_des_diffkin - q_des_model)
    qd_diff = np.abs(qd_des_diffkin - qd_des_model)
    
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
    fig3.suptitle('Absolute Difference between CartesianDiffKin and Neural Network', fontsize=14, fontweight='bold')
    
    for i in range(num_joints):
        axes3[0].plot(timestamps, q_diff[:, i], label=f'Joint {i+1}', linewidth=1.5, alpha=0.7)
    axes3[0].set_ylabel('|q_des difference| (rad)', fontsize=12)
    axes3[0].legend(loc='best', ncol=2)
    axes3[0].grid(True, alpha=0.3)
    
    for i in range(num_joints):
        axes3[1].plot(timestamps, qd_diff[:, i], label=f'Joint {i+1}', linewidth=1.5, alpha=0.7)
    axes3[1].set_ylabel('|qd_des difference| (rad/s)', fontsize=12)
    axes3[1].set_xlabel('Time (s)', fontsize=12)
    axes3[1].legend(loc='best', ncol=2)
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_qd_difference.png', dpi=300, bbox_inches='tight')
    print("Saved plot: q_qd_difference.png")
    
    plt.show()


def plot_cart_distance(cart_distances, timestamps):
    """
    Plot Cartesian distance to target over time
    """
    cart_distances = np.array(cart_distances)
    timestamps = np.array(timestamps)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, cart_distances, linewidth=2, color='navy', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cartesian Distance to Target (m)', fontsize=12)
    ax.set_title('Cartesian Distance to Target Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_dist = np.mean(cart_distances)
    final_dist = cart_distances[-1]
    ax.axhline(y=mean_dist, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_dist:.4f} m')
    ax.text(0.02, 0.98, f'Final distance: {final_dist:.4f} m', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('cart_distance_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved plot: cart_distance_over_time.png")
    
    plt.show()



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

    model = Part_2_Model(input_size=6, hidden_size=64, output_size=14)
    model = load_checkpoint(model, "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints/best_part_2_model.pth", device=torch.device('cuda'))

    # Data collection for plotting
    all_q_des_diffkin = []
    all_qd_des_diffkin = []
    all_q_des_model = []
    all_qd_des_model = []
    all_cart_distances = []
    all_timestamps = []

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

            # print("\n\n"+"="*20)

            q_des_diffkin, qd_des_clip_diffkin = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            # print(f"q_des from diff kin: \n{q_des_diffkin}")
            # print(f"qd_des_clip from diff kin: \n{qd_des_clip_diffkin}")

            prediction = model(torch.tensor(np.concatenate((cart_pos, desired_cartesian_pos)), dtype=torch.float32).to(torch.device('cuda')))

            q_des_model = prediction[:7].detach().cpu().numpy()
            qd_des_clip_model = prediction[7:].detach().cpu().numpy()
            # print(f"q_des from model: \n{q_des_model}")
            # print(f"qd_des_clip from model: \n{qd_des_clip_model}")
            
            # Use model predictions for control
            q_des = q_des_model
            qd_des_clip = qd_des_clip_model

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, k_p, k_d)
            # print(f"feedback lin ctrl tau_cmd: {tau_cmd}")
            

            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command
            current_time += time_step

            # cart_distance = sum((np.array(cart_pos) - desired_cartesian_pos)**2)**0.5
            
    #         # Collect data for plotting
    #         all_q_des_diffkin.append(q_des_diffkin.copy())
    #         all_qd_des_diffkin.append(qd_des_clip_diffkin.copy())
    #         all_q_des_model.append(q_des_model.copy())
    #         all_qd_des_model.append(qd_des_clip_model.copy())
    #         all_cart_distances.append(cart_distance)
    #         all_timestamps.append(current_time)

            # cart_distance = np.linalg.norm(np.array(cart_pos) - desired_cartesian_pos)
            # if cart_distance < 0.001:
            #     print(f"Not collecting data at step {t}, cart_distance: {cart_distance}")
            #     break
    
    # # Create plots
    # plot_comparison(all_q_des_diffkin, all_qd_des_diffkin, all_q_des_model, all_qd_des_model, all_timestamps)
    # plot_cart_distance(all_cart_distances, all_timestamps)



collect_data(num_poses=5, save_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/")