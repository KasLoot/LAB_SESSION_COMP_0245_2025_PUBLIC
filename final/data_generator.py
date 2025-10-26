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

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists

# Create data directory
DATA_DIR = FINAL_DIR / "data" / "raw_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist


PRINT_PLOTS = True  # Set to True to enable plotting
RECORDING = False  # Set to True to enable data recording

# downsample rate needs to be bigger than one (is how much I steps I skip when i downsample the data)
downsample_rate = 2

# Function to get downsample rate from the user without blocking the simulation loop
def get_downsample_rate():
    try:
        rate = int(input("Enter downsample rate (integer >=1): "))
        if rate < 1:
            print("Invalid downsample rate. Must be >= 1.")
            return None
        return rate
    except ValueError:
        print("Please enter a valid integer.")
        return None


def main():

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
    
    
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # P controller high level
    kp_pos = 100  # position
    kp_ori = 0    # orientation

    # PD controller gains low level (feedback gain)
    kp = 1000
    kd = 100
    


    # desired cartesian position
    # Generate random positions with specified ranges
    # Initial joint angles: [0.0, 1.0323, 0.0, 0.8247, 0.0, 1.57, 0.0]
    # Lower limits: [-2.8973, -1.7628, -2.8973, 0.0, -2.8973, -0.0175, -2.8973]
    # Upper limits: [2.8973, 1.7628, 2.8973, 3.002, 2.8973, 3.7525, 2.8973]
    # joint vel limits: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
    num_positions = 10
    list_of_desired_cartesian_positions = []
    for _ in range(num_positions):
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
    for _ in range(num_positions):
        list_of_desired_cartesian_orientations.append([0.0, 0.0, 0.0, 1.0])
    print(f"list_of_desired_cartesian_orientations: {list_of_desired_cartesian_orientations}")
    # for _ in range(num_positions):
    #     # Generate random quaternion components
    #     quat = np.random.randn(4)
    #     # Normalize to make it a unit quaternion
    #     quat = quat / np.linalg.norm(quat)
    #     list_of_desired_cartesian_orientations.append(quat.tolist())
    
    list_of_type_of_control = ["pos"] * num_positions # "pos",  "ori" or "both"
    list_of_duration_per_desired_cartesian_positions = [5.0] * num_positions # in seconds
    list_of_initialjoint_positions = [init_joint_angles] * num_positions

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all, tau_cmd_all = [], [], [], [], [], [], [], []

    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()

    # Convergence threshold for end-effector position (meters)
    cartesian_pos_tolerance = 0.001  # Adjust this value as needed

    for i in range(len(list_of_desired_cartesian_positions)):

        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        if list_of_initialjoint_positions[i] is None:
            init_position = init_joint_angles
        else:
            init_position = list_of_initialjoint_positions[i]
        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position, desired_cartesian_pos, np.zeros(3), desired_cartesian_ori, np.zeros(3), time_step, type_of_control, kp_pos, kp_ori, np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos/time_step)

        # reinitialize the robot to the initial position
        sim.ResetPose()
        if init_position is not None:
            sim.SetjointPosition(init_position)
        # Data collection loop
        for t in range(steps):
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)
            print(" torque measurement: ", tau_mes)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity
            # Compute desired joint positions and velocities using Cartesian differential kinematics
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            print(f"torque command from feedback lin ctrl: {tau_cmd}")

            checkpoint = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints/best_part_1_model.pth", weights_only=False)
            model = Part_1_Model(input_size=7, hidden_size=128, output_size=7)
            model_state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(model_state_dict)
            model.eval()
            q_diff_std, q_diff_mean = checkpoint["q_diff_std"], checkpoint["q_diff_mean"]
            qd_diff_std, qd_diff_mean = checkpoint["qd_diff_std"], checkpoint["qd_diff_mean"]
            tau_cmd_std, tau_cmd_mean = checkpoint["tau_cmd_std"], checkpoint["tau_cmd_mean"]
            print(f"tau_cmd_std: {tau_cmd_std}, tau_cmd_mean: {tau_cmd_mean}")
            # normalize q_mes - q_des
            q_input = (torch.tensor(q_mes - q_des) - q_diff_mean) / (q_diff_std + 1e-8)
            print(f"q_input: {q_input}")
            with torch.no_grad():
                tau_cmd = model(q_input.to(torch.float32))
                tau_cmd = tau_cmd * (tau_cmd_std + 1e-8) + tau_cmd_mean
            tau_cmd = np.array(tau_cmd)
            print(f"tau_cmd from model: {tau_cmd}")

            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command


            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            
            # Conditional data recording
            if RECORDING and t > 10:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)
                tau_cmd_all.append(tau_cmd)

            # Check if end-effector position has converged to desired values
            cartesian_error = np.linalg.norm(np.array(cart_pos) - np.array(desired_cartesian_pos))
            if cartesian_error < cartesian_pos_tolerance and t > 10:
                print(f"Trajectory {i}: End-effector position converged at step {t} (time: {current_time:.2f}s)")
                break

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            #print("Current time in seconds:", current_time)
    
        current_time = 0  # Reset current time for potential future use

        if len(q_mes_all) > 0:    
            print("Preparing to save data...")
            # Downsample data
            # Plot the downsampled data
            
            q_mes_all_downsampled = q_mes_all[::downsample_rate]
            qd_mes_all_downsampled = qd_mes_all[::downsample_rate]
            q_d_all_downsampled = q_d_all[::downsample_rate]
            qd_d_all_downsampled = qd_d_all[::downsample_rate]
            tau_mes_all_downsampled = tau_mes_all[::downsample_rate]
            cart_pos_all_downsampled = cart_pos_all[::downsample_rate]
            cart_ori_all_downsampled = cart_ori_all[::downsample_rate]
            tau_cmd_all_downsampled = tau_cmd_all[::downsample_rate]

            time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_downsampled))]

            # Save data to pickle file and for name use the current iteration number
            filename = DATA_DIR / f"data_{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump({
                    'time': time_array,
                    'q_mes_all': q_mes_all_downsampled,
                    'qd_mes_all': qd_mes_all_downsampled,
                    'q_d_all': q_d_all_downsampled,
                    'qd_d_all': qd_d_all_downsampled,
                    'tau_mes_all': tau_mes_all_downsampled,
                    'cart_pos_all': cart_pos_all_downsampled,
                    'cart_ori_all': cart_ori_all_downsampled,
                    'tau_cmd_all': tau_cmd_all_downsampled
                }, f)
            print(f"Data saved to {filename}")

            # Reinitialize data storage lists
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all, tau_cmd_all = [], [], [], [], [], [], [], []

        if PRINT_PLOTS:
            print("Plotting downsampled data...")
            # Plot joint positions for each joint
            num_joints = len(q_mes_all_downsampled[0])
            fig, axs = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
            fig.suptitle('Desired vs. Measured Joint Angles', fontsize=16)

            for joint_idx in range(num_joints):
                measured_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                desired_positions = [q[joint_idx] for q in q_d_all_downsampled]
                
                axs[joint_idx].plot(time_array, measured_positions, label=f'Measured Joint {joint_idx+1}')
                axs[joint_idx].plot(time_array, desired_positions, label=f'Desired Joint {joint_idx+1}', linestyle='--')
                axs[joint_idx].set_ylabel('Angle (rad)')
                axs[joint_idx].set_title(f'Joint {joint_idx+1}')
                axs[joint_idx].legend()
                axs[joint_idx].grid(True)

            plt.xlabel('Time (s)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.show()


            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_mes_all_downsampled[0])):
                joint_velocities = [qd[joint_idx] for qd in qd_mes_all_downsampled]
                plt.plot(time_array, joint_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    
    

if __name__ == '__main__':
    main()
    # test rollout loader
    rls = load_rollouts(indices=[0,1,2,3], directory=DATA_DIR)  # looks for ./data/data_1.pkl or ./data/1.pkl, up to 4
    print(f"Loaded {len(rls)} rollouts")
    print("First rollout keys lengths:",len(rls[0].time),len(rls[0].q_mes_all),len(rls[0].qd_mes_all))