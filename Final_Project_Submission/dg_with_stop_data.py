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
from rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists


PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

data_category = "test"  # "train" or "test" or "simulation"

if data_category == "train":
    FINAL_DIR = FINAL_DIR / "data" / "train_with_stop_data"
    torch.manual_seed(42)
    np.random.seed(42)
    num_of_poses = 200
elif data_category == "test":
    FINAL_DIR = FINAL_DIR / "data" / "test_with_stop_data"
    torch.manual_seed(56)
    np.random.seed(56)
    num_of_poses = 40
elif data_category == "simulation":
    torch.manual_seed(100)
    np.random.seed(100)
    num_of_poses = 5

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


def generate_data(num_poses, init_joint_angles=None):
    list_of_desired_cartesian_positions = []
    list_of_desired_cartesian_orientations = []
    list_of_type_of_control = [] # "pos",  "ori" or "both"
    list_of_duration_per_desired_cartesian_positions = []
    list_of_initialjoint_positions = []  # if None, use default initial joint angles
    for _ in range(num_poses):
        # x = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
        # y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
        x = np.random.uniform(0.2, 0.5)
        y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
        # Third element: [0.1:0.6]
        z = np.random.uniform(0.1, 0.6)
        list_of_desired_cartesian_positions.append([x, y, z])
        list_of_desired_cartesian_orientations.append([0.0, 0.0, 0.0, 1.0])
        list_of_type_of_control.append("pos")  # "pos",  "ori" or "both"
        list_of_duration_per_desired_cartesian_positions.append(2.0)  # in seconds
        list_of_initialjoint_positions.append(init_joint_angles)  # use default initial joint angles

    return list_of_desired_cartesian_positions, list_of_desired_cartesian_orientations, list_of_type_of_control, list_of_duration_per_desired_cartesian_positions, list_of_initialjoint_positions


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

    # # desired cartesian position
    # list_of_desired_cartesian_positions = [[0.5,0.0,0.1], 
    #                                        [0.4,0.5,0.1], 
    #                                        [0.4,-0.5,0.1], 
    #                                        [0.6,0.0,0.1]]
    # # desired cartesian orientation in quaternion (XYZW)
    # list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0],
    #                                           [0.0, 0.0, 0.0, 1.0],
    #                                           [0.0, 0.0, 0.0, 1.0],
    #                                           [0.0, 0.0, 0.0, 1.0]]
    # list_of_type_of_control = ["pos", "pos", "pos", "pos"] # "pos",  "ori" or "both"
    # list_of_duration_per_desired_cartesian_positions = [5.0, 5.0, 5.0, 5.0] # in seconds
    # list_of_initialjoint_positions = [init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles]

    list_of_desired_cartesian_positions, list_of_desired_cartesian_orientations, list_of_type_of_control, list_of_duration_per_desired_cartesian_positions, list_of_initialjoint_positions = generate_data(num_poses=num_of_poses, init_joint_angles=init_joint_angles)


    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()


    for i in range(len(list_of_desired_cartesian_positions)):
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all, final_cartesian_pos = [], [], [], [], [], [], [], []


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
        cart_distance = float('inf')  # Initialize with a large value
        reached_target = False  # Flag to track if target is reached
        steps_after_reaching_target = 0  # Counter for steps after reaching target
        extra_steps = 500  # Number of additional steps to collect after reaching target
        for t in range(steps):
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity
            # Compute desired joint positions and velocities using Cartesian differential kinematics
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
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
            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)
                final_cartesian_pos.append(desired_cartesian_pos)

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            #print("Current time in seconds:", current_time)

            cart_distance = np.linalg.norm(np.array(cart_pos) - desired_cartesian_pos)
            if cart_distance < 5e-5 and not reached_target:
                print(f"Reached desired position at time {current_time:.2f} seconds. Collecting {extra_steps} more steps...")
                reached_target = True
            
            # Count steps after reaching target
            if reached_target:
                steps_after_reaching_target += 1
                if steps_after_reaching_target >= extra_steps:
                    print(f"Collected {extra_steps} steps after reaching target. Moving to next trajectory.")
                    break

    
        current_time = 0  # Reset current time for potential future use

        q_mes_all_downsampled = torch.tensor(q_mes_all[::downsample_rate])
        qd_mes_all_downsampled = torch.tensor(qd_mes_all[::downsample_rate])
        q_d_all_downsampled = torch.tensor(q_d_all[::downsample_rate])
        qd_d_all_downsampled = torch.tensor(qd_d_all[::downsample_rate])
        tau_mes_all_downsampled = torch.tensor(tau_mes_all[::downsample_rate])
        cart_pos_all_downsampled = torch.tensor(cart_pos_all[::downsample_rate])
        cart_ori_all_downsampled = torch.tensor(cart_ori_all[::downsample_rate])
        final_cartesian_pos_downsampled = torch.tensor(final_cartesian_pos[::downsample_rate])

        time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_downsampled))]

        # Only save data if the final cart_distance is below the threshold
        save_flag = len(q_mes_all) > 0 and cart_distance < 5e-5
        print(f"Iteration {i}: cart_distance = {cart_distance:.6f}, save_flag = {save_flag}")

        if save_flag and data_category != "simulation":
            print("Preparing to save data...")

            filename = FINAL_DIR / f"data_{i}.pt"
            torch.save({
                # 'time': time_array,
                # 'qd_mes_all': qd_mes_all_downsampled,
                # 'tau_mes_all': tau_mes_all_downsampled,
                # 'cart_pos_all': cart_pos_all_downsampled,
                # 'cart_ori_all': cart_ori_all_downsampled,
                'q_mes_all': q_mes_all_downsampled,
                'final_cartesian_pos': final_cartesian_pos_downsampled,
                'q_d_all': q_d_all_downsampled,
                'qd_d_all': qd_d_all_downsampled
            }, filename)
            


            # print(f"Data saved to {filename}")
        else:
            print(f"Skipping data save for iteration {i}: cart_distance {cart_distance:.6f} exceeds threshold 5e-5")


        if PRINT_PLOTS:
            print("Plotting downsampled data...")
            # Plot joint positions
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_downsampled[0])):
                joint_mes_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                plt.plot(time_array, joint_mes_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Angles (rad)')
            plt.title('Downsampled measured Joint Angles')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(final_cartesian_pos_downsampled[0])):
                final_cartesian_pos = [qd[joint_idx] for qd in final_cartesian_pos_downsampled]
                plt.plot(time_array, final_cartesian_pos, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled final_cartesian_pos')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_d_all_downsampled[0])):
                joint_angles = [qd[joint_idx] for qd in q_d_all_downsampled]
                plt.plot(time_array, joint_angles, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Angles (rad)')
            plt.title('Downsampled Desired Joint Angles')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_d_all_downsampled[0])):
                joint_desired_velocities = [qd[joint_idx] for qd in qd_d_all_downsampled]
                plt.plot(time_array, joint_desired_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Desired Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_mes_all_downsampled[0])):
                joint_velocities = [qd[joint_idx] for qd in qd_mes_all_downsampled]
                plt.plot(time_array, joint_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Measured Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    
    

if __name__ == '__main__':
    main()
    # test rollout loader
    # rls = load_rollouts(indices=[0,1,2,3], directory=FINAL_DIR)  # looks for ./data_1.pkl or ./1.pkl, up to 4
    # print(f"Loaded {len(rls)} rollouts")
    # print("First rollout keys lengths:",len(rls[0].time),len(rls[0].q_mes_all),len(rls[0].qd_mes_all))