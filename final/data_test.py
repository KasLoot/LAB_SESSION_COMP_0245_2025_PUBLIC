import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists

# Create data directory
DATA_DIR = FINAL_DIR / "data" / "raw_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist


PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

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

import pinocchio as pin
import numpy as np

def solve_ik_pinocchio(
    model, data, frame_name, desired_pos, desired_ori,
    q_init, max_iter=200, eps=1e-3, damping=1e-2
):
    """
    稳定版基于阻尼伪逆的逆运动学求解
    """

    frame_id = model.getFrameId(frame_name)
    q = q_init.copy()

    for i in range(max_iter):
        # 前向运动学
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)
        oMf = data.oMf[frame_id]

        # --- 计算误差 ---
        # 平移误差
        pos_err = desired_pos - oMf.translation
        # 姿态误差（log3(Rd * Rc.T)）
        rot_err = pin.log3(desired_ori @ oMf.rotation.T)
        err = np.hstack((pos_err, rot_err))

        err_norm = np.linalg.norm(err)
        # print(f"Iter {i}: err = {err_norm:.4f}")

        # 收敛判定
        if err_norm < eps:
            print("✅ IK converged!")
            return q

        # --- 雅可比 ---
        # 使用 LOCAL_WORLD_ALIGNED 通常比 LOCAL 更稳定
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)

        # --- 阻尼伪逆 (Damped Least Squares) ---
        JJt = J @ J.T
        dq = J.T @ np.linalg.inv(JJt + (damping ** 2) * np.eye(6)) @ err

        # --- 步长限制 ---
        dq = np.clip(dq, -0.1, 0.1)
        q += dq

        # --- 限制在关节范围内 ---
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    print("❌ IK did not converge.")
    return None

def generate_valid_targets(
    dyn_model,
    controlled_frame_name,
    init_joint_angles,
    lower_limits,
    upper_limits,
    num_positions=20,
    local_ratio=0.7,
    workspace_bounds=None,
    cond_thresh=1000,
    verbose=True,
):
    """
    Generate a list of valid (pos, ori) targets for the robot, combining local and global samples.

    Args:
        dyn_model: PinWrapper object for FK/Jacobian.
        controlled_frame_name (str): target frame name (e.g. "panda_link8").
        init_joint_angles (np.ndarray): initial joint angles.
        lower_limits, upper_limits (list): joint angle limits.
        num_positions (int): total number of target points to generate.
        local_ratio (float): proportion of local perturbation targets (0~1).
        workspace_bounds (dict): bounds for global sampling.
            Example: {"x": [0.3, 0.7], "y": [-0.4, 0.4], "z": [0.1, 0.6]}.
        cond_thresh (float): condition number threshold for singularity filtering.
        verbose (bool): whether to print statistics.

    Returns:
        valid_positions (list[np.ndarray])
        valid_orientations (list[np.ndarray])
    """

    if workspace_bounds is None:
        workspace_bounds = {"x": [0.3, 0.7], "y": [-0.4, 0.4], "z": [0.1, 0.6]}

    num_local = int(num_positions * local_ratio)
    num_global = num_positions - num_local

    valid_positions = []
    valid_orientations = []

    # Get current position/orientation
    base_pos, base_ori = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)

    # 1️⃣ local perturbation sampling
    for _ in range(num_local):
        for _ in range(10):  # retry up to 10 times if invalid
            perturb = np.random.uniform(-0.05, 0.05, size=3)
            pos = base_pos + perturb
            quat = np.random.randn(4)
            quat /= np.linalg.norm(quat)

            print(f"Testing pos {pos.round(3)} ... ")
            try:
                q_sol = solve_ik_pinocchio(dyn_model.pin_model, dyn_model.pin_data, controlled_frame_name, pos, base_ori, np.array(init_joint_angles))
                print("IK result:", q_sol)
                if q_sol is None:
                    continue
                within_limits = np.all(q_sol >= lower_limits) and np.all(q_sol <= upper_limits)
                if not within_limits:
                    continue
                J_res = dyn_model.ComputeJacobian(q_sol, controlled_frame_name, "local_global")
                J = J_res.J
                cond = np.linalg.cond(J) 
                if cond > cond_thresh:
                    continue
                print(f"✅ cond={cond:.1f}")
                valid_positions.append(pos)
                valid_orientations.append(quat)
                break
            except Exception:
                import traceback
                print("❌ IK failed with error:")
                traceback.print_exc()
                exit(0)


    # 2️⃣ global workspace sampling
    for _ in range(num_global):
        for _ in range(10):
            pos = np.array([
                np.random.uniform(*workspace_bounds["x"]),
                np.random.uniform(*workspace_bounds["y"]),
                np.random.uniform(*workspace_bounds["z"])
            ])
            quat = np.random.randn(4)
            quat /= np.linalg.norm(quat)
            print(f"Testing pos {pos.round(3)} ... ")
            try:
                q_sol = solve_ik_pinocchio(dyn_model.pin_model, dyn_model.pin_data, controlled_frame_name,
                           pos, base_ori, np.array(init_joint_angles))
                print("IK result:", q_sol)
                if q_sol is None:
                    continue
                within_limits = np.all(q_sol >= lower_limits) and np.all(q_sol <= upper_limits)
                if not within_limits:
                    continue
                J_res = dyn_model.ComputeJacobian(q_sol, controlled_frame_name, "local_global")
                J = J_res.J
                cond = np.linalg.cond(J)
                if cond > cond_thresh:
                    continue
                print(f"✅ cond={cond:.1f}")
                valid_positions.append(pos)
                valid_orientations.append(quat)
                break
            except Exception:
                continue

    # 统计信息
    if verbose:
        if valid_positions:
            pos_array = np.array(valid_positions)
            print(f"✅ Generated {len(valid_positions)}/{num_positions} valid targets "
                  f"({len(valid_positions)/num_positions*100:.1f}% success rate)")
            print(f"  x range: {pos_array[:,0].min():.2f} ~ {pos_array[:,0].max():.2f}")
            print(f"  y range: {pos_array[:,1].min():.2f} ~ {pos_array[:,1].max():.2f}")
            print(f"  z range: {pos_array[:,2].min():.2f} ~ {pos_array[:,2].max():.2f}")
        else:
            print("⚠️ No valid targets found. Try loosening bounds or cond_thresh.")

    return valid_positions, valid_orientations


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
    
    # Convergence threshold for joint angles (radians)
    joint_angle_tolerance = 0.001  # Adjust this value as needed

    # desired cartesian position
    # Generate random positions with specified ranges
    num_positions = 20
    list_of_desired_cartesian_positions, list_of_desired_cartesian_orientations = generate_valid_targets(
        dyn_model=dyn_model,
        controlled_frame_name=controlled_frame_name,
        init_joint_angles=init_joint_angles,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        num_positions=num_positions,
        local_ratio=0.7,  # 70% 来自局部扰动，30% 来自全局采样
    )
    list_of_type_of_control = ["pos"] * num_positions
    list_of_duration_per_desired_cartesian_positions = [5.0] * num_positions
    list_of_initialjoint_positions = [init_joint_angles] * num_positions

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []

    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()


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
            if RECORDING and t>1000:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)

            # Check if joint angles have converged to desired values
            joint_error = np.abs(np.array(q_mes) - np.array(q_des))
            if np.all(joint_error < joint_angle_tolerance) and t > 1000:
                print(f"Trajectory {i}: Joint angles converged at step {t} (time: {current_time:.2f}s)")
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
                    'cart_ori_all': cart_ori_all_downsampled
                }, f)
            print(f"Data saved to {filename}")

            # Reinitialize data storage lists
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []

        if PRINT_PLOTS:
            print("Plotting downsampled data...")
            # Plot joint positions
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_downsampled[0])):
                joint_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Positions (rad)')
            plt.title('Downsampled Joint Positions')
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