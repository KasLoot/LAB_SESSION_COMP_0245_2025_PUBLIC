"""
Evaluation Script for Part 2 - Robot Control Model
===================================================

This script evaluates the trained neural network model by:
1. Running robot simulations with model predictions
2. Comparing model outputs against ground truth (desired joint positions/velocities)
3. Visualizing 3D trajectories of the end-effector

Author: Generated for COMP_0245_2025
Date: 29 October 2025
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path

# Import simulation and control modules
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# Import the trained model
from part_2 import P2_MLP


# ============================================================================
# Configuration Parameters
# ============================================================================

# Set random seed for reproducibility
np.random.seed(100)  # Using test seed for evaluation

# Directories
FINAL_DIR = Path(__file__).resolve().parent
MODEL_PATH = FINAL_DIR / "part2_best_model.pth"
RESULTS_DIR = FINAL_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
NUM_TEST_POSES = 10  # Number of different target positions to test
DURATION_PER_POSE = 5.0  # Duration in seconds for each pose

# Visualization parameters
SHOW_INTERACTIVE_3D = True  # Set to True to show interactive 3D plots (closes window to continue)


# ============================================================================
# Helper Functions
# ============================================================================

def generate_test_positions(num_poses, init_joint_angles):
    """
    Generate random test positions for evaluation.
    
    Args:
        num_poses (int): Number of test positions to generate
        init_joint_angles (array): Initial joint configuration
        
    Returns:
        list: List of desired Cartesian positions
        list: List of desired Cartesian orientations (quaternions)
        list: List of initial joint positions for each pose
    """
    desired_positions = []
    desired_orientations = []
    initial_joint_positions = []
    
    for _ in range(num_poses):
        # Generate random x, y coordinates in valid workspace
        x = np.random.uniform(0.3, 0.5)
        y = np.random.choice([np.random.uniform(0.3, 0.5), np.random.uniform(-0.5, -0.3)])
        z = np.random.uniform(0.1, 0.6)  # Height range
        
        desired_positions.append([x, y, z])
        desired_orientations.append([0.0, 0.0, 0.0, 1.0])  # Fixed orientation (XYZW)
        initial_joint_positions.append(init_joint_angles)
    
    return desired_positions, desired_orientations, initial_joint_positions


def load_trained_model(model_path, device):
    """
    Load the trained neural network model.
    
    Args:
        model_path (Path): Path to the saved model weights
        device (torch.device): Device to load the model on
        
    Returns:
        P2_MLP: Loaded model in evaluation mode
    """
    model = P2_MLP(input_size=10, output_size=14)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(torch.float64).to(device)
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    return model


def predict_with_model(model, q_mes, desired_cartesian_pos, device):
    """
    Get model predictions for desired joint positions and velocities.
    
    Args:
        model (P2_MLP): Trained neural network model
        q_mes (array): Current measured joint positions
        desired_cartesian_pos (array): Desired Cartesian position
        device (torch.device): Device for computation
        
    Returns:
        array: Predicted desired joint positions
        array: Predicted desired joint velocities
    """
    # Prepare input tensor
    input_data = torch.tensor(
        np.concatenate([q_mes, desired_cartesian_pos]), 
        dtype=torch.float64
    ).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_data)
    
    # Split output into positions and velocities
    output_np = output.cpu().numpy().squeeze()
    q_des = output_np[:7]  # First 7 values are joint positions
    qd_des = output_np[7:]  # Last 7 values are joint velocities
    
    return q_des, qd_des


def simulate_single_trajectory(sim, dyn_model, model, device, 
                               desired_cartesian_pos, init_joint_angles,
                               duration, controlled_frame_name,
                               kp=1000, kd=100):
    """
    Simulate robot trajectory for a single target position using the trained model.
    
    Args:
        sim: Simulation interface
        dyn_model: Dynamic model (PinWrapper)
        model: Trained neural network
        device: PyTorch device
        desired_cartesian_pos (array): Target Cartesian position
        init_joint_angles (array): Initial joint configuration
        duration (float): Simulation duration in seconds
        controlled_frame_name (str): Name of the controlled frame
        kp (float): Proportional gain for feedback control
        kd (float): Derivative gain for feedback control
        
    Returns:
        dict: Dictionary containing trajectory data
    """
    # Reset robot to initial position
    sim.ResetPose()
    
    # Wait for physics to settle after reset
    time.sleep(0.1)
    
    # Explicitly set joint positions to initial configuration
    sim.SetjointPosition(init_joint_angles)
    
    # Wait for 1 second after reset to stabilize
    time.sleep(1.0)
    
    # Storage for trajectory data
    trajectory_data = {
        'time': [],
        'q_mes': [],           # Measured joint positions
        'qd_mes': [],          # Measured joint velocities
        'q_des_model': [],     # Model-predicted desired joint positions
        'qd_des_model': [],    # Model-predicted desired joint velocities
        'cart_pos': [],        # Actual Cartesian positions
        'cart_ori': [],        # Actual Cartesian orientations
        'tau_cmd': [],         # Commanded torques
        'desired_cart_pos': desired_cartesian_pos
    }
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    steps = int(duration / time_step)
    cmd = MotorCommands()
    
    current_time = 0.0
    
    # Record initial state (at t=0, before any control is applied)
    q_mes_init = sim.GetMotorAngles(0)
    qd_mes_init = sim.GetMotorVelocities(0)
    cart_pos_init, cart_ori_init = dyn_model.ComputeFK(q_mes_init, controlled_frame_name)
    
    trajectory_data['time'].append(0.0)
    trajectory_data['q_mes'].append(q_mes_init.copy())
    trajectory_data['qd_mes'].append(qd_mes_init.copy())
    trajectory_data['q_des_model'].append(q_mes_init.copy())  # No control yet
    trajectory_data['qd_des_model'].append(qd_mes_init.copy())  # No control yet
    trajectory_data['cart_pos'].append(cart_pos_init.copy())
    trajectory_data['cart_ori'].append(cart_ori_init.copy())
    trajectory_data['tau_cmd'].append(np.zeros(7))  # No torque at start
    
    # Simulation loop
    for t in range(steps):
        # Measure current state at the beginning of this step
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        
        # Compute forward kinematics to get current Cartesian position
        cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
        
        # Use model to predict desired joint positions and velocities
        q_des, qd_des = predict_with_model(
            model, q_mes, desired_cartesian_pos, device
        )
        
        # Compute control torques using feedback linearization
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        
        # Send command and step simulation
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
        sim.Step(cmd, "torque")
        
        # Update time and wait
        current_time += time_step
        time.sleep(time_step)
        
        # Record data at the beginning of the NEXT step (which is the result of this step)
        q_mes_next = sim.GetMotorAngles(0)
        qd_mes_next = sim.GetMotorVelocities(0)
        cart_pos_next, cart_ori_next = dyn_model.ComputeFK(q_mes_next, controlled_frame_name)
        
        trajectory_data['time'].append(current_time)
        trajectory_data['q_mes'].append(q_mes_next.copy())
        trajectory_data['qd_mes'].append(qd_mes_next.copy())
        trajectory_data['q_des_model'].append(q_des.copy())
        trajectory_data['qd_des_model'].append(qd_des.copy())
        trajectory_data['cart_pos'].append(cart_pos_next.copy())
        trajectory_data['cart_ori'].append(cart_ori_next.copy())
        trajectory_data['tau_cmd'].append(tau_cmd.copy())
    
    # Convert lists to numpy arrays
    for key in ['time', 'q_mes', 'qd_mes', 'q_des_model', 'qd_des_model', 
                'cart_pos', 'cart_ori', 'tau_cmd']:
        trajectory_data[key] = np.array(trajectory_data[key])
    
    return trajectory_data


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_joint_positions_comparison(trajectory_data, pose_idx, save_dir):
    """
    Plot comparison of model-predicted vs measured joint positions.
    
    Args:
        trajectory_data (dict): Trajectory data dictionary
        pose_idx (int): Index of the current pose
        save_dir (Path): Directory to save plots
    """
    time_array = trajectory_data['time']
    q_mes = trajectory_data['q_mes']
    q_des_model = trajectory_data['q_des_model']
    
    num_joints = q_mes.shape[1]
    
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints))
    fig.suptitle(f'Joint Positions Comparison - Pose {pose_idx + 1}', 
                 fontsize=14, fontweight='bold')
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx] if num_joints > 1 else axes
        
        # Plot measured and model-predicted positions
        ax.plot(time_array, q_mes[:, joint_idx], 
                label='Measured', linewidth=2, color='blue', alpha=0.7)
        ax.plot(time_array, q_des_model[:, joint_idx], 
                label='Model Prediction', linewidth=2, color='red', 
                linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'Joint {joint_idx + 1}\nPosition (rad)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'joint_positions_pose_{pose_idx + 1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path.name}")
    plt.close()


def plot_joint_velocities_comparison(trajectory_data, pose_idx, save_dir):
    """
    Plot comparison of model-predicted vs measured joint velocities.
    
    Args:
        trajectory_data (dict): Trajectory data dictionary
        pose_idx (int): Index of the current pose
        save_dir (Path): Directory to save plots
    """
    time_array = trajectory_data['time']
    qd_mes = trajectory_data['qd_mes']
    qd_des_model = trajectory_data['qd_des_model']
    
    num_joints = qd_mes.shape[1]
    
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints))
    fig.suptitle(f'Joint Velocities Comparison - Pose {pose_idx + 1}', 
                 fontsize=14, fontweight='bold')
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx] if num_joints > 1 else axes
        
        # Plot measured and model-predicted velocities
        ax.plot(time_array, qd_mes[:, joint_idx], 
                label='Measured', linewidth=2, color='green', alpha=0.7)
        ax.plot(time_array, qd_des_model[:, joint_idx], 
                label='Model Prediction', linewidth=2, color='orange', 
                linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'Joint {joint_idx + 1}\nVelocity (rad/s)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'joint_velocities_pose_{pose_idx + 1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path.name}")
    plt.close()


def plot_3d_trajectory(trajectory_data, pose_idx, save_dir, interactive=True):
    """
    Plot 3D trajectory of end-effector position.
    
    Args:
        trajectory_data (dict): Trajectory data dictionary
        pose_idx (int): Index of the current pose
        save_dir (Path): Directory to save plots
        interactive (bool): If True, show interactive plot window
    """
    cart_pos = trajectory_data['cart_pos']
    desired_cart_pos = trajectory_data['desired_cart_pos']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x_traj = cart_pos[:, 0]
    y_traj = cart_pos[:, 1]
    z_traj = cart_pos[:, 2]
    
    # Plot trajectory with a single line (much faster than multiple segments)
    # Use a scatter plot with color gradient for time progression
    num_points = len(x_traj)
    time_normalized = np.linspace(0, 1, num_points)
    
    # Plot the trajectory line
    ax.plot(x_traj, y_traj, z_traj, 
            color='cyan', linewidth=2, alpha=0.6, label='Trajectory')
    
    # Add scatter points with color gradient (downsample for performance)
    downsample_factor = max(1, num_points // 50)  # Max 50 colored points
    scatter_indices = np.arange(0, num_points, downsample_factor)
    sc = ax.scatter(x_traj[scatter_indices], y_traj[scatter_indices], z_traj[scatter_indices],
                    c=time_normalized[scatter_indices], cmap='viridis', 
                    s=30, alpha=0.6, edgecolors='none')
    
    # Mark start position
    ax.scatter(x_traj[0], y_traj[0], z_traj[0], 
               color='green', s=200, marker='o', 
               label='Start Position', edgecolors='black', linewidths=2)
    
    # Mark end position
    ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], 
               color='blue', s=200, marker='s', 
               label='End Position', edgecolors='black', linewidths=2)
    
    # Mark desired target position
    ax.scatter(desired_cart_pos[0], desired_cart_pos[1], desired_cart_pos[2], 
               color='red', s=300, marker='*', 
               label='Target Position', edgecolors='black', linewidths=2)
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'3D End-Effector Trajectory - Pose {pose_idx + 1}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to start from zero
    # Find the maximum values to set upper limits
    all_x = np.concatenate([x_traj, [desired_cart_pos[0]]])
    all_y = np.concatenate([y_traj, [desired_cart_pos[1]]])
    all_z = np.concatenate([z_traj, [desired_cart_pos[2]]])
    
    x_max = max(abs(all_x.min()), abs(all_x.max()))
    y_max = max(abs(all_y.min()), abs(all_y.max()))
    z_max = max(0.1, all_z.max())  # Z should always be positive, min 0.1 for visibility
    
    # Set limits with origin at (0, 0, 0)
    ax.set_xlim(-x_max * 1.1, x_max * 1.1)
    ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    ax.set_zlim(0, z_max * 1.1)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save the figure first
    save_path = save_dir / f'3d_trajectory_pose_{pose_idx + 1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path.name}")
    
    # Show interactive plot if requested
    if interactive:
        print(f"  ℹ Showing interactive 3D plot (close window to continue)...")
        plt.show(block=True)  # Block until user closes the window
    
    plt.close()


def plot_cartesian_position_error(trajectory_data, pose_idx, save_dir):
    """
    Plot Cartesian position error over time.
    
    Args:
        trajectory_data (dict): Trajectory data dictionary
        pose_idx (int): Index of the current pose
        save_dir (Path): Directory to save plots
    """
    time_array = trajectory_data['time']
    cart_pos = trajectory_data['cart_pos']
    desired_cart_pos = trajectory_data['desired_cart_pos']
    
    # Compute errors
    errors = cart_pos - desired_cart_pos
    error_norms = np.linalg.norm(errors, axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Cartesian Position Error - Pose {pose_idx + 1}', 
                 fontsize=14, fontweight='bold')
    
    # Plot component-wise errors
    ax1 = axes[0]
    ax1.plot(time_array, errors[:, 0], label='X Error', linewidth=2, color='red')
    ax1.plot(time_array, errors[:, 1], label='Y Error', linewidth=2, color='green')
    ax1.plot(time_array, errors[:, 2], label='Z Error', linewidth=2, color='blue')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Position Error (m)', fontsize=10)
    ax1.set_title('Component-wise Position Error', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot error norm
    ax2 = axes[1]
    ax2.plot(time_array, error_norms, linewidth=2, color='purple')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Position Error Norm (m)', fontsize=10)
    ax2.set_title('Total Position Error Magnitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add final error annotation
    final_error = error_norms[-1]
    ax2.axhline(y=final_error, color='red', linestyle='--', alpha=0.5)
    ax2.text(time_array[-1] * 0.5, final_error * 1.1, 
             f'Final Error: {final_error:.4f} m', 
             fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / f'cartesian_error_pose_{pose_idx + 1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path.name}")
    plt.close()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    """
    Main evaluation function.
    """
    print("\n" + "="*70)
    print("PART 2 MODEL EVALUATION")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # 1. Setup Simulation and Model
    # -------------------------------------------------------------------------
    print("Step 1: Setting up simulation environment...")
    
    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    
    # Get active joint names
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    
    # Create dynamic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, 
                          source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    controlled_frame_name = "panda_link8"
    
    # Get initial joint configuration
    init_joint_angles = sim.GetInitMotorAngles()
    print(f"  ✓ Number of joints: {num_joints}")
    print(f"  ✓ Controlled frame: {controlled_frame_name}")
    print(f"  ✓ Initial joint angles: {init_joint_angles}")
    
    # -------------------------------------------------------------------------
    # 2. Load Trained Model
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading trained model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ Using device: {device}")
    
    if not MODEL_PATH.exists():
        print(f"  ✗ Error: Model file not found at {MODEL_PATH}")
        print("  Please train the model first by running part_2.py")
        return
    
    model = load_trained_model(MODEL_PATH, device)
    
    # -------------------------------------------------------------------------
    # 3. Generate Test Positions
    # -------------------------------------------------------------------------
    print(f"\nStep 3: Generating {NUM_TEST_POSES} test positions...")
    
    desired_positions, desired_orientations, initial_joint_positions = \
        generate_test_positions(NUM_TEST_POSES, init_joint_angles)
    
    for i, pos in enumerate(desired_positions):
        print(f"  Pose {i+1}: Target = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # -------------------------------------------------------------------------
    # 4. Run Simulations and Generate Plots
    # -------------------------------------------------------------------------
    print(f"\nStep 4: Running simulations and generating plots...")
    print(f"  Duration per pose: {DURATION_PER_POSE}s")
    print(f"  Results directory: {RESULTS_DIR}\n")
    
    all_final_errors = []
    
    for i in range(NUM_TEST_POSES):
        print(f"Evaluating Pose {i+1}/{NUM_TEST_POSES}...")
        
        # Run simulation
        trajectory_data = simulate_single_trajectory(
            sim=sim,
            dyn_model=dyn_model,
            model=model,
            device=device,
            desired_cartesian_pos=np.array(desired_positions[i]),
            init_joint_angles=init_joint_angles,
            duration=DURATION_PER_POSE,
            controlled_frame_name=controlled_frame_name
        )
        
        # Compute final error
        final_cart_pos = trajectory_data['cart_pos'][-1]
        target_pos = trajectory_data['desired_cart_pos']
        final_error = np.linalg.norm(final_cart_pos - target_pos)
        all_final_errors.append(final_error)
        
        print(f"  Final position error: {final_error:.4f} m")
        
        # Generate plots
        print(f"  Generating plots...")
        plot_joint_positions_comparison(trajectory_data, i, RESULTS_DIR)
        plot_joint_velocities_comparison(trajectory_data, i, RESULTS_DIR)
        plot_3d_trajectory(trajectory_data, i, RESULTS_DIR, interactive=SHOW_INTERACTIVE_3D)
        plot_cartesian_position_error(trajectory_data, i, RESULTS_DIR)
        print()
    
    # -------------------------------------------------------------------------
    # 5. Summary Statistics
    # -------------------------------------------------------------------------
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Number of test poses: {NUM_TEST_POSES}")
    print(f"Mean final position error: {np.mean(all_final_errors):.4f} m")
    print(f"Std final position error: {np.std(all_final_errors):.4f} m")
    print(f"Min final position error: {np.min(all_final_errors):.4f} m")
    print(f"Max final position error: {np.max(all_final_errors):.4f} m")
    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
