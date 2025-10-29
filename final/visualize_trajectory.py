import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import sys
from pathlib import Path

# Import the model architecture from part_2.py
from part_2 import P2_MLP

# Import simulation components
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# Get the FINAL_DIR
FINAL_DIR = Path(__file__).resolve().parent


class CartesianDiffKinApproximator:
    """
    Deep learning approximation of the CartesianDiffKin function.
    Uses the trained Part 2 model to predict q_des and qd_des.
    """
    def __init__(self, model_path='part2_best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load the trained model
        self.model = P2_MLP(input_size=10, output_size=14)
        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(torch.float64).to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, q_mes, desired_cartesian_pos):
        """
        Predict q_des and qd_des given current joint angles and desired Cartesian position.
        
        Args:
            q_mes: Current joint angles (numpy array of shape (7,))
            desired_cartesian_pos: Desired Cartesian position (numpy array of shape (3,))
        
        Returns:
            q_des: Desired joint angles (numpy array of shape (7,))
            qd_des: Desired joint velocities (numpy array of shape (7,))
        """
        # Prepare input: concatenate q_mes and desired_cartesian_pos
        input_data = np.concatenate([q_mes, desired_cartesian_pos])
        input_tensor = torch.tensor(input_data, dtype=torch.float64).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert output to numpy
        output_np = output.cpu().numpy().squeeze()
        
        # Split output into q_des and qd_des
        q_des = output_np[:7]
        qd_des = output_np[7:]
        
        return q_des, qd_des


def simulate_trajectory_with_model(
    model_approximator,
    desired_cartesian_pos,
    duration=2.0,
    conf_file_name="pandaconfig.json",
    kp=1000,
    kd=100,
    plot_3d=True,
    save_plot=True,
    position_threshold=0.0,
    early_stop=True
):
    """
    Simulate robot trajectory using the deep learning model to approximate CartesianDiffKin.
    
    Args:
        model_approximator: CartesianDiffKinApproximator instance
        desired_cartesian_pos: Target Cartesian position [x, y, z]
        duration: Duration of the trajectory in seconds
        conf_file_name: Robot configuration file
        kp: Proportional gain for PD controller
        kd: Derivative gain for PD controller
        plot_3d: Whether to plot the 3D trajectory
        save_plot: Whether to save the plot
        position_threshold: Distance threshold to consider target reached (in meters)
        early_stop: Whether to stop simulation when target is reached
    
    Returns:
        Dictionary containing trajectory data
    """
    root_dir = str(FINAL_DIR)
    
    # Initialize simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    
    # Get active joint names
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    
    # Create dynamic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    
    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    print(f"Initial Cartesian position: {init_cartesian_pos}")
    print(f"Desired Cartesian position: {desired_cartesian_pos}")
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    steps = int(duration / time_step)
    
    # Data storage
    q_mes_all = []
    qd_mes_all = []
    q_des_all = []
    qd_des_all = []
    tau_cmd_all = []
    cart_pos_all = []
    cart_ori_all = []
    time_array = []
    
    # Command structure
    cmd = MotorCommands()
    
    # Simulation loop
    current_time = 0
    target_reached = False
    reached_time = None
    print(f"\nSimulating trajectory for max {duration} seconds ({steps} steps)...")
    print(f"Early stop: {early_stop}, Position threshold: {position_threshold}m")
    
    for t in range(steps):
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
        
        # Use deep learning model to approximate CartesianDiffKin
        q_des, qd_des = model_approximator.predict(q_mes, desired_cartesian_pos)
        
        # Clip the desired joint velocities to be within the velocity limits
        qd_des = np.clip(np.array(qd_des), -np.array(joint_vel_limits), np.array(joint_vel_limits))
        
        # Compute control torques using PD controller
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        
        # Apply control command
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
        sim.Step(cmd, "torque")
        
        # Record data
        q_mes_all.append(q_mes.copy())
        qd_mes_all.append(qd_mes.copy())
        q_des_all.append(q_des.copy())
        qd_des_all.append(qd_des.copy())
        tau_cmd_all.append(tau_cmd.copy())
        cart_pos_all.append(cart_pos.copy())
        cart_ori_all.append(cart_ori.copy())
        time_array.append(current_time)
        
        # Check if target is reached
        cart_error = np.linalg.norm(desired_cartesian_pos - cart_pos)
        if cart_error < position_threshold and not target_reached:
            target_reached = True
            reached_time = current_time
            print(f"\n*** Target reached at time {current_time:.2f}s with error {cart_error:.4f}m ***")
            if early_stop:
                print(f"Early stopping at step {t}/{steps}")
                break
        
        # Update time
        current_time += time_step
        
        # Progress update
        if t % 100 == 0:
            print(f"Step {t}/{steps}, Time: {current_time:.2f}s, Cart. Error: {cart_error:.4f}m")
    
    # Final error
    final_cart_pos = cart_pos_all[-1]
    final_error = np.linalg.norm(desired_cartesian_pos - final_cart_pos)
    print(f"\nFinal Cartesian position: {final_cart_pos}")
    print(f"Final Cartesian error: {final_error:.4f}m")
    
    # Convert to numpy arrays
    cart_pos_all = np.array(cart_pos_all)
    
    # Plot 3D trajectory
    if plot_3d:
        plot_3d_trajectory(
            cart_pos_all,
            desired_cartesian_pos,
            init_cartesian_pos,
            save_plot=False  # Don't save yet, will be handled by caller
        )
    
    # Return trajectory data
    trajectory_data = {
        'time': time_array,
        'q_mes': q_mes_all,
        'qd_mes': qd_mes_all,
        'q_des': q_des_all,
        'qd_des': qd_des_all,
        'tau_cmd': tau_cmd_all,
        'cart_pos': cart_pos_all,
        'cart_ori': cart_ori_all,
        'desired_cart_pos': desired_cartesian_pos,
        'init_cart_pos': init_cartesian_pos,
        'final_error': final_error,
        'target_reached': target_reached,
        'reached_time': reached_time
    }
    
    return trajectory_data


def plot_3d_trajectory(cart_pos_all, desired_pos, init_pos, save_plot=True, save_name='3d_trajectory.png'):
    """
    Plot 3D trajectory of the end-effector.
    
    Args:
        cart_pos_all: Array of Cartesian positions (N, 3)
        desired_pos: Desired Cartesian position (3,)
        init_pos: Initial Cartesian position (3,)
        save_plot: Whether to save the plot
        save_name: Name of the saved plot file
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = cart_pos_all[:, 0]
    y = cart_pos_all[:, 1]
    z = cart_pos_all[:, 2]
    
    # Plot the trajectory
    ax.plot(x, y, z, 'b-', linewidth=2, label='End-effector trajectory', alpha=0.7)
    
    # Plot start point
    ax.scatter(init_pos[0], init_pos[1], init_pos[2], 
               c='green', marker='o', s=200, label='Start position', 
               edgecolors='black', linewidths=2)
    
    # Plot end point
    ax.scatter(x[-1], y[-1], z[-1], 
               c='blue', marker='o', s=200, label='Final position',
               edgecolors='black', linewidths=2)
    
    # Plot target point
    ax.scatter(desired_pos[0], desired_pos[1], desired_pos[2], 
               c='red', marker='*', s=400, label='Target position',
               edgecolors='black', linewidths=2)
    
    # Add a line from final position to target
    ax.plot([x[-1], desired_pos[0]], 
            [y[-1], desired_pos[1]], 
            [z[-1], desired_pos[2]], 
            'r--', linewidth=2, alpha=0.5, label='Final error')
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('3D End-Effector Trajectory\n(Model-based Cartesian Differential Kinematics)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_plot:
        save_path = FINAL_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n3D trajectory plot saved to '{save_path}'")
    
    plt.show()


def plot_position_vs_time(trajectory_data, save_plot=True, save_name='position_vs_time.png'):
    """
    Plot Cartesian position components vs time.
    
    Args:
        trajectory_data: Dictionary containing trajectory data
        save_plot: Whether to save the plot
        save_name: Name of the saved plot file
    """
    time = np.array(trajectory_data['time'])
    cart_pos = np.array(trajectory_data['cart_pos'])
    desired_pos = trajectory_data['desired_cart_pos']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, cart_pos[:, i], color=color, linewidth=2, label=f'{label} position')
        ax.axhline(y=desired_pos[i], color=color, linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Target {label}')
        ax.set_ylabel(f'{label} (m)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[0].set_title('Cartesian Position Components vs Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        save_path = FINAL_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position vs time plot saved to '{save_path}'")
    
    plt.show()


def plot_velocity_vs_time(trajectory_data, save_plot=True, save_name='velocity_vs_time.png'):
    """
    Plot joint velocities vs time.
    
    Args:
        trajectory_data: Dictionary containing trajectory data
        save_plot: Whether to save the plot
        save_name: Name of the saved plot file
    """
    time = np.array(trajectory_data['time'])
    qd_mes = np.array(trajectory_data['qd_mes'])
    qd_des = np.array(trajectory_data['qd_des'])
    
    num_joints = qd_mes.shape[1]
    
    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 2.5 * num_joints))
    
    # If only one joint, make axes iterable
    if num_joints == 1:
        axes = [axes]
    
    for i in range(num_joints):
        ax = axes[i]
        
        # Plot measured and desired velocities
        ax.plot(time, qd_mes[:, i], 'b-', linewidth=1.5, label='Measured velocity', alpha=0.7)
        ax.plot(time, qd_des[:, i], 'r--', linewidth=1.5, label='Desired velocity (clipped)', alpha=0.7)
        
        ax.set_ylabel(f'Joint {i+1}\nVelocity (rad/s)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[0].set_title('Joint Velocities vs Time (Measured vs Desired)', 
                      fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_plot:
        save_path = FINAL_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity vs time plot saved to '{save_path}'")
    
    plt.show()


def generate_random_cartesian_targets(num_targets=5, seed=None):
    """
    Generate random Cartesian target positions within robot workspace.
    
    Args:
        num_targets: Number of random targets to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of target positions (num_targets, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    targets = []
    for _ in range(num_targets):
        # Generate positions similar to data_generator.py
        x = np.random.choice([np.random.uniform(0.3, 0.5), np.random.uniform(-0.5, -0.3)])
        y = np.random.choice([np.random.uniform(0.4, 0.5), np.random.uniform(-0.5, -0.4)])
        z = np.random.uniform(0.1, 0.6)
        targets.append(np.array([x, y, z]))

    targets = [[0.5,0.0,0.1],
               [0.4,0.2,0.1], 
               [0.4,-0.2,0.1], 
               [0.5,0.0,0.1]]
    
    return targets


def plot_all_trajectories_3d(all_trajectory_data, save_plot=True):
    """
    Plot all trajectories in a single 3D plot.
    
    Args:
        all_trajectory_data: List of trajectory data dictionaries
        save_plot: Whether to save the plot
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_trajectory_data)))
    
    for i, (traj_data, color) in enumerate(zip(all_trajectory_data, colors)):
        cart_pos = traj_data['cart_pos']
        desired_pos = traj_data['desired_cart_pos']
        init_pos = traj_data['init_cart_pos']
        
        # Extract x, y, z coordinates
        x = cart_pos[:, 0]
        y = cart_pos[:, 1]
        z = cart_pos[:, 2]
        
        # Plot the trajectory
        ax.plot(x, y, z, color=color, linewidth=2, label=f'Trajectory {i+1}', alpha=0.7)
        
        # Plot target point
        ax.scatter(desired_pos[0], desired_pos[1], desired_pos[2], 
                   c=[color], marker='*', s=300, edgecolors='black', linewidths=1.5)
    
    # Plot initial position (should be same for all)
    init_pos = all_trajectory_data[0]['init_cart_pos']
    ax.scatter(init_pos[0], init_pos[1], init_pos[2], 
               c='green', marker='o', s=250, label='Start position', 
               edgecolors='black', linewidths=2, zorder=100)
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'All {len(all_trajectory_data)} Trajectories in 3D Space\n(Model-based Cartesian Differential Kinematics)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        save_path = FINAL_DIR / 'all_trajectories_3d.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAll trajectories 3D plot saved to '{save_path}'")
    
    plt.show()


def main():
    """Main function to run trajectory visualization."""
    
    # Configuration
    num_targets = 5  # Number of random targets to generate
    duration = 5.0   # Maximum duration per trajectory (seconds)
    position_threshold = 0.0  # Distance threshold to consider target reached (meters)
    early_stop = True  # Stop when target is reached
    random_seed = 420   # For reproducibility (set to None for random each time)
    
    # Model path
    model_path = FINAL_DIR / 'part2_best_model_v2.pth'
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have trained the Part 2 model.")
        return
    
    # Initialize model approximator
    print("Initializing CartesianDiffKin approximator...")
    model_approximator = CartesianDiffKinApproximator(model_path=str(model_path))
    
    # Generate random Cartesian targets
    print(f"\nGenerating {num_targets} random Cartesian targets...")
    desired_cartesian_positions = generate_random_cartesian_targets(num_targets, seed=random_seed)
    
    # Store all trajectory data
    all_trajectory_data = []
    
    # Simulate each trajectory
    for i, desired_pos in enumerate(desired_cartesian_positions):
        print("\n" + "="*70)
        print(f"TRAJECTORY {i+1}/{num_targets}")
        print(f"Target Cartesian position: {desired_pos}")
        print("="*70)
        
        # Simulate trajectory
        trajectory_data = simulate_trajectory_with_model(
            model_approximator=model_approximator,
            desired_cartesian_pos=desired_pos,
            duration=duration,
            kp=1000,
            kd=100,
            plot_3d=False,  # Don't plot individual trajectories yet
            save_plot=False,
            position_threshold=position_threshold,
            early_stop=early_stop
        )
        
        all_trajectory_data.append(trajectory_data)
        
        # Print summary
        print(f"\nTrajectory {i+1} Summary:")
        print(f"  - Final error: {trajectory_data['final_error']:.4f}m")
        print(f"  - Target reached: {trajectory_data['target_reached']}")
        if trajectory_data['target_reached']:
            print(f"  - Time to reach: {trajectory_data['reached_time']:.2f}s")
    
    # Now plot all individual trajectories
    print("\n" + "="*70)
    print("PLOTTING INDIVIDUAL TRAJECTORIES")
    print("="*70)
    for i, traj_data in enumerate(all_trajectory_data):
        print(f"\nPlotting trajectory {i+1}/{num_targets}...")
        
        # Plot 3D trajectory
        plot_3d_trajectory(
            traj_data['cart_pos'],
            traj_data['desired_cart_pos'],
            traj_data['init_cart_pos'],
            save_plot=True,
            save_name=f'3d_trajectory_{i+1}.png'
        )
        
        # Plot position vs time
        plot_position_vs_time(
            traj_data,
            save_plot=True,
            save_name=f'position_vs_time_{i+1}.png'
        )
        
        # Plot velocity vs time
        plot_velocity_vs_time(
            traj_data,
            save_plot=True,
            save_name=f'velocity_vs_time_{i+1}.png'
        )
    
    # Plot all trajectories together
    print("\n" + "="*70)
    print("PLOTTING ALL TRAJECTORIES TOGETHER")
    print("="*70)
    plot_all_trajectories_3d(all_trajectory_data, save_plot=True)
    
    # Save all trajectory data
    all_trajectories_file = FINAL_DIR / 'all_trajectories_data.pkl'
    with open(all_trajectories_file, 'wb') as f:
        pickle.dump(all_trajectory_data, f)
    print(f"\nAll trajectory data saved to '{all_trajectories_file}'")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total trajectories simulated: {num_targets}")
    print(f"Targets reached: {sum(t['target_reached'] for t in all_trajectory_data)}/{num_targets}")
    print(f"Average final error: {np.mean([t['final_error'] for t in all_trajectory_data]):.4f}m")
    avg_time = np.mean([t['reached_time'] for t in all_trajectory_data if t['reached_time'] is not None])
    if not np.isnan(avg_time):
        print(f"Average time to reach target: {avg_time:.2f}s")
    print("="*70)


if __name__ == '__main__':
    main()
