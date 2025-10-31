import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin
import torch
import sys
from pathlib import Path

# Import the MLP model from part_2.py
from part_2 import P2_MLP, P2_MLP_Previous

FINAL_DIR = Path(__file__).resolve().parent


def evaluate_model_in_simulation(
    model_path,
    num_test_poses=10,
    model_class=P2_MLP_Previous,
    successive_time_series_error_sigma=1.0,
    visualize=False,
    seed=100
):
    """
    Evaluate a trained MLP model in the simulation environment.
    
    Args:
        model_path: Path to the saved model weights
        num_test_poses: Number of random target poses to test
        model_class: The model class to instantiate (P2_MLP or P2_MLP_Previous)
        successive_time_series_error_sigma: Sigma parameter for the error metric
        visualize: Whether to show plots of trajectories
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize simulation
    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    
    # Get active joint names
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    
    # Create dynamic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    print(f"Initial cartesian position: {init_cartesian_pos}")
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(input_size=10, output_size=14)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(torch.float64).to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # PD controller gains for low-level control
    kp = 1000
    kd = 100
    
    # Time step
    time_step = sim.GetTimeStep()
    
    # Storage for evaluation metrics
    all_successive_errors = []
    all_final_distances = []
    all_trajectories = []
    
    # Generate random test poses
    test_poses = []
    for _ in range(num_test_poses):
        x = np.random.uniform(0.2, 0.5)
        y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
        z = np.random.uniform(0.1, 0.6)
        test_poses.append([x, y, z])
    
    print(f"\nEvaluating model on {num_test_poses} random target poses...")
    
    # Evaluate each test pose
    for pose_idx, desired_cartesian_pos in enumerate(test_poses):
        desired_cartesian_pos = np.array(desired_cartesian_pos)
        
        # Reset simulation to initial pose
        sim.ResetPose()
        sim.SetjointPosition(init_joint_angles)
        
        # Initialize command
        cmd = MotorCommands()
        
        # Storage for trajectory data
        cart_pos_trajectory = []
        q_mes_trajectory = []
        time_trajectory = []
        
        # Maximum duration and steps
        max_duration = 5.0  # seconds
        max_steps = int(max_duration / time_step)
        
        current_time = 0
        
        # Run simulation with model predictions
        for step in range(max_steps):
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            
            # Store trajectory data
            cart_pos_trajectory.append(cart_pos.copy())
            q_mes_trajectory.append(q_mes.copy())
            time_trajectory.append(current_time)
            
            # Prepare model input: [q_mes (7), desired_cartesian_pos (3)]
            model_input = torch.cat([
                torch.tensor(q_mes, dtype=torch.float64),
                torch.tensor(desired_cartesian_pos, dtype=torch.float64)
            ]).unsqueeze(0).to(device)
            
            # Get model prediction: [q_des (7), qd_des (7)]
            with torch.no_grad():
                model_output = model(model_input)
            
            q_des = model_output[0, :7].cpu().numpy()
            qd_des = model_output[0, 7:].cpu().numpy()
            
            # Apply PD control
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")
            
            # Check convergence
            # cart_distance = np.linalg.norm(cart_pos - desired_cartesian_pos)
            # if cart_distance < 1e-3:
            #     print(f"Pose {pose_idx + 1}/{num_test_poses}: Converged at step {step} (time={current_time:.3f}s)")
            #     break
            
            current_time += time_step
            time.sleep(time_step)
        
        # Convert to numpy arrays
        cart_pos_trajectory = np.array(cart_pos_trajectory)
        q_mes_trajectory = np.array(q_mes_trajectory)
        time_trajectory = np.array(time_trajectory)
        
        # Calculate successive time series error
        # successive_time_series_error = sum((|current_cart_pos - target_cart_pos| * sigma)^2)
        errors = cart_pos_trajectory - desired_cartesian_pos
        distances = np.linalg.norm(errors, axis=1)
        successive_error = np.sum((distances * successive_time_series_error_sigma) ** 2)
        
        # Calculate final distance to target
        final_distance = np.linalg.norm(cart_pos_trajectory[-1] - desired_cartesian_pos)
        
        all_successive_errors.append(successive_error)
        all_final_distances.append(final_distance)
        all_trajectories.append({
            'cart_pos': cart_pos_trajectory,
            'q_mes': q_mes_trajectory,
            'time': time_trajectory,
            'target': desired_cartesian_pos,
            'successive_error': successive_error,
            'final_distance': final_distance
        })
        
        print(f"Pose {pose_idx + 1}/{num_test_poses}: "
              f"Successive Error = {successive_error:.6f}, "
              f"Final Distance = {final_distance:.6f}")
    
    # Compute summary statistics
    results = {
        'successive_errors': all_successive_errors,
        'final_distances': all_final_distances,
        'mean_successive_error': np.mean(all_successive_errors),
        'std_successive_error': np.std(all_successive_errors),
        'mean_final_distance': np.mean(all_final_distances),
        'std_final_distance': np.std(all_final_distances),
        'max_successive_error': np.max(all_successive_errors),
        'min_successive_error': np.min(all_successive_errors),
        'trajectories': all_trajectories,
        'sigma': successive_time_series_error_sigma,
        'num_test_poses': num_test_poses
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of test poses: {num_test_poses}")
    print(f"Sigma parameter: {successive_time_series_error_sigma}")
    print(f"\nSuccessive Time Series Error:")
    print(f"  Mean:  {results['mean_successive_error']:.6f}")
    print(f"  Std:   {results['std_successive_error']:.6f}")
    print(f"  Min:   {results['min_successive_error']:.6f}")
    print(f"  Max:   {results['max_successive_error']:.6f}")
    print(f"\nFinal Distance to Target:")
    print(f"  Mean:  {results['mean_final_distance']:.6f}")
    print(f"  Std:   {results['std_final_distance']:.6f}")
    print("="*60)
    
    # Visualize if requested
    if visualize:
        visualize_results(results)
    
    return results


def visualize_results(results):
    """
    Visualize evaluation results including error distributions and sample trajectories.
    """
    trajectories = results['trajectories']
    
    # Plot 1: Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(results['successive_errors'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(results['mean_successive_error'], color='red', linestyle='--', 
                    label=f'Mean: {results["mean_successive_error"]:.4f}')
    axes[0].set_xlabel('Successive Time Series Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Successive Time Series Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(results['final_distances'], bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(results['mean_final_distance'], color='red', linestyle='--',
                    label=f'Mean: {results["mean_final_distance"]:.4f}')
    axes[1].set_xlabel('Final Distance to Target (m)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Final Distance to Target')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FINAL_DIR / 'evaluation_error_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved error distribution plot to {FINAL_DIR / 'evaluation_error_distributions.png'}")
    plt.show()
    
    # Plot 2: Sample trajectories (best, median, worst)
    sorted_indices = np.argsort(results['successive_errors'])
    best_idx = sorted_indices[0]
    median_idx = sorted_indices[len(sorted_indices) // 2]
    worst_idx = sorted_indices[-1]
    
    fig = plt.figure(figsize=(15, 5))
    
    for plot_idx, (traj_idx, title) in enumerate([
        (best_idx, 'Best (Lowest Error)'),
        (median_idx, 'Median Error'),
        (worst_idx, 'Worst (Highest Error)')
    ]):
        traj = trajectories[traj_idx]
        ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')
        
        # Plot trajectory
        cart_pos = traj['cart_pos']
        ax.plot(cart_pos[:, 0], cart_pos[:, 1], cart_pos[:, 2], 
                'b-', linewidth=2, label='Trajectory')
        
        # Plot start and end points
        ax.scatter(cart_pos[0, 0], cart_pos[0, 1], cart_pos[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(cart_pos[-1, 0], cart_pos[-1, 1], cart_pos[-1, 2],
                  c='blue', s=100, marker='x', label='End')
        
        # Plot target
        target = traj['target']
        ax.scatter(target[0], target[1], target[2],
                  c='red', s=100, marker='*', label='Target')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"{title}\nError: {traj['successive_error']:.4f}\n"
                    f"Final Dist: {traj['final_distance']:.6f}m")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FINAL_DIR / 'evaluation_sample_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to {FINAL_DIR / 'evaluation_sample_trajectories.png'}")
    plt.show()
    
    # Plot 3: Distance to target over time for sample trajectories
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for plot_idx, (traj_idx, title) in enumerate([
        (best_idx, 'Best'),
        (median_idx, 'Median'),
        (worst_idx, 'Worst')
    ]):
        traj = trajectories[traj_idx]
        distances = np.linalg.norm(traj['cart_pos'] - traj['target'], axis=1)
        
        axes[plot_idx].plot(traj['time'], distances, 'b-', linewidth=2)
        axes[plot_idx].set_xlabel('Time (s)')
        axes[plot_idx].set_ylabel('Distance to Target (m)')
        axes[plot_idx].set_title(f"{title} Trajectory")
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(FINAL_DIR / 'evaluation_distance_over_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved distance-over-time plot to {FINAL_DIR / 'evaluation_distance_over_time.png'}")
    plt.show()


def compare_models(model_paths, model_names, num_test_poses=10, sigma=1.0):
    """
    Compare multiple models using the same test poses.
    
    Args:
        model_paths: List of paths to model weights
        model_names: List of names for the models
        num_test_poses: Number of test poses to evaluate
        sigma: Sigma parameter for successive error calculation
    """
    results_list = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        results = evaluate_model_in_simulation(
            model_path=model_path,
            num_test_poses=num_test_poses,
            successive_time_series_error_sigma=sigma,
            visualize=False
        )
        results['model_name'] = model_name
        results_list.append(results)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean successive errors
    means = [r['mean_successive_error'] for r in results_list]
    stds = [r['std_successive_error'] for r in results_list]
    x_pos = np.arange(len(model_names))
    
    axes[0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Successive Time Series Error')
    axes[0].set_title('Model Comparison: Successive Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean final distances
    means_dist = [r['mean_final_distance'] for r in results_list]
    stds_dist = [r['std_final_distance'] for r in results_list]
    
    axes[1].bar(x_pos, means_dist, yerr=stds_dist, capsize=5, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('Final Distance (m)')
    axes[1].set_title('Model Comparison: Final Distance to Target')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FINAL_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {FINAL_DIR / 'model_comparison.png'}")
    plt.show()
    
    return results_list


if __name__ == "__main__":
    # Example usage: Evaluate a single model
    model_path = FINAL_DIR / "part2_best_model_with_stop_encoder_design.pth"
    
    if model_path.exists():
        results = evaluate_model_in_simulation(
            model_path=str(model_path),
            num_test_poses=10,
            model_class=P2_MLP,  # Change to P2_MLP if using the newer model
            successive_time_series_error_sigma=0.95,
            visualize=True,
            seed=100
        )
        
        # Save results
        # results_file = FINAL_DIR / 'simulation_evaluation_results.pt'
        # torch.save(results, results_file)
        # print(f"\nResults saved to {results_file}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable to point to your trained model.")
    
    # Example: Compare multiple models
    # Uncomment and modify the following to compare different models:
    """
    model_paths = [
        FINAL_DIR / "part2_best_model_with_stop.pth",
        FINAL_DIR / "part2_best_model_no_stop.pth",
    ]
    model_names = ["With Stop Data", "No Stop Data"]
    
    comparison_results = compare_models(
        model_paths=model_paths,
        model_names=model_names,
        num_test_poses=10,
        sigma=1.0
    )
    """
