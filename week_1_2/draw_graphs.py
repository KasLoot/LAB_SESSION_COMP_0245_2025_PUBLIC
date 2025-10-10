import os
import matplotlib.pyplot as plt
import numpy as np

def draw_plots(regressor_all: np.ndarray, tau_mes_all: np.ndarray, a_constructed: np.ndarray, time_step: float, cur_dir: str):

    # TODO plot the torque prediction error for each joint (optional)
    reg_pred = regressor_all.transpose((1, 0, 2))@a_constructed
    reg_pred = np.expand_dims(reg_pred, axis=-1)
    tau_mes_all = np.expand_dims(tau_mes_all.transpose((1, 0)), axis=-1)
    print(f"reg_pred shape: {reg_pred.shape}")
    print(f"tau_mes_all shape: {tau_mes_all.shape}")

    # Plot torque prediction vs measurement for each joint
    fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    fig.suptitle('Torque Prediction vs Measurement for Each Joint', fontsize=16)
    
    # Time array for x-axis
    time_array = np.arange(reg_pred.shape[1]) * time_step
    
    for joint_idx in range(7):
        ax = axes[joint_idx]
        
        # Extract prediction and measurement for this joint
        torque_pred = reg_pred[joint_idx, :, 0]
        torque_meas = tau_mes_all[joint_idx, :, 0]
        
        # Plot both curves
        ax.plot(time_array, torque_meas, 'b-', label='Measured', linewidth=1.5)
        ax.plot(time_array, torque_pred, 'r--', label='Predicted', linewidth=1.5)
        
        # Calculate and plot error
        error = torque_pred - torque_meas
        
        # Set labels and title
        ax.set_ylabel(f'Torque (Nm)', fontsize=10)
        ax.set_title(f'Joint {joint_idx + 1} (Error RMS: {np.sqrt(np.mean(error**2)):.4f} Nm)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if joint_idx == 6:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
    plot1_path = os.path.join(cur_dir, "part1_torque_prediction_vs_measurement.png")
    # plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    # print(f"Saved plot to {plot1_path}")
    plt.show()

    # Plot prediction errors for each joint
    fig2, axes2 = plt.subplots(7, 1, figsize=(12, 14))
    fig2.suptitle('Torque Prediction Error for Each Joint', fontsize=16)
    
    for joint_idx in range(7):
        ax = axes2[joint_idx]
        
        # Extract prediction and measurement for this joint
        torque_pred = reg_pred[joint_idx, :, 0]
        torque_meas = tau_mes_all[joint_idx, :, 0]
        
        # Calculate error
        error = torque_pred - torque_meas
        
        # Plot error
        ax.plot(time_array, error, 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Calculate statistics
        error_mean = np.mean(error)
        error_std = np.std(error)
        error_rms = np.sqrt(np.mean(error**2))
        
        # Set labels and title
        ax.set_ylabel(f'Error (Nm)', fontsize=10)
        ax.set_title(f'Joint {joint_idx + 1} (Mean: {error_mean:.4f}, Std: {error_std:.4f}, RMS: {error_rms:.4f} Nm)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if joint_idx == 6:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
    plot2_path = os.path.join(cur_dir, "part1_torque_prediction_error.png")
    # plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    # print(f"Saved plot to {plot2_path}")
    plt.show()

    # Plot error distribution histograms
    fig3, axes3 = plt.subplots(7, 1, figsize=(12, 14))
    fig3.suptitle('Torque Prediction Error Distribution for Each Joint', fontsize=16)
    
    for joint_idx in range(7):
        ax = axes3[joint_idx]
        
        # Extract prediction and measurement for this joint
        torque_pred = reg_pred[joint_idx, :, 0]
        torque_meas = tau_mes_all[joint_idx, :, 0]
        
        # Calculate error
        error = torque_pred - torque_meas
        
        # Plot histogram
        ax.hist(error, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
        
        # Calculate statistics
        error_mean = np.mean(error)
        error_std = np.std(error)
        
        # Set labels and title
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Joint {joint_idx + 1} (Mean: {error_mean:.4f}, Std: {error_std:.4f} Nm)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Only show x-label on bottom plot
        if joint_idx == 6:
            ax.set_xlabel('Error (Nm)', fontsize=10)
    
    plt.tight_layout()
    plot3_path = os.path.join(cur_dir, "part1_torque_error_distribution.png")
    # plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    # print(f"Saved plot to {plot3_path}")
    plt.show()

    # Plot overall error statistics comparison across joints
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle('Error Statistics Comparison Across Joints', fontsize=16)
    
    # Calculate statistics for all joints
    joint_labels = [f'Joint {i+1}' for i in range(7)]
    rms_errors = []
    mae_errors = []
    std_errors = []
    
    for joint_idx in range(7):
        torque_pred = reg_pred[joint_idx, :, 0]
        torque_meas = tau_mes_all[joint_idx, :, 0]
        error = torque_pred - torque_meas
        
        rms_errors.append(np.sqrt(np.mean(error**2)))
        mae_errors.append(np.mean(np.abs(error)))
        std_errors.append(np.std(error))
    
    # Plot RMS and MAE
    x_pos = np.arange(7)
    width = 0.35
    
    ax1.bar(x_pos - width/2, rms_errors, width, label='RMS Error', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, mae_errors, width, label='MAE', color='coral', alpha=0.8)
    ax1.set_xlabel('Joint', fontsize=12)
    ax1.set_ylabel('Error (Nm)', fontsize=12)
    ax1.set_title('RMS Error and MAE per Joint', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(joint_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot standard deviation
    ax2.bar(x_pos, std_errors, color='mediumseagreen', alpha=0.8)
    ax2.set_xlabel('Joint', fontsize=12)
    ax2.set_ylabel('Standard Deviation (Nm)', fontsize=12)
    ax2.set_title('Error Standard Deviation per Joint', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(joint_labels, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot4_path = os.path.join(cur_dir, "part1_error_statistics_comparison.png")
    # plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    # print(f"Saved plot to {plot4_path}")
    plt.show()