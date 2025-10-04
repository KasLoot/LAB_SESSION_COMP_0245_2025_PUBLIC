import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
    # frequencies = [0.49, 0.9, 0.4, 0.84, 0.24, 0.64, 0.03]  # Example frequencies for joints


    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        regressor_cur = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)        

        regressor_all.append(regressor_cur)
        tau_mes_all.append(tau_mes)

        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    tau_mes_all = np.array(tau_mes_all)
    new_tau_mes_all = tau_mes_all.reshape(-1)
    regressor_all = np.array(regressor_all)
    new_regressor_all = regressor_all.reshape(-1, num_joints*10)

    print(f"tau_mes_all shape: {tau_mes_all.shape}")
    print(f"new_tau_mes_all shape: {new_tau_mes_all.shape}")

    print(f"regressor_all shape: {regressor_all.shape}")
    print(f"new_regressor_all shape: {new_regressor_all.shape}")

    print(f"pinv regressor_all shape: {np.linalg.pinv(new_regressor_all).shape}")

    a = np.linalg.pinv(new_regressor_all)@new_tau_mes_all
    # a = np.load("./a2.npy")
    print(f"a shape: {a.shape}")
    np.save("./a.npy", a)

    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    isolated_new_regressor_all = regressor_all[:, -1, :]
    print(f"isolated_new_regressor_all shape: {isolated_new_regressor_all.shape}")
    isolated_new_torque_mes_all = tau_mes_all[:, -1]
    print(f"isolated_new_torque_mes_all shape: {isolated_new_torque_mes_all.shape}")

    a2 = a[60:]
    print(f"a2 shape: {a2.shape}")

    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    eval_regressor = isolated_new_regressor_all
    eval_tau = isolated_new_torque_mes_all
    metrics_source = "collected dataset"

    eval_regressor = np.asarray(eval_regressor)
    eval_tau = np.asarray(eval_tau)

    if eval_regressor.ndim != 2 or eval_regressor.shape[1] != a.shape[0]:
        eval_regressor = eval_regressor.reshape(-1, a.shape[0])

    torque_eval = eval_tau.reshape(-1)
    if torque_eval.size != eval_regressor.shape[0]:
        raise ValueError("Mismatch between evaluation regressor rows and torque samples")

    torque_pred_eval = eval_regressor @ a2
    residuals = torque_eval - torque_pred_eval

    sample_count = eval_regressor.shape[0]
    param_count = eval_regressor.shape[1]

    ss_res = np.dot(residuals, residuals)
    centered = torque_eval - torque_eval.mean()
    ss_tot = np.dot(centered, centered)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    if sample_count > param_count + 1:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (sample_count - 1) / (sample_count - param_count - 1)
    else:
        adjusted_r_squared = r_squared
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(ss_res / sample_count) if sample_count else 0.0

    metrics_output_path = os.path.join(cur_dir, "linear_model_metrics.txt")
    metrics_lines = [
        f"source: {metrics_source}",
        f"samples: {sample_count}",
        f"parameters: {param_count}",
        f"r_squared: {r_squared:.6f}",
        f"adjusted_r_squared: {adjusted_r_squared:.6f}",
        f"mae: {mae:.6f}",
        f"rmse: {rmse:.6f}",
    ]
    with open(metrics_output_path, "w", encoding="ascii") as metrics_file:
        metrics_file.write("\n".join(metrics_lines))

    print(f"Saved linear model metrics to {metrics_output_path}")

    # TODO plot the torque prediction error for each joint (optional)
    reg_pred = regressor_all.transpose((1, 0, 2))@a
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
    plot1_path = os.path.join(cur_dir, "torque_prediction_vs_measurement.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot1_path}")
    # plt.show()

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
    plot2_path = os.path.join(cur_dir, "torque_prediction_error.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot2_path}")
    # plt.show()

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
    plot3_path = os.path.join(cur_dir, "torque_error_distribution.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot3_path}")
    # plt.show()

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
    plot4_path = os.path.join(cur_dir, "error_statistics_comparison.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot4_path}")
    # plt.show()

if __name__ == '__main__':
    main()
