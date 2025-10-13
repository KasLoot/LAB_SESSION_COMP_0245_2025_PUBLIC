import numpy as np
# import time
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

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 12  # seconds
    
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
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the computed torque command
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

        
        # Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        if current_time > 2:
            regressor_all.append(cur_regressor)
            tau_mes_all.append(tau_mes)
        
        
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")

    print("\033[92m=============================== Parameters ================================\033[0m")
    # Stack all the regressors and all the torques, and compute the parameters 'a' using pseudoinverse
    regressor_all = np.vstack(regressor_all)  # Shape (N*7, p(features))
    print(f"Regressor shape: {regressor_all.shape}")
    tau_mes_all = np.vstack(tau_mes_all)      # Shape (N, 7)
    print(f"Torque measurements shape: {tau_mes_all.shape}")
    tau_mes_all_flat = tau_mes_all.flatten()  # Shape (N*7,)
    a = np.linalg.pinv(regressor_all) @ tau_mes_all_flat 
    print(f"Parameters 'a' for the linear model:\n{a}")
    print("\033[92m==================================== MSE ==================================\033[0m")
    tau_pred_all_flat = regressor_all @ a     # Shape (N*7,)
    mse = np.mean((tau_mes_all_flat - tau_pred_all_flat)**2)
    print(f"MSE for the linear model:{mse}")
    print("\033[92m================================= R-squared ===============================\033[0m")
    tss = np.sum((tau_mes_all_flat - np.mean(tau_mes_all_flat))**2)
    rss = np.sum((tau_mes_all_flat - tau_pred_all_flat)**2)
    print(f":rss: {rss}")
    r_squared = 1 - rss/tss
    print(f"R-squared for the linear model: {r_squared}")
    print("\033[92m================================= F-statistic =============================\033[0m")
    n = tau_mes_all_flat.shape[0]
    print(f"Number of samples: {n}")
    p = a.shape[0]
    f_stat = (tss - rss)/(rss/(n-p-1))
    print(f"F-statistic for the linear model: {f_stat}")
    print("\033[92m================= Confidence intervals of the parameters ==================\033[0m")
    # Estimate variance of the residuals
    sigma_squared = rss / (n - p)
    # Compute the covariance matrix of the parameter estimates
    XTX_inv = np.linalg.pinv(regressor_all.T @ regressor_all)
    # Covariance matrix of parameters
    param_var = sigma_squared * XTX_inv
    # Deal with negative variances due to numerical errors
    diag_elements = np.diag(param_var).copy()
    for i in range(len(diag_elements)):
        if diag_elements[i] < 0:
            print(f"All negative variance:\nNegative variance for parameter {i+1}: {diag_elements[i]}")
            diag_elements[i] = 0
    # Standard errors of parameters
    param_se = np.sqrt(diag_elements)
    # Plot confidence intervals for parameters
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(1, p+1), a, yerr=1.96*param_se, fmt='o', c='g', capsize=5, label='95% Confidence Intervals')
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Estimate (a)')
    plt.title('Confidence Intervals of Parameters')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # !Uncomment the following lines to save the plot
    # plt.savefig("images/noise_1/parameters_confidence_intervals.png")
    plt.show()
    
    ##################################################################################################
    # Compute confidence intervals for the prediction
    residuals = tau_mes_all_flat - tau_pred_all_flat
    std_residuals = np.std(residuals)
    lower_bounds_pred = tau_pred_all_flat - 1.96 * std_residuals
    upper_bounds_pred = tau_pred_all_flat + 1.96 * std_residuals
    
    # Plot the torque prediction error for each joint
    tau_pred_all = tau_pred_all_flat.reshape(-1, num_joints)  # Shape (N, 7)
    tau_mes_all = tau_mes_all_flat.reshape(-1, num_joints)
    # Compute the confidence intervals for the prediction
    lower_bounds_pred = lower_bounds_pred.reshape(-1, num_joints)
    upper_bounds_pred = upper_bounds_pred.reshape(-1, num_joints)
    
    samples = np.arange(n/7)
    # Plot the predicted torque and the measured torque for each joint
    _, axs = plt.subplots(num_joints, 1, figsize=(10, 15))
    for i in range(num_joints):
        axs[i].plot(samples, tau_mes_all[:, i], 'r-', label='Measured Torque', linewidth=1)
        axs[i].plot(samples, tau_pred_all[:, i], 'b-', label='Predicted Torque', linewidth=1)
        axs[i].legend()
        axs[i].set_title(f'Joint {i+1} Predicted Torque and Measured Torque')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Torque (Nm)')
        axs[i].grid(True)
    plt.tight_layout()
    # !Uncomment the following lines to save the plot
    # if not os.path.exists("images/noise_0.0001/"):
    #     os.makedirs("images/noise_0.0001/")
    # plt.savefig(f"images/noise_0.0001/prediction_measured.png")
    plt.show()
    
    # Plot Confidence intervals for the prediction of each joint
    _, axs = plt.subplots(num_joints, 1, figsize=(10, 15))
    for i in range(num_joints):
        axs[i].fill_between(samples, lower_bounds_pred[:, i], upper_bounds_pred[:, i], color='green', label='95% Confidence Interval')
        axs[i].set_title(f'Joint {i+1} 95% Confidence Interval')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Torque (Nm)')
        axs[i].grid(True)
    plt.tight_layout()
    # !Uncomment the following lines to save the plot
    # if not os.path.exists("images/noise_0.0001/"):
    #     os.makedirs("images/noise_0.0001/")
    # plt.savefig(f"images/noise_0.0001/prediction_confidence_interval.png")
    plt.show()
        
    print("\033[92m===========================================================================\033[0m")
    # Compute the error
    error_all = tau_mes_all - tau_pred_all  # Shape (N, 7)
    # Plot the error for each joint
    _, axs = plt.subplots(num_joints, 1, figsize=(10, 15))
    for i in range(num_joints):
        axs[i].plot(samples, error_all[:, i])
        axs[i].set_title(f'Torque Prediction Error for Joint {i+1}')
        axs[i].set_xlabel('Time Steps')
        axs[i].set_ylabel('Torque Error (Nm)')
        axs[i].grid(True)
    plt.tight_layout()
    # !Uncomment the following lines to save the plot
    # if not os.path.exists("images/noise_0.0001/"):
    #     os.makedirs("images/noise_0.0001/")
    # plt.savefig(f"images/noise_0.0001/error.png")
    plt.show()
    

if __name__ == '__main__':
    main()
