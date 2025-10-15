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
    # frequencies = [0.49, 0.9, 0.4, 0.84, 0.24, 0.64, 0.03]  # Example frequencies for joints
    # print(f"random frequencies: {frequencies}")

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.random.rand(7)
    print(f"random frequency: {frequency}")
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
        # print(f"Current time in seconds: {current_time:.2f}")

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

    # a = np.linalg.pinv(new_regressor_all)@new_tau_mes_all
    a = np.load("./a_constructed.npy")
    print(f"a shape: {a.shape}")
    # np.save("./a2.npy", a)

    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    isolated_new_regressor_all = regressor_all[:, :-1, :]
    print(isolated_new_regressor_all.shape)
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    
   
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
    plt.show()

if __name__ == '__main__':
    main()
