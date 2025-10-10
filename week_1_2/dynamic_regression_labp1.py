import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv

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
    # prqint(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
    
    link_true_paras=[[2.34, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [2.36, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [2.38, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [2.43, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [3.50, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [1.47, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3],
                    [0.45, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3]]


    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 2 # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = np.array([0]*7)                   #u 7t*1
    regressor_all = np.array([0]*490).reshape(7,70) #Y 7t*70

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)                       # measured values
        qd_mes = sim.GetMotorVelocities(0)                  # measured values
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        
        tau_mes = sim.GetMotorTorques(0)        # measured torque
        # print(tau_mes)
        tau_mes_all = np.concatenate((tau_mes_all,tau_mes))

        #regressor for all
        regressor = np.array(dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes))
        # print(regressor)
        regressor_all = np.vstack((regressor_all,regressor))


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
        
        
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joints
    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
  
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    
   
    # TODO plot the torque prediction error for each joint (optional)

    print(f"regressor_all shape: {regressor_all.shape}")    

    # --- Select data for the 7th joint only ---
    Y = regressor_all[7:]        # Skip initialization
    u = tau_mes_all[7:]          # Measured torques (flat array)

    print(f"Y shape: {Y.shape}")
    print(f"u shape: {u.shape}")

    # Each regressor row corresponds to 7 joints stacked vertically,
    # so extract only the 7th joint’s regressor rows.
    # Each timestep contributes 7 rows in regressor_all, one per joint.
    num_joints = 7
    joint_index = 6  # index for the 7th joint (0-based)

    # Select regressor and torque for the 7th joint
    Y7 = Y[joint_index::num_joints, joint_index::num_joints]     # every 7th row starting at joint 7
    u7 = u[joint_index::num_joints].reshape(-1, 1)

    print(f"Y7 shape: {Y7.shape}")
    print(f"u7 shape: {u7.shape}")

    # --- Fit Linear Regression model ---
    model = LinearRegression(fit_intercept=False)
    model.fit(Y7, u7)

    # --- Compute predictions and errors ---
    a_pred = model.coef_
    a_full = np.concatenate((np.array(link_true_paras[:6]).reshape(-1,1),a_pred.reshape(-1,1)))
    u7_pred = (Y @ a_full)[joint_index::num_joints]
    # u7_pred2 = model.predict(Y7)
    error = u7.flatten() - u7_pred.flatten()    
    # error2 = u7.flatten() - u7_pred2.flatten()

    # --- Print results ---
    print("\nEstimated physical parameters for Joint 7:")
    print(model.coef_.flatten())
    print(f"R² score (Joint 7): {model.score(Y7, u7):.4f}")

    # Compare with known true parameters
    PhyParams_true = link_true_paras[6]
    PhyParams_calc = model.coef_.flatten() # last 10 params correspond to link 7
    err_params = PhyParams_calc - PhyParams_true
    print("\nParameter estimation error (Joint 7):")
    print(err_params)

    # --- Plot torque prediction error ---
    time_axis = np.linspace(0, max_time, len(error))

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis[50:], error[50:], label='Torque Prediction Error (Joint 7)')
    # plt.plot(time_axis[50:], error2[50:], label='Torque Prediction Error2 (Joint 7)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Torque Error [Nm]")
    plt.title("Torque Prediction Error for Joint 7")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


