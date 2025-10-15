import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from draw_graphs import draw_plots

def initialize_robot(conf_file_name, cur_dir, sim: pb.SimInterface, source_names):

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    return dyn_model, num_joints


def collect_data_single_trajectory(dyn_model: PinWrapper, 
                                   cmd: MotorCommands,
                                   sim: pb.SimInterface, 
                                   ref: SinusoidalReference, 
                                   kp, 
                                   kd, 
                                   time_step, 
                                   max_time, 
                                   current_time, 
                                   regressor_all: list, 
                                   tau_mes_all: list,
                                   ):
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


def collect_data():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    dyn_model, num_joints = initialize_robot(conf_file_name=conf_file_name, cur_dir=cur_dir, sim=sim, source_names=source_names)

    traj_var = 1
    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    # frequency = np.random.rand(traj_var, 7)
    frequency = np.array([[0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]])
    ref = [SinusoidalReference(amplitude, f, sim.GetInitMotorAngles()) for f in frequency]  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    print(f"time step: {time_step}")
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
    for i in range(traj_var):
        print(f"Collecting data for trajectory {i+1}/{traj_var} with frequency: {frequency[i]}")
        current_time = 0  # Reset time for each trajectory
        collect_data_single_trajectory(dyn_model, cmd, sim, ref[i], kp, kd, time_step, max_time, current_time, regressor_all, tau_mes_all)


    # # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    tau_mes_all = np.array(tau_mes_all)[1000:]
    regressor_all = np.array(regressor_all)[1000:]
    print(f"\n\n\ntau_mes_all shape: {tau_mes_all.shape}")
    print(f"regressor_all shape: {regressor_all.shape}")

    regressor_6_joints = regressor_all[:, :6, :60]
    print(f"regressor_6_joints shape: {regressor_6_joints.shape}")
    regressor_last_joint = regressor_all[:, 6, :]
    print(f"regressor_last_joint shape: {regressor_last_joint.shape}")

    
    p_l1 = np.array([2.34, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
    p_l2 = np.array([2.36, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
    p_l3 = np.array([2.38, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
    p_l4 = np.array([2.43, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
    p_l5 = np.array([3.5, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
    p_l6 = np.array([1.47, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])

    a_known = np.concatenate((p_l1, p_l2, p_l3, p_l4, p_l5, p_l6))
    print(f"a_known shape: {a_known.shape}")

    

    tau_real = regressor_6_joints @ a_known
    print(f"tau_real shape: {tau_real.shape}")

    tau_real = tau_mes_all[:, :6] - tau_real


    tau_mixed = np.concatenate((tau_real, np.expand_dims(tau_mes_all[:, 6], axis=1)), axis=1)
    print(f"tau_mixed shape: {tau_mixed.shape}")

    a = np.linalg.pinv(np.vstack(regressor_all[:,:,60:])) @ np.hstack(tau_mixed)
    print(f"a computed shape: {a.shape}")

    a_last_joint = a
    print(f"a_last_joint shape: {a_last_joint.shape}")
    print(f"a_last_joint: {a_last_joint}")

    a = np.hstack((a_known, a_last_joint))

    np.save("./a_part1_test.npy", a)

    evaluate_model(sim, dyn_model, cur_dir, a)


def evaluate_model(sim: pb.SimInterface, dyn_model: PinWrapper, cur_dir: str, a: np.ndarray):
    amplitude = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    frequency = np.random.rand(1, 7)

    ref = SinusoidalReference(amplitude, frequency[0], sim.GetInitMotorAngles())  # Initialize the reference
    time_step = sim.GetTimeStep()
    max_time = 10  # seconds
    current_time = 0
    cmd = MotorCommands()  # Initialize command structure for motors
    kp = 1000
    kd = 100
    test_regressor_all = []
    test_tau_mes_all = []
    collect_data_single_trajectory(dyn_model, cmd, sim, ref, kp, kd, time_step, max_time, current_time, test_regressor_all, test_tau_mes_all)
    test_tau_mes_all = np.array(test_tau_mes_all)[1000:]
    test_regressor_all = np.array(test_regressor_all)[1000:]


    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file
    tau_pred = test_regressor_all @ a
    print(f"tau_pred shape: {tau_pred.shape}")
    residuals = test_tau_mes_all - tau_pred
    print(f"residuals shape: {residuals.shape}")

    rss = np.sum(residuals**2)
    tss = np.sum((test_tau_mes_all - np.mean(test_tau_mes_all))**2)
    r_squared = 1 - (rss / tss)
    print(f"R-squared: {r_squared}")

    n = test_regressor_all.shape[0]  # number of observations
    p = test_regressor_all.shape[1]  # number of predictors
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    print(f"Adjusted R-squared: {r_squared_adj}")

    # Compute F-statistic
    mss = tss - rss  # Model sum of squares
    f_statistic = (mss / p) / (rss / (n - p - 1))
    print(f"F-statistic: {f_statistic}")

    draw_plots(test_regressor_all, test_tau_mes_all, a, time_step, cur_dir)



def main():
    collect_data()


if __name__ == '__main__':
    main()
