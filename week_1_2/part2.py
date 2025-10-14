import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from draw_graphs import draw_plots


class PolynomialReference:
    """Polynomial trajectory: q(t) = a0 + a1*t + a2*t^2 + a3*t^3"""
    def __init__(self, coefficients, init_pos, duration=10):
        """
        coefficients: array of shape (n_joints, 4) containing polynomial coefficients
        init_pos: initial joint positions
        duration: time duration for the trajectory
        """
        self.coefficients = np.array(coefficients)
        self.init_pos = np.array(init_pos)
        self.duration = duration
        
    def get_values(self, t):
        t_normalized = (t % self.duration) / self.duration  # Normalize to [0, 1]
        q = (self.coefficients[:, 0] + 
             self.coefficients[:, 1] * t_normalized + 
             self.coefficients[:, 2] * t_normalized**2 + 
             self.coefficients[:, 3] * t_normalized**3)
        qd = (self.coefficients[:, 1] / self.duration + 
              2 * self.coefficients[:, 2] * t_normalized / self.duration + 
              3 * self.coefficients[:, 3] * t_normalized**2 / self.duration)
        return self.init_pos + q, qd


class StepReference:
    """Step trajectory: switches between positions at regular intervals"""
    def __init__(self, positions, step_duration, init_pos):
        """
        positions: list of target positions (each is an array of joint positions)
        step_duration: time duration for each step
        init_pos: initial joint positions
        """
        self.positions = np.array(positions)
        self.step_duration = step_duration
        self.init_pos = np.array(init_pos)
        self.n_steps = len(positions)
        
    def get_values(self, t):
        step_index = int(t / self.step_duration) % self.n_steps
        q = self.init_pos + self.positions[step_index]
        qd = np.zeros_like(q)  # Zero velocity for step changes
        return q, qd


class CompositeReference:
    """Combination of sinusoidal trajectories with different frequencies"""
    def __init__(self, amplitudes, frequencies, init_pos):
        """
        amplitudes: array of shape (n_components, n_joints)
        frequencies: array of shape (n_components, n_joints)
        init_pos: initial joint positions
        """
        self.amplitudes = np.array(amplitudes)
        self.frequencies = np.array(frequencies)
        self.init_pos = np.array(init_pos)
        
    def get_values(self, t):
        q = np.zeros(len(self.init_pos))
        qd = np.zeros(len(self.init_pos))
        
        for amp, freq in zip(self.amplitudes, self.frequencies):
            q += amp * np.sin(2 * np.pi * freq * t)
            qd += amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        
        return self.init_pos + q, qd


class LinearRampReference:
    """Linear ramp trajectory: smooth linear motion between positions"""
    def __init__(self, target_displacement, ramp_time, init_pos):
        """
        target_displacement: displacement from initial position
        ramp_time: time to complete the ramp
        init_pos: initial joint positions
        """
        self.target = np.array(target_displacement)
        self.ramp_time = ramp_time
        self.init_pos = np.array(init_pos)
        
    def get_values(self, t):
        t_mod = t % (2 * self.ramp_time)
        if t_mod < self.ramp_time:
            # Moving forward
            alpha = t_mod / self.ramp_time
            q = self.init_pos + alpha * self.target
            qd = self.target / self.ramp_time
        else:
            # Moving backward
            alpha = (t_mod - self.ramp_time) / self.ramp_time
            q = self.init_pos + (1 - alpha) * self.target
            qd = -self.target / self.ramp_time
        
        return q, qd

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
                                   ref,  # Generic reference trajectory object
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

    init_pos = sim.GetInitMotorAngles()
    
    # Create different types of trajectories for data collection
    trajectories = []
    traj_names = []
    
    # 1. Sinusoidal trajectories (original)
    n_sinusoidal = 1
    amplitude = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    frequency = np.array([[0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]])
    for i, f in enumerate(frequency):
        trajectories.append(SinusoidalReference(amplitude, f, init_pos))
        traj_names.append(f"Sinusoidal_{i+1}")
    
    # # 2. Polynomial trajectories
    # n_polynomial = 2
    # for i in range(n_polynomial):
    #     coeffs = np.random.randn(7, 4) * np.array([0.1, 0.2, 0.1, 0.05])  # Random coefficients
    #     trajectories.append(PolynomialReference(coeffs, init_pos, duration=10))
    #     traj_names.append(f"Polynomial_{i+1}")
    
    # # 3. Step trajectories
    # n_steps = 2
    # for i in range(n_steps):
    #     positions = [np.random.randn(7) * 0.3 for _ in range(4)]  # 4 different positions
    #     trajectories.append(StepReference(positions, step_duration=2.5, init_pos=init_pos))
    #     traj_names.append(f"Step_{i+1}")
    
    # # 4. Composite (multi-frequency) trajectories
    # n_composite = 2
    # for i in range(n_composite):
    #     n_components = 3
    #     amplitudes = np.random.rand(n_components, 7) * np.array([np.pi/8, np.pi/12, np.pi/8, np.pi/8, np.pi/8, np.pi/8, np.pi/8])
    #     frequencies = np.random.rand(n_components, 7) * 0.5
    #     trajectories.append(CompositeReference(amplitudes, frequencies, init_pos))
    #     traj_names.append(f"Composite_{i+1}")
    
    # # 5. Linear ramp trajectories
    # n_ramps = 2
    # for i in range(n_ramps):
    #     target_disp = np.random.randn(7) * 0.5
    #     trajectories.append(LinearRampReference(target_disp, ramp_time=5.0, init_pos=init_pos))
    #     traj_names.append(f"LinearRamp_{i+1}")
    
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

    # Data collection loop with different trajectory types
    total_trajectories = len(trajectories)
    for i, (ref, traj_name) in enumerate(zip(trajectories, traj_names)):
        print(f"Collecting data for trajectory {i+1}/{total_trajectories}: {traj_name}")
        collect_data_single_trajectory(dyn_model, cmd, sim, ref, kp, kd, time_step, max_time, current_time, regressor_all, tau_mes_all)


    # # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    tau_mes_all = np.array(tau_mes_all)
    regressor_all = np.array(regressor_all)

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    tau_mes_all = np.array(tau_mes_all)[1000:]
    new_tau_mes_all = tau_mes_all.reshape(-1)
    regressor_all = np.array(regressor_all)[1000:]
    new_regressor_all = regressor_all.reshape(-1, num_joints*10)

    a = np.linalg.pinv(new_regressor_all)@new_tau_mes_all
    np.save("./a_part2.npy", a)

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







   
