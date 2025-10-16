import numpy as np
from simulation_and_control import SinusoidalReference

class SinusoidalReference:
    def __init__(self, amplitude, frequency, q_init):
        self.amplitude = np.array(amplitude)
        self.frequency = np.array(frequency)
        self.q_init = np.array(q_init)

         # Check if all arrays have the same length
        if not (self.amplitude.size == self.frequency.size == self.q_init.size):
            expected_num_elements = self.q_init.size
            raise ValueError(f"All arrays must have the same number of elements. "
                             f"Expected number of elements (joints): {expected_num_elements}, "
                             f"Received - Amplitude: {self.amplitude.size}, "
                             f"Frequency: {self.frequency.size}, Q_init: {expected_num_elements}.")

        self.phase = np.full(self.q_init.shape, -np.pi / 2)
            

    def check_sinusoidal_feasibility(self, sim):
        """
        Check if the sinusoidal motion with the given amplitude is feasible within the joint limits.

        Args:
        amplitude (float or list of floats): The amplitude(s) of the sinusoidal motion.
        sim (object): Simulation object that provides joint limits and initial joint angles.

        Returns:
        bool: True if sinusoidal motion is feasible for all joints, False otherwise.
        """
        # Retrieve joint limits and initial angles from the simulation object
        lower_limits, upper_limits = sim.GetBotJointsLimit()
        velocity_limits = sim.GetBotJointsVelLimit()  # Retrieve velocity limits for all joints
        init_angles = sim.GetInitMotorAngles()
        
        # Iterate over each joint to check the feasibility of the given amplitude
        for i, init_angle in enumerate(init_angles):
            min_angle = init_angle - self.amplitude[i]
            max_angle = init_angle + self.amplitude[i]
            max_velocity_required = self.amplitude[i] * self.frequency[i] * 2 * np.pi
            
            if min_angle < lower_limits[i] or max_angle > upper_limits[i]:
                print(f"Joint {i+1}: Sinusoidal motion not possible. Limits exceeded.")
                print(f"    Expected range: {min_angle} to {max_angle}")
                print(f"    Actual limits: {lower_limits[i]} to {upper_limits[i]}")
                return False
            # Check if the maximum velocity is within limits
            if max_velocity_required > velocity_limits[i]:
                print(f"Joint {i+1}: Sinusoidal motion velocity too high. Maximum velocity exceeded.")
                print(f"    Required velocity: {max_velocity_required}")
                print(f"    Velocity limit: {velocity_limits[i]}")
                return False

        print("Sinusoidal motion is possible within the joint limits for all joints.")
        return True

    def get_values(self, time):
        """
        Calculate the position and velocity at a given time.

        Parameters:
        time (float or np.array): The time at which to evaluate the position and velocity.

        Returns:
        tuple: The position and velocity at the given time.
        """
        # Calculate the sinusoidal position around the initial position
        q_d = self.q_init + self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase)
        # Calculate the derivative of the position (velocity)
        qd_d = self.amplitude * 2 * np.pi * self.frequency * np.cos(2 * np.pi * self.frequency * time + self.phase)
        return q_d, qd_d

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

class FourierSeriesReference:
    """
    Generates a trajectory for a set of joints using a finite Fourier series.
    This allows for more complex and exciting motions for system identification.
    """
    def __init__(self, q_init, amplitudes, fundamental_frequency, num_harmonics):
        """
        Initializes the Fourier Series trajectory generator.

        Args:
            q_init (np.array): Array of initial/center positions for each joint.
            amplitudes (np.array): A 3D array of shape (num_joints, num_harmonics, 2)
                                  containing the amplitudes for the sin (a_ik) and cos (b_ik) terms.
            fundamental_frequency (float): The fundamental frequency (omega_f) in Hz for the series.
            num_harmonics (int): The number of harmonics (N) to sum in the series.
        """
        self.q_init = np.array(q_init)
        self.amplitudes = np.array(amplitudes)
        self.omega_f = 2 * np.pi * fundamental_frequency
        self.num_harmonics = num_harmonics
        self.num_joints = self.q_init.size

        # --- Input Validation ---
        if self.amplitudes.shape != (self.num_joints, self.num_harmonics, 2):
            raise ValueError(f"Amplitudes array has incorrect shape. "
                             f"Expected: {(self.num_joints, self.num_harmonics, 2)}, "
                             f"Received: {self.amplitudes.shape}")

    def get_values(self, time):
        """
        Calculate the position, velocity, and acceleration at a given time.

        Args:
            time (float or np.array): The time at which to evaluate the trajectory.

        Returns:
            tuple: The position (q_d), velocity (qd_d), and acceleration (qdd_d) at the given time.
        """
        # Ensure time is a numpy array for broadcasting
        time = np.asarray(time)
        if time.ndim == 0:
            time = time[np.newaxis] # Make it a 1D array if it's a scalar

        # Initialize outputs
        q_d = np.tile(self.q_init, (time.size, 1)).T
        qd_d = np.zeros((self.num_joints, time.size))
        qdd_d = np.zeros((self.num_joints, time.size))

        # Sum the contribution of each harmonic
        for k in range(1, self.num_harmonics + 1):
            a_k = self.amplitudes[:, k-1, 0][:, np.newaxis]
            b_k = self.amplitudes[:, k-1, 1][:, np.newaxis]
            
            angle = k * self.omega_f * time
            sin_term = np.sin(angle)
            cos_term = np.cos(angle)
            
            # Position
            q_d += a_k * sin_term + b_k * cos_term
            
            # Velocity
            qd_d += k * self.omega_f * (a_k * cos_term - b_k * sin_term)
            
            # Acceleration
            qdd_d += (k * self.omega_f)**2 * (-a_k * sin_term - b_k * cos_term)
            
        # return q_d.squeeze(), qd_d.squeeze(), qdd_d.squeeze()
        return q_d.squeeze(), qd_d.squeeze()


    def check_fourier_feasibility(self, sim, num_samples=1000):
        """
        Check if the Fourier series motion is feasible within the joint limits.
        This is done by simulating one full period of the trajectory and checking the extrema.

        Args:
            sim (object): A simulation object with GetBotJointsLimit() and GetBotJointsVelLimit().
            num_samples (int): The number of points to sample over one period for checking.

        Returns:
            bool: True if the trajectory is feasible, False otherwise.
        """
        lower_limits, upper_limits = sim.GetBotJointsLimit()
        velocity_limits = sim.GetBotJointsVelLimit()
        
        # The period is determined by the fundamental frequency
        period = 2 * np.pi / self.omega_f
        t_eval = np.linspace(0, period, num_samples)
        
        # Get the trajectory values over the entire period
        q_traj, qd_traj, _ = self.get_values(t_eval)

        # Find the min/max for each joint
        min_angles = np.min(q_traj, axis=1)
        max_angles = np.max(q_traj, axis=1)
        max_velocities = np.max(np.abs(qd_traj), axis=1)
        
        is_feasible = True
        for i in range(self.num_joints):
            if min_angles[i] < lower_limits[i] or max_angles[i] > upper_limits[i]:
                print(f"Joint {i+1}: Trajectory position limits exceeded.")
                print(f"    Calculated range: {min_angles[i]:.3f} to {max_angles[i]:.3f}")
                print(f"    Allowed limits:   {lower_limits[i]:.3f} to {upper_limits[i]:.3f}")
                is_feasible = False
            
            if max_velocities[i] > velocity_limits[i]:
                print(f"Joint {i+1}: Trajectory velocity limits exceeded.")
                print(f"    Calculated max velocity: {max_velocities[i]:.3f}")
                print(f"    Allowed velocity limit:  {velocity_limits[i]:.3f}")
                is_feasible = False

        if is_feasible:
            print("Fourier series trajectory is feasible within all joint limits.")
            
        return is_feasible
    
def generate_fourier_series_trajectory(init_pos, num_harmonics=5, fundamental_frequency=0.1):
    """
    Generates a randomized Fourier Series trajectory object for a 7-joint robot.
    
    Amplitudes for higher harmonics are scaled down to ensure smooth motion.

    Args:
        init_pos (np.array): The initial/center position for the 7 joints.
        num_harmonics (int): The number of harmonics to include in the series.
        fundamental_frequency (float): The base frequency in Hz.

    Returns:
        FourierSeriesReference: An instance of the trajectory class.
    """
    num_joints = len(init_pos)
    
    # Base amplitudes define the overall magnitude of motion for each joint
    base_amplitudes = np.array([np.pi/8, np.pi/10, np.pi/8, np.pi/8, np.pi/6, np.pi/6, np.pi/6])
    if len(base_amplitudes) != num_joints:
        raise ValueError(f"Length of base_amplitudes must match num_joints ({num_joints}).")

    amplitudes = np.zeros((num_joints, num_harmonics, 2))

    for i in range(num_joints):
        for k in range(num_harmonics):
            harmonic_k = k + 1
            # Scale down amplitudes for higher frequencies to ensure smoothness
            scale = base_amplitudes[i] / harmonic_k
            
            # Randomly generate sine (a_ik) and cosine (b_ik) coefficients
            a_ik = (np.random.rand() - 0.5) * 2 * scale 
            b_ik = (np.random.rand() - 0.5) * 2 * scale
            
            amplitudes[i, k, 0] = a_ik
            amplitudes[i, k, 1] = b_ik
            
    traj = FourierSeriesReference(init_pos, amplitudes, fundamental_frequency, num_harmonics)
    return traj

def generate_sinusoidal_trajectory(init_pos):
    amplitude = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    frequency = np.array(np.random.rand(7))
    traj = SinusoidalReference(amplitude, frequency, init_pos)
    return traj

def generate_polynomial_trajectory(init_pos):
    coeffs = np.random.randn(7, 4) * np.array([0.1, 0.2, 0.1, 0.05])  # Random coefficients
    traj = PolynomialReference(coeffs, init_pos, duration=10)
    return traj

def generate_step_trajectory(init_pos):
    positions = [np.random.randn(7) * 0.3 for _ in range(4)]  # 4 different positions
    traj = StepReference(positions, step_duration=2.5, init_pos=init_pos)
    return traj

def generate_composite_trajectory(init_pos):
    n_components = 3
    amplitudes = np.random.rand(n_components, 7) * np.array([np.pi/8, np.pi/12, np.pi/8, np.pi/8, np.pi/8, np.pi/8, np.pi/8])
    frequencies = np.random.rand(n_components, 7) * 0.5
    traj = CompositeReference(amplitudes, frequencies, init_pos)
    return traj

def generate_linear_ramp_trajectory(init_pos):
    target_disp = np.random.randn(7) * 0.5
    traj = LinearRampReference(target_disp, ramp_time=5.0, init_pos=init_pos)
    return traj

def generate_trajectories(init_pos,
                          num_trajectories: list[int],
                          mix: bool = False):
    
    # Create different types of trajectories for data collection
    trajectories = []
    traj_types = ['sinusoidal', 'polynomial', 'step', 'composite', 'linear_ramp', 'fourier_series']
    trajs = []
    for i, num in enumerate(num_trajectories):
        if num > 0:
            trajs.extend([traj_types[i]] * num)
    
    if mix:
        np.random.shuffle(trajs)

    for t in trajs:
        if t == 'sinusoidal':
            traj = generate_sinusoidal_trajectory(init_pos)
            trajectories.append(traj)
        if t == 'polynomial':
            traj = generate_polynomial_trajectory(init_pos)
            trajectories.append(traj)
        if t == 'step':
            traj = generate_step_trajectory(init_pos)
            trajectories.append(traj)
        if t == 'composite':
            traj = generate_composite_trajectory(init_pos)
            trajectories.append(traj)
        if t == 'linear_ramp':
            traj = generate_linear_ramp_trajectory(init_pos)
            trajectories.append(traj)
        if t == 'fourier_series':
            traj = generate_fourier_series_trajectory(init_pos)
            trajectories.append(traj)
    
    return trajectories
