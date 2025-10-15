import numpy as np
from simulation_and_control import SinusoidalReference

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
                          num_trajectories: list[int] =[1,1,1,1,1],
                          mix: bool = False):
    
    # Create different types of trajectories for data collection
    trajectories = []
    traj_types = ['sinusoidal', 'polynomial', 'step', 'composite', 'linear_ramp']
    trajs = []
    for i, num in enumerate(num_trajectories):
        if num > 0:
            trajs.append(traj_types[i]*num)
    
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
    
    return trajectories