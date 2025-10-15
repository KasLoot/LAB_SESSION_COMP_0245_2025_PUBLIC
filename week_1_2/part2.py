import numpy as np
import os
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from utils.draw_graphs import draw_plots
from utils.trajectory_generator import generate_trajectories
from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class Part1Config:
    conf_file_name: str = "pandaconfig.json"
    cur_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    source_names: list = field(default_factory=lambda: ["pybullet"])

    kp: float = 1000.0
    kd: float = 100.0
    collection_max_time_per_trajectory: float = 10.0  # seconds
    skip_initial: int = 1000  # number of initial samples to skip

    num_trajectories: list = field(default_factory=lambda: [1, 0, 0, 0, 0])  # Number of trajectories for each type
    evaluation_trajectories: list = field(default_factory=lambda: [1, 0, 0, 0, 0])  # Number of evaluation trajectories for each type

    train_mix: bool = False  # Whether to mix trajectory types during training
    evaluation_mix: bool = False   # Whether to mix trajectory types during evaluation

    run_evaluation: bool = True

    save_model: bool = True
    save_plots: bool = True
    show_plots: bool = False
    model_save_path: str = os.path.join(cur_dir, "checkpoints", "a_part2.npy")
    plots_save_dir: str = os.path.join(cur_dir, "plots_part2")


def initialize_robot(config: Part1Config):
    sim: pb.SimInterface = pb.SimInterface(config.conf_file_name, conf_file_path_ext=config.cur_dir)
    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(config.conf_file_name, "pybullet", ext_names, config.source_names, False, 0, config.cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    return sim, dyn_model, num_joints


def collect_data_single_trajectory(dyn_model: PinWrapper, 
                                   cmd: MotorCommands,
                                   sim: pb.SimInterface, 
                                   ref, 
                                   kp, 
                                   kd, 
                                   time_step, 
                                   max_time, 
                                   regressor_all: list, 
                                   tau_mes_all: list,
                                   skip_initial=1000
                                   ):
    current_time = 0  # Reset time for each trajectory
    skip_time = time_step * skip_initial
    with tqdm(total=max_time, desc="Collecting data", unit="s") as pbar:
        while current_time < max_time:
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

            # Compute sinusoidal reference trajectory
            q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
            cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque command
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
            if current_time >= skip_time:
                regressor_all.append(regressor_cur)
                tau_mes_all.append(tau_mes)
            pbar.update(time_step)
            current_time += time_step


def collect_data(config: Part1Config):

    sim, dyn_model, num_joints = initialize_robot(config)
    initial_motor_angles = sim.GetInitMotorAngles()

    ref = generate_trajectories(initial_motor_angles,
                                num_trajectories=config.num_trajectories,
                                mix=config.evaluation_mix)

    # Simulation parameters
    time_step = sim.GetTimeStep()
    print(f"time step: {time_step}")
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    for i in tqdm(range(len(ref)), desc="Collecting data from trajectories"):
        collect_data_single_trajectory(dyn_model=dyn_model,
                                       cmd=cmd,
                                       sim=sim,
                                       ref=ref[i],
                                       kp=config.kp,
                                       kd=config.kd,
                                       time_step=time_step,
                                       max_time=config.collection_max_time_per_trajectory,
                                       regressor_all=regressor_all,
                                       tau_mes_all=tau_mes_all,
                                       skip_initial=config.skip_initial)

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    tau_mes_all = np.array(tau_mes_all)
    new_tau_mes_all = tau_mes_all.reshape(-1)
    regressor_all = np.array(regressor_all)
    new_regressor_all = regressor_all.reshape(-1, num_joints*10)

    a = np.linalg.pinv(new_regressor_all)@new_tau_mes_all
    print(f"a shape: {a.shape}")

    if config.save_model:
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        np.save(config.model_save_path, a)
        print(f"Model parameters saved to {config.model_save_path}")

    if config.run_evaluation:
        evaluate_model(config, sim, dyn_model, time_step, initial_motor_angles, a)


def evaluate_model(config: Part1Config,
                   sim: pb.SimInterface,
                   dyn_model: PinWrapper,
                   time_step: float,
                   initial_motor_angles: np.ndarray,
                   a: np.ndarray,
                   ):
    sim.ResetPose()
    ref = generate_trajectories(initial_motor_angles,
                                num_trajectories=config.evaluation_trajectories,
                                mix=config.evaluation_mix)
    cmd = MotorCommands()
    test_regressor_all = []
    test_tau_mes_all = []
    for i in range(len(ref)):
        collect_data_single_trajectory(dyn_model=dyn_model,
                                       cmd=cmd,
                                       sim=sim,
                                       ref=ref[i],
                                       kp=config.kp,
                                       kd=config.kd,
                                       time_step=time_step,
                                       max_time=config.collection_max_time_per_trajectory,
                                       regressor_all=test_regressor_all,
                                       tau_mes_all=test_tau_mes_all,
                                       skip_initial=config.skip_initial)

    test_tau_mes_all = np.array(test_tau_mes_all)
    test_regressor_all = np.array(test_regressor_all)
    print(f"Evaluation samples collected: {test_tau_mes_all.shape[0]}")
    draw_plots(regressor_all=test_regressor_all, 
               tau_mes_all=test_tau_mes_all, 
               a=a, 
               time_step=time_step, 
               save_plots=config.save_plots,
               show_plots=config.show_plots,
               save_dir=config.plots_save_dir)

def run():
    config = Part1Config()
    collect_data(config)

def main():
    run()



if __name__ == '__main__':
    main()
