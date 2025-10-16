import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from simulation_and_control import (
    MotorCommands,
    PinWrapper,
    feedback_lin_ctrl,
    pb,
)
from utils.draw_graphs import draw_plots
from utils.trajectory_generator import generate_trajectories


@dataclass
class EvaluationConfig:
    """Runtime configuration for model evaluation."""

    conf_file_name: str = "pandaconfig.json"
    cur_dir: str = os.path.dirname(os.path.abspath(__file__))

    source_names: list = field(default_factory=lambda: ["pybullet"])

    kp: float = 1000.0
    kd: float = 100.0
    collection_max_time_per_trajectory: float = 10.0
    skip_initial: int = 1000

    evaluation_trajectories: list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    evaluation_mix: bool = False

    part = "part2"
    model_path: str = os.path.join(cur_dir, "checkpoints", f"a_{part}.npy")
    save_plots: bool = True
    show_plots: bool = False
    plots_save_dir: str = os.path.join(cur_dir, "plots_evaluation")


def initialize_robot(config: EvaluationConfig) -> Tuple[pb.SimInterface, PinWrapper, int]:
    """Initialise the simulator and dynamic model."""
    sim: pb.SimInterface = pb.SimInterface(
        config.conf_file_name, conf_file_path_ext=config.cur_dir
    )
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    dyn_model = PinWrapper(
        config.conf_file_name,
        "pybullet",
        ext_names,
        config.source_names,
        False,
        0,
        config.cur_dir,
    )
    num_joints = dyn_model.getNumberofActuatedJoints()

    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")
    return sim, dyn_model, num_joints


def collect_data_single_trajectory(
    dyn_model: PinWrapper,
    cmd: MotorCommands,
    sim: pb.SimInterface,
    ref,
    kp: float,
    kd: float,
    time_step: float,
    max_time: float,
    regressor_all: list,
    tau_mes_all: list,
    skip_initial: int = 1000,
) -> None:
    """Roll out one trajectory and store regressor/torque samples."""
    current_time = 0.0
    skip_time = time_step * skip_initial
    with tqdm(total=max_time, desc="Collecting data", unit="s") as pbar:
        while current_time < max_time:
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

            q_d, qd_d = ref.get_values(current_time)

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")

            tau_mes = sim.GetMotorTorques(0)

            if dyn_model.visualizer:
                for index in range(len(sim.bot)):
                    q = sim.GetMotorAngles(index)
                    dyn_model.DisplayModel(q)

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            q_key = ord("q")
            if q_key in keys and keys[q_key] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                break

            regressor_cur = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
            if current_time >= skip_time:
                regressor_all.append(regressor_cur)
                tau_mes_all.append(tau_mes)

            pbar.update(time_step)
            current_time += time_step


def load_model_parameters(model_path: str) -> np.ndarray:
    """Load the identified dynamic parameters from disk."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model parameters not found at {model_path}")
    params = np.load(model_path)
    if params.ndim != 1:
        raise ValueError(f"Expected 1-D parameter vector, got shape {params.shape}")
    print(f"Loaded model parameters from {model_path}")
    return params


def collect_evaluation_dataset(
    config: EvaluationConfig,
    sim: pb.SimInterface,
    dyn_model: PinWrapper,
    time_step: float,
    initial_motor_angles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather regressors and torques for the configured evaluation trajectories."""
    sim.ResetPose()
    references = generate_trajectories(
        initial_motor_angles,
        num_trajectories=config.evaluation_trajectories,
        mix=config.evaluation_mix,
    )

    cmd = MotorCommands()
    regressor_all = []
    tau_mes_all = []

    for ref in references:
        collect_data_single_trajectory(
            dyn_model=dyn_model,
            cmd=cmd,
            sim=sim,
            ref=ref,
            kp=config.kp,
            kd=config.kd,
            time_step=time_step,
            max_time=config.collection_max_time_per_trajectory,
            regressor_all=regressor_all,
            tau_mes_all=tau_mes_all,
            skip_initial=config.skip_initial,
        )

    regressor_np = np.array(regressor_all)
    tau_np = np.array(tau_mes_all)
    if regressor_np.shape[0] != tau_np.shape[0]:
        raise RuntimeError(
            "Mismatch between collected regressor and torque samples: "
            f"{regressor_np.shape[0]} vs {tau_np.shape[0]}"
        )
    print(f"Evaluation samples collected: {tau_np.shape[0]}")
    return regressor_np, tau_np


def compute_metrics(
    regressor_all: np.ndarray, tau_mes_all: np.ndarray, params: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute joint-wise prediction metrics for the evaluated model."""
    torque_pred = np.matmul(regressor_all, params)
    error = torque_pred - tau_mes_all

    num_samples, num_joints = tau_mes_all.shape
    dof_model = params.size
    dof_resid_joint = num_samples - dof_model - 1

    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    max_abs = np.max(np.abs(error), axis=0)

    y_mean = np.mean(tau_mes_all, axis=0)
    ss_tot = np.sum((tau_mes_all - y_mean) ** 2, axis=0)
    ss_res = np.sum(error**2, axis=0)
    ss_reg = ss_tot - ss_res

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1 - ss_res / ss_tot
    # Handle degenerate cases where variance is ~0
    r2 = np.where(ss_tot <= np.finfo(float).eps, 1.0, r2)

    if dof_resid_joint > 0 and dof_model > 0:
        adj_r2 = 1 - (1 - r2) * (num_samples - 1) / dof_resid_joint
        with np.errstate(divide="ignore", invalid="ignore"):
            f_stat = (ss_reg / dof_model) / (ss_res / dof_resid_joint)
    else:
        adj_r2 = np.full(r2.shape, np.nan, dtype=float)
        f_stat = np.full(r2.shape, np.nan, dtype=float)

    overall_rmse = np.sqrt(np.mean(error**2))
    overall_mae = np.mean(np.abs(error))

    overall_mean = np.mean(tau_mes_all)
    overall_ss_tot = np.sum((tau_mes_all - overall_mean) ** 2)
    overall_ss_res = np.sum(error**2)
    overall_ss_reg = overall_ss_tot - overall_ss_res

    with np.errstate(divide="ignore", invalid="ignore"):
        overall_r2 = 1 - overall_ss_res / overall_ss_tot
    if overall_ss_tot <= np.finfo(float).eps:
        overall_r2 = 1.0

    total_observations = num_samples * num_joints
    dof_resid_overall = total_observations - dof_model - 1
    if dof_resid_overall > 0 and dof_model > 0:
        overall_adj_r2 = 1 - (1 - overall_r2) * (total_observations - 1) / dof_resid_overall
        with np.errstate(divide="ignore", invalid="ignore"):
            overall_f = (overall_ss_reg / dof_model) / (overall_ss_res / dof_resid_overall)
    else:
        overall_adj_r2 = np.nan
        overall_f = np.nan

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
        "overall_adj_r2": overall_adj_r2,
        "overall_f": overall_f,
        "torque_pred": torque_pred,
        "error": error,
        "dof_model": dof_model,
        "dof_resid_joint": dof_resid_joint,
        "dof_resid_overall": dof_resid_overall,
    }
    return metrics


def evaluate_model(config: EvaluationConfig) -> Dict[str, np.ndarray]:
    """Full evaluation pipeline: load model, collect data, compute metrics, draw plots."""
    params = load_model_parameters(config.model_path)

    sim, dyn_model, _ = initialize_robot(config)
    time_step = sim.GetTimeStep()
    initial_motor_angles = sim.GetInitMotorAngles()

    regressor_all, tau_mes_all = collect_evaluation_dataset(
        config, sim, dyn_model, time_step, initial_motor_angles
    )

    metrics = compute_metrics(regressor_all, tau_mes_all, params)

    os.makedirs(config.plots_save_dir, exist_ok=True)
    draw_plots(
        regressor_all=regressor_all,
        tau_mes_all=tau_mes_all,
        a=params,
        time_step=time_step,
        save_plots=config.save_plots,
        show_plots=config.show_plots,
        save_dir=config.plots_save_dir,
        part=config.part,
    )

    rmse = metrics["rmse"]
    mae = metrics["mae"]
    max_abs = metrics["max_abs"]
    r2 = metrics["r2"]
    adj_r2 = metrics["adj_r2"]
    f_stat = metrics["f_stat"]

    dof_model = metrics["dof_model"]
    dof_resid_joint = metrics["dof_resid_joint"]
    dof_resid_overall = metrics["dof_resid_overall"]

    for joint_idx in range(rmse.shape[0]):
        print(
            f"Joint {joint_idx + 1}: MAE={mae[joint_idx]:.4f} Nm, "
            f"RMSE={rmse[joint_idx]:.4f} Nm, "
            f"Max |error|={max_abs[joint_idx]:.4f} Nm, "
            f"R²={r2[joint_idx]:.4f}, "
            f"Adj. R²={adj_r2[joint_idx]:.4f}, "
            f"F={f_stat[joint_idx]:.4f}"
        )
    print(
        f"Overall MAE={metrics['overall_mae']:.4f} Nm, "
        f"Overall RMSE={metrics['overall_rmse']:.4f} Nm, "
        f"R²={metrics['overall_r2']:.4f}, "
        f"Adj. R²={metrics['overall_adj_r2']:.4f}, "
        f"F={metrics['overall_f']:.4f}"
    )
    print(
        f"Degrees of freedom: model={dof_model}, "
        f"residual (per joint)={dof_resid_joint}, "
        f"residual (overall)={dof_resid_overall}"
    )
    return metrics


def run() -> Dict[str, np.ndarray]:
    """Entry point used for parity with the training scripts."""
    config = EvaluationConfig()
    return evaluate_model(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
