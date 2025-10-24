import numpy as np
import json
import os
from tqdm import tqdm
from typing import Tuple
from simulation_and_control import pb, PinWrapper, MotorCommands
from part1 import Part1Config, initialize_robot, generate_trajectories, collect_data_single_trajectory
from part2 import Part2Config
from evaluation import EvaluationConfig, compute_metrics


def collect_dataset(
    config: Part1Config | EvaluationConfig,
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
    print(f"Samples collected: {tau_np.shape[0]}")
    return regressor_np, tau_np


def check_ill_conditioning(Y, experiment_name=""):
    """
    Check if the regressor matrix Y is ill-conditioned.
    """
    print(f"--- Checking ill-conditioning for {experiment_name} ---")

    I_est = np.linalg.pinv(Y) @ Y
    I = np.eye(I_est.shape[0])
    # 这会输出 Frobenius 范数，也就是整体偏离程度；如果偏大（比如 >1e-5），则可能意味着 Y 存在病态或冗余列
    print("Difference from identity (Frobenius norm):",
          np.linalg.norm(I_est - I, 'fro'))

    cond = np.linalg.cond(Y)
    # 如果 > 1e8：几乎肯定病态
    print("Condition number:", cond)

    rank = np.linalg.matrix_rank(Y)
    # 如果 rank < 70（比如只有 55 或 60），说明有 10~15 个参数根本不可辨识。
    print("Matrix rank:", rank, "out of", Y.shape[1])


def run_all_regression_methods(Y, tau):
    # --- 防御性检查 ---
    if not np.isfinite(Y).all():
        print("⚠️ Warning: Y contains NaN or Inf values!")
    if not np.isfinite(tau).all():
        print("⚠️ Warning: tau contains NaN or Inf values!")

    a_adj = []

    print("\n--- 方案 0：原始数据 ---")
    check_ill_conditioning(Y, "Original regressor")
    a_est = np.linalg.pinv(Y) @ tau
    a_adj.append(a_est)

    # --- 方案 1：列归一化 ---
    print("\n--- 方案 1：列归一化（Z-score / L2 归一化） ---")
    col_norms = np.linalg.norm(Y, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    Y_norm = Y / col_norms
    check_ill_conditioning(Y_norm, "After column normalization")
    a_est = np.linalg.pinv(Y_norm) @ tau
    a_adj.append(a_est)

    # --- 方案 2：rcond 控制伪逆 ---
    print("\n--- 方案 2：rcond 控制伪逆 ---")
    I_est = np.linalg.pinv(Y, rcond=1e-6) @ Y
    I = np.eye(I_est.shape[1])
    diff = np.linalg.norm(I_est - I, 'fro')
    print(f"Difference from identity (Frobenius norm): {diff:.4e}")
    a_est = np.linalg.pinv(Y, rcond=1e-6) @ tau
    a_adj.append(a_est)

    # --- 方案 3：正则化伪逆（Damped Least Squares） ---
    print("\n--- 方案 3：正则化伪逆（Damped Least Squares） ---")
    alpha = 1e-3  # 可调参数
    a_est = np.linalg.solve(
        Y.T @ Y + alpha * np.eye(Y.shape[1]), Y.T @ tau)
    a_adj.append(a_est)

    # --- 方案 4：截断奇异值（Truncated SVD） ---
    print("\n--- 方案 4：截断奇异值（Truncated SVD） ---")
    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    print("Singular values (top 10):", s[:10])
    tol = 1e-2
    rank = np.sum(s > tol)
    print(f"Effective rank (s > {tol}): {rank}")
    S_inv = np.diag([1/x if x > tol else 0 for x in s])
    a_est = vh.T @ S_inv @ u.T @ tau
    a_adj.append(a_est)

    # 方案4：激励轨迹设计（Excitation trajectory design）
    # 选择关节轨迹让每个参数对应的列都能独立激励。
    # 常见方法：优化轨迹的谱特性或基于 Fisher information 矩阵。

    print("\n✅ All methods completed.\n")
    return a_adj


def compute_model_parameters(part, tau_mes_all, regressor_all):
    assert tau_mes_all.shape[0] == regressor_all.shape[0], "Mismatch in number of samples between tau and regressor"
    print(f"Train samples collected: {tau_mes_all.shape[0]}")

    a = None
    if part == "part1":
        regressor_6_joints = regressor_all[:, :6, :60]

        p_l1 = np.array([2.34, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
        p_l2 = np.array([2.36, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
        p_l3 = np.array([2.38, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
        p_l4 = np.array([2.43, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
        p_l5 = np.array([3.5, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])
        p_l6 = np.array([1.47, 0, 0, 0, 0.3, 0, 0.3, 0.0, 0.0, 0.3])

        a_known = np.concatenate((p_l1, p_l2, p_l3, p_l4, p_l5, p_l6))
        tau_real = regressor_6_joints @ a_known
        tau_real = tau_mes_all[:, :6] - tau_real

        tau_mixed = np.concatenate(
            (tau_real, np.expand_dims(tau_mes_all[:, 6], axis=1)), axis=1)

        a_last_joint = np.linalg.pinv(np.vstack(regressor_all[:,:,60:])) @ np.hstack(tau_mixed)
        print(f"a_last_joint shape: {a_last_joint.shape}")
        print(f"a_last_joint: {a_last_joint}")

        a = np.hstack((a_known, a_last_joint))
    elif part == "part2":
        tau_mes_all = np.array(tau_mes_all)
        new_tau_mes_all = tau_mes_all.reshape(-1)
        regressor_all = np.array(regressor_all)
        new_regressor_all = regressor_all.reshape(-1, 70)

        a = np.linalg.pinv(new_regressor_all)@new_tau_mes_all
        print(f"a shape: {a.shape}")
        print(a[60:])

        run_all_regression_methods(new_regressor_all, new_tau_mes_all)

    return a


def format_output(metrics):
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


def drop_rate_train():
    cfg = Part1Config(show_plots=False, save_plots=False, skip_initial=0,
                      num_trajectories=[1, 0, 0, 0, 0, 0])
    eval_cfg = EvaluationConfig(part="part1", show_plots=False, skip_initial=1000,
                                save_plots=False, evaluation_trajectories=[1, 1, 1, 1, 1, 1])
    sim, dyn_model, num_joints = initialize_robot(cfg)
    initial_motor_angles = sim.GetInitMotorAngles()
    time_step = sim.GetTimeStep()

    drop_num = [0, 100, 200, 500, 1000, 2000]
    regressor_all, tau_mes_all = collect_dataset(
        cfg, sim, dyn_model, time_step, initial_motor_angles)
    regressor_test, tau_mes_test = collect_dataset(
        eval_cfg, sim, dyn_model, time_step, initial_motor_angles)
    
    for d in drop_num:
        print(f"\n\n--- Drop Rate: {d} ---")
        a = compute_model_parameters(cfg, tau_mes_all[d:,:], regressor_all[d:,:,:])
        metrics = compute_metrics(regressor_test, tau_mes_test, a)
        format_output(metrics)


def drop_rate_test():
    cfg = Part1Config(show_plots=False, save_plots=False, skip_initial=1000,
                      num_trajectories=[1, 0, 0, 0, 0, 0])
    eval_cfg = EvaluationConfig(part="part1", show_plots=False, skip_initial=0,
                                save_plots=False, evaluation_trajectories=[1, 1, 1, 1, 1, 1])
    sim, dyn_model, num_joints = initialize_robot(cfg)
    initial_motor_angles = sim.GetInitMotorAngles()
    time_step = sim.GetTimeStep()

    drop_num = [0, 100, 200, 500, 1000, 2000]
    regressor_all, tau_mes_all = collect_dataset(
        cfg, sim, dyn_model, time_step, initial_motor_angles)
    regressor_test, tau_mes_test = collect_dataset(
        eval_cfg, sim, dyn_model, time_step, initial_motor_angles)
    
    for d in drop_num:
        print(f"\n\n--- Drop Rate: {d} ---")
        processed_regressor_test = []
        processed_tau_mes_test = []
        chunk_size = 10001
        for i in range(6):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 5 else 60006

            processed_regressor_test.append(regressor_test[start_idx+d:end_idx,:,:])
            processed_tau_mes_test.append(tau_mes_test[start_idx+d:end_idx,:])

        processed_regressor_test = np.vstack(processed_regressor_test)
        processed_tau_mes_test = np.vstack(processed_tau_mes_test)

        a = compute_model_parameters(cfg, tau_mes_all, regressor_all)
        metrics = compute_metrics(processed_regressor_test, processed_tau_mes_test, a)
        format_output(metrics)


def ill_conditioning_test():
    cfg = Part2Config(show_plots=False, save_plots=False,
                      num_trajectories=[1, 1, 1, 1, 1, 1])
    eval_cfg = EvaluationConfig(part="part2", show_plots=False,
                                save_plots=False, evaluation_trajectories=[1, 1, 1, 1, 1, 1])
    sim, dyn_model, num_joints = initialize_robot(cfg)
    initial_motor_angles = sim.GetInitMotorAngles()
    time_step = sim.GetTimeStep()

    regressor_all, tau_mes_all = collect_dataset(
        cfg, sim, dyn_model, time_step, initial_motor_angles)
    regressor_test, tau_mes_test = collect_dataset(
        eval_cfg, sim, dyn_model, time_step, initial_motor_angles)

    a = compute_model_parameters("part2", tau_mes_all, regressor_all)

    metrics = compute_metrics(regressor_test, tau_mes_test, a)
    format_output(metrics)


# def time_step():
#     cfg = Part1Config(show_plots=False, save_plots=False)
#     eval_cfg = EvaluationConfig(part="part1", show_plots=False, save_plots=False, evaluation_trajectories=[1,0,0,0,0,0])
#     time_steps = [0.001,]

def trajectory():
    cfg = Part1Config(show_plots=False, save_plots=False, train_mix=True,
                    num_trajectories=[1, 1, 1, 1, 1, 1])
    eval_cfg = EvaluationConfig(part="part1", show_plots=False, evaluation_mix=True,
                                save_plots=False, evaluation_trajectories=[1, 1, 1, 1, 1, 1])
    sim, dyn_model, num_joints = initialize_robot(cfg)
    initial_motor_angles = sim.GetInitMotorAngles()
    time_step = sim.GetTimeStep()

    regressor_all, tau_mes_all = collect_dataset(
        cfg, sim, dyn_model, time_step, initial_motor_angles)
    regressor_test, tau_mes_test = collect_dataset(
        eval_cfg, sim, dyn_model, time_step, initial_motor_angles)
    
    a = compute_model_parameters(cfg, tau_mes_all, regressor_all)
    metrics = compute_metrics(regressor_test, tau_mes_test, a)
    format_output(metrics)



if __name__ == "__main__":
    ill_conditioning_test()
