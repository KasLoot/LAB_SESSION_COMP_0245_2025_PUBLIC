import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference


def collect_data_single_trajectory(
    dyn_model: PinWrapper,
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
    drop_initial=100
):
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        # Compute sinusoidal reference trajectory
        # Desired position and velocity
        q_d, qd_d = ref.get_values(current_time)

        # Control command
        tau_cmd = feedback_lin_ctrl(
            dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                # Update the display of the robot model
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        # TODO Compute regressor and store it
        regressor_cur = dyn_model.ComputeDynamicRegressor(
            q_mes, qd_mes, qdd_mes)

        regressor_all.append(regressor_cur)
        tau_mes_all.append(tau_mes)

        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")
    return np.asarray(regressor_all[drop_initial:]), np.asarray(tau_mes_all[drop_initial:])


def compute_a_only(regressor_all, tau_mes_all):
    p_l1 = np.array([2.34, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    p_l2 = np.array([2.36, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    p_l3 = np.array([2.38, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    p_l4 = np.array([2.43, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    p_l5 = np.array([3.5, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    p_l6 = np.array([1.47, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_known = np.hstack((p_l1, p_l2, p_l3, p_l4, p_l5, p_l6))

    tau_real = regressor_all[:, :6, :].reshape(-1, 70)[:, :60] @ a_known
    tau_real = tau_real.reshape(-1, 6)
    tau_mixed = np.hstack(
        (tau_real, tau_mes_all[:, 6].reshape(-1, 1))).reshape(-1)

    a_est = np.linalg.pinv(regressor_all.reshape(-1, 70)) @ tau_mixed
    a_last = a_est[60:]
    print("Estimated parameters for last link with known others:", a_last)
    return a_est


def compute_metrics(Y, a_est, tau_mes, experiment_name=""):
    """
    Compute and print R-squared and Adjusted R-squared for the linear model.
    """
    print(f"--- Computing metrics for {experiment_name} ---")
    # print(f"a_est shape: {a_est.shape}")
    # print(f"Y shape: {Y.shape}")
    # Y = Y.reshape(10001, 7, 70)
    # tau_mes = tau_mes.reshape(10001, 7)

    tau_pred = Y @ a_est
    # print(f"tau_pred shape: {tau_pred.shape}")
    residuals = tau_mes - tau_pred
    # print(f"residuals shape: {residuals.shape}")

    rss = np.sum(residuals**2)
    tss = np.sum((tau_mes - np.mean(tau_mes))**2)
    r_squared = 1 - (rss / tss)
    print(f"R-squared: {r_squared}")

    n = Y.shape[0]  # number of observations
    p = Y.shape[1]  # number of predictors
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    print(f"Adjusted R-squared: {r_squared_adj}")


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


def run_all_regression_methods(Y, tau_mixed, tau):
    # --- 防御性检查 ---
    if not np.isfinite(Y).all():
        print("⚠️ Warning: Y contains NaN or Inf values!")
    if not np.isfinite(tau).all():
        print("⚠️ Warning: tau contains NaN or Inf values!")

    if True:
        tau_compute = tau_mixed
    else:
        tau_compute = tau

    print("\n--- 方案 0：原始数据 ---")
    check_ill_conditioning(Y, "Original regressor")
    a_est = np.linalg.pinv(Y) @ tau_compute
    compute_metrics(Y, a_est, tau, "Original")

    # --- 方案 1：列归一化 ---
    print("\n--- 方案 1：列归一化（Z-score / L2 归一化） ---")
    col_norms = np.linalg.norm(Y, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    Y_norm = Y / col_norms
    a_est = np.linalg.pinv(Y_norm) @ tau_compute
    check_ill_conditioning(Y_norm, "After column normalization")
    compute_metrics(Y_norm, a_est, tau, "Column normalization")

    # --- 方案 2：rcond 控制伪逆 ---
    print("\n--- 方案 2：rcond 控制伪逆 ---")
    I_est = np.linalg.pinv(Y, rcond=1e-6) @ Y
    I = np.eye(I_est.shape[1])
    diff = np.linalg.norm(I_est - I, 'fro')
    print(f"Difference from identity (Frobenius norm): {diff:.4e}")
    a_est = np.linalg.pinv(Y, rcond=1e-6) @ tau_compute
    compute_metrics(Y, a_est, tau, "rcond pseudo-inverse")

    # --- 方案 3：正则化伪逆（Damped Least Squares） ---
    print("\n--- 方案 3：正则化伪逆（Damped Least Squares） ---")
    alpha = 1e-3  # 可调参数
    a_est = np.linalg.solve(
        Y.T @ Y + alpha * np.eye(Y.shape[1]), Y.T @ tau_compute)
    compute_metrics(Y, a_est, tau, f"Damped least squares (alpha={alpha})")

    # --- 方案 4：截断奇异值（Truncated SVD） ---
    print("\n--- 方案 4：截断奇异值（Truncated SVD） ---")
    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    print("Singular values (top 10):", s[:10])
    tol = 1e-2
    rank = np.sum(s > tol)
    print(f"Effective rank (s > {tol}): {rank}")
    S_inv = np.diag([1/x if x > tol else 0 for x in s])
    a_est = vh.T @ S_inv @ u.T @ tau_compute
    compute_metrics(Y, a_est, tau, f"Truncated SVD (tol={tol})")

    # 方案4：激励轨迹设计（Excitation trajectory design）
    # 选择关节轨迹让每个参数对应的列都能独立激励。
    # 常见方法：优化轨迹的谱特性或基于 Fisher information 矩阵。

    print("\n✅ All methods completed.\n")


def statistic_analysis(Y, tau, a):
    Y_A, tau_A, a_A = Y[0], tau[0], a[0]
    Y_B, tau_B, a_B = Y[1], tau[1], a[1]
    Y_C, tau_C, a_C = Y[2], tau[2], a[2]

    # 计算欧几里得范数差
    diff_AB = np.linalg.norm(a_A - a_B)
    diff_BC = np.linalg.norm(a_B - a_C)
    diff_AC = np.linalg.norm(a_A - a_C)

    # 相对差异（便于比较）
    rel_diff_AB = diff_AB / np.linalg.norm(a_B)
    rel_diff_BC = diff_BC / np.linalg.norm(a_B)

    print(f"||a_A - a_B|| = {diff_AB:.4f} ({rel_diff_AB*100:.2f}%)")
    print(f"||a_B - a_C|| = {diff_BC:.4f} ({rel_diff_BC*100:.2f}%)")

    from scipy.stats import ks_2samp

    # 假设三个实验得到的残差：
    res_A = tau_A - Y_A @ a_A
    res_B = tau_B - Y_B @ a_B
    res_C = tau_C - Y_C @ a_C

    # 统计检验（Kolmogorov–Smirnov）
    ks_stat, p_value = ks_2samp(res_A.flatten(), res_B.flatten())
    print(f"KS test statistic = {ks_stat:.4f}, p = {p_value:.4e}")
    ks_stat, p_value = ks_2samp(res_B.flatten(), res_C.flatten())
    print(f"KS test statistic = {ks_stat:.4f}, p = {p_value:.4e}")


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    # Adjust the shape for compatibility
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet",
                           ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi /
                  4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    # Example frequencies for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    # Initialize the reference
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())

    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = [900, 1000, 1100]
    kd = [90, 100, 110]

    # Initialize data storage
    tau_mes_all_list = []
    regressor_all_list = []
    a_est_list = []

    # Data collection loop
    # for kp, kd in zip(kp, kd):
    #     current_time = 0   # 重置时间
    #     sim.ResetPose()
    #     print(f"--- Collecting data with kp={kp}, kd={kd} ---")
    #     regressor_all, tau_mes_all = collect_data_single_trajectory(
    #         dyn_model, cmd, sim, ref, kp, kd, time_step, max_time, current_time, [], [])
    #     a = compute_a_only(regressor_all, tau_mes_all)
    #     regressor_all_list.append(regressor_all)
    #     tau_mes_all_list.append(tau_mes_all)
    #     a_est_list.append(a)
    #     compute_metrics(regressor_all.reshape(-1, 70), a, tau_mes_all.reshape(-1), f"kp={kp}, kd={kd}")
    #     check_ill_conditioning(regressor_all.reshape(-1, 70), f"kp={kp}, kd={kd}")

    drop = [0, 100, 200]
    for d in drop:
        current_time = 0   # 重置时间
        sim.ResetPose()
        print(f"--- Collecting data dropping first {d} samples ---")
        regressor_all, tau_mes_all = collect_data_single_trajectory(
            dyn_model, cmd, sim, ref, 1000, 100, time_step, max_time, current_time, [], [], d)
        a = compute_a_only(regressor_all, tau_mes_all)
        regressor_all_list.append(regressor_all)
        tau_mes_all_list.append(tau_mes_all)
        a_est_list.append(a)
        compute_metrics(regressor_all.reshape(-1, 70), a,
                        tau_mes_all.reshape(-1), f"drop first {d} samples")
        check_ill_conditioning(
            regressor_all.reshape(-1, 70), f"drop first {d} samples")

    statistic_analysis(regressor_all_list, tau_mes_all_list, a_est_list)

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    # regressor_all = np.asarray(regressor_all)
    # tau_mes_all = np.asarray(tau_mes_all)

    # Y = regressor_all.reshape(-1, num_joints*10)
    # tau = tau_mes_all.reshape(-1)

    # a_est = np.linalg.pinv(Y) @ tau
    # # print("Estimated parameters for all links:", a_est)
    # a_est_last = a_est[-10:]
    # print("Estimated parameters for last link:", a_est_last)

    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters

    # valid_cols = np.std(Y, axis=0) > 1e-8
    # Y_valid = Y[:, valid_cols]
    # a_est = np.linalg.pinv(Y_valid) @ tau_mixed

    # run_all_regression_methods(Y_valid, tau_mixed, tau)

    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file
    # compute_metrics(Y, a_est, tau, "All joints with known others")
    # run_all_regression_methods(Y, tau_mixed, tau)

    # TODO plot the torque prediction error for each joint (optional)


if __name__ == '__main__':
    main()
