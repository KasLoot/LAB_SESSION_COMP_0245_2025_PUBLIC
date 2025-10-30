import os
import time
from pathlib import Path

import numpy as np
# Project-specific imports (keep these as in original environment)
from simulation_and_control import (
    pb,
    MotorCommands,
    PinWrapper,
    feedback_lin_ctrl,
    CartesianDiffKin,
)

import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure project root on path
from rollout_loader import load_rollouts  # keep for compatibility if needed


# --------------------------
# Configuration (centralized)
# --------------------------
# Edit these to change data/category/seeds/limits without touching main logic.
DATA_CATEGORY = "train"  # "train" | "test" | "simulation"
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "train": {
        "final_dir": BASE_DIR / "data" / "train",
        "torch_seed": 42,
        "np_seed": 42,
        "num_poses": 200,
        "sampling": "hybrid",
    },
    "test": {
        "final_dir": BASE_DIR / "data" / "test",
        "torch_seed": 56,
        "np_seed": 56,
        "num_poses": 40,
        "sampling": "random",
    },
    "simulation": {
        "final_dir": BASE_DIR / "data" / "simulation",
        "torch_seed": 100,
        "np_seed": 100,
        "num_poses": 5,
        "sampling": "random",
    },
}

# Default workspace bounds used by generate_data if none provided
DEFAULT_WORKSPACE_BOUNDS = (0.3, 0.6, -0.4, 0.4, 0.1, 0.6)  # x_min,x_max,y_min,y_max,z_min,z_max

# Runtime flags
PRINT_PLOTS = False
RECORDING = True
DOWN_SAMPLE_RATE = 2  # integer >= 1
EXTRA_STEPS_AFTER_REACH = 500  # steps to keep collecting after reaching the target
SAVE_THRESHOLD = 5e-5  # final Cartesian distance threshold for saving


# --------------------------
# Utility functions
# --------------------------
def ensure_config():
    """Set seeds and ensure final directory exists based on DATA_CATEGORY."""
    cfg = CONFIG.get(DATA_CATEGORY, CONFIG["train"])
    torch.manual_seed(cfg["torch_seed"])
    np.random.seed(cfg["np_seed"])
    final_dir = cfg["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    return cfg, final_dir


def generate_data(num_poses, init_joint_angles=None, workspace_bounds=None, jitter=0.05):
    """
    Sample points in workspace cuboid using a hybrid strategy and return lists for data collection.

    - num_poses: number of sample points required
    - workspace_bounds: (x_min,x_max,y_min,y_max,z_min,z_max)
    - jitter: maximum uniform jitter added to each coordinate (meters)

    Sampling behaviour:
      - Uses CONFIG[DATA_CATEGORY].get('sampling') if present ('uniform', 'random', or 'hybrid').
      - Defaults: 'train' -> 'hybrid', others -> 'random'.
    Returns five lists: positions, orientations, control_types, durations, initial_positions
    """
    if workspace_bounds is None:
        x_min, x_max, y_min, y_max, z_min, z_max = DEFAULT_WORKSPACE_BOUNDS
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds

    # Determine sampling mode (prefer config entry if present)
    cfg_entry = CONFIG.get(DATA_CATEGORY, {})
    sampling_mode = cfg_entry.get(
        "sampling", "hybrid" if DATA_CATEGORY == "train" else "random"
    ).lower()

    # ---------------------------
    # 1. Random sampling
    # ---------------------------
    def random_samples(n):
        return np.column_stack(
            (
                np.random.uniform(x_min, x_max, n),
                np.random.uniform(y_min, y_max, n),
                np.random.uniform(z_min, z_max, n),
                # np.random.uniform(0.2, 0.5, n),
                # np.random.choice([1, -1], n) * np.random.uniform(0.2, 0.5, n),
                # np.random.uniform(0.1, 0.6, n),
            )
        )

    # ---------------------------
    # 2. Uniform grid sampling
    # ---------------------------
    def uniform_grid_samples(n):
        n_axis = int(np.ceil(n ** (1.0 / 3.0)))
        xs = np.linspace(x_min, x_max, n_axis)
        ys = np.linspace(y_min, y_max, n_axis)
        zs = np.linspace(z_min, z_max, n_axis)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

        total_grid = grid_pts.shape[0]
        if total_grid > n:
            indices = np.linspace(0, total_grid - 1, n).astype(int)
            grid_pts = grid_pts[indices]
        elif total_grid < n:
            extra = n - total_grid
            extra_pts = random_samples(extra)
            grid_pts = np.vstack((grid_pts, extra_pts))
        return grid_pts

    # ---------------------------
    # 3. Hybrid sampling
    # ---------------------------
    if sampling_mode == "hybrid":
        n_rand = int(num_poses * 0.7)
        n_grid = num_poses - n_rand
        rand_pts = random_samples(n_rand)
        grid_pts = uniform_grid_samples(n_grid)
        pts = np.vstack((rand_pts, grid_pts))
        np.random.shuffle(pts)  # mix grid and random samples
    elif sampling_mode == "random":
        pts = random_samples(num_poses)
    else:  # 'uniform'
        pts = uniform_grid_samples(num_poses)

    # ---------------------------
    # 4. Add jitter (optional)
    # ---------------------------
    if jitter and jitter > 0.0:
        noise = (np.random.rand(num_poses, 3) - 0.5) * 2.0 * jitter
        pts = pts + noise

    # ---------------------------
    # 5. Build return lists
    # ---------------------------
    positions = pts.tolist()
    orientations = [[0.0, 0.0, 0.0, 1.0] for _ in range(num_poses)]
    control_types = ["pos"] * num_poses
    durations = [2.0] * num_poses
    initial_positions = [init_joint_angles] * num_poses

    return positions, orientations, control_types, durations, initial_positions


def setup_sim(conf_file_name: str, root_dir: str):
    """
    Initialize simulation and dynamic model; return sim, dyn_model and initial state info.
    """
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    init_angles = sim.GetInitMotorAngles()
    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    # joint_vel_limits = [vel * 100 for vel in joint_vel_limits]
    return sim, dyn_model, init_angles, joint_vel_limits, time_step


def collect_trajectory_for_target(
    sim,
    dyn_model,
    controlled_frame_name,
    init_position,
    desired_cartesian_pos,
    desired_cartesian_ori,
    duration,
    joint_vel_limits,
    kp_pos=100,
    kp_ori=0,
    kp=1000,
    kd=100,
    downsample_rate=2,
    extra_steps_after_reach=500,
):
    """
    Run simulation until duration or until target reached + extra steps.
    Returns a dict with downsampled tensors and metrics including final cart distance.
    """
    # Reset and set initial pose
    sim.ResetPose()
    if init_position is not None:
        sim.SetjointPosition(init_position)

    time_step = sim.GetTimeStep()
    steps = int(duration / time_step)

    # Initialize data buffers
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all, final_cart_pos, tau_cmd_all = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    cmd = MotorCommands()
    diff_kin = None

    cart_distance = float("inf")
    reached_target = False
    steps_after_reaching_target = 0

    for t in range(steps):
        q_mes = sim.GetMotorAngles(0)
        cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = dyn_model.ComputeMotorAccelerationTMinusOne(0) if hasattr(dyn_model, "ComputeMotorAccelerationTMinusOne") else None
        tau_mes = np.asarray(sim.GetMotorTorques(0), dtype=float)

        # Compute desired joint positions/velocities from Cartesian diff kinematics
        pd_d = [0.0, 0.0, 0.0]
        ori_d_des = [0.0, 0.0, 0.0]
        q_des, qd_des_clip = CartesianDiffKin(
            dyn_model,
            controlled_frame_name,
            q_mes,
            desired_cartesian_pos,
            pd_d,
            desired_cartesian_ori,
            ori_d_des,
            time_step,
            "pos",
            kp_pos,
            kp_ori,
            np.array(joint_vel_limits),
        )

        # Feedback control command and simulation step
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"] * len(q_mes))
        sim.Step(cmd, "torque")

        # Optionally record data
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des_clip)
        tau_mes_all.append(tau_mes)
        cart_pos_all.append(cart_pos)
        cart_ori_all.append(cart_ori)
        final_cart_pos.append(desired_cartesian_pos)
        tau_cmd_all.append(tau_cmd)

        # Update time and check reaching condition
        cart_distance = np.linalg.norm(np.array(cart_pos) - desired_cartesian_pos)
        if cart_distance < 5e-5 and not reached_target:
            reached_target = True

        if reached_target:
            steps_after_reaching_target += 1
            if steps_after_reaching_target >= extra_steps_after_reach:
                break

        # Small sleep to match real-time step (keeps same behaviour as original)
        time.sleep(time_step)

    # Downsample and convert to tensors
    def to_tensor_downsample(lst):
        if len(lst) == 0:
            return torch.tensor([])
        arr = np.array(lst)[::downsample_rate]
        return torch.tensor(arr)

    results = {
        "q_mes": to_tensor_downsample(q_mes_all),
        "qd_mes": to_tensor_downsample(qd_mes_all),
        "q_des": to_tensor_downsample(q_d_all),
        "qd_des": to_tensor_downsample(qd_d_all),
        "tau_mes": to_tensor_downsample(tau_mes_all),
        "tau_cmd": to_tensor_downsample(tau_cmd_all),
        "cart_pos": to_tensor_downsample(cart_pos_all),
        "cart_ori": to_tensor_downsample(cart_ori_all),
        "final_cart_pos": to_tensor_downsample(final_cart_pos),
        "final_cart_distance": float(cart_distance),
        "saved_steps": len(q_mes_all),
    }

    return results


def save_dataset(index: int, data: dict, final_dir: Path):
    """Save a single dataset file as a Torch .pt file."""
    filename = final_dir / f"data_{index}.pt"
    torch.save(
        {
            "q_mes_all": data["q_mes"],
            "final_cartesian_pos": data["final_cart_pos"],
            "q_d_all": data["q_des"],
            "qd_d_all": data["qd_des"],
            "tau_cmd_all": data["tau_cmd"],
        },
        filename,
    )
    return filename


# --------------------------
# Plot helpers (optional)
# --------------------------
def plot_time_series(time_array, datasets, labels, title, ylabel):
    plt.figure(figsize=(10, 5))
    for d, lab in zip(datasets, labels):
        plt.plot(time_array, d, label=lab)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_trajectory(actual_traj, desired_traj, save_path=None):
    """Plot 3D trajectory with Start/End/Desired markers and display final distance in title."""
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass

    if actual_traj.size == 0 or desired_traj.size == 0:
        print("Not enough data to plot 3D trajectory.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], label="Actual trajectory", color="tab:blue", linewidth=2)
    ax.scatter(desired_traj[:, 0], desired_traj[:, 1], desired_traj[:, 2], label="Desired (per-step)", color="tab:orange", marker="x", s=40)

    start = actual_traj[0]
    end = actual_traj[-1]
    ax.scatter(start[0], start[1], start[2], color="green", marker="o", s=80, label="Start")
    ax.scatter(end[0], end[1], end[2], color="red", marker="o", s=80, label="End")

    final_desired = desired_traj[-1]
    ax.scatter(final_desired[0], final_desired[1], final_desired[2], color="purple", marker="*", s=140, label="Desired (final)")

    final_distance = np.linalg.norm(end - final_desired)
    ax.set_title(f"3D Trajectory — Final distance: {final_distance:.4f} m")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"3D trajectory saved to '{save_path}'")
    plt.show()


# --------------------------
# Main flow
# --------------------------
def main():
    cfg, final_dir = ensure_config()
    num_poses = cfg["num_poses"]

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize simulation and model
    sim, dyn_model, init_joint_angles, joint_vel_limits, time_step = setup_sim(conf_file_name, root_dir)
    controlled_frame_name = "panda_link8"

    # Print basic info (kept for debugging)
    print(f"Initial joint angles: {init_joint_angles}")
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    print(f"joint vel limits: {joint_vel_limits}")

    # Generate workspace targets
    (
        list_of_desired_cartesian_positions,
        list_of_desired_cartesian_orientations,
        list_of_type_of_control,
        list_of_duration_per_desired_cartesian_positions,
        list_of_initialjoint_positions,
    ) = generate_data(num_poses=num_poses, init_joint_angles=init_joint_angles)

    # Iterate targets and collect data
    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        init_pos_for_this = (
            init_joint_angles if list_of_initialjoint_positions[i] is None else list_of_initialjoint_positions[i]
        )

        # Collect trajectory for this target
        results = collect_trajectory_for_target(
            sim=sim,
            dyn_model=dyn_model,
            controlled_frame_name=controlled_frame_name,
            init_position=init_pos_for_this,
            desired_cartesian_pos=desired_cartesian_pos,
            desired_cartesian_ori=desired_cartesian_ori,
            duration=duration,
            joint_vel_limits=joint_vel_limits,
            kp_pos=100,
            kp_ori=0,
            kp=1000,
            kd=100,
            downsample_rate=DOWN_SAMPLE_RATE,
            extra_steps_after_reach=EXTRA_STEPS_AFTER_REACH,
        )

        print(f"Iteration {i}: final_cart_distance = {results['final_cart_distance']:.6f}, saved_steps = {results['saved_steps']}")

        # Decide whether to save
        save_flag = results["saved_steps"] > 0 and results["final_cart_distance"] < SAVE_THRESHOLD
        if save_flag and DATA_CATEGORY != "simulation":
            saved_file = save_dataset(i, results, final_dir)
            print(f"Saved data {saved_file}")
        else:
            print(f"Skipping data save for iteration {i}: distance {results['final_cart_distance']:.6f} exceeds threshold")

        # Optional plotting for diagnostics (per-iteration)
        if PRINT_PLOTS and results["q_mes"].numel() > 0:
            time_array = [time_step * DOWN_SAMPLE_RATE * idx for idx in range(results["q_mes"].shape[0])]
            # Plot a single joint angle trace as example
            plot_time_series(time_array, [results["q_mes"][:, 0].numpy()], ["Joint 1"], "Measured Joint 1", "Angle (rad)")
            # 3D trajectory plot
            actual_traj = results["cart_pos"].numpy() if results["cart_pos"].numel() else np.array([])
            desired_traj = results["final_cart_pos"].numpy() if results["final_cart_pos"].numel() else np.array([])
            if actual_traj.size and desired_traj.size:
                plot_3d_trajectory(actual_traj, desired_traj, save_path=final_dir / f"trajectory_{i}.png")


if __name__ == "__main__":
    main()