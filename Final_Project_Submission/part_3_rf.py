"""
Evaluation script for RandomForest model (part_2_rf).
"""
import os
import time
from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt

from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# reuse plotting helpers from part_3
from part_3 import (
    generate_test_positions,
    plot_joint_positions_comparison,
    plot_joint_velocities_comparison,
    plot_3d_trajectory,
    plot_cartesian_position_error,
)

# configuration
FINAL_DIR = Path(__file__).resolve().parent
RF_MODEL_PATH = FINAL_DIR / "part2_rf_model.pkl"
RESULTS_DIR = FINAL_DIR / "evaluation_results_rf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_TEST_POSES = 10
DURATION_PER_POSE = 5.0
SHOW_INTERACTIVE_3D = False
SEED = 100
np.random.seed(SEED)


# -------------------------
# RF helper functions
# -------------------------
def load_rf_model(model_path: Path):
    """Load RandomForest model saved with joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"RF model not found: {model_path}")
    rf = joblib.load(model_path)
    print(f"✓ RandomForest model loaded from {model_path}")
    return rf


def predict_with_rf(rf_model, q_mes, desired_cartesian_pos):
    """
    Predict using RF model.
    Input: q_mes (7,), desired_cartesian_pos (3,) -> concatenated (10,)
    Output: q_des (7,), qd_des (7,)
    """
    inp = np.concatenate([np.asarray(q_mes).ravel(), np.asarray(desired_cartesian_pos).ravel()])[None, :]
    pred = rf_model.predict(inp)  # shape (1, 14) or (14,)
    pred = np.asarray(pred).reshape(-1)
    q_des = pred[:7].copy()
    qd_des = pred[7:14].copy()
    return q_des, qd_des


def simulate_single_trajectory_rf(
    sim: pb.SimInterface,
    dyn_model,
    rf_model,
    desired_cartesian_pos,
    init_joint_angles,
    duration,
    controlled_frame_name,
    kp=1000,
    kd=100,
):
    """
    Simulate trajectory using RF predictions as desired joint references.
    Returns trajectory_data dict compatible with plotting helpers from part_3.

    Added runtime checks and logging to avoid silent hangs:
      - verify time_step > 0
      - cap max steps
      - print occasional progress
      - detect NaN/Inf in control outputs and clamp to zero
    """
    # reset and set initial pose
    sim.ResetPose()
    if init_joint_angles is not None:
        sim.SetjointPosition(init_joint_angles)
    time.sleep(0.05)

    time_step = sim.GetTimeStep()
    if not (isinstance(time_step, (int, float)) and time_step > 0):
        print(f"[WARN] sim.GetTimeStep() returned {time_step!r}, falling back to 0.001s")
        time_step = 1e-3

    steps = int(max(1, round(duration / time_step)))

    # safety cap to avoid runaway long loops
    MAX_STEPS = 20000
    if steps > MAX_STEPS:
        print(f"[WARN] requested steps={steps} exceeds cap {MAX_STEPS}, capping.")
        steps = MAX_STEPS

    cmd = MotorCommands()

    # store
    traj = {
        "time": [],
        "q_mes": [],
        "qd_mes": [],
        "q_des_model": [],
        "qd_des_model": [],
        "cart_pos": [],
        "cart_ori": [],
        "tau_cmd": [],
        "desired_cart_pos": np.array(desired_cartesian_pos),
    }

    # initial measurement
    q_mes_init = sim.GetMotorAngles(0)
    qd_mes_init = sim.GetMotorVelocities(0)
    cart_pos_init, cart_ori_init = dyn_model.ComputeFK(q_mes_init, controlled_frame_name)

    print(f"[INFO] start sim: duration={duration}s time_step={time_step}s steps={steps}")
    print(f"[INFO] init q_mes={np.asarray(q_mes_init)} desired_pos={np.asarray(desired_cartesian_pos)}")

    traj["time"].append(0.0)
    traj["q_mes"].append(np.array(q_mes_init).copy())
    traj["qd_mes"].append(np.array(qd_mes_init).copy())
    traj["q_des_model"].append(np.array(q_mes_init).copy())  # prior to control
    traj["qd_des_model"].append(np.array(qd_mes_init).copy())
    traj["cart_pos"].append(np.array(cart_pos_init).copy())
    traj["cart_ori"].append(np.array(cart_ori_init).copy())
    traj["tau_cmd"].append(np.zeros(len(q_mes_init)))

    for step in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)

        # current fk
        cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)

        # RF prediction
        q_des, qd_des = predict_with_rf(rf_model, q_mes, desired_cartesian_pos)

        # detect bad predictions
        if np.any(np.isnan(q_des)) or np.any(np.isnan(qd_des)) or np.any(np.isinf(q_des)) or np.any(np.isinf(qd_des)):
            print(f"[WARN] Bad RF output at step {step}: NaN/Inf detected. Replacing with zeros.")
            q_des = np.zeros_like(q_mes)
            qd_des = np.zeros_like(q_mes)

        # control (feedback linearization as in part_3)
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        if np.any(np.isnan(tau_cmd)) or np.any(np.isinf(tau_cmd)):
            print(f"[WARN] Bad tau_cmd at step {step}: NaN/Inf detected. Clamping to zeros.")
            tau_cmd = np.zeros_like(tau_cmd)

        cmd.SetControlCmd(tau_cmd, ["torque"] * len(q_mes))
        sim.Step(cmd, "torque")

        # wait for physics step to complete (keep as small sleep to avoid busy-wait)
        current_time = (step + 1) * time_step
        time.sleep(time_step)

        # record post-step state
        q_mes_next = sim.GetMotorAngles(0)
        qd_mes_next = sim.GetMotorVelocities(0)
        cart_pos_next, cart_ori_next = dyn_model.ComputeFK(q_mes_next, controlled_frame_name)

        traj["time"].append(current_time)
        traj["q_mes"].append(np.array(q_mes_next).copy())
        traj["qd_mes"].append(np.array(qd_mes_next).copy())
        traj["q_des_model"].append(np.array(q_des).copy())
        traj["qd_des_model"].append(np.array(qd_des).copy())
        traj["cart_pos"].append(np.array(cart_pos_next).copy())
        traj["cart_ori"].append(np.array(cart_ori_next).copy())
        traj["tau_cmd"].append(np.array(tau_cmd).copy())

        # occasional progress log
        if (step + 1) % 200 == 0 or step == steps - 1:
            dist = np.linalg.norm(np.asarray(cart_pos_next) - np.asarray(desired_cartesian_pos))
            print(f"[DEBUG] step {step+1}/{steps} time={current_time:.3f}s cart_dist={dist:.6f}m")

    # convert lists to arrays
    for k in ["time", "q_mes", "qd_mes", "q_des_model", "qd_des_model", "cart_pos", "cart_ori", "tau_cmd"]:
        traj[k] = np.asarray(traj[k])

    return traj


def main():
    """
    Main evaluation function.
    """
    print("\n" + "="*70)
    print("PART 2 MODEL EVALUATION")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # 1. Setup Simulation and Model
    # -------------------------------------------------------------------------
    print("Step 1: Setting up simulation environment...")
    
    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    
    # Get active joint names
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    
    # Create dynamic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, 
                          source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    controlled_frame_name = "panda_link8"
    
    # Get initial joint configuration
    init_joint_angles = sim.GetInitMotorAngles()
    print(f"  ✓ Number of joints: {num_joints}")
    print(f"  ✓ Controlled frame: {controlled_frame_name}")
    print(f"  ✓ Initial joint angles: {init_joint_angles}")
    
    # -------------------------------------------------------------------------
    # 2. Load Trained Model
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading trained model...")
    rf_model = load_rf_model(RF_MODEL_PATH)
    
    # -------------------------------------------------------------------------
    # 3. Generate Test Positions
    # -------------------------------------------------------------------------
    print(f"\nStep 3: Generating {NUM_TEST_POSES} test positions...")
    
    desired_positions, desired_orientations, initial_joint_positions = \
        generate_test_positions(NUM_TEST_POSES, init_joint_angles)
    
    for i, pos in enumerate(desired_positions):
        print(f"  Pose {i+1}: Target = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # -------------------------------------------------------------------------
    # 4. Run Simulations and Generate Plots
    # -------------------------------------------------------------------------
    print(f"\nStep 4: Running simulations and generating plots...")
    print(f"  Duration per pose: {DURATION_PER_POSE}s")
    print(f"  Results directory: {RESULTS_DIR}\n")
    
    all_final_errors = []
    
    for i in range(NUM_TEST_POSES):
        print(f"Evaluating Pose {i+1}/{NUM_TEST_POSES}...")
        
        # Run simulation
        trajectory_data = simulate_single_trajectory_rf(
            sim=sim,
            dyn_model=dyn_model,
            rf_model=rf_model,
            desired_cartesian_pos=desired_positions[i],
            init_joint_angles=init_joint_angles,
            duration=DURATION_PER_POSE,
            controlled_frame_name=controlled_frame_name,
        )
        
        # Compute final error
        final_cart_pos = trajectory_data['cart_pos'][-1]
        target_pos = trajectory_data['desired_cart_pos']
        final_error = np.linalg.norm(final_cart_pos - target_pos)
        all_final_errors.append(final_error)
        
        print(f"  Final position error: {final_error:.4f} m")
        
        # Generate plots
        print(f"  Generating plots...")
        plot_joint_positions_comparison(trajectory_data, i, RESULTS_DIR)
        plot_joint_velocities_comparison(trajectory_data, i, RESULTS_DIR)
        plot_3d_trajectory(trajectory_data, i, RESULTS_DIR, interactive=SHOW_INTERACTIVE_3D)
        plot_cartesian_position_error(trajectory_data, i, RESULTS_DIR)
        print()
    
    # -------------------------------------------------------------------------
    # 5. Summary Statistics
    # -------------------------------------------------------------------------
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Number of test poses: {NUM_TEST_POSES}")
    print(f"Mean final position error: {np.mean(all_final_errors):.4f} m")
    print(f"Std final position error: {np.std(all_final_errors):.4f} m")
    print(f"Min final position error: {np.min(all_final_errors):.4f} m")
    print(f"Max final position error: {np.max(all_final_errors):.4f} m")
    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()