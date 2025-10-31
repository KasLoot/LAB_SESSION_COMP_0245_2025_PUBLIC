# Model Evaluation Script (eval_p2.py)

## Overview

This script evaluates trained neural network models for robot control by running simulations with the Panda robot. It compares the model's predictions against real-time robot behavior and generates comprehensive visualizations and performance metrics.

## Purpose

The evaluation script:
1. Loads a trained neural network model
2. Generates random test target positions
3. Uses the model to predict desired joint positions/velocities
4. Executes robot control based on model predictions
5. Visualizes trajectories and performance
6. Saves plots and results to disk

## Key Features

- **Real-time Simulation**: Tests model predictions in actual robot simulation
- **Comprehensive Visualization**: Generates plots for joint positions, velocities, 3D trajectories, and errors
- **Multiple Test Poses**: Evaluates model on various target positions
- **Performance Metrics**: Computes Cartesian position errors and final distances
- **Interactive 3D Plots**: Optional interactive visualization of end-effector trajectories

## Usage

### Prerequisites

Before running the evaluation, ensure you have:
1. A trained model file (`.pth` format)
2. The corresponding model class definition in `part_2.py`

### Basic Usage

Run the script directly:

```bash
python eval_p2.py
```

The script will:
1. Load the trained model
2. Generate random test positions
3. Simulate robot trajectories using model predictions
4. Save visualization plots to the results directory

### Output

Results are saved in the directory specified by `RESULTS_DIR`:
- Joint position comparison plots
- Joint velocity comparison plots
- 3D trajectory visualizations
- Cartesian position error plots

## Key Parameters to Tune

### 1. Model Configuration

**Location**: Lines 23-27

```python
MODEL_CLASS = "none_encoder"  # encoder or none_encoder
MODEL_PATH = FINAL_DIR / "part2_none_encoder_with_stop.pth"
RESULTS_DIR = FINAL_DIR / "part2_none_encoder_with_stop_results"
```

**What it does**:
- `MODEL_CLASS`: Specifies which model architecture to use
  - `"encoder"`: Use `P2_MLP_Encoder` class
  - `"none_encoder"`: Use `P2_MLP_None_Encoder` class
- `MODEL_PATH`: Path to the trained model weights file
- `RESULTS_DIR`: Directory where plots and results will be saved

**When to change**:
- Update `MODEL_PATH` to evaluate a different trained model
- Change `MODEL_CLASS` to match your model architecture
- Modify `RESULTS_DIR` to organize results from different experiments

---

### 2. Number of Test Poses

**Location**: Line 31

```python
NUM_TEST_POSES = 10  # Number of different target positions to test
```

**What it does**: Determines how many random target positions to test.

**When to change**:
- Increase (e.g., 20-50) for more comprehensive evaluation
- Decrease (e.g., 3-5) for quick testing
- More poses = better statistical evaluation but longer runtime

---

### 3. Duration Per Pose

**Location**: Line 32

```python
DURATION_PER_POSE = 5.0  # Duration in seconds for each pose
```

**What it does**: Maximum simulation time for each target position.

**When to change**:
- Increase (e.g., 8.0-10.0) if robot needs more time to reach targets
- Decrease (e.g., 3.0) for faster evaluation
- Should match or exceed the duration used during training data generation

---

### 4. Random Seed

**Location**: Line 18

```python
np.random.seed(78)  # Using test seed for evaluation
```

**What it does**: Sets the random seed for reproducible test positions.

**When to change**:
- Use different seed for different test sets
- Keep constant for reproducible comparisons between models
- Recommended: Use a seed different from training (42) and test data generation (56)

---

### 5. Interactive Visualization

**Location**: Line 35

```python
SHOW_INTERACTIVE_3D = False  # Set to True to show interactive 3D plots
```

**What it does**: Controls whether interactive 3D trajectory plots are displayed.

**When to change**:
- Set to `True` to interactively explore 3D trajectories
- Keep `False` for batch processing (all plots saved automatically)
- **Note**: When `True`, you must close each plot window to continue

---

### 6. Test Position Ranges

**Location**: Lines 73-80 in `generate_test_positions()` function

```python
x = np.random.uniform(0.2, 0.5)
y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
z = np.random.uniform(0.1, 0.6)
```

**What it does**: Defines the workspace for random test targets.

**When to change**:
- Should match the ranges used during training data generation
- Adjust to test model performance in specific workspace regions
- Can intentionally use different ranges to test generalization

---

### 7. Controller Gains

**Location**: Lines 145-146 (function signature) and actual values passed in main()

```python
kp=1000  # Proportional gain
kd=100   # Derivative gain
```

**What it does**: Low-level PD controller gains for torque control.

**When to change**:
- Should match the gains used during training data generation
- Increase for stiffer tracking of model predictions
- Decrease if robot oscillates or becomes unstable

---

## Workflow for Model Evaluation

### Step 1: Prepare Your Model

Ensure your trained model file exists:
```bash
ls final_1/part2_none_encoder_with_stop.pth
```

### Step 2: Configure Evaluation Parameters

Edit the script to set:
1. Correct `MODEL_CLASS` (line 24)
2. Correct `MODEL_PATH` (line 25)
3. Desired `RESULTS_DIR` (line 26)
4. Number of test poses (line 31)

### Step 3: Run Evaluation

```bash
python eval_p2.py
```

### Step 4: Review Results

Check the results directory:
```bash
ls final_1/part2_none_encoder_with_stop_results/
```

You should see:
- `joint_positions_pose_1.png`, `joint_positions_pose_2.png`, ...
- `joint_velocities_pose_1.png`, `joint_velocities_pose_2.png`, ...
- `3d_trajectory_pose_1.png`, `3d_trajectory_pose_2.png`, ...
- `cartesian_error_pose_1.png`, `cartesian_error_pose_2.png`, ...

### Step 5: Analyze Performance

Look at the plots to assess:
1. **Joint Positions**: How well does the robot follow model predictions?
2. **Joint Velocities**: Are velocity predictions smooth and accurate?
3. **3D Trajectories**: Does the end-effector reach the target? Is the path reasonable?
4. **Cartesian Errors**: How close does the robot get to the target position?

## Understanding the Plots

### Joint Positions Comparison

Shows measured vs. model-predicted joint angles over time for all 7 joints.
- **Good**: Measured and predicted lines closely overlap
- **Bad**: Large gaps between measured and predicted

### Joint Velocities Comparison

Shows measured vs. model-predicted joint velocities over time.
- **Good**: Smooth velocities, no oscillations
- **Bad**: Jerky motion, oscillations, or large discrepancies

### 3D Trajectory

Shows the end-effector path in 3D space.
- **Green circle**: Start position
- **Blue square**: End position
- **Red star**: Target position
- **Color gradient**: Time progression (dark to light)

**Good performance**: Blue square very close to red star

### Cartesian Position Error

- **Top plot**: Component-wise errors (X, Y, Z)
- **Bottom plot**: Total error magnitude over time

**Good performance**: Error decreases to near zero by the end

## Performance Metrics

The script prints metrics during execution:

```
Final Cartesian Error: X.XXXX m
```

**Interpretation**:
- < 0.001 m (1 mm): Excellent
- 0.001-0.005 m (1-5 mm): Good
- 0.005-0.01 m (5-10 mm): Acceptable
- > 0.01 m (10 mm): Poor (model needs improvement)

## Troubleshooting

### Issue: Model file not found

**Error**: `FileNotFoundError` or similar

**Solutions**:
- Check that `MODEL_PATH` points to the correct file
- Ensure the model has been trained and saved
- Use absolute path if relative path fails

### Issue: Robot doesn't reach targets

**Possible causes**:
- Model not trained properly
- Test positions outside training distribution
- Controller gains mismatch between training and evaluation

**Solutions**:
- Increase `DURATION_PER_POSE` to give more time
- Check that test position ranges match training ranges
- Verify model training loss converged
- Ensure controller gains match training configuration

### Issue: Robot oscillates or is unstable

**Solutions**:
- Decrease `kp` and/or `kd` values
- Check if model predictions are reasonable (not NaN or extremely large)
- Verify model loaded correctly

### Issue: Plots not saved

**Solutions**:
- Check that `RESULTS_DIR` is created (should be automatic)
- Ensure write permissions for the results directory
- Check for errors in plot generation code

### Issue: Out of memory

**Solutions**:
- Decrease `NUM_TEST_POSES`
- Close plot windows if `SHOW_INTERACTIVE_3D = True`
- Run on a machine with more RAM

## Comparing Multiple Models

To compare different models:

1. Evaluate Model 1:
   ```python
   MODEL_PATH = FINAL_DIR / "model1.pth"
   RESULTS_DIR = FINAL_DIR / "model1_results"
   ```
   Run the script.

2. Evaluate Model 2:
   ```python
   MODEL_PATH = FINAL_DIR / "model2.pth"
   RESULTS_DIR = FINAL_DIR / "model2_results"
   ```
   Run the script.

3. Compare the results directories visually or quantitatively.

## Advanced Usage

### Evaluating with Custom Test Positions

To test specific positions instead of random ones, modify `generate_test_positions()`:

```python
def generate_test_positions(num_poses, init_joint_angles):
    desired_positions = [
        [0.4, 0.3, 0.2],  # Custom position 1
        [0.3, -0.3, 0.4], # Custom position 2
        # Add more...
    ]
    desired_orientations = [[0.0, 0.0, 0.0, 1.0]] * len(desired_positions)
    initial_joint_positions = [init_joint_angles] * len(desired_positions)
    return desired_positions, desired_orientations, initial_joint_positions
```

### Saving Quantitative Metrics

To save numerical results to a file, add at the end of `main()`:

```python
import json

metrics = {
    'num_poses': NUM_TEST_POSES,
    'final_errors': [],  # Collect from each trajectory
    'mean_error': 0.0,
    # Add more metrics...
}

with open(RESULTS_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Tips for Best Results

1. **Use the same seed for fair comparison**: Keep random seed constant when comparing models
2. **Match training conditions**: Use same controller gains and target ranges as training
3. **Run multiple evaluations**: Different random seeds give different test sets
4. **Check all plot types**: Different plots reveal different aspects of performance
5. **Compare against baseline**: Evaluate both with and without certain features

## Expected Runtime

- **Per pose**: ~5-10 seconds (depending on `DURATION_PER_POSE`)
- **10 poses**: ~1-2 minutes total
- **50 poses**: ~5-10 minutes total

Actual time depends on your hardware and whether interactive plots are enabled.
