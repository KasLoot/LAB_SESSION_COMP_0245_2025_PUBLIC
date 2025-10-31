# Data Generator (dg.py)

## Overview

This script generates training and test datasets for robot control by simulating a Panda robotic arm reaching random target positions in Cartesian space. The generated data includes joint positions, desired joint positions, and velocities that can be used to train neural network models for robot control.

## Purpose

The data generator creates trajectories where the robot moves from its initial position to randomly generated target positions using Cartesian differential kinematics. It records the robot's state throughout each trajectory and saves the data for machine learning purposes.

## Key Features

- **Random Target Generation**: Generates random Cartesian positions within the robot's workspace
- **Cartesian Control**: Uses differential kinematics to compute desired joint positions from Cartesian targets
- **Data Downsampling**: Configurable downsampling rate to reduce dataset size
- **Quality Control**: Only saves trajectories that successfully reach the target position
- **Configurable Dataset Size**: Separate configurations for training and test data

## Usage

### Basic Usage

Run the script directly to generate data:

```bash
python dg.py
```

The script will:
1. Initialize the Panda robot simulation
2. Generate random target positions
3. Simulate trajectories to each target
4. Save successful trajectories to disk

### Output

Generated data files are saved in:
- **Training data**: `final_1/data/train/data_0.pt`, `data_1.pt`, etc.
- **Test data**: `final_1/data/test/data_0.pt`, `data_1.pt`, etc.

Each `.pt` file contains:
- `q_mes_all`: Measured joint positions (7 joints)
- `final_cartesian_pos`: Target Cartesian position (x, y, z)
- `q_d_all`: Desired joint positions
- `qd_d_all`: Desired joint velocities

## Key Parameters to Tune

### 1. Data Category

**Location**: Line 23
```python
data_category = "test"  # "train" or "test" or "simulation"
```

**Options**:
- `"train"`: Generate training dataset (200 poses, seed=42)
- `"test"`: Generate test dataset (40 poses, seed=56)
- `"simulation"`: Quick simulation mode (5 poses, seed=100)

**When to change**: Switch between generating training data and test data.

---

### 2. Number of Poses

**Location**: Lines 25-33

**Training data**:
```python
num_of_poses = 200
```

**Test data**:
```python
num_of_poses = 40
```

**When to change**: 
- Increase for larger datasets (more training data)
- Decrease for faster generation or smaller datasets
- Recommended: 200-500 for training, 40-100 for testing

---

### 3. Downsample Rate

**Location**: Line 39
```python
downsample_rate = 2
```

**What it does**: Controls how many simulation steps to skip when saving data. A value of 2 means every other step is saved.

**When to change**:
- Increase (e.g., 5-10) to reduce dataset size and memory usage
- Decrease (e.g., 1) for higher temporal resolution
- Trade-off: Higher values = smaller files but less detailed trajectories

---

### 4. Target Position Ranges

**Location**: Lines 56-66 in `generate_data()` function

```python
x = np.random.uniform(0.2, 0.5)
y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
z = np.random.uniform(0.1, 0.6)
```

**What it does**: Defines the workspace bounds for random target generation.
- `x`: 0.2 to 0.5 meters (forward reach)
- `y`: 0.2 to 0.5 OR -0.5 to -0.2 meters (left/right)
- `z`: 0.1 to 0.6 meters (height)

**When to change**:
- Adjust based on robot's reachable workspace
- Increase ranges to explore more diverse positions
- Decrease ranges to focus on specific workspace regions
- **Warning**: Targets outside reachable workspace will fail

---

### 5. Duration Per Trajectory

**Location**: Line 69
```python
list_of_duration_per_desired_cartesian_positions.append(2.0)  # in seconds
```

**What it does**: Maximum time allowed for reaching each target.

**When to change**:
- Increase (e.g., 3.0-5.0) for distant targets or slower motion
- Decrease (e.g., 1.0-1.5) for faster data generation
- Trade-off: Longer duration = smoother trajectories but slower generation

---

### 6. Controller Gains

**Location**: Lines 126-132

**High-level Cartesian controller**:
```python
kp_pos = 100  # position gain
kp_ori = 0    # orientation gain (not used in position-only control)
```

**Low-level PD controller**:
```python
kp = 1000  # proportional gain
kd = 100   # derivative gain
```

**When to change**:
- **kp_pos**: Higher values = faster convergence to target (but may oscillate)
- **kp/kd**: Higher values = stiffer control (but may cause instability)
- Start with default values and adjust if:
  - Robot moves too slowly: Increase `kp_pos`
  - Robot oscillates: Decrease `kp` or increase `kd`
  - Robot is unstable: Decrease both `kp` and `kd`

---

### 7. Convergence Threshold

**Location**: Line 219
```python
if cart_distance < 5e-5:
```

**What it does**: Distance threshold (in meters) to consider target "reached".

**When to change**:
- Decrease (e.g., 1e-5) for more precise trajectories
- Increase (e.g., 1e-4) to accept less precise convergence
- Trade-off: Smaller threshold = higher quality data but more failed trajectories

---

### 8. Visualization and Recording

**Location**: Lines 20-21
```python
PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True     # Set to True to enable data recording
```

**When to change**:
- Set `PRINT_PLOTS = True` to visualize trajectories during generation
- Set `RECORDING = False` to run simulation without saving data (testing)

---

## Workflow for Generating Data

### Step 1: Generate Training Data

1. Edit `data_category = "train"` (line 23)
2. Optionally adjust `num_of_poses` for training set size
3. Run the script:
   ```bash
   python dg.py
   ```
4. Wait for completion (may take several minutes)

### Step 2: Generate Test Data

1. Edit `data_category = "test"` (line 23)
2. Optionally adjust `num_of_poses` for test set size
3. Run the script:
   ```bash
   python dg.py
   ```

### Step 3: Verify Generated Data

Check the output directories:
```bash
ls final_1/data/train/  # Should see data_0.pt, data_1.pt, ...
ls final_1/data/test/   # Should see data_0.pt, data_1.pt, ...
```

## Troubleshooting

### Issue: Many trajectories failing (cart_distance > threshold)

**Solutions**:
- Increase `duration_per_desired_cartesian_positions` (allow more time)
- Increase convergence threshold (line 219)
- Reduce target position ranges (make targets easier to reach)
- Increase `kp_pos` (faster convergence)

### Issue: Robot oscillates or becomes unstable

**Solutions**:
- Decrease `kp` and/or `kd` values
- Decrease `kp_pos` value
- Increase `duration_per_desired_cartesian_positions`

### Issue: Data generation too slow

**Solutions**:
- Decrease `num_of_poses`
- Increase `downsample_rate`
- Decrease `duration_per_desired_cartesian_positions`
- Set `PRINT_PLOTS = False`

### Issue: Files too large

**Solutions**:
- Increase `downsample_rate` (e.g., from 2 to 5 or 10)
- Decrease `duration_per_desired_cartesian_positions`
- Reduce `num_of_poses`

## Advanced Configuration

### Custom Initial Joint Angles

To start from different initial configurations, modify line 70:
```python
list_of_initialjoint_positions.append(init_joint_angles)  # Use custom angles here
```

### Different Control Types

Change control type from position-only to include orientation (line 68):
```python
list_of_type_of_control.append("both")  # "pos", "ori", or "both"
```

## Data Format

Each saved `.pt` file is a PyTorch tensor dictionary with:
- **Shape**: All arrays have shape `(num_timesteps, num_joints)`
- **num_timesteps**: Varies per trajectory (depends on convergence time and downsample_rate)
- **num_joints**: 7 (Panda robot has 7 joints)

Example loading:
```python
import torch
data = torch.load("final_1/data/train/data_0.pt")
print(data['q_mes_all'].shape)  # e.g., (500, 7)
print(data['final_cartesian_pos'].shape)  # e.g., (500, 3)
```

## Notes

- The script uses fixed random seeds for reproducibility
- Only trajectories that successfully reach the target are saved
- The simulation runs in real-time (with `time.sleep(time_step)`)
- Press 'q' during simulation to exit early
