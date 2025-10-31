# Data Generator with Stop Data (dg_with_stop_data.py)

## Overview

This script is an enhanced version of the basic data generator (`dg.py`) that collects additional data **after** the robot reaches its target position. This "stop data" captures the robot's behavior at the target, which can be valuable for training models to understand steady-state behavior and improve final positioning accuracy.

## Purpose

The key difference from the basic data generator is that this script continues collecting data for a specified number of steps **after** the robot successfully reaches the target position. This creates datasets with both:
1. **Approach phase**: Robot moving toward the target
2. **Settled phase**: Robot maintaining position at the target

## Key Features

- All features from `dg.py`, plus:
- **Extended data collection**: Continues recording after reaching target
- **Steady-state data**: Captures robot behavior when stationary at goal
- **Configurable stop duration**: Adjustable number of steps to collect after convergence

## Usage

### Basic Usage

Run the script directly:

```bash
python dg_with_stop_data.py
```

The script operates identically to `dg.py` but saves longer trajectories that include the "stop" phase.

### Output

Generated data files are saved in:
- **Training data**: `final_1/data/train_with_stop_data/data_0.pt`, `data_1.pt`, etc.
- **Test data**: `final_1/data/test_with_stop_data/data_0.pt`, `data_1.pt`, etc.

**Note**: The directory names include `_with_stop_data` to distinguish from regular data.

## Key Differences from dg.py

### Main Change: Extra Steps After Convergence

**Location**: Lines 181-196

The critical addition is:

```python
reached_target = False  # Flag to track if target is reached
steps_after_reaching_target = 0  # Counter for steps after reaching target
extra_steps = 500  # Number of additional steps to collect after reaching target
```

And in the main loop:

```python
if cart_distance < 5e-5 and not reached_target:
    print(f"Reached desired position at time {current_time:.2f} seconds. Collecting {extra_steps} more steps...")
    reached_target = True

# Count steps after reaching target
if reached_target:
    steps_after_reaching_target += 1
    if steps_after_reaching_target >= extra_steps:
        print(f"Collected {extra_steps} steps after reaching target. Moving to next trajectory.")
        break
```

**What this does**:
1. Robot moves toward target normally
2. When `cart_distance < 5e-5`, flag is set but simulation continues
3. Collects `extra_steps` more data points
4. Then moves to next trajectory

## Key Parameters to Tune

All parameters from `dg.py` apply, plus:

### Extra Steps After Reaching Target

**Location**: Line 183

```python
extra_steps = 500  # Number of additional steps to collect after reaching target
```

**What it does**: Specifies how many simulation steps to continue after reaching the target.

**When to change**:
- **Increase** (e.g., 1000-2000):
  - For longer steady-state periods
  - If you want more data showing robot at target
  - For training models to maintain position
  
- **Decrease** (e.g., 200-300):
  - For faster data generation
  - If you only need a brief confirmation that target was reached
  - To reduce file sizes

**Time calculation**:
```
actual_time = extra_steps * time_step
```
With default `time_step = 0.001` seconds:
- 500 steps = 0.5 seconds of stop data
- 1000 steps = 1.0 seconds of stop data
- 2000 steps = 2.0 seconds of stop data

**Recommendation**: Start with 500 and adjust based on:
- Model performance (does it need more steady-state examples?)
- File size constraints
- Data generation time

---

### Other Important Parameters

All other parameters are identical to `dg.py`. Refer to the `README_dg.md` for:
- Data category selection
- Number of poses
- Downsample rate
- Target position ranges
- Duration per trajectory
- Controller gains
- Convergence threshold

## When to Use This Script vs. Regular dg.py

### Use `dg_with_stop_data.py` when:

1. **Training models for precise positioning**: The stop data shows the model what "success" looks like
2. **Learning steady-state behavior**: Important for applications requiring stable positioning
3. **Improving final accuracy**: Extra examples at the target help models learn to stay there
4. **Training with velocity information**: The stop data has near-zero velocities, helping models learn to stop

### Use regular `dg.py` when:

1. **Only need trajectory data**: If you only care about the motion, not the final state
2. **Faster data generation**: Regular version finishes each trajectory quicker
3. **Smaller file sizes**: Without stop data, trajectories are shorter
4. **Training for motion planning**: If focus is on path, not destination maintenance

## Workflow for Generating Data with Stop

### Step 1: Configure the Script

1. Set `data_category` (train, test, or simulation)
2. Adjust `extra_steps` based on your needs
3. Set other parameters as needed (see `README_dg.md`)

### Step 2: Generate Training Data with Stop

```python
# In the script
data_category = "train"
extra_steps = 500  # Adjust as needed
```

```bash
python dg_with_stop_data.py
```

### Step 3: Generate Test Data with Stop

```python
# In the script
data_category = "test"
extra_steps = 500  # Keep same as training
```

```bash
python dg_with_stop_data.py
```

### Step 4: Verify Data

Check that files contain the extra timesteps:

```python
import torch

# Load a file
data = torch.load("final_1/data/train_with_stop_data/data_0.pt")

# Check length
print(f"Number of timesteps: {len(data['q_mes_all'])}")
print(f"Expected minimum: {500 / 2}")  # extra_steps / downsample_rate
```

## Understanding the Generated Trajectories

### Trajectory Structure

With `extra_steps = 500` and `downsample_rate = 2`:

```
[--- Approach Phase ---][--- Stop Phase ---]
0 .................. t_reach ......... t_end

Approach: Variable length (depends on target distance)
Stop: ~250 timesteps (500 steps / downsample_rate)
```

### Data Characteristics

**Approach Phase**:
- Joint positions changing toward desired configuration
- Non-zero joint velocities
- Cartesian position error decreasing

**Stop Phase**:
- Joint positions nearly constant (at target)
- Joint velocities near zero
- Cartesian position error < 5e-5 meters
- Model learns: "When at target, maintain position with zero velocity"

## Comparing Data With and Without Stop

### File Size Comparison

Approximate sizes (depends on trajectory length):

| Dataset | Without Stop | With Stop (500 steps) | Increase |
|---------|--------------|----------------------|----------|
| Training (200 poses) | ~50-100 MB | ~80-150 MB | ~50% |
| Test (40 poses) | ~10-20 MB | ~15-30 MB | ~50% |

### Training Performance

**Benefits of stop data**:
- ✅ Better final positioning accuracy
- ✅ Model learns to maintain position
- ✅ More balanced dataset (includes static states)
- ✅ Helps with stability at target

**Potential drawbacks**:
- ❌ Larger files
- ❌ Longer training time (more data points)
- ❌ May overfit to staying at target if `extra_steps` too large

## Tuning Tips

### Finding Optimal extra_steps

**Too small** (< 200):
- Model may not learn steady-state well
- Less improvement in final positioning

**Too large** (> 2000):
- Unnecessarily large files
- Slower training
- Diminishing returns

**Recommended**: 500-1000 steps

### Balancing with downsample_rate

If you increase `extra_steps`, consider increasing `downsample_rate` to keep file sizes manageable:

```python
# Example 1: High resolution stop data
extra_steps = 1000
downsample_rate = 2
# Result: 500 stop timesteps per trajectory

# Example 2: Same number of stop timesteps, faster generation
extra_steps = 2000
downsample_rate = 4
# Result: 500 stop timesteps per trajectory
```

## Troubleshooting

### Issue: Trajectories too long / files too large

**Solutions**:
- Decrease `extra_steps`
- Increase `downsample_rate`
- Decrease `num_of_poses`

### Issue: Model doesn't improve with stop data

**Possible causes**:
- Too much stop data (model overfits to staying still)
- Not enough variety in approach trajectories

**Solutions**:
- Decrease `extra_steps` (e.g., from 500 to 200)
- Increase `num_of_poses` to get more diverse approach trajectories
- Try training without stop data for comparison

### Issue: Data generation very slow

**Solutions**:
- The extra steps add time. Either:
  - Decrease `extra_steps`
  - Decrease `num_of_poses`
  - Use parallel data generation (run multiple instances)

## Advanced: Custom Stop Conditions

You can modify the stop condition to collect data based on other criteria:

```python
# Example: Stop based on velocity instead of position
if np.linalg.norm(qd_mes) < 1e-3 and not reached_target:
    print("Robot velocity near zero, collecting stop data...")
    reached_target = True

# Example: Stop based on time at target
time_at_target = 0
if cart_distance < 5e-5:
    time_at_target += time_step
    if time_at_target >= 0.5 and not reached_target:  # 0.5 seconds at target
        reached_target = True
```

## Comparison with Other Data Generation Methods

| Method | Approach Data | Stop Data | Use Case |
|--------|---------------|-----------|----------|
| `dg.py` | ✓ | ✗ | Motion planning, trajectory learning |
| `dg_with_stop_data.py` | ✓ | ✓ | Precise positioning, regulation tasks |
| Continuous motion | ✓ | ✗ | Tracking, continuous control |

## Expected Results

With properly generated stop data, your trained models should:
- Reach targets more accurately (smaller final error)
- Maintain position at target with minimal oscillation
- Have smoother velocity profiles near the target
- Better understand the relationship between position error and required velocity

## Notes

- Keep `extra_steps` consistent between training and test data generation
- The convergence threshold (`5e-5`) should remain the same as regular data generation
- Stop data is most beneficial for tasks requiring precise final positioning
- Consider your application requirements when choosing between regular and stop data generation
