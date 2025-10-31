# Successive Time Series Error Evaluation (succesive_time_series_error.py)

## Overview

This script provides a comprehensive evaluation framework for trained robot control models using a custom metric called **Successive Time Series Error**. Unlike simple final position error, this metric penalizes the entire trajectory, encouraging models that reach targets quickly and smoothly.

## Purpose

The script evaluates models by:
1. Running robot simulations with model predictions
2. Computing successive time series error across the entire trajectory
3. Measuring final distance to target
4. Comparing multiple models using the same test cases
5. Providing statistical summaries of model performance

## Key Features

- **Trajectory-based evaluation**: Considers entire path, not just final position
- **Weighted error metric**: Configurable sigma parameter to tune error sensitivity
- **Statistical analysis**: Mean, std, min, and max across multiple test poses
- **Model comparison**: Framework for comparing different trained models
- **Reproducible testing**: Fixed random seeds for consistent evaluation

## The Successive Time Series Error Metric

### Formula

```
Successive Error = Σ (σ * ||current_position - target_position||)²
```

Where:
- Sum is over all timesteps in the trajectory
- `σ` (sigma) is a weighting parameter
- `||·||` is the Euclidean distance in Cartesian space

### Interpretation

**Lower is better**:
- Small values indicate robot quickly reaches and stays near target
- Large values indicate slow convergence or large position errors

**What the metric captures**:
- **Speed of convergence**: Faster convergence = smaller cumulative error
- **Accuracy**: Staying close to target = smaller errors
- **Smoothness**: Direct paths generally have lower total error

**Comparison to final distance**:
- Final distance only measures end state
- Successive error measures the entire trajectory quality

## Usage

### Basic Usage

Edit the script to specify your model path and configuration, then run:

```bash
python succesive_time_series_error.py
```

### Single Model Evaluation

```python
model_path = FINAL_DIR / "part2_encoder_with_stop.pth"

results = evaluate_model_in_simulation(
    model_path=str(model_path),
    num_test_poses=10,
    model_class=P2_MLP_Encoder,
    successive_time_series_error_sigma=1.0,
    seed=78
)
```

### Comparing Multiple Models

```python
model_paths = [
    FINAL_DIR / "part2_best_model_with_stop.pth",
    FINAL_DIR / "part2_best_model_no_stop.pth",
]
model_classes = [P2_MLP_Encoder, P2_MLP_None_Encoder]

comparison_results = compare_models(
    model_paths=model_paths,
    model_classes=model_classes,
    num_test_poses=10,
    sigma=1.0
)
```

## Key Parameters to Tune

### 1. Model Configuration

**Location**: Line 252-254 (in `__main__` section)

```python
model_path = FINAL_DIR / "part2_encoder_with_stop.pth"
model_class = P2_MLP_Encoder  # or P2_MLP_None_Encoder
```

**What it does**: Specifies which trained model to evaluate.

**When to change**:
- Update `model_path` to evaluate different models
- Match `model_class` to your model architecture
- For comparison, prepare multiple model paths

---

### 2. Number of Test Poses

**Location**: Line 255-257

```python
num_test_poses=10
```

**What it does**: Number of random target positions to test.

**When to change**:
- **Increase** (20-50): More comprehensive evaluation, better statistics
- **Decrease** (3-5): Quick testing during development
- **Recommendation**: Use 10-20 for final evaluation

**Trade-off**: More poses = better statistics but longer runtime

---

### 3. Sigma Parameter

**Location**: Line 258

```python
successive_time_series_error_sigma=1.0
```

**What it does**: Weighting factor in the error metric formula.

**Impact**:
- **Larger sigma** (e.g., 10.0): Amplifies errors, more sensitive to deviations
- **Smaller sigma** (e.g., 0.1): Reduces errors, less sensitive
- **sigma = 1.0**: Neutral weighting (recommended default)

**When to change**:
- Use larger sigma to emphasize differences between models
- Use smaller sigma if errors are already very large
- Keep constant when comparing multiple models

**Example interpretation**:
```
With sigma = 1.0:
- Error of 0.01m contributes 0.0001 to total
- Error of 0.1m contributes 0.01 to total

With sigma = 10.0:
- Error of 0.01m contributes 0.01 to total  
- Error of 0.1m contributes 1.0 to total
```

**Recommendation**: Start with 1.0 and adjust if needed to spread out model rankings.

---

### 4. Random Seed

**Location**: Line 259

```python
seed=78
```

**What it does**: Sets random seed for test position generation.

**When to change**:
- Use different seeds to test on different random test sets
- Keep constant for reproducible comparisons
- Should be different from training (42) and test data generation (56)

---

### 5. Maximum Simulation Duration

**Location**: Line 135 (inside `evaluate_model_in_simulation`)

```python
max_duration = 5.0  # seconds
```

**What it does**: Maximum time allowed for each trajectory.

**When to change**:
- **Increase** (7.0-10.0): If robot needs more time to reach difficult targets
- **Decrease** (3.0): For faster evaluation if targets are easy
- Should match duration used during training data generation

---

### 6. Test Position Ranges

**Location**: Lines 116-119 (inside `evaluate_model_in_simulation`)

```python
x = np.random.uniform(0.2, 0.5)
y = np.random.choice([np.random.uniform(0.2, 0.5), np.random.uniform(-0.5, -0.2)])
z = np.random.uniform(0.1, 0.6)
```

**When to change**:
- Should match training data ranges for fair evaluation
- Can use different ranges to test generalization
- Ensure ranges are within robot's workspace

---

## Understanding the Output

### Console Output

The script prints:

```
Evaluating model on 10 random target poses...
Pose 1/10: Successive Error = 0.123456, Final Distance = 0.001234
Pose 2/10: Successive Error = 0.234567, Final Distance = 0.002345
...

==============================================================
EVALUATION SUMMARY
==============================================================
Number of test poses: 10
Sigma parameter: 1.0

Successive Time Series Error:
  Mean:  0.123456
  Std:   0.012345
  Min:   0.100000
  Max:   0.150000

Final Distance to Target:
  Mean:  0.001234
  Std:   0.000123
  Min:   0.001000
  Max:   0.001500
==============================================================
```

### Interpreting Results

#### Successive Time Series Error

**Scale depends on sigma**, but generally:
- **< 0.1**: Excellent performance
- **0.1 - 0.5**: Good performance
- **0.5 - 2.0**: Acceptable performance
- **> 2.0**: Poor performance (model needs improvement)

**Note**: These ranges assume `sigma = 1.0` and `max_duration = 5.0s`

#### Final Distance to Target

Measured in meters:
- **< 0.001** (1 mm): Excellent
- **0.001 - 0.005** (1-5 mm): Good
- **0.005 - 0.01** (5-10 mm): Acceptable
- **> 0.01** (> 10 mm): Poor

### Statistical Metrics

- **Mean**: Average performance across all test poses
- **Std**: Consistency (lower = more consistent)
- **Min/Max**: Best and worst case performance

**What to look for**:
- Low mean error (good average performance)
- Low standard deviation (consistent performance)
- Small gap between min and max (reliable)

## Comparing Models

### Using the compare_models Function

```python
model_paths = [
    FINAL_DIR / "model_a.pth",
    FINAL_DIR / "model_b.pth",
    FINAL_DIR / "model_c.pth",
]
model_classes = [
    P2_MLP_Encoder,
    P2_MLP_None_Encoder,
    P2_MLP_Encoder,
]

results = compare_models(
    model_paths=model_paths,
    model_classes=model_classes,
    num_test_poses=20,
    sigma=1.0
)
```

### Analyzing Comparison Results

The function returns a list of result dictionaries. You can analyze them:

```python
# Extract mean errors for each model
for i, result in enumerate(results):
    print(f"Model {i+1} ({result['model_class'].__name__}):")
    print(f"  Mean Successive Error: {result['mean_successive_error']:.4f}")
    print(f"  Mean Final Distance: {result['mean_final_distance']:.4f}")
    print()

# Find best model
best_idx = np.argmin([r['mean_successive_error'] for r in results])
print(f"Best model: Model {best_idx + 1}")
```

## Workflow for Model Evaluation

### Step 1: Train Your Models

Ensure you have trained model files:
```bash
ls final_1/*.pth
```

### Step 2: Configure Evaluation

Edit the script's `__main__` section:

```python
model_path = FINAL_DIR / "your_model.pth"
model_class = P2_MLP_Encoder  # Match your architecture
```

### Step 3: Run Evaluation

```bash
python succesive_time_series_error.py
```

### Step 4: Analyze Results

Review the printed statistics and compare with other models.

### Step 5: Compare Multiple Models (Optional)

Uncomment and configure the comparison section:

```python
model_paths = [...]
model_classes = [...]
comparison_results = compare_models(...)
```

## Advanced Usage

### Custom Test Positions

To test specific challenging positions:

```python
# Modify inside evaluate_model_in_simulation
test_poses = [
    [0.5, 0.5, 0.3],   # Far right
    [0.5, -0.5, 0.3],  # Far left
    [0.3, 0.0, 0.6],   # High
    [0.4, 0.0, 0.1],   # Low
]
```

### Saving Results to File

Add after evaluation:

```python
import json

with open(FINAL_DIR / 'evaluation_results.json', 'w') as f:
    # Remove non-serializable data
    save_results = {
        'mean_successive_error': results['mean_successive_error'],
        'std_successive_error': results['std_successive_error'],
        'mean_final_distance': results['mean_final_distance'],
        'std_final_distance': results['std_final_distance'],
        'sigma': results['sigma'],
        'num_test_poses': results['num_test_poses'],
    }
    json.dump(save_results, f, indent=2)
```

### Plotting Comparison Results

```python
import matplotlib.pyplot as plt

model_names = ['Model A', 'Model B', 'Model C']
mean_errors = [r['mean_successive_error'] for r in comparison_results]
std_errors = [r['std_successive_error'] for r in comparison_results]

plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_errors, yerr=std_errors, capsize=5)
plt.ylabel('Mean Successive Time Series Error')
plt.title('Model Comparison')
plt.grid(True, alpha=0.3)
plt.savefig(FINAL_DIR / 'model_comparison.png')
plt.show()
```

## Troubleshooting

### Issue: All errors are very large

**Possible causes**:
- Model not trained properly
- Wrong model class specified
- Sigma parameter too large

**Solutions**:
- Verify model loaded correctly (check file path)
- Try `sigma = 0.1` to see if errors are more reasonable
- Check model training loss - should be low
- Ensure test positions are within training distribution

### Issue: High variance in errors

**Possible causes**:
- Some test positions harder than others
- Model not generalizing well
- Not enough test poses

**Solutions**:
- Increase `num_test_poses` for better statistics
- Check if certain workspace regions are problematic
- Train on more diverse data

### Issue: Final distance good but successive error high

**Interpretation**:
- Model eventually reaches target but takes inefficient path
- May oscillate or move slowly

**Solutions**:
- Visualize trajectories to see behavior
- Consider retraining with different hyperparameters
- May need more training data or different architecture

### Issue: Successive error good but final distance poor

**Note**: This is unusual - successive error includes final distance

**Check**:
- Verify metric calculation is correct
- May indicate numerical issues

## Tuning Sigma for Your Application

### Choosing Sigma

The sigma parameter should be chosen based on your requirements:

**For precise positioning tasks** (e.g., assembly, surgery):
- Use larger sigma (5.0-10.0)
- Heavily penalizes any deviation from target
- Emphasizes models with tight tracking

**For general motion tasks** (e.g., pick-and-place):
- Use moderate sigma (1.0-2.0)
- Balanced evaluation of path and final position

**For rough positioning tasks**:
- Use smaller sigma (0.1-0.5)
- More lenient evaluation
- Focuses on whether target is eventually reached

### Sigma and Model Comparison

**Important**: Use the same sigma for all models you compare!

Different sigma values change the relative ranking of models.

## Performance Expectations

### Well-trained model (on test data similar to training):
- Mean successive error: 0.05 - 0.2 (sigma=1.0)
- Mean final distance: < 0.002 m
- Std successive error: < 0.05
- Max final distance: < 0.005 m

### Poorly-trained model:
- Mean successive error: > 1.0 (sigma=1.0)
- Mean final distance: > 0.01 m
- High variance in both metrics

### Model trained without stop data vs. with stop data:
- With stop data typically has:
  - Lower final distance (better steady-state)
  - Possibly similar or slightly higher successive error (depends on trajectory efficiency)

## Expected Runtime

- **Per pose**: 5-10 seconds (depends on `max_duration`)
- **10 poses**: ~1-2 minutes
- **50 poses**: ~5-10 minutes
- **Model comparison** (3 models, 10 poses each): ~5-10 minutes

## Integration with Other Evaluation Scripts

This script complements `eval_p2.py`:

- **eval_p2.py**: Visual analysis with plots
- **succesive_time_series_error.py**: Quantitative analysis with metrics

**Recommended workflow**:
1. Use `succesive_time_series_error.py` to rank models quantitatively
2. Use `eval_p2.py` to visualize best model's behavior
3. Iterate on training based on insights from both

## Notes

- The metric is sensitive to trajectory length (longer = higher error possible)
- Models should be evaluated with the same `max_duration` for fair comparison
- Consider both successive error and final distance - a model could be good at one but not the other
- The script uses double precision (float64) for numerical accuracy
