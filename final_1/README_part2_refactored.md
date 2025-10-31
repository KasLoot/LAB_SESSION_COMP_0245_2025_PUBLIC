# Part 2: Neural Network Training for Robot Control

A refactored and user-friendly implementation for training MLP models to predict desired joint positions and velocities from measured joint positions and desired Cartesian positions.

## 🎯 Overview

This script trains deep learning models to map robot sensor data to control commands. It supports two different model architectures and provides comprehensive training, validation, and evaluation capabilities.

## ✨ Features

- **Centralized Configuration**: All hyperparameters in one place (`Config` class)
- **Easy Model Switching**: Toggle between two architectures with a single parameter
- **Comprehensive Logging**: Detailed training progress and evaluation metrics
- **Visualization**: Automatic generation of loss curves
- **Reproducibility**: Fixed random seeds for consistent results
- **GPU Support**: Automatic CUDA detection and usage

## 🏗️ Model Architectures

### 1. P2_MLP_Encoder (Smaller Model)
```
Input (10) → 256 → 128 → 64 → Output (14)
```
- Decreasing layer sizes
- Fewer parameters
- Faster training

### 2. P2_MLP_None_Encoder (Larger Model)
```
Input (10) → 256 → 256 → 256 → Output (14)
```
- Constant hidden layer sizes
- More parameters
- Higher capacity

## 📊 Data Format

### Input Features (10 dimensions)
- `q_mes`: Measured joint positions (7 dimensions)
- `desired_cartesian_pos`: Target Cartesian position (3 dimensions)

### Output Targets (14 dimensions)
- `q_des`: Desired joint positions (7 dimensions)
- `qd_des`: Desired joint velocities (7 dimensions)

### Data Files
Data should be stored in `.pt` files containing dictionaries with keys:
- `q_mes_all`: Array of measured joint positions
- `final_cartesian_pos`: Array of desired Cartesian positions
- `q_d_all`: Array of desired joint positions
- `qd_d_all`: Array of desired joint velocities

## 🚀 Quick Start

### 1. Configure Your Training

Edit the `Config` class at the top of the file:

```python
class Config:
    # Choose model: 'encoder' or 'none_encoder'
    MODEL_TYPE = 'encoder'
    
    # Data paths
    TRAIN_DATA_DIR = './data/train_with_stop_data/'
    TEST_DATA_DIR = './data/test_with_stop_data/'
    
    # Training hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 200
    
    # Saving options
    SAVE_MODEL = True
    MODEL_SAVE_PATH = 'part2_model.pth'
    LOSS_PLOT_PATH = 'loss_curves.png'
```

### 2. Train the Model

```python
if __name__ == "__main__":
    train()
```

Run the script:
```bash
python part_2_refactored.py
```

### 3. Evaluate the Model

```python
if __name__ == "__main__":
    evaluate_best_model()
```

Or with custom parameters:
```python
evaluate_best_model(
    model_path='custom_model.pth',
    data_dir='./data/custom_test_data/'
)
```

## ⚙️ Configuration Parameters

### Model Selection
| Parameter | Options | Description |
|-----------|---------|-------------|
| `MODEL_TYPE` | `'encoder'`, `'none_encoder'` | Choose model architecture |

### Data Paths
| Parameter | Type | Description |
|-----------|------|-------------|
| `TRAIN_DATA_DIR` | str | Path to training data directory |
| `TEST_DATA_DIR` | str | Path to test data directory |

### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_SIZE` | 10 | Input dimension size |
| `OUTPUT_SIZE` | 14 | Output dimension size |

### Training Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 128 | Number of samples per batch |
| `LEARNING_RATE` | 0.0001 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 200 | Number of training epochs |
| `TRAIN_VAL_SPLIT` | 0.8 | Train/validation split ratio |

### System Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | 10 | Random seed for reproducibility |
| `DEVICE` | Auto-detect | Use 'cuda' or 'cpu' |

### Saving Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAVE_MODEL` | True | Whether to save best model |
| `MODEL_SAVE_PATH` | str | Path to save model weights |
| `LOSS_PLOT_PATH` | str | Path to save loss curve plot |

## 📈 Output and Metrics

### During Training
- Real-time progress bars for each epoch
- Per-epoch train/validation/test losses
- Automatic saving of best model (based on test loss)

### Loss Curves
Automatically generated plot showing:
- Training loss
- Validation loss
- Test loss

Saved to the path specified in `LOSS_PLOT_PATH`.

### Evaluation Metrics
- **Per-dimension MSE**: Mean squared error for each output dimension
- **Per-dimension R²**: Coefficient of determination for each dimension
- **Overall MSE**: Global mean squared error
- **Overall R²**: Global coefficient of determination

### Example Output
```
==================================================================
EVALUATION RESULTS
==================================================================

Per-dimension MSE:
  Dim  0: 1.234567e-04
  Dim  1: 2.345678e-04
  ...

Per-dimension R²:
  Dim  0: 0.987654
  Dim  1: 0.976543
  ...

==================================================================
Overall MSE: 1.567890e-04
Overall R²:  0.982345
==================================================================
```

## 🔄 Comparing Models

To compare both architectures:

1. **Train the first model:**
   ```python
   # In Config class
   MODEL_TYPE = 'encoder'
   MODEL_SAVE_PATH = 'encoder_model.pth'
   LOSS_PLOT_PATH = 'encoder_loss.png'
   
   train()
   ```

2. **Train the second model:**
   ```python
   # In Config class
   MODEL_TYPE = 'none_encoder'
   MODEL_SAVE_PATH = 'none_encoder_model.pth'
   LOSS_PLOT_PATH = 'none_encoder_loss.png'
   
   train()
   ```

3. **Evaluate both:**
   ```python
   print("Evaluating Encoder Model:")
   metrics_encoder = evaluate_best_model(model_path='encoder_model.pth')
   
   print("\nEvaluating None-Encoder Model:")
   metrics_none = evaluate_best_model(model_path='none_encoder_model.pth')
   ```

## 📁 Project Structure

```
final_1/
├── part_2_refactored.py          # Main refactored script
├── README_part2_refactored.md    # This file
├── data/
│   ├── train_with_stop_data/     # Training data (.pt files)
│   └── test_with_stop_data/      # Test data (.pt files)
├── *.pth                          # Saved model weights
└── *_loss_curves.png              # Generated loss plots
```

## 🛠️ Requirements

```python
torch
numpy
matplotlib
tqdm
pickle  # (standard library)
```

Install dependencies:
```bash
pip install torch numpy matplotlib tqdm
```

## 💡 Tips and Best Practices

1. **Start with fewer epochs** (e.g., 20) to verify everything works before full training
2. **Monitor both validation and test loss** to detect overfitting
3. **Experiment with learning rate** if loss doesn't decrease
4. **Try both architectures** to see which performs better for your data
5. **Use GPU** when available for faster training
6. **Check data paths** before training to avoid errors

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in Config
- Use smaller model (`MODEL_TYPE = 'encoder'`)

### Loss Not Decreasing
- Lower `LEARNING_RATE`
- Increase `NUM_EPOCHS`
- Check data normalization

### File Not Found Errors
- Verify `TRAIN_DATA_DIR` and `TEST_DATA_DIR` paths
- Ensure `.pt` files exist in directories

## 📝 License

This code is part of LAB_SESSION_COMP_0245_2025_PUBLIC.

## 🤝 Contributing

For questions or improvements, please contact the course instructors.

---

**Last Updated:** October 31, 2025
