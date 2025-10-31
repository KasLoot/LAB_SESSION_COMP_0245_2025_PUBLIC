"""
Part 2: Neural Network Training and Evaluation for Robot Control
================================================================

This script trains and evaluates MLP models to predict desired joint positions
and velocities from measured joint positions and desired Cartesian positions.

Two model architectures are available:
1. P2_MLP_Encoder: Smaller architecture (256->128->64->output)
2. P2_MLP_None_Encoder: Larger architecture (256->256->256->output)
"""

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION SECTION - Modify parameters here
# ============================================================================

class Config:
    """Central configuration for all hyperparameters and settings"""
    
    # Model Selection
    # Choose: 'encoder' or 'none_encoder'
    MODEL_TYPE = 'none_encoder'
    
    # Torch Random Seed
    RANDOM_SEED = 10
    
    # Data Paths
    TRAIN_DATA_DIR = './data/train/'
    TEST_DATA_DIR = './data/test/'
    
    # Model Architecture
    INPUT_SIZE = 10   # q_mes (7) + desired_cartesian_pos (3)
    OUTPUT_SIZE = 14  # q_des (7) + qd_des (7)
    
    # Training Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 200
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Saving Options
    SAVE_MODEL = True
    MODEL_SAVE_PATH = f'part2_{MODEL_TYPE}_no_stop.pth'
    LOSS_PLOT_PATH = f'part2_{MODEL_TYPE}_no_stop_loss_curves.png'

    @classmethod
    def get_model_class(cls):
        """Returns the model class based on MODEL_TYPE"""
        if cls.MODEL_TYPE == 'encoder':
            return P2_MLP_Encoder
        elif cls.MODEL_TYPE == 'none_encoder':
            return P2_MLP_None_Encoder
        else:
            raise ValueError(f"Unknown model type: {cls.MODEL_TYPE}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"Model Type:          {cls.MODEL_TYPE}")
        print(f"Random Seed:         {cls.RANDOM_SEED}")
        print(f"Train Data Dir:      {cls.TRAIN_DATA_DIR}")
        print(f"Test Data Dir:       {cls.TEST_DATA_DIR}")
        print(f"Batch Size:          {cls.BATCH_SIZE}")
        print(f"Learning Rate:       {cls.LEARNING_RATE}")
        print(f"Number of Epochs:    {cls.NUM_EPOCHS}")
        print(f"Train/Val Split:     {cls.TRAIN_VAL_SPLIT}")
        print(f"Device:              {cls.DEVICE}")
        print(f"Save Model:          {cls.SAVE_MODEL}")
        print(f"Model Save Path:     {cls.MODEL_SAVE_PATH}")
        print(f"Loss Plot Path:      {cls.LOSS_PLOT_PATH}")
        print("="*70 + "\n")


# ============================================================================
# DATASET DEFINITION
# ============================================================================

class P2_dataset(Dataset):
    """
    Dataset for Part 2 - loads trajectory data from .pt files
    
    Each sample contains:
    - q_mes: measured joint positions (7,)
    - desired_cartesian_pos: target Cartesian position (3,)
    - q_des: desired joint positions (7,)
    - qd_des: desired joint velocities (7,)
    """
    
    def __init__(self, data_dir):
        print(f"\nInitializing P2_dataset from: {data_dir}")
        
        self.q_mes_all = []
        self.desired_cartesian_pos_all = []
        self.q_des_all = []
        self.qd_des_all = []
        self.num_trajectories = 0
        
        # Load all .pt files from directory
        for filename in os.listdir(data_dir):
            if filename.endswith('.pt'):
                file_path = os.path.join(data_dir, filename)
                data = torch.load(file_path, weights_only=False)
                
                self.q_mes_all.append(data['q_mes_all'])
                self.desired_cartesian_pos_all.append(data['final_cartesian_pos'])
                self.q_des_all.append(data['q_d_all'])
                self.qd_des_all.append(data['qd_d_all'])
                self.num_trajectories += 1
        
        # Concatenate and convert to tensors
        self.q_mes_all = torch.tensor(
            np.concatenate(self.q_mes_all, axis=0), 
            dtype=torch.float64
        )
        self.desired_cartesian_pos_all = torch.tensor(
            np.concatenate(self.desired_cartesian_pos_all, axis=0), 
            dtype=torch.float64
        )
        self.q_des_all = torch.tensor(
            np.concatenate(self.q_des_all, axis=0), 
            dtype=torch.float64
        )
        self.qd_des_all = torch.tensor(
            np.concatenate(self.qd_des_all, axis=0), 
            dtype=torch.float64
        )
        
        # Print dataset statistics
        print(f"✓ Loaded {self.num_trajectories} trajectories")
        print(f"✓ Total samples: {self.q_mes_all.shape[0]}")
        print(f"  - q_mes shape:                {self.q_mes_all.shape}")
        print(f"  - desired_cartesian_pos shape: {self.desired_cartesian_pos_all.shape}")
        print(f"  - q_des shape:                {self.q_des_all.shape}")
        print(f"  - qd_des shape:               {self.qd_des_all.shape}")
        
        # Verify data consistency
        assert (self.q_mes_all.shape[0] == 
                self.desired_cartesian_pos_all.shape[0] == 
                self.q_des_all.shape[0] == 
                self.qd_des_all.shape[0]), \
                "Mismatch in number of samples among data arrays."
    
    def __len__(self):
        return self.q_mes_all.shape[0]
    
    def __getitem__(self, idx):
        return (
            self.q_mes_all[idx], 
            self.desired_cartesian_pos_all[idx], 
            self.q_des_all[idx], 
            self.qd_des_all[idx]
        )


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class P2_MLP_Encoder(nn.Module):
    """
    Smaller MLP architecture with decreasing layer sizes
    Architecture: 10 -> 256 -> 128 -> 64 -> 14
    """
    
    def __init__(self, input_size=10, output_size=14):
        super(P2_MLP_Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.mlp(x)


class P2_MLP_None_Encoder(nn.Module):
    """
    Larger MLP architecture with constant hidden layer sizes
    Architecture: 10 -> 256 -> 256 -> 256 -> 14
    """
    
    def __init__(self, input_size=10, output_size=14):
        super(P2_MLP_None_Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_loss_curves(train_losses, val_losses, test_losses, save_path):
    """
    Plot and save training, validation, and test loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_losses: List of test losses per epoch
        save_path: Path to save the plot image
    """
    num_epochs = len(train_losses)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training, Validation, and Test Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Loss curves saved to '{save_path}'")
    plt.close()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    """
    Main training loop for the neural network
    """
    # Set random seed for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    
    # Print configuration
    Config.print_config()
    
    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading training dataset...")
    dataset = P2_dataset(data_dir=Config.TRAIN_DATA_DIR)
    
    # Split into train and validation
    train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = P2_dataset(data_dir=Config.TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model_class = Config.get_model_class()
    model = model_class(input_size=Config.INPUT_SIZE, output_size=Config.OUTPUT_SIZE)
    model.to(torch.float64).to(device)
    
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model '{Config.MODEL_TYPE}' initialized")
    print(f"  Trainable parameters: {model_param_count:,}")
    
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training tracking
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        # ==================== Training Phase ====================
        model.train()
        epoch_train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]"):
            q_mes, desired_cartesian_pos, q_des, qd_des = batch
            
            # Prepare inputs and targets
            inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(device)
            targets = torch.cat((q_des, qd_des), dim=1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # ==================== Validation Phase ====================
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(device)
                targets = torch.cat((q_des, qd_des), dim=1).to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                epoch_val_losses.append(loss.item())
        
        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        # ==================== Test Phase ====================
        epoch_test_losses = []
        
        with torch.no_grad():
            for batch in test_loader:
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(device)
                targets = torch.cat((q_des, qd_des), dim=1).to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                epoch_test_losses.append(loss.item())
        
        avg_test_loss = np.mean(epoch_test_losses)
        test_losses.append(avg_test_loss)
        
        # ==================== Logging ====================
        print(f"Epoch [{epoch+1:3d}/{Config.NUM_EPOCHS}] | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"Test: {avg_test_loss:.6f}")
        
        # ==================== Model Saving ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            if Config.SAVE_MODEL:
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  → Best model saved (test loss: {best_test_loss:.6f})")
    
    # ==================== Training Complete ====================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best test loss:       {best_test_loss:.6f}")
    print(f"Model saved to:       '{Config.MODEL_SAVE_PATH}'")
    print(f"{'='*70}\n")
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, test_losses, Config.LOSS_PLOT_PATH)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_best_model(model_path=None, data_dir=None, batch_size=64, device=None):
    """
    Evaluate a saved model on test data.
    
    Computes per-dimension and overall MSE and R² metrics.
    
    Args:
        model_path: Path to saved model weights (default: Config.MODEL_SAVE_PATH)
        data_dir: Directory containing test data (default: Config.TEST_DATA_DIR)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on (default: Config.DEVICE)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Use defaults from Config if not specified
    if model_path is None:
        model_path = Config.MODEL_SAVE_PATH
    if data_dir is None:
        data_dir = Config.TEST_DATA_DIR
    if device is None:
        device = torch.device(Config.DEVICE)
    
    print(f"\n{'='*70}")
    print("MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Model path:  {model_path}")
    print(f"Data dir:    {data_dir}")
    print(f"Device:      {device}")
    print(f"{'='*70}\n")
    
    # Load dataset
    dataset = P2_dataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model_class = Config.get_model_class()
    model = model_class(input_size=Config.INPUT_SIZE, output_size=Config.OUTPUT_SIZE)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(torch.float64).to(device)
    model.eval()
    
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded successfully")
    print(f"  Trainable parameters: {model_param_count:,}\n")
    
    # Collect predictions and targets
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            q_mes, desired_cartesian_pos, q_des, qd_des = batch
            
            inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(device)
            targets = torch.cat((q_des, qd_des), dim=1).to(device)
            
            outputs = model(inputs)
            
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    errors = preds - targets
    
    # ==================== Compute Metrics ====================
    # Per-dimension MSE
    mse_per_dim = np.mean(errors**2, axis=0)
    
    # Overall MSE
    mse_overall = np.mean(errors**2)
    
    # Per-dimension R²
    y_mean = np.mean(targets, axis=0)
    ss_res = np.sum((targets - preds)**2, axis=0)
    ss_tot = np.sum((targets - y_mean)**2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r2_per_dim = 1.0 - ss_res / ss_tot
    r2_per_dim = np.where(ss_tot <= np.finfo(float).eps, 1.0, r2_per_dim)
    
    # Overall R²
    overall_ss_res = np.sum((targets - preds)**2)
    overall_ss_tot = np.sum((targets - np.mean(targets))**2)
    overall_r2 = 1.0 - overall_ss_res / overall_ss_tot if overall_ss_tot > np.finfo(float).eps else 1.0
    
    # ==================== Print Results ====================
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    np.set_printoptions(precision=6, suppress=True)
    
    print("Per-dimension MSE:")
    for i, mse in enumerate(mse_per_dim):
        print(f"  Dim {i:2d}: {mse:.6e}")
    
    print(f"\nPer-dimension R²:")
    for i, r2 in enumerate(r2_per_dim):
        print(f"  Dim {i:2d}: {r2:.6f}")
    
    print(f"\n{'='*70}")
    print(f"Overall MSE: {mse_overall:.6e}")
    print(f"Overall R²:  {overall_r2:.6f}")
    print(f"{'='*70}\n")
    
    # Return metrics
    metrics = {
        "mse_per_dim": mse_per_dim,
        "mse_overall": mse_overall,
        "r2_per_dim": r2_per_dim,
        "r2_overall": overall_r2,
        "preds": preds,
        "targets": targets,
    }
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Train the model
    train()
    
    # Evaluate the best saved model
    # evaluate_best_model()
    
    # Evaluate with custom parameters
    evaluate_best_model(
        model_path=Config.MODEL_SAVE_PATH,
        data_dir='./data/test_with_stop_data/'
    )
