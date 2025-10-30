"""
Refactored Part 2: MLP for Robot Trajectory Prediction
This module provides a clean, modular implementation with configurable hyperparameters.
"""

import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(10)


@dataclass
class ModelConfig:
    """Configuration for the MLP model architecture."""
    input_size: int = 10
    output_size: int = 14
    hidden_sizes: List[int] = None
    activation: str = 'leaky_relu'
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256, 128]


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 0.0001
    num_epochs: int = 100
    train_split: float = 0.8
    device: str = 'auto'  # 'auto', 'cuda', or 'cpu'
    
    # Data paths
    train_data_dir: str = './data/train_with_stop_data/'
    test_data_dir: str = './data/test_with_stop_data/'
    
    # Output paths
    best_model_path: str = 'part2_best_model_with_stop.pth'
    loss_curves_path: str = 'part2_with_stop_loss_curves.png'
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


class P2_Dataset(Dataset):
    """Dataset for robot trajectory data."""
    
    def __init__(self, data_dir: str, verbose: bool = True):
        """
        Initialize the dataset by loading all trajectory files.
        
        Args:
            data_dir: Directory containing .pt files with trajectory data
            verbose: Whether to print loading information
        """
        self.data_dir = data_dir
        self.verbose = verbose
        
        if self.verbose:
            print(f"Initializing P2_Dataset from: {data_dir}")
        
        # Initialize data containers
        self.q_mes_all = []
        self.desired_cartesian_pos_all = []
        self.q_des_all = []
        self.qd_des_all = []
        self.num_trajectories = 0
        
        # Load all trajectory files
        self._load_data()
        
        # Convert to tensors
        self._convert_to_tensors()
        
        # Validate data
        self._validate_data()
        
        if self.verbose:
            self._print_summary()
    
    def _load_data(self):
        """Load data from all .pt files in the data directory."""
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.pt'):
                file_path = os.path.join(self.data_dir, filename)
                data = torch.load(file_path, weights_only=False)
                
                self.q_mes_all.append(data['q_mes_all'])
                self.desired_cartesian_pos_all.append(data['final_cartesian_pos'])
                self.q_des_all.append(data['q_d_all'])
                self.qd_des_all.append(data['qd_d_all'])
                self.num_trajectories += 1
    
    def _convert_to_tensors(self):
        """Convert loaded data to PyTorch tensors."""
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
    
    def _validate_data(self):
        """Validate that all data arrays have consistent shapes."""
        num_samples = self.q_mes_all.shape[0]
        assert self.desired_cartesian_pos_all.shape[0] == num_samples, \
            "Mismatch in desired_cartesian_pos samples"
        assert self.q_des_all.shape[0] == num_samples, \
            "Mismatch in q_des samples"
        assert self.qd_des_all.shape[0] == num_samples, \
            "Mismatch in qd_des samples"
    
    def _print_summary(self):
        """Print summary of loaded data."""
        print(f"Loaded {self.num_trajectories} trajectories")
        print(f"Total samples: {len(self)}")
        print(f"  q_mes shape: {self.q_mes_all.shape}")
        print(f"  desired_cartesian_pos shape: {self.desired_cartesian_pos_all.shape}")
        print(f"  q_des shape: {self.q_des_all.shape}")
        print(f"  qd_des shape: {self.qd_des_all.shape}")
    
    def __len__(self) -> int:
        return self.q_mes_all.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single data sample.
        
        Returns:
            Tuple of (q_mes, desired_cartesian_pos, q_des, qd_des)
        """
        return (
            self.q_mes_all[idx], 
            self.desired_cartesian_pos_all[idx], 
            self.q_des_all[idx], 
            self.qd_des_all[idx]
        )


class P2_MLP(nn.Module):
    """Multi-Layer Perceptron for trajectory prediction."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the MLP model.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super(P2_MLP, self).__init__()
        self.config = config
        
        # Build the network layers
        layers = []
        input_size = config.input_size
        
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self._get_activation())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config.output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(self.config.activation, nn.LeakyReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.mlp(x)


class Trainer:
    """Handles model training, validation, and evaluation."""
    
    def __init__(
        self, 
        model_config: ModelConfig, 
        training_config: TrainingConfig
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Configuration for model architecture
            training_config: Configuration for training process
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = training_config.get_device()
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = P2_MLP(model_config)
        self.model.to(torch.float64).to(self.device)
        
        # Loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=training_config.learning_rate
        )
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.best_val_loss = float('inf')
        self.best_test_loss = float('inf')
    
    def setup_data(self):
        """Load and prepare data loaders."""
        print("\n=== Setting up data ===")
        
        # Load training dataset
        train_dataset = P2_Dataset(self.training_config.train_data_dir)
        
        # Split into train and validation
        train_size = int(self.training_config.train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_data, 
            batch_size=self.training_config.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_data, 
            batch_size=self.training_config.batch_size, 
            shuffle=False
        )
        
        # Load test dataset
        test_dataset = P2_Dataset(self.training_config.test_data_dir)
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=False
        )
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.training_config.num_epochs}"
        )
        
        for batch in progress_bar:
            q_mes, desired_cartesian_pos, q_des, qd_des = batch
            
            # Prepare inputs and targets
            inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(self.device)
            targets = torch.cat((q_des, qd_des), dim=1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return np.mean(epoch_losses)
    
    def evaluate(self, loader: DataLoader) -> float:
        """
        Evaluate the model on a data loader.
        
        Args:
            loader: Data loader to evaluate on
            
        Returns:
            Average loss
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in loader:
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(self.device)
                targets = torch.cat((q_des, qd_des), dim=1).to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def train(self):
        """Main training loop."""
        print("\n=== Starting training ===")
        
        for epoch in range(self.training_config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            # Test
            test_loss = self.evaluate(self.test_loader)
            self.test_losses.append(test_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{self.training_config.num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Test Loss:  {test_loss:.6f}")
            
            # Update best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Save best model based on test loss
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.save_model()
                print(f"  ✓ New best model saved (test loss: {test_loss:.6f})")
            
            print()
        
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best test loss: {self.best_test_loss:.6f}")
        
        # Plot and save loss curves
        self.plot_loss_curves()
    
    def save_model(self):
        """Save the current model state."""
        torch.save(
            self.model.state_dict(), 
            self.training_config.best_model_path
        )
    
    def plot_loss_curves(self):
        """Plot and save training, validation, and test loss curves."""
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'g-', label='Validation Loss', linewidth=2)
        plt.plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training, Validation, and Test Loss Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.training_config.loss_curves_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to '{self.training_config.loss_curves_path}'")
        plt.show()


class Evaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(
        self, 
        model_path: str, 
        model_config: ModelConfig,
        device: torch.device = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to saved model weights
            model_config: Configuration for model architecture
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = P2_MLP(model_config)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(torch.float64).to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Evaluation device: {self.device}")
    
    def evaluate(
        self, 
        data_dir: str, 
        batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_dir: Directory containing evaluation data
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n=== Evaluating on: {data_dir} ===")
        
        # Load dataset
        dataset = P2_Dataset(data_dir, verbose=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Collect predictions and targets
        preds_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(self.device)
                targets = torch.cat((q_des, qd_des), dim=1).to(self.device)
                
                outputs = self.model(inputs)
                
                preds_list.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # Concatenate all batches
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(preds, targets)
        
        # Print results
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        preds: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate evaluation metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing various metrics
        """
        errors = preds - targets
        
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
        overall_r2 = 1.0 - overall_ss_res / overall_ss_tot \
            if overall_ss_tot > np.finfo(float).eps else 1.0
        
        return {
            'mse_per_dim': mse_per_dim,
            'mse_overall': mse_overall,
            'r2_per_dim': r2_per_dim,
            'r2_overall': overall_r2,
            'preds': preds,
            'targets': targets,
            'errors': errors
        }
    
    def _print_metrics(self, metrics: Dict[str, np.ndarray]):
        """Print evaluation metrics in a readable format."""
        np.set_printoptions(precision=6, suppress=True)
        
        print("\n=== Evaluation Results ===")
        print(f"\nOverall MSE: {metrics['mse_overall']:.6e}")
        print(f"Overall R²:  {metrics['r2_overall']:.6f}")
        
        print(f"\nPer-dimension MSE (14 dimensions):")
        print(metrics['mse_per_dim'])
        
        print(f"\nPer-dimension R² (14 dimensions):")
        print(metrics['r2_per_dim'])
        
        print(f"\nMean absolute error: {np.mean(np.abs(metrics['errors'])):.6e}")
        print(f"Max absolute error:  {np.max(np.abs(metrics['errors'])):.6e}")


def main():
    """Main function to run training and evaluation."""
    
    # Configure model architecture
    model_config = ModelConfig(
        input_size=10,
        output_size=14,
        hidden_sizes=[256, 256, 128],
        activation='leaky_relu'
    )
    
    # Configure training
    training_config = TrainingConfig(
        batch_size=128,
        learning_rate=0.0001,
        num_epochs=100,
        train_split=0.8,
        device='auto',
        train_data_dir='./data/train_with_stop_data/',
        test_data_dir='./data/test_with_stop_data/',
        best_model_path='part2_best_model_with_stop.pth',
        loss_curves_path='part2_with_stop_loss_curves.png'
    )
    
    # Print configurations
    print("="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    for key, value in asdict(model_config).items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in asdict(training_config).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Train model
    trainer = Trainer(model_config, training_config)
    trainer.setup_data()
    trainer.train()
    
    # Evaluate best model
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    evaluator = Evaluator(
        model_path=training_config.best_model_path,
        model_config=model_config,
        device=training_config.get_device()
    )
    
    metrics = evaluator.evaluate(
        data_dir=training_config.test_data_dir,
        batch_size=training_config.batch_size
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
