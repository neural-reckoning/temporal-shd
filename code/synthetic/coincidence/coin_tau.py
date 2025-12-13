"""
Coin Test Training Script with Learnable Tau (SLAYER Implementation)
==========================================================

This script implements a spiking neural network (SNN) for the coin test dataset
using SLAYER framework with learnable tau parameters, following the training logic from 
isi_tau.py and adding coin-specific functionality.

Key Features:
- SLAYER SNN framework with ProbSpikes loss
- Single hidden layer with 3 neurons and learnable tau parameters
- Learnable tau parameters for temporal processing via PSP filter
- Lambda-based dataset splitting for disturbance evaluation
- Comprehensive gradient logging and analysis tools

Dataset: Coin test with 3 classes (A, B, C), 60 neurons, 1000 time steps
Authors: Based on slayerPytorch framework and isi_tau.py
Date: 2025
"""

import os
import sys
import pickle
import random
import json
import numpy as np
from scipy import io
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add SLAYER to path
CURRENT_DIR = os.getcwd()
sys.path.append(os.path.join(CURRENT_DIR, "../../src"))
import slayerSNN as snn

ms = 1e-3  # Time constant

# === Parameters ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_neurons = 60  # 3 groups of 20 neurons each
num_classes = 3   # Classes A, B, C
T = 1000          # Time steps

# === SLAYER Parameters ===
sim_param = {'Ts': 1, 'tSample': 1000}  # Match 1000 time steps
lif_param = {'type': 'SRMALPHA', 'theta': 1, 'tauSr': 1, 'tauRho': 1, 'tauRef': 1, 'scaleRef': 2, 'scaleRho': 1}

print(f"Device: {device}")

# === Load Data ===
def load_coin_data(data_file="coin_data.mat"):
    """Load coin dataset"""
    try:
        data = io.loadmat(data_file)
        X = data['X']                      # shape: (N, num_neurons, T)
        Y = data['Y'].ravel()              # shape: (N,)
        lambdas = data['lambda'].ravel()   # shape: (N,)
        
        print(f"Loaded {data_file}: X={X.shape}, Y={Y.shape}")
        print(f"Lambda values: {np.unique(lambdas)}")
        print(f"Classes: {np.unique(Y)}")
        
        return X, Y, lambdas
    except FileNotFoundError:
        print(f"File {data_file} not found. Using fallback data...")
        # Create synthetic data as fallback
        N, num_neurons, T = 1000, 60, 1000
        X = np.random.binomial(1, 0.1, (N, num_neurons, T)).astype(np.float32)
        Y = np.random.randint(0, 3, N)
        lambdas = np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], N)
        return X, Y, lambdas

# === Dataset Class ===
class SpikeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (num_neurons, T)
        y = self.Y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# === Network with Learnable Tau (SLAYER-based implementation) ===
class Network(nn.Module):
    def __init__(self, num_neurons, num_classes, hidden_units=3, learn_tau=False):
        super().__init__()
        self.learn_tau = learn_tau
        slayer = snn.layer(lif_param, sim_param)
        self.slayer = slayer
        
        # Single hidden layer with specified units
        self.fc1 = nn.utils.weight_norm(slayer.dense(num_neurons, hidden_units), name='weight')
        self.fc2 = nn.utils.weight_norm(slayer.dense(hidden_units, num_classes), name='weight')
        
        # Learnable tau parameters using SLAYER's pspFilter
        if learn_tau:
            # Use SLAYER's built-in learnable temporal filter
            self.psp_filter = slayer.pspFilter(nFilter=1, filterLength=50, filterScale=1)
            self._initialize_alpha_filter()
        
    def _initialize_alpha_filter(self):
        """Initialize the PSP filter with exponential decay function"""
        if self.learn_tau:
            # Create exponential decay function with tau = 50ms as starting point
            tau = 50 * ms
            Ts = self.slayer.simulation['Ts'] * ms
            filter_length = self.psp_filter.weight.shape[-1]
            
            # 使用指数衰减函数: epsVal = mult * exp(-t / tau)
            exp_kernel = []
            mult = 1.0  # 乘数因子
            for t in np.arange(0, filter_length * Ts, Ts):
                # 指数衰减函数，从t=0开始直接衰减
                val = mult * np.exp(-t / tau)
                exp_kernel.append(val)
            
            # Normalize and set weights
            exp_kernel = np.array(exp_kernel)
            if np.max(np.abs(exp_kernel)) > 0:
                # 适当缩放，防止电压值过大
                exp_kernel = exp_kernel / np.max(np.abs(exp_kernel)) * 0.1
            
            # Set the filter weights (note: SLAYER expects flipped kernel for convolution)
            with torch.no_grad():
                self.psp_filter.weight.data = torch.FloatTensor(
                    np.flip(exp_kernel).copy()
                ).reshape(self.psp_filter.weight.shape)

    def get_tau(self):
        """Estimate effective tau from learned exponential decay filter"""
        if self.learn_tau:
            # Get filter weights and estimate tau
            weights = self.psp_filter.weight.data.squeeze().cpu().numpy()
            weights = np.flip(weights)  # Unflip to get actual temporal response
            
            if len(weights) > 0 and np.max(np.abs(weights)) > 1e-6:
                # For exponential decay: find time constant from decay curve
                # Method: Find where amplitude drops to 1/e of initial value
                max_val = np.max(np.abs(weights))
                target_val = max_val / np.e  # 1/e of max value
                
                # Find index where value drops to target
                decay_indices = np.where(np.abs(weights) <= target_val)[0]
                if len(decay_indices) > 0:
                    tau_estimate = decay_indices[0] * self.slayer.simulation['Ts'] * ms
                else:
                    # Fallback: use weighted center of mass for tau estimation
                    times = np.arange(len(weights)) * self.slayer.simulation['Ts'] * ms
                    weights_abs = np.abs(weights)
                    if np.sum(weights_abs) > 0:
                        tau_estimate = np.average(times, weights=weights_abs)
                    else:
                        tau_estimate = 50 * ms
                
                # 添加上下限约束 - 这很重要！
                tau_min = 10 * ms   # 最小值 10ms
                tau_max = 100 * ms  # 最大值 100ms
                
                return torch.tensor(max(min(tau_estimate, tau_max), tau_min))
            else:
                return torch.tensor(50 * ms)
        else:
            return torch.tensor(50 * ms)

    def forward(self, x):
        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x)
        if x.dim() == 3: 
            x = x.unsqueeze(2).unsqueeze(3)
        x = x.float().to(device)
        
        # Forward through first layer with learnable or fixed PSP
        if self.learn_tau:
            # Use learnable PSP filter instead of fixed PSP
            x_filtered = self.psp_filter(x)
            x = self.slayer.spike(self.fc1(x_filtered))
        else:
            # Use standard SLAYER PSP
            x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        
        # Second layer uses standard PSP
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        return x

# === Data Preprocessing ===
def get_split_by_lambda(lam, X_all, Y_all, lambda_all, 
                       train_ratio=0.6, val_ratio=0.15, test_ratio=0.25, seed=42):
    """Split dataset by lambda value with consistent seeding"""
    # Create independent random number generator for consistent results
    rng = np.random.RandomState(seed)
    
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for cls in range(num_classes):
        indices = np.where((np.isclose(lambda_all, lam)) & (Y_all == cls))[0]
        rng.shuffle(indices)  # Use independent RNG
        n = len(indices)
        if n == 0:
            print(f"Warning: no samples for lambda={lam}, class={cls}")
            continue

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train.extend(X_all[train_idx])
        Y_train.extend(Y_all[train_idx])
        X_val.extend(X_all[val_idx])
        Y_val.extend(Y_all[val_idx])
        X_test.extend(X_all[test_idx])
        Y_test.extend(Y_all[test_idx])

    print(f"Lambda {lam}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return (np.array(X_train), np.array(Y_train),
            np.array(X_val), np.array(Y_val),
            np.array(X_test), np.array(Y_test))

def preprocess_data(lam=0.0, X_all=None, Y_all=None, lambda_all=None, seed=42):
    """Preprocess data with lambda-based splitting"""
    np.random.seed(seed)
    
    # Split by lambda value
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_split_by_lambda(
        lam, X_all, Y_all, lambda_all, seed=seed)
    
    # Create datasets
    train_ds = SpikeDataset(X_train, Y_train)
    val_ds = SpikeDataset(X_val, Y_val)
    test_ds = SpikeDataset(X_test, Y_test)
    
    return train_ds, val_ds, test_ds

# === Training Function ===
def train_model(lam=0.0, X_all=None, Y_all=None, lambda_all=None, 
                epochs=100, batch_size=32, lr=0.002, hidden_units=3, 
                learn_tau=False, seed=42):
    """Train the SLAYER SNN model with learnable tau"""
    from slayerSNN import spikeLoss, utils
    
    print(f"\n=== Training model with lambda={lam}, epochs={epochs}, learn_tau={learn_tau}, seed={seed} ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using device: {device}")
    
    # Preprocess data
    train_ds, val_ds, test_ds = preprocess_data(lam=lam, X_all=X_all, Y_all=Y_all, lambda_all=lambda_all, seed=seed)
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Create SLAYER network
    net = Network(num_neurons, num_classes, hidden_units, learn_tau).to(device)
    
    # SLAYER loss function (use ProbSpikes)
    loss_fn = snn.spikeLoss.spikeLoss({
        'neuron': lif_param, 
        'simulation': sim_param,
        'training': {
            'error': {
                'type': 'ProbSpikes'
            }
        }
    }).to(device)
    
    # Optimizer with different learning rates for tau parameters
    if learn_tau and hasattr(net, 'psp_filter'):
        optimizer = utils.optim.Nadam([
            {'params': [p for name, p in net.named_parameters() if 'psp_filter' not in name], 'lr': lr},
            {'params': [p for name, p in net.named_parameters() if 'psp_filter' in name], 'lr': lr * 10}  # Higher LR for tau
        ])
    else:
        optimizer = utils.optim.Nadam(net.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.5)
    
    # Training tracking
    best_val_loss = float('inf')
    best_model_state = None
    tau_history = []
    
    # Initialize gradient logging
    gradient_log = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'grad_norm_total': [],
        'grad_norm_layers': {},
        'param_changes': {},
        'tau_values': []
    }
    
    # Initialize layer-wise gradient tracking
    for name, param in net.named_parameters():
        if param.requires_grad:
            gradient_log['grad_norm_layers'][name] = []
            gradient_log['param_changes'][name] = []
    
    # Store initial parameters for change tracking
    prev_params = {}
    for name, param in net.named_parameters():
        if param.requires_grad:
            prev_params[name] = param.data.clone()
    
    total_steps = epochs * len(train_loader)
    
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            # Training phase
            net.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for x_batch, y_batch in train_loader:
                # Prepare data for SLAYER
                if x_batch.dim() == 3:
                    x_batch = x_batch.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device)
                
                # Create target spike patterns
                target = torch.zeros((x_batch.size(0), num_classes, 1, 1, 1), device=device)
                target.scatter_(1, y_batch.long()[:, None, None, None, None], 1.0)
                
                # Forward pass
                outputs = net(x_batch)
                loss = loss_fn.probSpikes(outputs, target)
                epoch_loss += loss.item()
                batch_count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Log gradients (every 50 epochs to save space)
                if epoch % 50 == 0 and batch_count == 1:  # Only log from first batch of epoch
                    total_grad_norm = 0.0
                    layer_grad_norms = {}
                    
                    for name, param in net.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            layer_grad_norms[name] = grad_norm
                            total_grad_norm += grad_norm ** 2
                    
                    total_grad_norm = total_grad_norm ** 0.5
                    
                    # Store gradient information
                    gradient_log['grad_norm_total'].append(total_grad_norm)
                    for name in layer_grad_norms:
                        gradient_log['grad_norm_layers'][name].append(layer_grad_norms[name])
                
                optimizer.step()
                pbar.update(1)
            
            # Calculate parameter changes (every 50 epochs)
            if epoch % 50 == 0:
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        param_change = (param.data - prev_params[name]).norm(2).item()
                        gradient_log['param_changes'][name].append(param_change)
                        prev_params[name] = param.data.clone()
            
            # Validation phase
            net.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    if x_batch.dim() == 3:
                        x_batch = x_batch.unsqueeze(2).unsqueeze(3)
                    x_batch = x_batch.to(device).float()
                    y_batch = y_batch.to(device)
                    
                    target = torch.zeros((x_batch.size(0), num_classes, 1, 1, 1), device=device)
                    target.scatter_(1, y_batch.long()[:, None, None, None, None], 1.0)
                    
                    outputs = net(x_batch)
                    val_loss += loss_fn.probSpikes(outputs, target).item()
                
                # Record tau value
                if learn_tau:
                    tau_val = net.get_tau().item() / ms
                    tau_history.append(tau_val)
            
            val_loss /= len(val_loader)
            epoch_loss /= batch_count
            
            # Log training metrics (every 50 epochs)
            if epoch % 50 == 0:
                gradient_log['epoch'].append(epoch)
                gradient_log['train_loss'].append(epoch_loss)
                gradient_log['val_loss'].append(val_loss)
                gradient_log['tau_values'].append(tau_history[-1] if tau_history else 50.0)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = net.state_dict()
            
            scheduler.step()
            pbar.set_postfix(
                epoch=epoch + 1, 
                val_loss=f"{val_loss:.4f}", 
                best=f"{best_val_loss:.4f}",
                tau=f"{tau_history[-1]:.1f}ms" if tau_history else "N/A"
            )
    
    # Save gradient log to file
    log_filename = f"gradient_log_coin_tau_lam{int(lam*10):02d}_tau_{'learnable' if learn_tau else 'fixed'}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    for key in gradient_log:
        if isinstance(gradient_log[key], dict):
            for subkey in gradient_log[key]:
                if isinstance(gradient_log[key][subkey], list):
                    gradient_log[key][subkey] = [float(x) if not isinstance(x, list) else x for x in gradient_log[key][subkey]]
        elif isinstance(gradient_log[key], list):
            gradient_log[key] = [float(x) if not isinstance(x, list) else x for x in gradient_log[key]]
    
    with open(log_filename, 'w') as f:
        json.dump(gradient_log, f, indent=2)
    
    print(f"\nGradient log saved to {log_filename}")
    
    # Visualize tau evolution if learnable
    if learn_tau and tau_history:
        plt.figure(figsize=(8, 4))
        plt.plot(tau_history)
        plt.xlabel('Epoch')
        plt.ylabel('Tau (ms)')
        plt.title(f'Tau Evolution (lambda={lam})')
        plt.grid(True)
        # plt.show()
    
    # Load best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    
    return net, test_loader

# === Test Function ===
def test_model(net, test_loader):
    """Test the trained SLAYER model"""
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if x_batch.dim() == 3:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3)
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device)
            
            outputs = net(x_batch)
            predicted = snn.predict.getClass(outputs)
            
            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# === Model Analysis ===
def analyze_learned_parameters(net, dataset_name=""):
    """Analyze the learned tau parameters"""
    print(f"\n=== Learned Parameters Analysis {dataset_name} ===")
    
    if hasattr(net, 'learn_tau') and net.learn_tau:
        tau_val = net.get_tau().item() / ms
        print(f"Learned Tau: {tau_val:.2f} ms")
        
        # Analyze PSP filter weights if available
        if hasattr(net, 'psp_filter'):
            filter_weights = net.psp_filter.weight.data.squeeze().cpu().numpy()
            filter_max = np.max(np.abs(filter_weights))
            filter_mean = np.mean(filter_weights)
            print(f"PSP Filter: max_weight={filter_max:.4f}, mean_weight={filter_mean:.4f}")
    else:
        print("Fixed Tau: 50.0 ms")
    
    # Weight statistics
    for name, param in net.named_parameters():
        if 'weight' in name:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            print(f"{name} stats: mean={weight_mean:.4f}, std={weight_std:.4f}")

# === Main Script ===
if __name__ == "__main__":
    print(f"Starting Coin test training with SLAYER SNN and learnable tau")
    print(f"Device: {device}")
    
    # Quick test of SLAYER network functionality
    print("\n=== Quick Network Test ===")
    test_net = Network(num_neurons, num_classes, hidden_units=3, learn_tau=True).to(device)
    
    # Test forward pass
    test_input = torch.randn(2, num_neurons, T).unsqueeze(2).unsqueeze(3).to(device)
    test_output = test_net(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    
    # Test spike prediction
    pred = snn.predict.getClass(test_output)
    print(f"Predicted classes: {pred}")
    
    # Test tau extraction
    tau_val = test_net.get_tau()
    print(f"Initial tau: {tau_val.item() / ms:.2f} ms")
    
    print("Network test passed!")
    
    # Load coin data
    print("\n=== Loading Data ===")
    X_all, Y_all, lambda_all = load_coin_data("coin_data.mat")
    
    # Training parameters - lambda values represent disturbance/overlap levels
    lambda_values = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # Start with just one value for testing
    
    results = {}
    
    for lam in lambda_values:
        print(f"\n{'='*60}")
        print(f"Training with lambda = {lam} (disturbance level)")
        print(f"{'='*60}")
        
        # Train model
        net, test_loader = train_model(
            lam=lam,
            X_all=X_all,
            Y_all=Y_all,
            lambda_all=lambda_all,
            epochs=501,
            batch_size=32,
            lr=0.001,
            hidden_units=3,
            learn_tau=True,
            seed=42
        )
        
        # Test model
        test_acc = test_model(net, test_loader)
        
        # Analyze learned parameters
        analyze_learned_parameters(net, f"lambda={lam}")
        
        # Save model
        model_path = f"coin_tau_lam{int(lam*10):02d}.pt"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Get final tau value
        final_tau = net.get_tau().item() / ms if net.learn_tau else 50.0
        
        results[lam] = {
            'accuracy': test_acc,
            'tau_value': final_tau
        }
    
    print("\nTraining completed! Model and logs saved.")
