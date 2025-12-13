"""
CCISI Training Script
=====================

This script implements a spiking neural network (SNN) for CCISI dataset
using SLAYER framework, following the training logic from ccisi_ori.py
and the network architecture from ccisi.ipynb.

Key Features:
- SLAYER SNN framework with ProbSpikes loss
- Single hidden layer with learnable tau parameters
- Partial spike timing randomization for robustness evaluation
- Cross-entropy loss for classification
- Comprehensive visualization and analysis tools

Dataset: CCISI test with multiple classes, neurons, time steps
Authors: Based on slayerPytorch framework and ccisi.ipynb
Date: 2025
"""

import os
import sys
import pickle
import random
import numpy as np
from scipy import io
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

# Add SLAYER to path
CURRENT_DIR = os.getcwd()
sys.path.append(os.path.join(CURRENT_DIR, "../../src"))
import slayerSNN as snn

ms = 1e-3  # Time constant

# === Parameters ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_range = (0.0, 0.6)
validation_range = (0.6, 0.75)
testing_range = (0.75, 0.9)

# === SLAYER Parameters ===
sim_param = {'Ts': 1, 'tSample': 1000}  # Match 1000 time steps
lif_param = {'type': 'SRMALPHA', 'theta': 1, 'tauSr': 1, 'tauRho': 1, 'tauRef': 1, 'scaleRef': 2, 'scaleRho': 1}

print(f"Device: {device}")

# === Load Data ===
def load_ccisi_data(data_file="ccisi.h5"):
    """Load CCISI dataset from HDF5 file"""
    try:
        with h5py.File(data_file, "r") as f:
            X = f["X"][:]  # (N, num_neurons, T)
            Y = f["Y"][:].ravel()
            firing_rates_all = f["firing_rates"][:] if "firing_rates" in f else None
            isis_all = f["isis"][:] if "isis" in f else None

        print(f"Loaded {data_file}: X={X.shape}, Y={Y.shape}")
        print(f"Classes: {np.unique(Y)}")

        return X, Y, firing_rates_all, isis_all
    except FileNotFoundError:
        print(f"File {data_file} not found. Using fallback data...")
        # Create synthetic data as fallback
        N, num_neurons, T = 1000, 10, 1000
        X = np.random.binomial(1, 0.1, (N, num_neurons, T)).astype(np.float32)
        Y = np.random.randint(0, 5, N)  # More classes for CCISI
        return X, Y, None, None

# === Partial Randomization Function ===
def partial_randomize_spike_train(spike_train, f=0.0, max_attempts=50):
    """
    Partially randomize spike timings with probability f
    """
    if f <= 0:
        return spike_train
    
    num_neurons, T = spike_train.shape
    new_train = np.copy(spike_train)
    
    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        
        # Randomly select spikes to move
        num_to_randomize = int(len(spike_times) * f)
        if num_to_randomize > 0:
            random_indices = np.random.choice(spike_times, size=num_to_randomize, replace=False)
            new_train[neuron_idx, random_indices] = 0
            
            # Place spikes in new random locations
            for _ in range(max_attempts * num_to_randomize):
                new_t = np.random.randint(0, T)
                if np.sum(new_train[neuron_idx, :]) >= len(spike_times):
                    break
                if new_train[neuron_idx, new_t] == 0:
                    new_train[neuron_idx, new_t] = 1
    
    return new_train

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

# === Network (SLAYER-based version with Learnable PSP) ===
class Network(nn.Module):
    def __init__(self, num_neurons, num_classes, hidden_units=100, learn_tau=False):
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
        """Initialize the PSP filter with alpha function"""
        if self.learn_tau:
            # Create alpha function with tau = 50ms as starting point
            tau = 50 * ms
            Ts = self.slayer.simulation['Ts'] * ms
            filter_length = self.psp_filter.weight.shape[-1]
            
            alpha_kernel = []
            for t in np.arange(0, filter_length * Ts, Ts):
                val = t / tau * np.exp(1 - t / tau)
                alpha_kernel.append(val)
            
            # Normalize and set weights
            alpha_kernel = np.array(alpha_kernel)
            if np.max(np.abs(alpha_kernel)) > 0:
                alpha_kernel = alpha_kernel / np.max(np.abs(alpha_kernel))
            
            # Set the filter weights (note: SLAYER expects flipped kernel for convolution)
            with torch.no_grad():
                self.psp_filter.weight.data = torch.FloatTensor(
                    np.flip(alpha_kernel).copy()
                ).reshape(self.psp_filter.weight.shape)

    def get_tau(self):
        """Estimate effective tau from learned filter"""
        if self.learn_tau:
            # Get filter weights and estimate tau
            weights = self.psp_filter.weight.data.squeeze().cpu().numpy()
            weights = np.flip(weights)  # Unflip to get actual temporal response
            
            if len(weights) > 0:
                # Find peak and estimate tau (simple method)
                peak_idx = np.argmax(np.abs(weights))
                # Tau estimation: approximately 3 times the peak location
                estimated_tau = 3 * peak_idx * self.slayer.simulation['Ts'] * ms
                return torch.tensor(max(estimated_tau, 10*ms))  # Clamp to reasonable range
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
def get_indices(rng, total):
    return np.arange(int(rng[0] * total), int(rng[1] * total))

def preprocess_data(f=0.0, X_all=None, Y_all=None, seed=42):
    """Preprocess data with partial randomization"""
    np.random.seed(seed)
    
    # Apply partial randomization
    X_proc = np.array([partial_randomize_spike_train(x, f) for x in X_all])
    
    # Split data
    train_idx = get_indices(training_range, len(X_all))
    val_idx = get_indices(validation_range, len(X_all))
    test_idx = get_indices(testing_range, len(X_all))
    
    # Create datasets
    train_ds = SpikeDataset(X_proc[train_idx], Y_all[train_idx])
    val_ds = SpikeDataset(X_proc[val_idx], Y_all[val_idx])
    test_ds = SpikeDataset(X_proc[test_idx], Y_all[test_idx])
    
    return train_ds, val_ds, test_ds

# === Training Function ===
# === Training Function ===
def train_model(f=0.0, X_all=None, Y_all=None, num_neurons=10, num_classes=2,
                epochs=100, batch_size=32, lr=0.002, hidden_units=100, 
                learn_tau=False, seed=42):
    """Train the SLAYER SNN model"""
    from slayerSNN import spikeLoss, utils
    
    print(f"\n=== Training model with f={f}, epochs={epochs}, learn_tau={learn_tau}, seed={seed} ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using device: {device}")
    
    # Preprocess data
    train_ds, val_ds, test_ds = preprocess_data(f=f, X_all=X_all, Y_all=Y_all, seed=seed)
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
    if learn_tau and hasattr(net, 'logit_tau'):
        optimizer = utils.optim.Nadam([
            {'params': [p for name, p in net.named_parameters() if 'logit_tau' not in name], 'lr': lr},
            {'params': [net.logit_tau], 'lr': lr * 10}  # Higher LR for tau
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
    log_filename = f"gradient_log_ccisi_tau_f{int(f*10):02d}_tau_{'learnable' if learn_tau else 'fixed'}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    import json
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
        plt.title(f'Tau Evolution (f={f})')
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

# === Visualization Functions ===
def plot_predictions(net, test_loader, max_samples=8):
    """Plot model predictions for visualization"""
    plt.figure(figsize=(16, 6))
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            if batch_idx > 0:  # Only process first batch
                break
                
            if x_batch.dim() == 3:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3)
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device)
            
            outputs = net(x_batch)
            predicted = snn.predict.getClass(outputs)
            
            batch_size = min(x_batch.size(0), max_samples)
            
            for b in range(batch_size):
                plt.subplot(2, 4, b + 1)
                plt.axhline(y_batch[b].item(), color='k', linestyle='--', label='True label')
                plt.axhline(predicted[b].item(), color='r', linestyle='-', label='Predicted')
                plt.ylim(-0.5, 1.5)
                plt.title(f"Sample {b}, True={y_batch[b].item()}, Pred={predicted[b].item()}")
                plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()

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
    print(f"Starting CCISI training with SLAYER SNN")
    print(f"Device: {device}")
    
    # Quick test of SLAYER network functionality
    print(f"\n=== Quick Network Test ===")
    test_net = Network(10, 5, hidden_units=100, learn_tau=True).to(device)  # 5 classes for CCISI
    
    # Test forward pass
    test_input = torch.randn(2, 10, 1000).unsqueeze(2).unsqueeze(3).to(device)
    test_output = test_net(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    
    # Test spike prediction
    pred = snn.predict.getClass(test_output)
    print(f"Predicted classes: {pred}")
    
    print("Network test passed!")
    
    # Load CCISI data
    print(f"\n=== Loading Data ===")
    X_all, Y_all, firing_rates_all, isis_all = load_ccisi_data("ccisi.h5")
    
    num_neurons = X_all.shape[1] if X_all is not None else 10
    num_classes = len(np.unique(Y_all)) if Y_all is not None else 5  # More classes for CCISI
    
    print(f"Network config: {num_neurons} neurons, {num_classes} classes")
    
    # Training parameters - f values represent randomization levels
    f_values = [0]  # Different randomization levels

    # Train with learnable tau only
    learn_tau = True
    tau_str = "learnable"
    print(f"\n{'='*60}")
    print(f"Training with {tau_str} tau parameters")
    print(f"{'='*60}")
    
    results = {}
    
    for f in f_values:
        print(f"\n{'='*60}")
        print(f"Training with f = {f} (randomization level), tau = {tau_str}")
        print(f"{'='*60}")
        
        # Train model with 300 epochs
        net, test_loader = train_model(
            f=f,
            X_all=X_all,
            Y_all=Y_all,
            num_neurons=num_neurons,
            num_classes=num_classes,
            epochs=301,
            batch_size=32,
            lr=0.001,
            hidden_units=100,
            learn_tau=learn_tau,
            seed=42
        )
        
        # Test model
        test_acc = test_model(net, test_loader)
        
        # Analyze learned parameters
        analyze_learned_parameters(net, f"f={f}")
        
        # Save model
        tau_suffix = "learnable" if learn_tau else "fixed"
        model_path = f"ccisi_tau_f{int(f*10):02d}.pt"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        results[f] = test_acc

# === Evaluation and Visualization ===
def evaluate_robustness(f_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], repeats=3, 
                       epochs=50, batch_size=64, hidden_units=128):
    """Evaluate model robustness across different randomization levels"""
    results = {}
    
    for f in f_values:
        print(f"\n{'='*50}")
        print(f"Evaluating f = {f}")
        print(f"{'='*50}")
        
        f_results = []
        
        for repeat in range(repeats):
            print(f"Repeat {repeat + 1}/{repeats}")
            
            # Train model
            net, train_loader, val_loader, test_loader = train_model(
                f=f,
                epochs=epochs,
                batch_size=batch_size,
                lr=0.001,
                hidden_units=hidden_units,
                use_custom_net=True,
                seed=42 + repeat  # Different seed for each repeat
            )
            
            # Test model
            test_acc = test_model(net, test_loader, use_custom_net=True)
            f_results.append(test_acc)
        
        results[f] = {
            'mean': np.mean(f_results),
            'std': np.std(f_results),
            'results': f_results
        }
        
        print(f"f = {f}: Mean Accuracy = {results[f]['mean']:.4f} ± {results[f]['std']:.4f}")
    
    # Plot robustness curve
    plt.figure(figsize=(10, 6))
    
    f_vals = list(results.keys())
    means = [results[f]['mean'] for f in f_vals]
    stds = [results[f]['std'] for f in f_vals]
    
    plt.errorbar(f_vals, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
    plt.xlabel('Randomization Level (f)')
    plt.ylabel('Test Accuracy')
    plt.title('Model Robustness to Spike Timing Randomization')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add text annotations
    for f, mean, std in zip(f_vals, means, stds):
        plt.annotate(f'{mean:.3f}±{std:.3f}', 
                    (f, mean), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ccisi_robustness_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return results

# === Model Analysis ===
def analyze_learned_parameters(net, dataset_name=""):
    """Analyze the learned tau parameters"""
    print(f"\n=== Learned Parameters Analysis {dataset_name} ===")
    
    for i, layer in enumerate(net.networks):
        print(f"\nLayer {i}:")
        print(f"  Input size: {layer.n_in}")
        print(f"  Output size: {layer.n_out}")
        print(f"  Spiking: {layer.spiking}")
        
        if hasattr(layer, 'logit_tau') and layer.logit_tau is not None:
            tau_val = layer.tau.item()
            logit_val = layer.logit_tau.item()
            print(f"  Tau: {tau_val/ms:.2f} ms (learnable, logit={logit_val:.3f})")
        else:
            tau_val = layer.tau.item()
            print(f"  Tau: {tau_val/ms:.2f} ms (fixed)")
        
        # Weight statistics
        weight_mean = layer.weight.data.mean().item()
        weight_std = layer.weight.data.std().item()
        print(f"  Weight stats: mean={weight_mean:.4f}, std={weight_std:.4f}")
        
        if hasattr(layer, 'bias'):
            bias_mean = layer.bias.data.mean().item()
            bias_std = layer.bias.data.std().item()
            print(f"  Bias stats: mean={bias_mean:.4f}, std={bias_std:.4f}")

def visualize_network_activity(net, test_loader, num_samples=3):
    """Visualize network activity for a few test samples"""
    net.eval()
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            if batch_idx > 0:  # Only process first batch
                break
                
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get network outputs with recording
            outputs, recordings = net(x_batch[:num_samples], record=True)
            
            # Plot activity for each sample
            fig, axes = plt.subplots(num_samples, len(recordings[0]), 
                                   figsize=(15, 3*num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for sample_idx in range(num_samples):
                for time_idx in range(min(10, len(recordings))):  # Show first 10 time steps
                    for layer_idx, layer_activity in enumerate(recordings[time_idx]):
                        if sample_idx < axes.shape[0] and layer_idx < axes.shape[1]:
                            activity = layer_activity[sample_idx].cpu().numpy()
                            axes[sample_idx, layer_idx].bar(range(len(activity)), activity)
                            if sample_idx == 0:
                                if layer_idx == 0:
                                    axes[sample_idx, layer_idx].set_title(f'Input (t={time_idx})')
                                else:
                                    axes[sample_idx, layer_idx].set_title(f'Layer {layer_idx} (t={time_idx})')
                            axes[sample_idx, layer_idx].set_ylabel(f'Sample {sample_idx}\nLabel: {y_batch[sample_idx].item()}')
            
            plt.tight_layout()
            plt.savefig('ccisi_network_activity.png', dpi=300, bbox_inches='tight')
            # plt.show()
            break