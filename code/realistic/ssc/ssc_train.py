"""
==============================================
Spiking Neural Network Training (SSC Dataset)
==============================================

Example Usage:
--------------
# Train with delay
python ssc_train.py whole delay f=[0,0.1,0.5,1.0]

# Train without delay
python ssc_train.py whole nodelay f=[0.0,0.5,1.0]

# Train on 'part' dataset
python ssc_train.py part delay f=[0.0,0.5,1.0]

Description:
------------
This script loads spike-based datasets from HDF5 files (train/val/test),
trains a SLAYER-based spiking neural network with optional delay layers,
and evaluates the model for multiple temporal perturbation factors (f values).
"""

import os
import sys
import ast
import random
import numpy as np
import h5py
import torch
from torch import nn
from tqdm import tqdm
import slayerSNN as snn


# ======================================================
# 1. Dataset Loader
# ======================================================
class SpikeDataset(torch.utils.data.Dataset):
    """
    A simple dataset wrapper for HDF5 spike data.

    Each HDF5 file must contain:
        - 'X': spike trains, shape (N, neurons, time_steps)
        - 'Y': labels, shape (N,)
    """
    def __init__(self, h5_path):
        super().__init__()
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            self.X = np.array(f['X'], dtype=np.float32)
            self.Y = np.array(f['Y']).astype(int).ravel()

        # Convert X to torch tensor later during __getitem__
        self.num_samples = len(self.Y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).unsqueeze(2).unsqueeze(3)  # (neurons, time, 1, 1)
        y = torch.tensor(self.Y[idx]).long()
        return x, y


# ======================================================
# 2. Configurations
# ======================================================
DATASET_CONFIGS = {
    "whole": {
        "dir": "/root/autodl-tmp/data_split_whole",
        "model_prefix": "sscdelay_whole",
    },
    "part": {
        "dir": "/root/autodl-tmp/data_split_part",
        "model_prefix": "sscdelay_part",
    },
    "val": {
        "dir": "/root/autodl-tmp/data_split_val",
        "model_prefix": "sscdelay_val",
    },
}


# ======================================================
# 3. CLI argument parsing
# ======================================================
def parse_f_list(tokens):
    """Parse 'f' values from command-line arguments, e.g. f=[0,0.1,1.0]."""
    if not tokens:
        return [0.0]
    raw = None
    for token in tokens:
        if token.startswith("f=") or token.startswith("--f="):
            raw = token.split("=", 1)[1]
            break
        if token.startswith("[") and token.endswith("]"):
            raw = token
            break
    if raw is None:
        return [0.0]
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        return [float(val)]
    except Exception:
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [float(x) for x in s.split(",") if x]


def parse_cli(argv):
    """Return (dataset_key, mode, f_values) parsed from CLI."""
    tokens = list(argv[1:])
    dataset = "whole"
    mode = "delay"

    if tokens:
        first = tokens[0].lower()
        if first in DATASET_CONFIGS:
            dataset = first
            tokens = tokens[1:]
            if tokens and tokens[0].lower() in {"delay", "nodelay"}:
                mode = tokens[0].lower()
                tokens = tokens[1:]
        elif first in {"delay", "nodelay"}:
            mode = first
            tokens = tokens[1:]
    return dataset, mode, parse_f_list(tokens)


# ======================================================
# 4. Network definition
# ======================================================
class Network(nn.Module):
    """Three-layer SLAYER-based SNN with optional delay modules."""
    def __init__(self, sim_param, lif_param, device, input_dim, output_size, use_delay=True):
        super().__init__()
        slayer = snn.layer(lif_param, sim_param)
        self.slayer = slayer
        self.fc1 = nn.utils.weight_norm(slayer.dense(input_dim, 128), name='weight')
        self.fc2 = nn.utils.weight_norm(slayer.dense(128, 128), name='weight')
        self.fc3 = nn.utils.weight_norm(slayer.dense(128, output_size), name='weight')

        self.use_delay = use_delay
        if use_delay:
            self.delay1 = slayer.delay(128)
            self.delay2 = slayer.delay(128)
        self._device = device

    def forward(self, x):
        x = x.float().to(self._device)
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def clamp(self, thea1, thea2):
        if not self.use_delay:
            return
        self.delay1.delay.data.clamp_(0, thea1)
        self.delay2.delay.data.clamp_(0, thea2)


# ======================================================
# 5. Utility functions
# ======================================================
def build_loss_and_optim(sim_param, lif_param, net, lr=0.1):
    """Build spike rate loss, optimizer, and scheduler."""
    from slayerSNN import spikeLoss, utils
    error_cfg = {
        'neuron': lif_param,
        'simulation': sim_param,
        'training': {
            'error': {
                'type': 'SpikeRate',
                'tgtSpikeRegion': {'start': 0, 'stop': 200},
                'tgtSpikeRate': {True: 0.2, False: 0.02}
            }
        }
    }
    loss_fn = spikeLoss.spikeLoss(error_cfg)
    optimizer = utils.optim.Nadam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
    return loss_fn, optimizer, scheduler


def set_seed(seed):
    """Ensure reproducibility."""
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = False


def test_accuracy_cached(net, test_loader, device):
    """Compute classification accuracy on test set."""
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = snn.predict.getClass(net(xb))
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / max(1, total)


# ======================================================
# 6. Main training loop
# ======================================================
def main(argv):
    dataset_key, mode, f_values = parse_cli(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_param = dict(Ts=1, tSample=200)
    lif_param = dict(type='SRMALPHA', theta=10, tauSr=1, tauRho=0.1,
                     tauRef=2, scaleRef=2, scaleRho=0.1)

    config = DATASET_CONFIGS[dataset_key]
    data_dir = config["dir"]
    use_delay = (mode == "delay")
    model_prefix = config["model_prefix"] if use_delay else config["model_prefix"].replace("delay", "nodelay")

    print(f"Device: {device}")
    print(f"Dataset: {dataset_key} | Mode: {mode} | f values: {f_values}")

    for f in f_values:
        f_str = f"{int(f * 10):02d}"
        print(f"\n[INFO] Loading dataset for f = {f_str}")

        train_ds = SpikeDataset(os.path.join(data_dir, f"train_f{f_str}.h5"))
        val_ds   = SpikeDataset(os.path.join(data_dir, f"val_f{f_str}.h5"))
        test_ds  = SpikeDataset(os.path.join(data_dir, f"test_f{f_str}.h5"))

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

        # Infer input dimension and number of classes automatically
        num_neurons = train_ds[0][0].shape[1]
        num_classes = len(np.unique(train_ds.Y))

        print(f"Input neurons: {num_neurons} | Classes: {num_classes}")

        # Build network, loss, optimizer
        net = Network(sim_param, lif_param, device, num_neurons, num_classes, use_delay=use_delay).to(device)
        loss_fn, optimizer, scheduler = build_loss_and_optim(sim_param, lif_param, net, lr=0.1)
        loss_fn = loss_fn.to(device)

        best_val = float('inf')
        best_state = None
        thea1 = thea2 = 64
        patience = 300
        stop_counter = 0

        for epoch in range(1000):
            net.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                target = torch.zeros((len(yb), num_classes, 1, 1, 1), device=device)
                target.scatter_(1, yb[:, None, None, None, None], 1.0)
                out = net(xb)
                loss = loss_fn.spikeRate(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Validation phase
            net.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    target = torch.zeros((len(yb), num_classes, 1, 1, 1), device=device)
                    target.scatter_(1, yb[:, None, None, None, None], 1.0)
                    out = net(xb)
                    val_loss += loss_fn.spikeRate(out, target).item()
                    pred = snn.predict.getClass(out)
                    correct += (pred == yb).sum().item()
                    total += len(yb)
            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)

            print(f"Epoch {epoch+1:03d} | Train {np.mean(losses):.3f} | Val {val_loss:.3f} | Acc {val_acc:.2%}")
            scheduler.step()

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_state = net.state_dict()
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    break

        # Load best model and evaluate
        if best_state:
            net.load_state_dict(best_state)

        acc = test_accuracy_cached(net, test_loader, device)
        model_name = f"{model_prefix}_f{f_str}.pt"
        torch.save(net.state_dict(), os.path.join(data_dir, model_name))
        print(f"[✓] Saved {model_name} | Test Accuracy: {acc:.2%}")


if __name__ == '__main__':
    main(sys.argv)
