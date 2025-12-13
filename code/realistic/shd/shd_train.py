import os
import sys
import ast
import math
import random
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from tqdm import tqdm

# Try importing slayerSNN (must be installed/available on PYTHONPATH)
import slayerSNN as snn

# Quick guide:
# 1) Choose a dataset key: whole / part / norm
# 2) Run inside this folder, e.g. `python shd_train.py whole delay f=[0,0.1,...,1.0]`
#    Use `... nodelay ...` to switch to the no-delay network variant (delay is the default if omitted).
# 3) Checkpoints are saved as `shdelay_<dataset>_fXX.pt` (delay) or `shnodelay_<dataset>_fXX.pt` (no-delay).

DATASET_CONFIGS = {
    "whole": {
        "mat_file": "shd_whole.mat",
        "input_dim": 700,
        "model_prefix": "shdelay_whole",
    },
    "part": {
        "mat_file": "shd_part.mat",
        "input_dim": 224,
        "model_prefix": "shdelay_part",
    },
    "norm": {
        "mat_file": "shd_norm.mat",
        "input_dim": 224,
        "model_prefix": "shdelay_norm",
    },
}


def parse_f_list(tokens):
    """Parse f list from CLI, expecting e.g. "f=[0,0.1,...,1.0]" or "--f=[...]".
    Returns list of floats. If not provided, default to [0.0].
    """
    if not tokens:
        return [0.0]
    raw = None
    for token in tokens:
        if token.startswith("f=") or token.startswith("--f="):
            raw = token.split("=", 1)[1]
            break
        # Also allow passing just the list literal
        if token.startswith("[") and token.endswith("]"):
            raw = token
            break
    if raw is None:
        return [0.0]
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        # single number
        return [float(val)]
    except Exception:
        # Fallback: split by comma after stripping brackets
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [float(x) for x in s.split(",") if x]


def parse_cli(argv):
    """Return (dataset_key, mode, f_values) extracted from CLI arguments."""
    tokens = list(argv[1:])
    dataset = "whole"
    mode = "delay"

    if tokens:
        first = tokens[0]
        lowered = first.lower()
        if lowered in DATASET_CONFIGS:
            dataset = lowered
            tokens = tokens[1:]
            if tokens and tokens[0].lower() in {"delay", "nodelay"}:
                mode = tokens[0].lower()
                tokens = tokens[1:]
        elif lowered in {"delay", "nodelay"}:
            mode = lowered
            tokens = tokens[1:]
        elif first.startswith("f") or (first.startswith("[") and first.endswith("]")):
            # Dataset omitted; default to "whole"
            pass
        else:
            raise ValueError(f"Unknown dataset '{first}'. Choose from: {', '.join(DATASET_CONFIGS)}")
    elif not tokens:
        # nothing provided, keep defaults
        pass

    # If mode still default and next token spells it, consume it now.
    if tokens and tokens[0].lower() in {"delay", "nodelay"}:
        mode = tokens[0].lower()
        tokens = tokens[1:]

    return dataset, mode, parse_f_list(tokens)


def partial_randomize_spike_train(spike_train, f=0.0, max_attempts=50):
    if f <= 0:
        return spike_train

    num_neurons, T = spike_train.shape
    new_train = np.copy(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        for old_time in spike_times:
            if np.random.rand() < f:
                new_train[neuron_idx, old_time] = 0
                inserted = False
                attempts = 0
                while not inserted and attempts < max_attempts:
                    attempts += 1
                    new_t = np.random.randint(0, T)
                    if new_train[neuron_idx, new_t] == 0:
                        new_train[neuron_idx, new_t] = 1
                        inserted = True
    return new_train


def preprocess_full_dataset(X, Y, f, training_range=(0.0, 0.6), validation_range=(0.6, 0.75), testing_range=(0.75, 0.9), seed=42):
    np.random.seed(seed)

    N = len(Y)
    train_start, train_end = training_range
    val_start, val_end = validation_range
    test_start, test_end = testing_range

    train_indices = np.arange(int(N * train_start), int(N * train_end))
    val_indices = np.arange(int(N * val_start), int(N * val_end))
    test_indices = np.arange(int(N * test_start), int(N * test_end))

    np.random.shuffle(train_indices)

    def process_indices(indices):
        X_out = np.array([partial_randomize_spike_train(X[i], f) for i in indices])
        Y_out = Y[indices]
        return X_out, Y_out

    X_train, Y_train = process_indices(train_indices)
    X_val, Y_val = process_indices(val_indices)
    X_test, Y_test = process_indices(test_indices)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def build_loss_and_optim(sim_param, lif_param, net, lr=0.1):
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


class Network(nn.Module):
    def __init__(self, sim_param, lif_param, device, input_dim, use_delay=True):
        super().__init__()
        slayer = snn.layer(lif_param, sim_param)
        self.slayer = slayer
        self.fc1 = nn.utils.weight_norm(slayer.dense(input_dim, 128), name='weight')
        self.fc2 = nn.utils.weight_norm(slayer.dense(128, 128), name='weight')
        self.fc3 = nn.utils.weight_norm(slayer.dense(128, 20), name='weight')
        self.use_delay = use_delay
        if use_delay:
            self.delay1 = slayer.delay(128)
            self.delay2 = slayer.delay(128)
        self._device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
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


def set_seed(seed):
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


def train_model(X_train, Y_train, X_val, Y_val, device, sim_param, lif_param, input_dim, epochs=1000, bs=128, lr=0.1, seed=42, use_delay=True):
    set_seed(seed)
    num_classes = len(np.unique(Y_train))
    net = Network(sim_param, lif_param, device, input_dim, use_delay=use_delay).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optim(sim_param, lif_param, net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val = float('inf')
    best_state = None
    update1 = update2 = 0
    thea1 = thea2 = 64
    early_stop_counter = 0
    early_stop_patience = 300

    total_steps = epochs * max(1, (len(X_train) // bs))
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            net.train()
            indices = np.arange(len(Y_train))
            np.random.shuffle(indices)
            batch_losses = []

            for b in range(0, len(indices), bs):
                batch_idx = indices[b:b+bs]
                xb = X_train[batch_idx]
                yb = Y_train[batch_idx]

                x = torch.tensor(xb).unsqueeze(2).unsqueeze(3).float().to(device)
                y = torch.tensor(yb).long().to(device)
                target = torch.zeros((len(y), num_classes, 1, 1, 1), device=device)
                target.scatter_(1, y[:, None, None, None, None], 1.0)

                out = net(x)
                loss = loss_fn.spikeRate(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                pbar.update(1)

            # Clamp delay logic
            if use_delay:
                if epoch <= 250:
                    net.clamp(64, 64)
                else:
                    update1 += 1
                    update2 += 1
                    for name, param in net.named_parameters():
                        if "delay1.delay" in name and update1 > 150:
                            sorted_ = torch.sort(torch.floor(param.detach().flatten()))[0]
                            thea1_val = torch.max(sorted_)
                            if sorted_[108] > (thea1_val - 5):
                                thea1 = int(thea1_val.item()) + 1
                                update1 = 0
                        elif "delay2.delay" in name and update2 > 150:
                            sorted_ = torch.sort(torch.floor(param.detach().flatten()))[0]
                            thea2_val = torch.max(sorted_)
                            if sorted_[108] > (thea2_val - 5):
                                thea2 = int(thea2_val.item()) + 1
                                update2 = 0
                    net.clamp(thea1, thea2)

            # Validation
            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for b in range(0, len(Y_val), bs):
                    xb = X_val[b:b+bs]
                    yb = Y_val[b:b+bs]
                    x = torch.tensor(xb).unsqueeze(2).unsqueeze(3).float().to(device)
                    y = torch.tensor(yb).long().to(device)
                    target = torch.zeros((len(y), num_classes, 1, 1, 1), device=device)
                    target.scatter_(1, y[:, None, None, None, None], 1.0)
                    out = net(x)
                    val_loss += loss_fn.spikeRate(out, target).item()
                    pred = snn.predict.getClass(out)
                    correct += (pred.cpu() == y.cpu()).sum().item()
                    total += len(y)
            val_loss /= max(1, len(Y_val) // bs)
            val_acc = correct / max(1, total)

            pbar.set_postfix_str(f"Ep {epoch+1} | Train {np.mean(batch_losses):.3f} | Val {val_loss:.3f} | Val Acc {val_acc:.2%}")
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_state = net.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    break

    if best_state:
        net.load_state_dict(best_state)
    return net


def test_accuracy_cached(net, X_test, Y_test, device, batch_size=64):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b in range(0, len(Y_test), batch_size):
            xb = X_test[b:b+batch_size]
            yb = Y_test[b:b+batch_size]
            x = torch.tensor(xb).unsqueeze(2).unsqueeze(3).float().to(device)
            pred = snn.predict.getClass(net(x))
            correct += (pred.cpu().numpy() == yb).sum()
            total += len(yb)
    return correct / max(1, total)


def main(argv):
    # Settings
    try:
        dataset_key, mode, f_values = parse_cli(argv)
    except ValueError as exc:
        print(exc)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Simulation and neuron params (match notebook)
    sim_param = dict(Ts=1, tSample=200)
    lif_param = dict(type='SRMALPHA', theta=10, tauSr=1, tauRho=0.1, tauRef=2, scaleRef=2, scaleRho=0.1)

    config = DATASET_CONFIGS[dataset_key]
    data_dir = os.path.dirname(__file__)

    # Load dataset
    mat_path = os.path.join(data_dir, config["mat_file"])
    data = loadmat(mat_path)
    X = data['X']
    Y = data['Y'].ravel()

    # Pad time dimension to 200 as in notebook
    n_samples, n_neurons, T = X.shape
    if n_neurons != config["input_dim"]:
        raise ValueError(f"Expected {config['input_dim']} input neurons for dataset '{dataset_key}', got {n_neurons}.")
    T_target = 200
    if T < T_target:
        padded = np.zeros((n_samples, n_neurons, T_target), dtype=X.dtype)
        padded[:, :, :T] = X
        X = padded

    use_delay = (mode == "delay")
    if use_delay:
        model_prefix = config["model_prefix"]
    else:
        base_prefix = config["model_prefix"]
        model_prefix = base_prefix.replace("delay", "nodelay") if "delay" in base_prefix else f"{base_prefix}_nodelay"

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_key} | mode: {mode} | samples: {n_samples} | input neurons: {n_neurons}")
    print(f"Training models for f values: {f_values}")

    for f in f_values:
        f = float(f)
        print(f"\nPreprocessing for f = {f}")
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = preprocess_full_dataset(
            X, Y, f,
            training_range=(0.0, 0.6),
            validation_range=(0.6, 0.75),
            testing_range=(0.75, 0.9),
            seed=42,
        )

        print(f"Training for f = {f}")
        net = train_model(
            X_train, Y_train, X_val, Y_val,
            device=device,
            sim_param=sim_param,
            lif_param=lif_param,
            input_dim=config["input_dim"],
            epochs=1000,
            bs=128,
            lr=0.1,
            seed=42,
            use_delay=use_delay,
        )

        model_name = f"{model_prefix}_f{int(f*10):02d}.pt"
        model_path = os.path.join(data_dir, model_name)
        torch.save(net.state_dict(), model_path)

        acc = test_accuracy_cached(net, X_test, Y_test, device=device, batch_size=64)
        print(f"Saved {os.path.basename(model_path)} | Test Accuracy: {acc:.2%}")


if __name__ == '__main__':
    main(sys.argv)

