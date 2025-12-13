import os, torch, torch.nn as nn, numpy as np, h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import slayerSNN as snn
import gc

# === Params ===
sim_param = {'Ts': 1, 'tSample': 200}
lif_param = {'type': 'SRMALPHA', 'theta': 1, 'tauSr': 1, 'tauRho': 1, 'tauRef': 1, 'scaleRef': 2, 'scaleRho': 1}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset (loaded fully into memory) ===
class SpikeDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.X = np.array(f['X'])  # [N, C, T]
            self.Y = np.array(f['Y'])

    def __len__(self): return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx][None, :, None, None, :]  # → [1, C, 1, 1, T]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

# === Network ===
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        slayer = snn.layer(lif_param, sim_param)
        self.slayer = slayer
        self.fc1 = nn.utils.weight_norm(slayer.dense(input_size, 128), name='weight')
        self.fc2 = nn.utils.weight_norm(slayer.dense(128, 128), name='weight')
        self.fc3 = nn.utils.weight_norm(slayer.dense(128, output_size), name='weight')
        self.delay1 = slayer.delay(128)
        self.delay2 = slayer.delay(128)

    def forward(self, x):
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

# === Training ===
def train_model(f):
    from slayerSNN import spikeLoss
    f_str = f"{int(f * 10):02d}"
    log_path = f"trainlog_f{f_str}.txt"

    # === Load datasets ===
    train_ds = SpikeDataset(f"/root/autodl-tmp/data_split_part/train_f{f_str}.h5")
    val_ds   = SpikeDataset(f"/root/autodl-tmp/data_split_part/val_f{f_str}.h5")
    test_ds  = SpikeDataset(f"/root/autodl-tmp/data_split_part/test_f{f_str}.h5")

    num_neurons = train_ds[0][0].shape[1]
    T = train_ds[0][0].shape[-1]
    num_classes = len(np.unique(train_ds.Y))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # === Model and training setup ===
    net = Network(num_neurons, num_classes).to(device)
    loss_fn = spikeLoss.spikeLoss({
        'neuron': lif_param,
        'simulation': sim_param,
        'training': {'error': {'type': 'ProbSpikes'}}
    }).to(device)
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    best_val, best_model = float('inf'), None
    total_steps = 150 * len(train_loader)

    # === Open log file ===
    with open(log_path, "w") as log_file, tqdm(total=total_steps, desc=f"Training f={f:.1f}") as pbar:
        for epoch in range(150):
            net.train()
            epoch_losses = []

            for xb, yb in train_loader:
                x = xb.squeeze(1).to(device)
                y = yb.to(device)
                tgt = torch.zeros((x.size(0), num_classes, 1, 1, 1), device=device)
                tgt.scatter_(1, y[:, None, None, None, None], 1.0)

                out = net(x)
                loss = loss_fn.probSpikes(out, tgt)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                epoch_losses.append(loss.item())
                pbar.update(1)

            scheduler.step()

            # === Validation ===
            train_loss = np.mean(epoch_losses)
            net.eval(); val_loss = 0.
            with torch.no_grad():
                for xb, yb in val_loader:
                    x = xb.squeeze(1).to(device)
                    y = yb.to(device)
                    tgt = torch.zeros((x.size(0), num_classes, 1, 1, 1), device=device)
                    tgt.scatter_(1, y[:, None, None, None, None], 1.0)
                    val_loss += loss_fn.probSpikes(net(x), tgt).item()
            val_loss /= len(val_loader)

            # === Save best model ===
            if val_loss < best_val:
                best_val, best_model = val_loss, net.state_dict()

            # === Logging ===
            log_file.write(f"epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}\n")
            log_file.flush()
            pbar.set_postfix(epoch=epoch+1, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", best=f"{best_val:.4f}")

    # === Load best and test ===
    if best_model: net.load_state_dict(best_model)

    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.squeeze(1).to(device)
            y = yb.to(device)
            pred = snn.predict.getClass(net(x))
            correct += (pred.cpu() == y.cpu()).sum().item()
            total += len(y)

    acc = correct / total
    print(f"f = {f:.1f} | Test Accuracy: {acc:.2%}")
    torch.save(net.state_dict(), f"slayer_part_new_f{f_str}.pt")

# === Main ===
if __name__ == "__main__":
    f_values = [round(0.1 * i, 1) for i in range(11)]
    for f in f_values:
        train_model(f)
