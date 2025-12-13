import os
import numpy as np
import h5py
import gc
# because the data is large, we will split and save smaller files with different f values
# === Configurations ===
data_path = "ssc_part.h5"  # 原始大文件路径
save_dir = "/root/autodl-tmp/data_split_part"     # 存储输出路径
T_target = 200              # 目标时间步
#f_values = [round(0.1 * i, 1) for i in range(11)]  # f = 0.0, 0.1, ..., 1.0
f_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# === 读取原始数据 ===
with h5py.File(data_path, 'r') as f:
    X = f["X"][:].astype(np.float32)  # shape: (N, C, T)
    Y = f["Y"][:].ravel()

N, num_neurons, T = X.shape
print(f"[Data] Loaded X shape: {X.shape}, dtype: {X.dtype}")
print(f"[Data] Total size: {X.nbytes / 1024**2:.2f} MB")

# === Zero Padding ===
if T < T_target:
    X_pad = np.zeros((X.shape[0], num_neurons, T_target), dtype=np.float32)
    X_pad[:, :, :T] = X
    X = X_pad
    T = T_target
    print(f"[Data] Padded to T = {T_target}")

# === 数据集划分比例 ===
def get_indices(rng, total):
    return np.arange(int(rng[0] * total), int(rng[1] * total))

training_range   = (0.0, 0.6)
validation_range = (0.6, 0.75)
testing_range    = (0.75, 0.9)

# === 扰动函数 ===
def partial_randomize(spike_train, f, max_attempts=50):
    if f <= 0: return spike_train
    train = np.copy(spike_train)
    for i in range(train.shape[0]):
        spikes = np.where(train[i] == 1)[0]
        for t in spikes:
            if np.random.rand() < f:
                train[i, t] = 0
                for _ in range(max_attempts):
                    new_t = np.random.randint(0, train.shape[1])
                    if train[i, new_t] == 0:
                        train[i, new_t] = 1
                        break
    return train

# === 主函数：处理并保存子集 ===
def preprocess_and_save_split(f, prefix="data_split"):
    print(f"\n[INFO] Preprocessing f = {f:.1f}")
    os.makedirs(prefix, exist_ok=True)

    # 打乱索引
    all_indices = np.arange(N)
    np.random.seed(42)  # 保证每次一致
    np.random.shuffle(all_indices)
    
    # 按比例划分打乱后的索引
    n_train = int(training_range[1] * N)
    n_val   = int((validation_range[1] - validation_range[0]) * N)
    n_test  = int((testing_range[1] - testing_range[0]) * N)
    
    train_idx = all_indices[:n_train]
    val_idx   = all_indices[n_train:n_train + n_val]
    test_idx  = all_indices[n_train + n_val:n_train + n_val + n_test]

    for name, idx in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
        out_path = f"{prefix}/{name}_f{int(f*10):02d}.h5"
        subset_X = X[idx]  # [N_subset, C, T]
        subset_Y = Y[idx]

        if f > 0:
            for i in range(subset_X.shape[0]):
                subset_X[i] = partial_randomize(subset_X[i], f=f)

        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("X", data=subset_X, compression="gzip")
            hf.create_dataset("Y", data=subset_Y, compression="gzip")

        print(f"  → Saved {name:5s} | f = {f:.1f} | shape = {subset_X.shape} | file: {out_path}")
        
        del subset_X, subset_Y
        gc.collect()
# === 主流程 ===
if __name__ == "__main__":
    for f in f_values:
        preprocess_and_save_split(f, save_dir)
