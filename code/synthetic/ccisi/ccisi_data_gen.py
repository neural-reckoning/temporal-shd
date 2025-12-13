# generate_ccisi_parallel_h5.py (修改版, 使用通用直线 y = slope*x + b 作为分界)
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py

# ================== 配置 ==================
ms = 1e-3  # 常量: 1ms
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 参数
num_points = 5000   # 初始生成点数
radius = 10         # 平面边界 [-radius, radius]
min_distance = 2.5    # 与直线的最小距离
slope = -1/2          # 直线斜率
b = 0               # 直线截距
T = 10000            # 时间步长 (ms)
num_neurons = 20    # 神经元数量（必须为偶数）
max_attempts = 500  # 放置 spike 尝试次数
num_workers = 10

# ================== 数据点生成 ==================
X = np.random.uniform(low=-radius, high=radius, size=(num_points, 2))

# 距离计算: 点到直线 ax + by + c = 0 的距离
# 直线: y = slope*x + b  => slope*x - y + b = 0
a = slope
c = b
distances = np.abs(a * X[:, 0] - X[:, 1] + c) / np.sqrt(a ** 2 + 1)

# 过滤点
valid_indices = distances >= min_distance
X = X[valid_indices]
num_valid_points = X.shape[0]

# 分类: y > slope*x + b 属于 Class 0，否则 Class 1
y = np.where(X[:, 1] > slope * X[:, 0] + b, 0, 1).astype(np.uint8)

# ================== 映射函数 ==================
def linear_map_to_half_steps(value, old_min, old_max, new_min, new_max, step=1):
    mapped = (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    stepped = np.round(mapped / step) * step
    return np.clip(stepped, new_min, new_max)

X_min = X.min(axis=0)
X_max = X.max(axis=0)

firing_rates = linear_map_to_half_steps(X[:, 0], X_min[0], X_max[0], 0.1, 3, step=0.1)
isis = linear_map_to_half_steps(X[:, 1], X_min[1], X_max[1], 1, 100, step=1)

# ================== 可视化检查 ==================
plt.figure(figsize=(8, 6))
plt.scatter(firing_rates[y == 0], isis[y == 0], color='blue', label='Class 0')
plt.scatter(firing_rates[y == 1], isis[y == 1], color='red', label='Class 1')

# 画分界线
x_line = np.linspace(-radius, radius, 100)
y_line = slope * x_line + b
plt.plot(x_line, y_line, color='black', linestyle='--', label=f'y = {slope}x + {b}')

plt.xlabel('Firing Rate (Hz)')
plt.ylabel('ISI (ms)')
plt.legend()
plt.title('Two-Class Dataset (Firing Rate & ISI Scaled)')
plt.grid()
plt.show()

# ================== Spike Train 生成函数 (新逻辑) ==================
def generate_spike_train(f, isi, num_neurons=20, T=10000, max_attempts=100):
    """
    CCISI模式:
    - neuron_a 在偶数索引
    - neuron_b 在对应奇数索引，延迟 isi 时间
    - firing rate f 按 pair 计算（Hz，允许小数）
    """
    assert num_neurons % 2 == 0, "Number of neurons must be even for pairing"
    spike_trains = np.zeros((num_neurons, T), dtype=np.uint8)

    # === 修正 firing rate 计算，支持小数 Hz ===
    duration_s = T / 1000.0
    pairs = int(np.round(f * duration_s))   # 期望的 pair 数
    total_spikes = 2 * pairs                # 每个 pair 包含 2 个 spike

    isi_steps = max(1, int(round(isi)))

    # 遍历所有 neuron pairs
    for pair_idx in range(0, num_neurons, 2):
        neuron_a = pair_idx
        neuron_b = pair_idx + 1
        occupied_times = set()

        for _ in range(pairs):
            placed = False
            for _ in range(max_attempts):
                start_t = np.random.randint(0, T - isi_steps)
                # 检查 start_t 附近是否有冲突
                if all(
                    t not in occupied_times
                    for t in range(start_t - isi_steps, start_t + isi_steps + 1)
                ):
                    spike_trains[neuron_a, start_t] = 1
                    spike_trains[neuron_b, start_t + isi_steps] = 1
                    for conflict_t in range(start_t - isi_steps, start_t + isi_steps + 1):
                        occupied_times.add(conflict_t)
                    placed = True
                    break
            if not placed:
                pass  # 放不下就跳过

    return spike_trains


# 包装成单样本任务
def worker_task(i):
    f_val = firing_rates[i]
    isi_val = isis[i]
    return generate_spike_train(f_val, isi_val, num_neurons, T=T, max_attempts=max_attempts)

# ================== 主流程 ==================
print(f"Generating dataset with {num_valid_points} valid samples using {num_workers} workers...")

spike_trains_per_sample = [None] * num_valid_points
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(worker_task, i): i for i in range(num_valid_points)}
    for future in tqdm(as_completed(futures), total=num_valid_points, desc="Generating spike trains"):
        idx = futures[future]
        spike_trains_per_sample[idx] = future.result()

spike_trains_per_sample = np.array(spike_trains_per_sample, dtype=np.uint8)

# ================== 保存为 HDF5 ==================
h5_filename = "ccisi.h5"
with h5py.File(h5_filename, "w") as f:
    f.create_dataset("X", data=spike_trains_per_sample, compression="gzip")
    f.create_dataset("Y", data=y, compression="gzip")
    f.create_dataset("firing_rates", data=firing_rates, compression="gzip")
    f.create_dataset("isis", data=isis, compression="gzip")

print(
    f"Dataset saved to {h5_filename}, X shape = {spike_trains_per_sample.shape}, dtype={spike_trains_per_sample.dtype}"
)
