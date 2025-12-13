import numpy as np
import torch
from scipy import io
from scipy.stats import poisson

# ==== 参数 ====
sig = 2
scale_on = 3
scale_off = 3

def on_prob(k, rate):
    return poisson.pmf(k, rate)

def off_prob(k, rate):
    return poisson.pmf(k, rate)

def normalize_probs(prob_fn, max_k):
    raw = np.array([prob_fn(k) for k in range(max_k + 1)])
    return raw / raw.sum()

def compute_overlap(on_probs, off_probs):
    return np.sum(np.minimum(on_probs, off_probs))

# ==== 计算 lambda 映射 ====
n_neurons = 20  # 用于计算映射的神经元数量
lambda_grid = np.linspace(0, 1.0, 1001)
target_overlaps = np.round(np.arange(0.1, 1.01, 0.1), 2)
found_overlap = {v: (None, float('inf')) for v in target_overlaps}

for lam in lambda_grid:
    on_mu = (1 - lam) * 12 + lam * 5
    off_mu = (1 - lam) * 2 + lam * 5
    on_probs = normalize_probs(lambda k: on_prob(k, on_mu), n_neurons)
    off_probs = normalize_probs(lambda k: off_prob(k, off_mu), n_neurons)
    overlap = compute_overlap(on_probs, off_probs)

    for target in target_overlaps:
        dist = abs(overlap - target)
        if dist < found_overlap[target][1]:
            found_overlap[target] = (lam, dist)

# 构造映射表
lambda_mapping = {0.0: 0.0}
for target, (lam, err) in found_overlap.items():
    lambda_mapping[round(target, 2)] = round(lam, 4)

# 数据生成参数
n_neurons = 60  # 3组，每组20个neuron
group_size = 20
n_timesteps = 4000

def generate_sample(lam, class_type):
    on_mu = (1 - lam) * 12 + lam * 5
    off_mu = (1 - lam) * 2 + lam * 5
    on_probs = normalize_probs(lambda k: on_prob(k, on_mu), group_size)
    off_probs = normalize_probs(lambda k: off_prob(k, off_mu), group_size)

    spikes = np.zeros((n_neurons, n_timesteps))
    group1 = np.arange(0, 20)
    group2 = np.arange(20, 40)
    group3 = np.arange(40, 60)

    window_size = 200
    n_windows = n_timesteps // window_size
    
    for w in range(n_windows):
        start = w * window_size
        spike_start = start + 50
        spike_end = min(start + 150, n_timesteps)
        spike_window = np.arange(spike_start, spike_end)
        
        if len(spike_window) == 0 or spike_start >= n_timesteps:
            continue

        current_state = np.random.rand() < 0.5

        if class_type == 'A':
            g1_dist = on_probs if current_state else off_probs
            g2_dist = g1_dist
            g3_dist = off_probs if current_state else on_probs
        elif class_type == 'B':
            g1_dist = on_probs if current_state else off_probs
            g3_dist = g1_dist
            g2_dist = off_probs if current_state else on_probs
        elif class_type == 'C':
            g2_dist = on_probs if current_state else off_probs
            g3_dist = g2_dist
            g1_dist = off_probs if current_state else on_probs

        for group_neurons, dist in zip([group1, group2, group3], [g1_dist, g2_dist, g3_dist]):
            k = np.random.choice(np.arange(group_size + 1), p=dist)
            if k > 0:
                active = np.random.choice(group_size, size=k, replace=False)
                time_slots = np.random.choice(spike_window, size=k, replace=True)
                for neuron_idx, t in zip(group_neurons[active], time_slots):
                    spikes[neuron_idx, t] = 1

    return spikes

# ==== 数据生成和保存 ====
n_samples_per_class = 500
lambda_prime_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
class_map = {'A': 0, 'B': 1, 'C': 2}

# ==== 3. 分批生成和保存每个lambda值的数据 ====
for lam_prime in lambda_prime_values:
    lam_true = lambda_mapping[round(lam_prime, 1)]  # 映射为 mu-weighted λ
    print(f"Generating for λ′ = {lam_prime:.1f} → μ = {lam_true:.4f}")
    
    # 为当前lambda值初始化数据容器
    X_current = []
    Y_current = []
    lambda_current = []
    
    # 生成当前lambda值的所有数据
    for cls in ['A', 'B', 'C']:
        for _ in range(n_samples_per_class):
            spike = generate_sample(lam_true, cls)
            X_current.append(spike)
            Y_current.append(class_map[cls])
            lambda_current.append(lam_prime)  # 这里保存的是 λ′
    
    # 转换为numpy数组
    X_current = np.array(X_current, dtype=np.uint8)
    Y_current = np.array(Y_current, dtype=np.int32)
    lambda_current = np.array(lambda_current, dtype=np.float32)
    
    # 保存当前lambda值的数据
    filename = f"coin_data_lam{int(lam_prime * 10):02d}.pt"
    data_dict = {
        'X': X_current,
        'Y': Y_current,
        'lambda': lambda_current
    }
    
    torch.save(data_dict, filename)
    print(f"Saved {X_current.shape[0]} samples to {filename}")
    print(f"Shape: X={X_current.shape}, Y={Y_current.shape}, lambda={lambda_current.shape}")
    
    # 清理内存
    del X_current, Y_current, lambda_current
    
print("All lambda datasets generated and saved individually!")

# ==== 4. 合并所有文件为一个完整数据集 ====
print("\n=== Combining all files into one dataset ===")
X_all = []
Y_all = []
lambda_all = []

for lam_prime in lambda_prime_values:
    filename = f"coin_data_lam{int(lam_prime * 10):02d}.pt"
    data = torch.load(filename)
    X_all.extend(data['X'])
    Y_all.extend(data['Y'])
    lambda_all.extend(data['lambda'])

X_all = np.array(X_all, dtype=np.uint8)
Y_all = np.array(Y_all, dtype=np.int32)
lambda_all = np.array(lambda_all, dtype=np.float32)

io.savemat("coin_data.mat", {
    'X': X_all,
    'Y': Y_all,
    'lambda': lambda_all
})
print(f"Combined dataset saved to coin_data.mat with shape: X={X_all.shape}")