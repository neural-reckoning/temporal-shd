import numpy as np
import h5py
import os

# ==== Load SSC Whole ====
# after running ssc_whole_data_gen.py and getting ssc_whole.h5
input_path = "***"
with h5py.File(input_path, "r") as f:
    X_all = f["X"][:]  # (N, F, T)
    Y_all = f["Y"][:].ravel()

print(f"Loaded SSC whole: X.shape={X_all.shape}, Y.shape={Y_all.shape}")

# ==== Min-count utils ====
def do_min_count(X, Y):
    N, F, T = X.shape
    count_all = X.sum(axis=2)
    min_counts = count_all.min(axis=0)

    X_min = np.zeros_like(X)
    for f_idx in range(F):
        N_f = min_counts[f_idx]
        if N_f == 0:
            continue
        for i_idx in range(N):
            spike_times = np.where(X[i_idx, f_idx, :] == 1)[0]
            if len(spike_times) > N_f:
                chosen_times = np.random.choice(spike_times, size=N_f, replace=False)
                X_min[i_idx, f_idx, chosen_times] = 1
            else:
                X_min[i_idx, f_idx, spike_times] = 1
    return X_min, Y

def create_min_count_dataset_return_mask(X, Y, neuron_threshold=2, max_frac_for_neuron=0.02, max_samples_to_remove=20000):
    N, F, T = X.shape
    counts = X.sum(axis=2)
    min_counts_per_neuron = counts.min(axis=0)

    bad_neurons = np.where(min_counts_per_neuron < neuron_threshold)[0]
    print(f"Found {len(bad_neurons)} neurons with min_count < {neuron_threshold}.")

    keep_idxs = np.arange(N)
    if len(bad_neurons) > 0:
        samples_to_remove = set()
        for f_idx in bad_neurons:
            neuron_counts = counts[:, f_idx]
            i_bad = np.where(neuron_counts < neuron_threshold)[0]
            frac = len(i_bad) / N
            if frac <= max_frac_for_neuron:
                samples_to_remove.update(i_bad)

        if 0 < len(samples_to_remove) < max_samples_to_remove:
            keep_idxs = np.setdiff1d(np.arange(N), list(samples_to_remove))
            X = X[keep_idxs]
            Y = Y[keep_idxs]
            print(f"Removing {len(samples_to_remove)} samples.")
        else:
            print("NOT removing any samples.")

    X_min, Y_min = do_min_count(X, Y)
    return X_min, Y_min, keep_idxs

# ==== Balance util ====
def balance_and_save(X_in, Y_in, output_path):
    N, num_neurons, T = X_in.shape
    print(f"\nBalancing: X.shape=({N},{num_neurons},{T}), Y.shape=({len(Y_in)})")

    unique_labels = np.unique(Y_in)
    counts_per_class = {c: np.sum(Y_in == c) for c in unique_labels}
    min_count = min(counts_per_class.values())

    print("--- Sample counts per class ---")
    for c in sorted(unique_labels):
        print(f"Class {c}: {counts_per_class[c]} samples")
    print(f"Using min_count = {min_count}")

    X_list = []
    Y_list = []
    for c in sorted(unique_labels):
        idxs_c = np.where(Y_in == c)[0]
        np.random.shuffle(idxs_c)
        selected = idxs_c[:min_count]
        X_list.append(X_in[selected])
        Y_list.append(Y_in[selected])

    X_bal = np.concatenate(X_list, axis=0)
    Y_bal = np.concatenate(Y_list, axis=0)

    perm = np.random.permutation(len(Y_bal))
    X_bal = X_bal[perm]
    Y_bal = Y_bal[perm]

    print(f"Final balanced shape: X={X_bal.shape}, Y={Y_bal.shape}")
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("X", data=X_bal, compression="gzip")
        f_out.create_dataset("Y", data=Y_bal, compression="gzip")
    print(f"Saved balanced dataset to: {output_path}")

X_min, Y_min, keep_idxs = create_min_count_dataset_return_mask(
    X_all, Y_all,
    neuron_threshold=2,
    max_frac_for_neuron=0.02,
    max_samples_to_remove=20000
)
sum_over_samples_time = X_min.sum(axis=(0, 2))
non_zero_mask = (sum_over_samples_time > 0)

# ==== Create PART version and save ====
X_part = X_all[keep_idxs][:, non_zero_mask, :]
Y_part = Y_all[keep_idxs]

balance_and_save(X_part, Y_part, "***")

# ==== Create NORM version and save ====
X_norm = X_min[:, non_zero_mask, :]
Y_norm = Y_min

print(f"\nAfter min-count + drop zero neurons => X_norm.shape = {X_norm.shape}")

balance_and_save(X_norm, Y_norm, "***")
