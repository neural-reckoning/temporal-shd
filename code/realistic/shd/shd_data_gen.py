import os
import numpy as np
import h5py
import scipy.io as io
from utils import get_shd_dataset


# ==============================
# Step 1. Download SHD dataset
# ==============================
def prepare_shd_dataset(base_dir="~/shd_data"):
    base_dir = os.path.expanduser(base_dir)
    cache_subdir = "hdspikes"
    os.makedirs(base_dir, exist_ok=True)
    get_shd_dataset(base_dir, cache_subdir)

    base_path = os.path.join(base_dir, cache_subdir)
    train_path = os.path.join(base_path, "shd_train.h5")
    test_path = os.path.join(base_path, "shd_test.h5")
    assert os.path.exists(train_path), f"train file not found: {train_path}"
    assert os.path.exists(test_path), f"test file not found: {test_path}"
    return train_path, test_path


# ==============================
# Step 2. Convert to dense spike trains
# ==============================
def sparse_to_dense(X, y, nb_steps=100, nb_units=700, max_time=1.4, batch_size=256):
    labels_ = np.array(y, dtype=np.int32)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))
    firing_times = X['times']
    units_fired = X['units']
    time_bins = np.linspace(0, max_time, num=nb_steps)
    dense_list, label_list = [], []

    for counter in range(number_of_batches):
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        dense_batch = np.zeros((batch_size, nb_units, nb_steps), dtype=np.uint8)
        y_batch = []

        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            times[times >= nb_steps] = nb_steps - 1
            dense_batch[bc, units, times] = 1
            y_batch.append(labels_[idx])

        dense_list.append(dense_batch)
        label_list.append(np.array(y_batch, dtype=np.uint8))

    X_all = np.concatenate(dense_list, axis=0)
    Y_all = np.concatenate(label_list, axis=0)
    return X_all, Y_all


# ==============================
# Step 3. Combine and save as shd_whole.mat
# ==============================
def create_whole_dataset(train_path, test_path, save_path):
    train_file = h5py.File(train_path, "r")
    test_file = h5py.File(test_path, "r")

    X_train_all, Y_train_all = sparse_to_dense(train_file['spikes'], train_file['labels'])
    X_test_all, Y_test_all = sparse_to_dense(test_file['spikes'], test_file['labels'])

    X_all = np.concatenate([X_train_all, X_test_all], axis=0)
    Y_all = np.concatenate([Y_train_all, Y_test_all], axis=0)

    io.savemat(save_path, {'X': X_all, 'Y': Y_all})
    print(f"[✓] Saved whole dataset: {save_path}, X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all


# ==============================
# Step 4. Min-count filtering
# ==============================
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


def create_min_count_dataset_return_mask(X, Y,
                                         neuron_threshold=2,
                                         max_frac_for_neuron=0.01,
                                         max_samples_to_remove=2000):
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


# ==============================
# Step 5. Class balancing
# ==============================
def balance_dataset(input_path, output_path):
    print(f"\nProcessing: {input_path}")
    data = io.loadmat(input_path)
    X_full = data["X"]
    Y_full = data["Y"].ravel()
    N, F, T = X_full.shape

    print(f"Loaded dataset: X.shape=({N},{F},{T}), Y.shape=({len(Y_full)})")

    unique_labels = np.unique(Y_full)
    counts_per_class = {c: np.sum(Y_full == c) for c in unique_labels}
    min_count = min(counts_per_class.values())

    print("--- Sample counts per class ---")
    for c in sorted(unique_labels):
        print(f"Class {c}: {counts_per_class[c]} samples")
    print(f"Using min_count = {min_count}")

    X_list, Y_list = [], []
    for c in sorted(unique_labels):
        idxs_c = np.where(Y_full == c)[0]
        np.random.shuffle(idxs_c)
        selected = idxs_c[:min_count]
        X_list.append(X_full[selected])
        Y_list.append(Y_full[selected])

    X_bal = np.concatenate(X_list, axis=0)
    Y_bal = np.concatenate(Y_list, axis=0)
    perm = np.random.permutation(len(Y_bal))
    X_bal = X_bal[perm]
    Y_bal = Y_bal[perm]

    io.savemat(output_path, {"X": X_bal, "Y": Y_bal})
    print(f"Final balanced shape: X={X_bal.shape}, Y={Y_bal.shape}")
    print(f"Saved balanced dataset to: {output_path}")
    return X_bal, Y_bal


# ==============================
# Step 6. Full pipeline
# ==============================
def main():
    base_dir = os.path.expanduser("~/shd_data")
    os.makedirs(base_dir, exist_ok=True)

    print("[Step 1] Downloading SHD dataset...")
    train_path, test_path = prepare_shd_dataset(base_dir)

    print("[Step 2] Building dense spike train dataset...")
    shd_whole_path = os.path.join(base_dir, "shd_whole.mat")
    X_all, Y_all = create_whole_dataset(train_path, test_path, shd_whole_path)

    print("[Step 3] Applying min-count filtering...")
    X_min, Y_min, keep_idxs = create_min_count_dataset_return_mask(X_all, Y_all)
    sum_over_samples_time = X_min.sum(axis=(0, 2))
    non_zero_mask = (sum_over_samples_time > 0)

    # === Apply masks ===
    X_norm = X_min[:, non_zero_mask, :]
    io.savemat(os.path.join(base_dir, "shd_norm_ori.mat"), {"X": X_norm, "Y": Y_min})
    print(f"[✓] Saved shd_norm_ori.mat: X={X_norm.shape}, Y={Y_min.shape}")

    X_part = X_all[keep_idxs][:, non_zero_mask, :]
    Y_part = Y_all[keep_idxs]
    io.savemat(os.path.join(base_dir, "shd_part_ori.mat"), {"X": X_part, "Y": Y_part})
    print(f"[✓] Saved shd_part_ori.mat: X={X_part.shape}, Y={Y_part.shape}")

    # Step 4. Balance both datasets
    print("[Step 4] Balancing datasets...")
    balance_dataset(os.path.join(base_dir, "shd_part_ori.mat"),
                    os.path.join(base_dir, "shd_part_new.mat"))
    balance_dataset(os.path.join(base_dir, "shd_norm_ori.mat"),
                    os.path.join(base_dir, "shd_norm_new.mat"))

    print("\n=== Final Summary ===")
    print(f"shd_whole.mat -> X: {X_all.shape}, Y: {Y_all.shape}")
    print(f"shd_norm_new.mat -> balanced")
    print(f"shd_part_new.mat -> balanced")


if __name__ == "__main__":
    main()
