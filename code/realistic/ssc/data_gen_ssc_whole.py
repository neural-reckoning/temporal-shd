import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

 # download from https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

base_path = "***"
train_path = os.path.join(base_path, "ssc_train.h5")
test_path = os.path.join(base_path, "ssc_test.h5")
valid_path = os.path.join(base_path, "ssc_valid.h5")

train_file = h5py.File(train_path, "r")
test_file = h5py.File(test_path, "r")
valid_file = h5py.File(valid_path, "r")

x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']
x_valid = valid_file['spikes']
y_valid = valid_file['labels']

batch_size = 256
nb_steps = 100
nb_units = 700
max_time = 1.4

def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    labels_ = np.array(y, dtype=np.int32)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))
    firing_times = X['times']
    units_fired = X['units']
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        dense_batch = np.zeros((batch_size, nb_units, nb_steps), dtype=np.uint8)
        y_batch = []

        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            times[times >= nb_steps] = nb_steps - 1 
            dense_batch[bc, units, times] = 1
            y_batch.append(labels_[idx])

        yield dense_batch, np.array(y_batch, dtype=np.uint8)
        counter += 1

def collect_all(X_h5, Y_h5):
    X_all = []
    Y_all = []
    for x_batch, y_batch in sparse_data_generator_from_hdf5_spikes(
            X_h5, Y_h5, batch_size, nb_steps, nb_units, max_time, shuffle=False):
        X_all.append(x_batch)
        Y_all.append(y_batch)
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all

# === Collect all data ===
X_train_all, Y_train_all = collect_all(x_train, y_train)
X_test_all, Y_test_all = collect_all(x_test, y_test)
X_valid_all, Y_valid_all = collect_all(x_valid, y_valid)

# === Concatenate ===
X_all = np.concatenate([X_train_all, X_test_all, X_valid_all], axis=0)
Y_all = np.concatenate([Y_train_all, Y_test_all, Y_valid_all], axis=0)

# === Save ===
import h5py

save_path = "***"

with h5py.File(save_path, "w") as f:
    f.create_dataset("X", data=X_all, compression="gzip")
    f.create_dataset("Y", data=Y_all)

print(f"Saved to: {save_path}")
print(f"X shape: {X_all.shape}, Y shape: {Y_all.shape}")