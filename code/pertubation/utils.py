"""
Spike-train perturbation helpers.

This module provides a few common perturbation operators for spike rasters
represented as NumPy arrays of shape (num_neurons, T):

- `jitter_per_spike`: add Gaussian jitter to every spike independently
- `jitter_per_neuron`: apply the same Gaussian shift to all spikes of a neuron
- `deletion_per_spike`: delete each spike with probability `f`

How to use in training
----------------------
These functions are designed to be drop‑in replacements for the data
augmentation used in training. If your training loop currently calls a
function named `partial_randomize_spike_train(...)`, you can replace that call
with one of the functions in this file to switch the perturbation type, for
example:

    # Old (example):
    # spikes_aug = partial_randomize_spike_train(spikes, f=5)

    # New: per‑spike jitter
    # spikes_aug = jitter_per_spike(spikes, f=5)

    # New: per‑neuron jitter
    # spikes_aug = jitter_per_neuron(spikes, f=5)

    # New: per‑spike deletion
    # spikes_aug = deletion_per_spike(spikes, f=0.5)

No other code changes are required, as long as your training code expects and
returns spike trains in the same shape.
"""

def jitter_per_neuron(spike_train, f=5):
    """Per‑neuron Gaussian jitter.

    Applies the same random Gaussian shift (std `f`) to all spikes of each
    neuron independently. Useful for testing robustness to neuron‑specific
    timing shifts.
    """
    D=f
    shift_amount = 20
    num_neurons, T = spike_train.shape

    final_train = np.zeros_like(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        if len(spike_times) == 0:
            continue

        d = int(round(np.random.normal(0, D))) if D > 0 else 0

        shifted_times_final = np.clip(spike_times + shift_amount + d, 0, T - 1)
        shifted_times_final = np.unique(shifted_times_final)

        final_train[neuron_idx, shifted_times_final] = 1

    return final_train

def jitter_per_spike(spike_train, f=1.0, max_attempts=500):
    """Per‑spike Gaussian jitter.

    Independently jitters each spike time by N(0, f) ms. Collisions are
    resolved by retrying up to `max_attempts` times; if still conflicting, the
    original time is kept.
    """
    num_neurons, T = spike_train.shape
    new_train = np.zeros_like(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        if len(spike_times) == 0:
            continue

        for old_time in spike_times:
            inserted = False
            attempts = 0
            while not inserted and attempts < max_attempts:
                attempts += 1
                jittered_time = int(round(old_time + np.random.normal(0, f)))
                jittered_time = np.clip(jittered_time, 0, T - 1)

                if new_train[neuron_idx, jittered_time] == 0:
                    new_train[neuron_idx, jittered_time] = 1
                    inserted = True

            if not inserted:
                new_train[neuron_idx, old_time] = 1

    return new_train

def deletion_per_spike(spike_train, f=0.5):
    """
    Per‑spike deletion.

    Randomly deletes each spike with probability `f` (0 ≤ f ≤ 1) and returns a
    new spike train.
    """
    if f <= 0:
        return spike_train.copy()
    if f >= 1:
        return np.zeros_like(spike_train)

    num_neurons, T = spike_train.shape
    new_train = np.zeros_like(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        if len(spike_times) == 0:
            continue

        keep_mask = np.random.rand(len(spike_times)) > f
        kept_spikes = spike_times[keep_mask]

        new_train[neuron_idx, kept_spikes] = 1

    return new_train
