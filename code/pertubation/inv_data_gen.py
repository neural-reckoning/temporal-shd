"""
Inverse SHD dataset generator.

This script reproduces the inverse-data construction used in
`pertubation/inverse/inverse_shd.ipynb` and saves three inverted datasets:

- shd_whole_inv.mat
- shd_part_inv.mat
- shd_norm_inv.mat

Method (per sample):
- Find the global active window [t_start, t_end] across all neurons where any
  spike occurs (first and last spike times).
- For every neuron, reverse the spike segment within this window. This is a
  time inversion of the active portion only (outside the window is unchanged).

Usage:
- Place `shd_whole.mat`, `shd_part.mat`, and `shd_norm.mat` in the same folder
  as this script (or pass custom paths via CLI args).
- Run the script. It will save corresponding `*_inv.mat` files next to the
  inputs.

Evaluation note:
- To test on the inverted datasets, you only need to load your models trained
  on the original datasets (weights from `shd_whole.mat`, `shd_part.mat`, and
  `shd_norm.mat`) and evaluate them on `*_inv.mat`. No retraining is required.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
from scipy.io import loadmat, savemat


def invert_sample_time_window(sample: np.ndarray) -> np.ndarray:
    """Invert spikes within the global active time window for a sample.

    Expects `sample` of shape (num_neurons, T) with binary spikes.
    Returns a new array with the same shape.
    """
    num_neurons, T = sample.shape

    # Find all spike times across all neurons for this sample
    spike_where = np.where(sample == 1)
    if spike_where[0].size == 0:
        return sample.copy()

    # Active window is from first to last spike time
    t_start = int(np.min(spike_where[1]))
    t_end = int(np.max(spike_where[1]))

    out = sample.copy()
    # Reverse each neuron's activity within [t_start, t_end]
    seg_slice = slice(t_start, t_end + 1)
    for j in range(num_neurons):
        seg = out[j, seg_slice]
        out[j, seg_slice] = seg[::-1]
    return out


def generate_inverse_dataset(in_mat: str, out_mat: str) -> Tuple[int, int, int]:
    """Load a SHD .mat dataset, invert each sample's active window, and save.

    The .mat file must contain keys:
      - `X`: shaped (N, num_neurons, T)
      - `Y`: labels, shaped (N,) or (N, 1)

    Returns a short tuple with dataset dimensions (N, num_neurons, T).
    """
    data = loadmat(in_mat)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"Input {in_mat} must contain keys 'X' and 'Y'.")

    X = data["X"]
    Y = data["Y"].ravel()

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D (N, num_neurons, T), got shape {X.shape}"
        )

    N, num_neurons, T = X.shape

    # Process each sample independently
    X_inv = np.empty_like(X)
    for i in range(N):
        X_inv[i] = invert_sample_time_window(X[i])

    savemat(out_mat, {"X": X_inv, "Y": Y})
    return (N, num_neurons, T)


def main():
    parser = argparse.ArgumentParser(
        description="Generate inverse SHD datasets (whole/part/norm)."
    )
    parser.add_argument(
        "--whole",
        default="shd_whole.mat",
        help="Path to shd_whole.mat (input)",
    )
    parser.add_argument(
        "--part",
        default="shd_part.mat",
        help="Path to shd_part.mat (input)",
    )
    parser.add_argument(
        "--norm",
        default="shd_norm.mat",
        help="Path to shd_norm.mat (input)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory (defaults to input file directory).",
    )
    args = parser.parse_args()

    # Build output paths next to inputs unless an explicit out-dir is given.
    cfg = [
        (args.whole, "shd_whole_inv.mat"),
        (args.part, "shd_part_inv.mat"),
        (args.norm, "shd_norm_inv.mat"),
    ]

    for in_path, default_out_name in cfg:
        if not os.path.isfile(in_path):
            print(f"[Skip] Input not found: {in_path}")
            continue
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(in_path))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, default_out_name)

        print(f"Generating inverse dataset for: {in_path}")
        N, C, T = generate_inverse_dataset(in_path, out_path)
        print(f"  Saved: {out_path} | shape: (N={N}, neurons={C}, T={T})")

    print(
        "\nNote: To evaluate effects on inverted data, simply load models trained "
        "with 'shd_whole.mat', 'shd_part.mat', or 'shd_norm.mat' and run "
        "inference on the corresponding '*_inv.mat' datasets. No retraining "
        "is needed."
    )


if __name__ == "__main__":
    main()

