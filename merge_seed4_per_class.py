#!/usr/bin/env python3
"""Merge per-class SEED-IV generated npz files into one synthetic bundle."""

import argparse
from pathlib import Path

import numpy as np

LABEL_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge per-class generated SEED-IV DE npz files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Generated npz files, one per class")
    parser.add_argument("--labels", nargs="+", type=int, required=True,
                        help="Class label assigned to each input file")
    parser.add_argument("--output", required=True,
                        help="Output merged npz path")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle merged samples after concatenation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    all_data, all_labels = [], []
    for path_str, label in zip(args.inputs, args.labels):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        bundle = np.load(path)
        data = bundle["data"].astype(np.float32)
        labels = np.full(data.shape[0], label, dtype=np.int64)
        all_data.append(data)
        all_labels.append(labels)
        name = LABEL_NAMES.get(label, str(label))
        print(f"[merge] {name:<7s} label={label}: {data.shape[0]} samples from {path}")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if args.shuffle:
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(data.shape[0])
        data = data[order]
        labels = labels[order]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, data=data, labels=labels)

    print(f"[merge] saved: {output}")
    print(f"[merge] data shape: {data.shape}")
    for label in sorted(set(labels.tolist())):
        name = LABEL_NAMES.get(label, str(label))
        print(f"[merge]   {name:<7s}: {(labels == label).sum()}")


if __name__ == "__main__":
    main()
