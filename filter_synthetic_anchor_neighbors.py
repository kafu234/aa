"""Rank generated DE samples by consistency with target pseudo-label anchors."""

import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def get_dataset_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import (
            LABEL_NAMES, NUM_CLASSES, load_synthetic_de)
        return NUM_CLASSES, tuple(
            LABEL_NAMES[class_id] for class_id in range(NUM_CLASSES)
        ), load_synthetic_de
    from eval_de_classifier import load_synthetic_de
    return 3, ("negative", "neutral", "positive"), load_synthetic_de


def balanced_counts(total, num_classes):
    base = total // num_classes
    counts = [base] * num_classes
    for index in range(total - base * num_classes):
        counts[index] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Create fixed-size synthetic subsets ranked by target-anchor consistency")
    parser.add_argument("--dataset", choices=["seed", "seed4"], default="seed")
    parser.add_argument("--synthetic_path", required=True)
    parser.add_argument("--anchor_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_count", type=int, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.15, 0.2])
    parser.add_argument("--neighbors", type=int, default=11)
    parser.add_argument("--pca_variance", type=float, default=0.95)
    args = parser.parse_args()

    num_classes, class_names, load_synthetic_de = get_dataset_api(args.dataset)
    synthetic_x, synthetic_y = load_synthetic_de(args.synthetic_path)
    anchor_bundle = np.load(args.anchor_path)
    anchor_x = np.clip(
        anchor_bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32)
    anchor_y = anchor_bundle["labels"].astype(np.int64)

    expected_classes = list(range(num_classes))
    if sorted(np.unique(synthetic_y).tolist()) != expected_classes:
        raise ValueError(
            f"Synthetic data must contain all {num_classes} {args.dataset} classes")
    if sorted(np.unique(anchor_y).tolist()) != expected_classes:
        raise ValueError(
            f"Anchor data must contain all {num_classes} {args.dataset} classes")
    if not 1 <= args.neighbors <= len(anchor_y):
        raise ValueError("--neighbors must be within the anchor sample count")

    anchor_flat = anchor_x.reshape(len(anchor_x), -1)
    synthetic_flat = synthetic_x.reshape(len(synthetic_x), -1)
    scaler = StandardScaler().fit(anchor_flat)
    anchor_scaled = scaler.transform(anchor_flat)
    synthetic_scaled = scaler.transform(synthetic_flat)
    pca = PCA(
        n_components=args.pca_variance,
        svd_solver="full",
        random_state=42,
    ).fit(anchor_scaled)
    anchor_features = pca.transform(anchor_scaled)
    synthetic_features = pca.transform(synthetic_scaled)

    neighbors = NearestNeighbors(
        n_neighbors=args.neighbors, metric="euclidean", n_jobs=-1)
    neighbors.fit(anchor_features)
    distances, indices = neighbors.kneighbors(synthetic_features)
    neighbor_labels = anchor_y[indices]
    same_class = neighbor_labels == synthetic_y[:, None]
    purity = same_class.mean(axis=1)

    # Distance to the nearest same-label anchor. If no same-label anchor occurs
    # among the k nearest neighbors, use infinity so it ranks last.
    same_distance = np.where(same_class, distances, np.inf).min(axis=1)

    ranked_by_class = {}
    print(
        f"Anchors={len(anchor_y)}, synthetic={len(synthetic_y)}, "
        f"PCA components={pca.n_components_}, k={args.neighbors}")
    for class_id, name in enumerate(class_names):
        class_indices = np.flatnonzero(synthetic_y == class_id)
        # Primary key: purity descending. Secondary key: same-class distance ascending.
        order = np.lexsort((same_distance[class_indices], -purity[class_indices]))
        ranked = class_indices[order]
        ranked_by_class[class_id] = ranked
        print(
            f"{name}: n={len(class_indices)}, "
            f"purity mean={purity[class_indices].mean():.3f}, "
            f"p50={np.median(purity[class_indices]):.3f}, "
            f"fully_consistent={(purity[class_indices] == 1.0).mean():.2%}")

    os.makedirs(args.output_dir, exist_ok=True)
    for ratio in args.ratios:
        requested = int(args.source_count * ratio)
        counts = balanced_counts(requested, num_classes)
        selected_parts = []
        print(f"\nratio={ratio:.3f}, requested={requested}")
        for class_id, count in enumerate(counts):
            ranked = ranked_by_class[class_id]
            if count > len(ranked):
                raise ValueError(
                    f"ratio {ratio} requests {count} samples for class {class_id}, "
                    f"but only {len(ranked)} exist")
            chosen = ranked[:count]
            selected_parts.append(chosen)
            print(
                f"  {class_names[class_id]}: kept={count}, "
                f"purity mean={purity[chosen].mean():.3f}, "
                f"minimum={purity[chosen].min():.3f}, "
                f"same-distance mean={same_distance[chosen].mean():.3f}")

        selected = np.concatenate(selected_parts)
        rng = np.random.RandomState(42)
        rng.shuffle(selected)
        ratio_name = f"{ratio:.3f}".replace(".", "p")
        output = os.path.join(
            args.output_dir, f"generated_anchor_ranked_r{ratio_name}.npz")
        np.savez(
            output,
            data=(synthetic_x[selected] * 5.0).astype(np.float32),
            labels=synthetic_y[selected].astype(np.int64),
            anchor_purity=purity[selected].astype(np.float32),
            nearest_same_anchor_distance=same_distance[selected].astype(np.float32),
            original_indices=selected.astype(np.int64),
        )
        print(f"  saved: {output}")


if __name__ == "__main__":
    main()
