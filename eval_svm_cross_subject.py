"""Sample-level Linear SVM LOSO evaluation for SEED DE features."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from eval_de_classifier import load_de_data, load_synthetic_de


def balanced_subset(data, labels, count, seed):
    if count is None or count <= 0 or count >= len(labels):
        return data, labels
    rng = np.random.RandomState(seed)
    selected = []
    classes = np.unique(labels)
    per_class = count // len(classes)
    for i, class_id in enumerate(classes):
        candidates = np.where(labels == class_id)[0]
        n = per_class if i < len(classes) - 1 else count - len(selected)
        selected.extend(rng.choice(candidates, min(n, len(candidates)), replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return data[selected], labels[selected]


def main():
    parser = argparse.ArgumentParser(
        description="Sample-level Linear SVM cross-subject DE evaluation")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_subject", type=int, required=True)
    parser.add_argument("--synthetic_path", default=None)
    parser.add_argument("--syn_ratio", type=float, default=None,
                        help="synthetic count as a ratio of source count")
    parser.add_argument("--num_synthetic", type=int, default=None,
                        help="exact synthetic count; overrides --syn_ratio")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_x, source_y, target_x, target_y = load_de_data(
        args.data_root, args.seed, "subject", test_subject=args.test_subject)

    train_x, train_y = source_x, source_y
    method = "source_only"
    if args.synthetic_path:
        syn_x, syn_y = load_synthetic_de(args.synthetic_path)
        requested = args.num_synthetic
        if requested is None and args.syn_ratio is not None:
            requested = int(len(source_y) * args.syn_ratio)
        syn_x, syn_y = balanced_subset(syn_x, syn_y, requested, args.seed)
        train_x = np.concatenate([source_x, syn_x], axis=0)
        train_y = np.concatenate([source_y, syn_y], axis=0)
        method = "source+synthetic"

    train_x = train_x.reshape(len(train_x), -1)
    target_x = target_x.reshape(len(target_x), -1)

    print(f"[SVM] metric=sample_level_no_trial_voting, method={method}, "
          f"train={len(train_y)}, test={len(target_y)}, "
          f"features={train_x.shape[1]}, C={args.C}, max_iter={args.max_iter}")
    print(f"  Train labels: {dict(zip(*np.unique(train_y, return_counts=True)))}")
    print(f"  Test  labels: {dict(zip(*np.unique(target_y, return_counts=True)))}")

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=args.C,
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
            dual=False,
        ),
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(target_x)

    acc = accuracy_score(target_y, pred)
    f1 = f1_score(target_y, pred, average="macro")
    names = ["negative", "neutral", "positive"]
    per_class = {
        names[c]: float((pred[target_y == c] == c).mean())
        if (target_y == c).any() else 0.0
        for c in range(3)
    }
    print(f"result: acc={acc:.4f}, f1={f1:.4f}, per_class={per_class}")


if __name__ == "__main__":
    main()
