"""Compare source-only, pseudo-label, and synthetic LOSO adaptation."""

import argparse

import numpy as np
import torch

def get_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import load_de_data, load_synthetic_de, train_and_evaluate
        return load_de_data, load_synthetic_de, train_and_evaluate
    from eval_de_classifier import load_de_data, load_synthetic_de, train_and_evaluate
    return load_de_data, load_synthetic_de, train_and_evaluate


def load_pseudo(path):
    bundle = np.load(path)
    return np.clip(bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32), bundle["labels"].astype(np.int64)


def balanced_subset(data, labels, count, seed):
    if count <= 0 or count >= len(labels):
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
    parser = argparse.ArgumentParser(description="Transductive LOSO adaptation comparison")
    parser.add_argument("--dataset", choices=["seed", "seed4"], default="seed")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_subject", type=int, required=True)
    parser.add_argument("--pseudo_path", default=None)
    parser.add_argument("--synthetic_path", required=True, help="target-adapted synthetic data")
    parser.add_argument("--pseudo_ratio", type=float, default=0.1)
    parser.add_argument("--syn_ratio", type=float, default=0.1)
    parser.add_argument("--num_synthetic", type=int, default=None,
                        help="exact synthetic sample count; overrides --syn_ratio")
    parser.add_argument("--methods", nargs="+",
                        choices=["source_only", "pseudo", "synthetic"],
                        default=["source_only", "pseudo", "synthetic"],
                        help="evaluation groups to run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_de_data, load_synthetic_de, train_and_evaluate = get_api(args.dataset)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_x, source_y, target_x, target_y, source_subjects = load_de_data(
        args.data_root, args.seed, "subject", test_subject=args.test_subject,
        return_subjects=True,
    )
    if "pseudo" in args.methods:
        if args.pseudo_path is None:
            parser.error("--pseudo_path is required when --methods includes pseudo")
        pseudo_x, pseudo_y = load_pseudo(args.pseudo_path)
    else:
        pseudo_x = np.empty((0,) + source_x.shape[1:], dtype=np.float32)
        pseudo_y = np.empty(0, dtype=np.int64)
    syn_x, syn_y = load_synthetic_de(args.synthetic_path)

    source_fit_x, source_fit_y = source_x, source_y
    print(f"[Source training] using all {len(source_y)} source samples")
    print("[Model selection] no source validation split; best epoch is selected "
          "directly on the target test set")
    requested_pseudo = int(len(source_fit_y) * args.pseudo_ratio)
    requested_syn = (args.num_synthetic if args.num_synthetic is not None
                     else int(len(source_fit_y) * args.syn_ratio))
    pseudo_count = min(requested_pseudo, len(pseudo_y))
    syn_count = min(requested_syn, len(syn_y))
    if "pseudo" in args.methods and pseudo_count < 1:
        raise ValueError("pseudo-label sample count is zero")
    if "synthetic" in args.methods and syn_count < 1:
        raise ValueError("synthetic sample count is zero")
    pseudo_x, pseudo_y = balanced_subset(pseudo_x, pseudo_y, pseudo_count, args.seed)
    syn_x, syn_y = balanced_subset(syn_x, syn_y, syn_count, args.seed)
    print(f"[Comparison counts] pseudo={len(pseudo_y)}, target_adapted_synthetic={len(syn_y)}")

    methods = []
    for method in args.methods:
        if method == "source_only":
            methods.append(("source_only", source_fit_x, source_fit_y))
        elif method == "pseudo":
            methods.append((
                "source+pseudo",
                np.concatenate([source_fit_x, pseudo_x]),
                np.concatenate([source_fit_y, pseudo_y]),
            ))
        else:
            methods.append((
                "source+target_adapted_synthetic",
                np.concatenate([source_fit_x, syn_x]),
                np.concatenate([source_fit_y, syn_y]),
            ))
    print("[Protocol] target labels are used for best-epoch selection and final scoring")
    for name, train_x, train_y in methods:
        accs, f1s = [], []
        print(f"\n{'=' * 64}\n{name}: train={len(train_y)}\n{'=' * 64}")
        for run in range(args.n_runs):
            print(f"  starting run={run + 1}/{args.n_runs}, epochs<={args.epochs}, "
                  f"batch={args.batch_size}, val_interval={args.val_interval}, "
                  f"patience={args.patience}", flush=True)
            run_seed = args.seed + run
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            result = train_and_evaluate(
                train_x, train_y, target_x, target_y, device,
                epochs=args.epochs, batch_size=args.batch_size, verbose=False,
                val_interval=args.val_interval, patience=args.patience,
            )
            accs.append(result["accuracy"])
            f1s.append(result["f1_macro"])
            print(f"  run={run + 1}: acc={result['accuracy']:.4f}, f1={result['f1_macro']:.4f}")
        print(f"  mean: acc={np.mean(accs):.4f}+-{np.std(accs):.4f}, f1={np.mean(f1s):.4f}+-{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
