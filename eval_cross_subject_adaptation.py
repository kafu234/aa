"""Compare source-only, pseudo-label, and synthetic LOSO adaptation."""

import argparse

import numpy as np
import torch

def get_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import (
            load_de_data, load_synthetic_de, train_and_evaluate,
            train_with_validation,
        )
        return (
            load_de_data, load_synthetic_de, train_and_evaluate,
            train_with_validation, 4,
        ), [
            "neutral", "sad", "fear", "happy"]
    from eval_de_classifier import (
        load_de_data, load_synthetic_de, train_and_evaluate,
        train_with_validation,
    )
    return (
        load_de_data, load_synthetic_de, train_and_evaluate,
        train_with_validation, 3,
    ), [
        "negative", "neutral", "positive"]


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
    parser.add_argument(
        "--run_offset",
        type=int,
        default=0,
        help="zero-based run offset used to resume an interrupted evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_selection",
        choices=["target_test", "source_subject"],
        default="target_test",
        help=(
            "target_test preserves the legacy, label-leaking protocol; "
            "source_subject holds out one labeled source subject per run"
        ),
    )
    parser.add_argument(
        "--model",
        choices=[
            "dgcnn", "dan", "nsal_dgat", "dgat_bls", "gcbnet", "pgcn",
        ],
        default="dgcnn",
    )
    args = parser.parse_args()
    if args.run_offset < 0:
        parser.error("--run_offset must be non-negative")
    if args.dataset == "seed4" and args.model in ("dan", "nsal_dgat"):
        parser.error(f"{args.model} is currently wired for 3-class SEED DE features")

    ((load_de_data, load_synthetic_de, train_and_evaluate,
      train_with_validation, num_classes), class_names) = get_api(args.dataset)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loaded = load_de_data(
        args.data_root, args.seed, "subject", test_subject=args.test_subject,
        return_subjects=args.model_selection == "source_subject",
    )
    if args.model_selection == "source_subject":
        source_x, source_y, target_x, target_y, source_subjects = loaded
        validation_subjects = np.unique(source_subjects)
    else:
        source_x, source_y, target_x, target_y = loaded
        source_subjects = None
        validation_subjects = None
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
    if args.model_selection == "source_subject":
        print(
            "[Model selection] one source subject is held out per run; "
            "target labels are used only for final scoring"
        )
    else:
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
            methods.append(("source_only", None, None))
        elif method == "pseudo":
            methods.append(("source+pseudo", pseudo_x, pseudo_y))
        else:
            methods.append(("source+target_adapted_synthetic", syn_x, syn_y))
    if args.model_selection == "source_subject":
        print(
            "[Protocol] transductive target features may be used by adaptation models; "
            "target labels are used only for final scoring"
        )
    else:
        print("[Protocol] target labels are used for best-epoch selection and final scoring")
    for name, bridge_x, bridge_y in methods:
        accs, f1s = [], []
        bridge_count = 0 if bridge_y is None else len(bridge_y)
        print(
            f"\n{'=' * 64}\n{name}: "
            f"source={len(source_fit_y)}, bridge={bridge_count}\n{'=' * 64}"
        )
        for run in range(args.n_runs):
            run_index = args.run_offset + run
            print(f"  starting run={run_index + 1}, epochs<={args.epochs}, "
                  f"batch={args.batch_size}, val_interval={args.val_interval}, "
                  f"patience={args.patience}", flush=True)
            run_seed = args.seed + run_index
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            if args.model_selection == "source_subject":
                validation_subject = validation_subjects[
                    run_index % len(validation_subjects)
                ]
                validation_mask = source_subjects == validation_subject
                train_x = source_fit_x[~validation_mask]
                train_y = source_fit_y[~validation_mask]
                validation_x = source_fit_x[validation_mask]
                validation_y = source_fit_y[validation_mask]
                if bridge_y is not None:
                    train_x = np.concatenate([train_x, bridge_x])
                    train_y = np.concatenate([train_y, bridge_y])
                print(
                    f"  validation_subject={validation_subject}, "
                    f"train={len(train_y)}, validation={len(validation_y)}",
                    flush=True,
                )
                result = train_with_validation(
                    train_x, train_y, target_x, target_y,
                    validation_x, validation_y, device,
                    model_type=args.model, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    dropout=args.dropout, verbose=False,
                    val_interval=args.val_interval, patience=args.patience,
                )
            else:
                if bridge_y is None:
                    train_x, train_y = source_fit_x, source_fit_y
                else:
                    train_x = np.concatenate([source_fit_x, bridge_x])
                    train_y = np.concatenate([source_fit_y, bridge_y])
                result = train_and_evaluate(
                    train_x, train_y, target_x, target_y, device,
                    model_type=args.model, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    dropout=args.dropout, verbose=False,
                    val_interval=args.val_interval, patience=args.patience,
                )
            accs.append(result["accuracy"])
            f1s.append(result["f1_macro"])
            print(
                f"  run={run_index + 1}: acc={result['accuracy']:.4f}, "
                f"f1={result['f1_macro']:.4f}"
            )
        print(f"  mean: acc={np.mean(accs):.4f}+-{np.std(accs):.4f}, f1={np.mean(f1s):.4f}+-{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
