"""
Soft synthetic-data filter for SEED and SEED-IV.

The filter keeps a fixed amount per class instead of letting the weakest class
shrink every class. It mixes label-consistent high-confidence samples with
random samples to retain diversity, and can average scores from multiple DGCNNs.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_dataset_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import (
            LABEL_NAMES,
            NUM_CLASSES,
            load_de_data,
            load_synthetic_de,
            train_and_evaluate,
        )
        return NUM_CLASSES, LABEL_NAMES, load_de_data, load_synthetic_de, train_and_evaluate

    from eval_de_classifier import load_de_data, load_synthetic_de, train_and_evaluate
    names = {0: "negative", 1: "neutral", 2: "positive"}
    return 3, names, load_de_data, load_synthetic_de, train_and_evaluate


def predict_probabilities(models, data, device, batch_size=1024):
    loader = DataLoader(TensorDataset(torch.from_numpy(data).float()), batch_size=batch_size)
    probability_sum = None

    for model in models:
        model.eval()
        model_probabilities = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = model(xb.to(device))
                model_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        probabilities = np.concatenate(model_probabilities)
        probability_sum = probabilities if probability_sum is None else probability_sum + probabilities

    return probability_sum / len(models)


def select_class_samples(class_indices, class_scores, class_correct, target_count,
                         high_conf_fraction, rng):
    high_count = min(target_count, int(round(target_count * high_conf_fraction)))

    correct_local = np.where(class_correct)[0]
    correct_order = correct_local[np.argsort(-class_scores[correct_local])]
    selected_local = correct_order[:high_count].tolist()

    # Backfill the high-confidence part by label confidence when too few samples
    # are classified correctly by the scoring ensemble.
    if len(selected_local) < high_count:
        selected_set = set(selected_local)
        score_order = np.argsort(-class_scores)
        for idx in score_order:
            if int(idx) not in selected_set:
                selected_local.append(int(idx))
                selected_set.add(int(idx))
            if len(selected_local) == high_count:
                break

    selected_set = set(selected_local)
    remaining = np.array([i for i in range(len(class_indices)) if i not in selected_set], dtype=np.int64)
    random_count = target_count - len(selected_local)
    if random_count > 0:
        random_local = rng.choice(remaining, size=min(random_count, len(remaining)), replace=False)
        selected_local.extend(random_local.tolist())

    return class_indices[np.asarray(selected_local, dtype=np.int64)]


def main():
    parser = argparse.ArgumentParser(description="柔性质量过滤: 高置信与随机样本混合")
    parser.add_argument("--dataset", choices=["seed", "seed4"], default="seed")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--synthetic_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--split_mode", default="trial")
    parser.add_argument("--train_trials", default=None)
    parser.add_argument("--test_trials", default=None)
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="每类保留比例，默认 0.5")
    parser.add_argument("--high_conf_fraction", type=float, default=0.7,
                        help="保留数据中高置信部分的比例，其余随机选择")
    parser.add_argument("--min_per_class", type=int, default=300,
                        help="每类至少保留数量")
    parser.add_argument("--max_per_class", type=int, default=0,
                        help="每类最多保留数量，0 表示不限制")
    parser.add_argument("--score_runs", type=int, default=3,
                        help="用于平均打分的独立 DGCNN 数量")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_eval", action="store_true",
                        help="过滤后快速评估多个增强比例")
    args = parser.parse_args()

    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep_ratio 必须在 (0, 1] 范围内")
    if not 0 <= args.high_conf_fraction <= 1:
        raise ValueError("--high_conf_fraction 必须在 [0, 1] 范围内")
    if args.score_runs < 1:
        raise ValueError("--score_runs 必须至少为 1")

    num_classes, names, load_de_data, load_synthetic_de, train_and_evaluate = \
        get_dataset_api(args.dataset)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    tr_d, tr_l, te_d, te_l = load_de_data(
        args.data_root, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject,
    )
    syn_data, syn_labels = load_synthetic_de(args.synthetic_path)
    invalid = sorted(set(np.unique(syn_labels).tolist()) - set(range(num_classes)))
    if invalid:
        raise ValueError(f"合成数据包含超出 {args.dataset} 类别范围的标签: {invalid}")

    print(f"\n训练 {args.score_runs} 个 DGCNN 为合成数据打分...")
    models = []
    baseline_accs = []
    for run in range(args.score_runs):
        run_seed = args.seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        result = train_and_evaluate(
            tr_d, tr_l, te_d, te_l, device, model_type="dgcnn",
            epochs=args.epochs, split_seed=run_seed, verbose=False,
        )
        models.append(result["model"])
        baseline_accs.append(result["accuracy"])
        print(f"  Scorer {run + 1}: baseline acc={result['accuracy']:.4f}")

    probabilities = predict_probabilities(models, syn_data, device)
    predictions = probabilities.argmax(axis=1)
    rng = np.random.RandomState(args.seed)
    selected_indices = []

    print(f"\n{'=' * 66}")
    print("  柔性过滤统计")
    print(f"  keep_ratio={args.keep_ratio}, high_conf_fraction={args.high_conf_fraction}, "
          f"min_per_class={args.min_per_class}")
    print(f"{'=' * 66}")

    for class_id in range(num_classes):
        class_indices = np.where(syn_labels == class_id)[0]
        if len(class_indices) == 0:
            raise ValueError(f"类别 {class_id} ({names[class_id]}) 没有生成样本")

        target = max(args.min_per_class, int(round(len(class_indices) * args.keep_ratio)))
        if args.max_per_class > 0:
            target = min(target, args.max_per_class)
        target = min(target, len(class_indices))

        class_scores = probabilities[class_indices, class_id]
        class_correct = predictions[class_indices] == class_id
        chosen = select_class_samples(
            class_indices, class_scores, class_correct, target,
            args.high_conf_fraction, rng,
        )
        selected_indices.extend(chosen.tolist())

        chosen_scores = probabilities[chosen, class_id]
        chosen_correct = predictions[chosen] == class_id
        print(f"  {names[class_id]:>8}: source={len(class_indices):4d}, "
              f"scorer_correct={class_correct.mean():.3f}, kept={len(chosen):4d}, "
              f"kept_correct={chosen_correct.mean():.3f}, "
              f"confidence={chosen_scores.mean():.3f}")

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices)
    filtered_data = syn_data[selected_indices]
    filtered_labels = syn_labels[selected_indices]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        data=(filtered_data * 5.0).astype(np.float32),
        labels=filtered_labels.astype(np.int64),
    )
    print(f"\n过滤后: {filtered_data.shape}")
    print(f"类别分布: {dict(zip(*np.unique(filtered_labels, return_counts=True)))}")
    print(f"保存至: {args.output}")

    if args.quick_eval:
        print(f"\nBaseline acc: {np.mean(baseline_accs):.4f}±{np.std(baseline_accs):.4f}")
        for ratio in [0.1, 0.25, 0.5, 1.0]:
            count = max(1, int(len(filtered_labels) * ratio))
            subset = rng.choice(len(filtered_labels), size=count, replace=False)
            aug_data = np.concatenate([tr_d, filtered_data[subset]])
            aug_labels = np.concatenate([tr_l, filtered_labels[subset]])
            accs = []
            for run in range(3):
                run_seed = args.seed + run
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                result = train_and_evaluate(
                    aug_data, aug_labels, te_d, te_l, device,
                    model_type="dgcnn", epochs=args.epochs,
                    split_seed=run_seed, verbose=False,
                )
                accs.append(result["accuracy"])
            print(f"  ratio={ratio:.2f}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")


if __name__ == "__main__":
    main()
