"""Create high-confidence pseudo labels for an unlabeled LOSO target subject."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Utils.Data_utils.group_split import group_holdout


def get_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import load_de_data, train_and_evaluate, NUM_CLASSES, LABEL_NAMES
        return load_de_data, train_and_evaluate, NUM_CLASSES, LABEL_NAMES
    from eval_de_classifier import load_de_data, train_and_evaluate
    return load_de_data, train_and_evaluate, 3, {0: "negative", 1: "neutral", 2: "positive"}


def predict_probabilities(models, data, device, batch_size=1024):
    loader = DataLoader(TensorDataset(torch.from_numpy(data).float()), batch_size=batch_size)
    all_probabilities = []
    for model in models:
        model.eval()
        parts = []
        with torch.no_grad():
            for (xb,) in loader:
                parts.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
        all_probabilities.append(np.concatenate(parts))
    stacked = np.stack(all_probabilities)
    probabilities = stacked.mean(axis=0)
    predictions = probabilities.argmax(axis=1)
    agreement = (stacked.argmax(axis=2) == predictions[None, :]).mean(axis=0)
    return probabilities, agreement


def select_pseudo_labels(probs, agreement, threshold, min_agreement, min_per_class, max_per_class, seed):
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in range(probs.shape[1]):
        candidates = np.where(predictions == class_id)[0]
        if len(candidates) == 0:
            raise ValueError(f"source ensemble predicted no target samples for class {class_id}")
        score = confidence[candidates] * agreement[candidates]
        order = candidates[np.argsort(-score)]
        high = order[(confidence[order] >= threshold) & (agreement[order] >= min_agreement)]
        keep_n = max(min_per_class, len(high))
        if max_per_class > 0:
            keep_n = min(keep_n, max_per_class)
        keep_n = min(keep_n, len(order))
        chosen = np.concatenate([high, order[~np.isin(order, high)]])[:keep_n]
        selected.extend(chosen.tolist())
        print(f"  class={class_id}: predicted={len(candidates)}, reliable={len(high)}, kept={len(chosen)}, confidence={confidence[chosen].mean():.3f}, agreement={agreement[chosen].mean():.3f}")
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected, predictions[selected], confidence[selected], agreement[selected]


def main():
    parser = argparse.ArgumentParser(description="LOSO target pseudo-label generation without target-label supervision")
    parser.add_argument("--dataset", choices=["seed", "seed4"], default="seed")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_subject", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score_runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min_agreement", type=float, default=0.67)
    parser.add_argument("--min_per_class", type=int, default=100)
    parser.add_argument("--max_per_class", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_de_data, train_and_evaluate, num_classes, names = get_api(args.dataset)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_x, source_y, target_x, _target_y, source_subjects = load_de_data(
        args.data_root, args.seed, "subject", test_subject=args.test_subject,
        return_subjects=True,
    )
    print(f"[Protocol] source labels are used; target subject {args.test_subject} labels are ignored")
    source_fit_idx, source_val_idx = group_holdout(
        source_subjects, val_ratio=0.05, seed=args.seed)
    source_fit_x, source_fit_y = source_x[source_fit_idx], source_y[source_fit_idx]
    source_val_x, source_val_y = source_x[source_val_idx], source_y[source_val_idx]
    print(f"[Source validation] held-out subjects={sorted(np.unique(source_subjects[source_val_idx]).tolist())}")

    models = []
    dummy_target_labels = np.zeros(len(target_x), dtype=np.int64)
    for run in range(args.score_runs):
        run_seed = args.seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        result = train_and_evaluate(
            source_fit_x, source_fit_y, target_x, dummy_target_labels, device,
            model_type="dgcnn", epochs=args.epochs, verbose=False,
            split_seed=run_seed, val_data=source_val_x, val_labels=source_val_y,
        )
        models.append(result["model"])
        print(f"  scorer {run + 1}/{args.score_runs}: source-val={result['best_val_accuracy']:.4f}")

    probabilities, agreement = predict_probabilities(models, target_x, device)
    if probabilities.shape[1] != num_classes:
        raise ValueError("classifier output class count mismatch")
    selected, pseudo_labels, confidence, selected_agreement = select_pseudo_labels(
        probabilities, agreement, args.threshold, args.min_agreement,
        args.min_per_class, args.max_per_class, args.seed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        data=(target_x[selected] * 5.0).astype(np.float32),
        labels=pseudo_labels.astype(np.int64),
        confidence=confidence.astype(np.float32),
        agreement=selected_agreement.astype(np.float32),
        target_indices=selected,
        test_subject=np.int64(args.test_subject),
    )
    print(f"Saved {len(selected)} pseudo-labeled target anchors to {args.output}")
    for class_id in range(num_classes):
        mask = pseudo_labels == class_id
        print(f"  {names[class_id]}: {mask.sum()}, confidence={confidence[mask].mean():.3f}, agreement={selected_agreement[mask].mean():.3f}")


if __name__ == "__main__":
    main()
