"""Create high-confidence pseudo labels for an unlabeled LOSO target subject."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset



def get_api(dataset):
    if dataset == "seed4":
        from eval_de_classifier_seed4 import load_de_data, train_and_evaluate, NUM_CLASSES, LABEL_NAMES
        return load_de_data, train_and_evaluate, NUM_CLASSES, LABEL_NAMES
    from eval_de_classifier import load_de_data, train_and_evaluate
    return load_de_data, train_and_evaluate, 3, {0: "negative", 1: "neutral", 2: "positive"}


def fit_temperature(model, data, labels, device, batch_size=1024):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).long()),
        batch_size=batch_size,
    )
    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(model(xb.to(device)))
            targets.append(yb.to(device))
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    before_nll = criterion(logits, targets).item()
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 100.0).item())
    after_nll = criterion(logits / temperature, targets).item()
    return temperature, before_nll, after_nll


def predict_probabilities(models, temperatures, data, device, batch_size=1024):
    loader = DataLoader(TensorDataset(torch.from_numpy(data).float()), batch_size=batch_size)
    all_probabilities = []
    for model, temperature in zip(models, temperatures):
        model.eval()
        parts = []
        with torch.no_grad():
            for (xb,) in loader:
                parts.append(torch.softmax(model(xb.to(device)) / temperature, dim=1).cpu().numpy())
        all_probabilities.append(np.concatenate(parts))
    stacked = np.stack(all_probabilities)
    probabilities = stacked.mean(axis=0)
    predictions = probabilities.argmax(axis=1)
    agreement = (stacked.argmax(axis=2) == predictions[None, :]).mean(axis=0)
    return probabilities, agreement


def print_confidence_distribution(probs, agreement):
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    print("[Target confidence distribution before filtering]")
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999):
        count = int((confidence >= threshold).sum())
        print(f"  confidence >= {threshold:.4f}: {count}/{len(confidence)} ({count / len(confidence):.2%})")
    quantiles = np.quantile(confidence, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1))
    print("  quantiles min/p10/p25/p50/p75/p90/p99/max: "
          + ", ".join(f"{value:.6f}" for value in quantiles))
    for class_id in range(probs.shape[1]):
        mask = predictions == class_id
        class_quantiles = np.quantile(confidence[mask], (0.25, 0.5, 0.75))
        print(f"  class={class_id}: predicted={mask.sum()}, confidence p25/p50/p75="
              + "/".join(f"{value:.6f}" for value in class_quantiles)
              + f", agreement_mean={agreement[mask].mean():.3f}")


def select_pseudo_labels(probs, agreement, threshold, min_agreement, min_per_class, max_per_class, seed, balance_classes=False):
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    rng = np.random.RandomState(seed)
    selected_by_class = []
    for class_id in range(probs.shape[1]):
        candidates = np.where(predictions == class_id)[0]
        if len(candidates) == 0:
            raise ValueError(f"source ensemble predicted no target samples for class {class_id}")
        score = confidence[candidates] * agreement[candidates]
        order = candidates[np.argsort(-score)]
        high = order[(confidence[order] >= threshold) & (agreement[order] >= min_agreement)]
        if len(high) == 0:
            raise ValueError(
                f"class {class_id} has no pseudo-labels passing confidence/agreement thresholds")
        if len(high) < min_per_class:
            print(f"  warning: class={class_id} reliable={len(high)} < requested minimum "
                  f"{min_per_class}; keeping only reliable samples")
        keep_n = min(len(high), max_per_class) if max_per_class > 0 else len(high)
        selected_by_class.append(high[:keep_n])
        print(f"  class={class_id}: predicted={len(candidates)}, reliable={len(high)}, "
              f"before_balance={keep_n}")

    if balance_classes:
        balanced_n = min(len(indices) for indices in selected_by_class)
        selected_by_class = [indices[:balanced_n] for indices in selected_by_class]
        print(f"  balanced reliable pseudo-labels to {balanced_n} per class")

    for class_id, chosen in enumerate(selected_by_class):
        print(f"  class={class_id}: kept={len(chosen)}, confidence={confidence[chosen].mean():.3f}, "
              f"agreement={agreement[chosen].mean():.3f}")
    selected = np.concatenate(selected_by_class).astype(np.int64)
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
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="label smoothing used only while training pseudo-label scorers")
    parser.add_argument("--temperature_calibration", action="store_true",
                        help="train each scorer without one source subject and calibrate its temperature on that subject")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min_agreement", type=float, default=0.67)
    parser.add_argument("--min_per_class", type=int, default=100,
                        help="warn when a class has fewer reliable samples; never backfills")
    parser.add_argument("--max_per_class", type=int, default=1000,
                        help="maximum reliable pseudo-labels retained per predicted class; 0 means unlimited")
    parser.add_argument("--balance_classes", action="store_true",
                        help="truncate reliable pseudo-labels to the smallest class count")
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
    source_fit_x, source_fit_y = source_x, source_y
    print(f"[Source pool] {len(source_y)} labeled samples; target labels remain unused")
    models = []
    temperatures = []
    dummy_target_labels = np.zeros(len(target_x), dtype=np.int64)
    source_subject_ids = np.unique(source_subjects)
    for run in range(args.score_runs):
        run_seed = args.seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        validation_subject = source_subject_ids[run % len(source_subject_ids)]
        validation_mask = source_subjects == validation_subject
        scorer_train_x = source_x[~validation_mask]
        scorer_train_y = source_y[~validation_mask]
        validation_x = source_x[validation_mask]
        validation_y = source_y[validation_mask]
        print(f"  scorer {run + 1}: train={len(scorer_train_y)} samples, "
              f"validation_subject={validation_subject}, validation={len(validation_y)} samples")
        print(f"  starting scorer {run + 1}/{args.score_runs}: epochs={args.epochs}, batch={args.batch_size}", flush=True)
        result = train_and_evaluate(
            scorer_train_x, scorer_train_y, target_x, dummy_target_labels, device,
            model_type="dgcnn", epochs=args.epochs, batch_size=args.batch_size,
            verbose=False,
            split_seed=run_seed, use_validation=True,
            val_data=validation_x, val_labels=validation_y,
            label_smoothing=args.label_smoothing,
        )
        models.append(result["model"])
        if args.temperature_calibration:
            temperature, before_nll, after_nll = fit_temperature(
                result["model"], validation_x, validation_y, device, args.batch_size)
            temperatures.append(temperature)
            print(f"  scorer {run + 1}: temperature={temperature:.4f}, "
                  f"calibration_nll={before_nll:.4f}->{after_nll:.4f}")
        else:
            temperatures.append(1.0)
        print(f"  scorer {run + 1}/{args.score_runs}: best_epoch={result['best_epoch']}, "
              f"best_val_acc={result['best_val_accuracy']:.4f}, batch={args.batch_size}")

    probabilities, agreement = predict_probabilities(
        models, temperatures, target_x, device, args.batch_size)
    if probabilities.shape[1] != num_classes:
        raise ValueError("classifier output class count mismatch")
    print_confidence_distribution(probabilities, agreement)
    selected, pseudo_labels, confidence, selected_agreement = select_pseudo_labels(
        probabilities, agreement, args.threshold, args.min_agreement,
        args.min_per_class, args.max_per_class, args.seed, args.balance_classes)

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
