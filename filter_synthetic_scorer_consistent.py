"""Filter generated DE samples using the pseudo-label scorer ensemble."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval_de_classifier import load_de_data, load_synthetic_de, train_with_validation
from pseudo_label_target import fit_temperature


def predict_ensemble(models, temperatures, data, device, batch_size):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities = []
    predictions = []
    for model, temperature in zip(models, temperatures):
        model.eval()
        parts = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = model(xb.to(device)) / temperature
                parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(parts)
        probabilities.append(probs)
        predictions.append(probs.argmax(axis=1))
    return np.stack(probabilities), np.stack(predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Keep generated samples whose labels agree with all source scorers")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--synthetic_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test_subject", type=int, required=True)
    parser.add_argument("--score_runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_x, source_y, target_x, _target_y, source_subjects = load_de_data(
        args.data_root, args.seed, "subject",
        test_subject=args.test_subject, return_subjects=True)
    synthetic_x, synthetic_y = load_synthetic_de(args.synthetic_path)

    source_subject_ids = np.unique(source_subjects)
    dummy_target_labels = np.zeros(len(target_x), dtype=np.int64)
    models = []
    temperatures = []
    for run in range(args.score_runs):
        run_seed = args.seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        validation_subject = source_subject_ids[run % len(source_subject_ids)]
        validation_mask = source_subjects == validation_subject
        train_x = source_x[~validation_mask]
        train_y = source_y[~validation_mask]
        validation_x = source_x[validation_mask]
        validation_y = source_y[validation_mask]
        print(
            f"scorer {run + 1}/{args.score_runs}: train={len(train_y)}, "
            f"validation_subject={validation_subject}, "
            f"validation={len(validation_y)}",
            flush=True,
        )
        result = train_with_validation(
            train_x, train_y, target_x, dummy_target_labels,
            validation_x, validation_y, device,
            model_type="dgcnn", epochs=args.epochs,
            batch_size=args.batch_size, verbose=False,
        )
        temperature, before_nll, after_nll = fit_temperature(
            result["model"], validation_x, validation_y,
            device, args.batch_size)
        models.append(result["model"])
        temperatures.append(temperature)
        print(
            f"  best_epoch={result['best_epoch']}, "
            f"validation_acc={result['best_val_accuracy']:.4f}, "
            f"temperature={temperature:.4f}, "
            f"nll={before_nll:.4f}->{after_nll:.4f}",
            flush=True,
        )

    probabilities, predictions = predict_ensemble(
        models, temperatures, synthetic_x, device, args.batch_size)
    mean_probabilities = probabilities.mean(axis=0)
    assigned_confidence = mean_probabilities[
        np.arange(len(synthetic_y)), synthetic_y]
    all_agree = (predictions == synthetic_y[None, :]).all(axis=0)
    keep = all_agree & (assigned_confidence >= args.min_confidence)

    print("\nScorer-consistency filtering")
    for class_id, name in enumerate(("negative", "neutral", "positive")):
        class_mask = synthetic_y == class_id
        kept = class_mask & keep
        print(
            f"  {name}: kept={kept.sum()}/{class_mask.sum()} "
            f"({kept.sum() / class_mask.sum():.2%}), "
            f"confidence={assigned_confidence[kept].mean():.4f}"
            if kept.any() else
            f"  {name}: kept=0/{class_mask.sum()}",
        )

    kept_x = synthetic_x[keep]
    kept_y = synthetic_y[keep]
    if len(np.unique(kept_y)) != 3:
        raise ValueError("Filtering removed an entire class")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        data=(kept_x * 5.0).astype(np.float32),
        labels=kept_y.astype(np.int64),
        scorer_confidence=assigned_confidence[keep].astype(np.float32),
    )
    print(f"Saved {len(kept_y)} samples to {args.output}", flush=True)


if __name__ == "__main__":
    main()
