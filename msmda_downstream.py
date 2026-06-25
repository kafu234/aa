"""MS-MDA downstream classifier adapted from LibEER.

Each real source subject owns a domain-specific feature extractor and
classifier. Target-style synthetic samples, when supplied, form one additional
labeled source domain instead of being mixed into a real subject.
"""

import copy
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}


def _libeer_electrode_normalize(data):
    """Match LibEER.data_utils.preprocess.ele_normalize."""
    original_shape = data.shape
    flattened = data.reshape(len(data), -1)
    minimum = flattened.min(axis=0, keepdims=True)
    value_range = flattened.max(axis=0, keepdims=True) - minimum
    value_range[value_range == 0] = 1.0
    return ((flattened - minimum) / value_range).reshape(
        original_shape).astype(np.float32)


def load_libeer_seed_session(data_root, test_subject, session=1):
    """Load one SEED session exactly in LibEER MS-MDA's LOSO layout."""
    if session not in (1, 2, 3):
        raise ValueError("SEED session must be 1, 2, or 3")
    files = [
        path for path in glob.glob(os.path.join(data_root, "*.mat"))
        if "label" not in os.path.basename(path).lower()
    ]
    grouped = {}
    for path in files:
        subject = int(os.path.basename(path).split("_", 1)[0])
        grouped.setdefault(subject, []).append(path)
    if sorted(grouped) != list(range(1, 16)):
        raise ValueError("Expected all 15 SEED subjects in data_root")

    from scipy import io as sio

    subject_data = {}
    subject_labels = {}
    for subject in sorted(grouped):
        subject_files = sorted(grouped[subject])
        if len(subject_files) < session:
            raise ValueError(
                f"Subject {subject} does not contain session {session}")
        mat_data = sio.loadmat(subject_files[session - 1])
        samples = []
        labels = []
        for trial_index, raw_label in enumerate(SEED_LABELS, 1):
            key = f"de_LDS{trial_index}"
            if key not in mat_data:
                raise KeyError(f"{key} is missing from {subject_files[session - 1]}")
            trial = mat_data[key].transpose(1, 0, 2).astype(np.float32)
            samples.append(trial)
            labels.extend(
                [SEED_LABEL_MAP[raw_label]] * len(trial))
        data = np.concatenate(samples, axis=0)
        subject_data[subject] = _libeer_electrode_normalize(data)
        subject_labels[subject] = np.asarray(labels, dtype=np.int64)

    source_subjects = [
        subject for subject in sorted(subject_data) if subject != test_subject]
    source_data = np.concatenate(
        [subject_data[subject] for subject in source_subjects], axis=0)
    source_labels = np.concatenate(
        [subject_labels[subject] for subject in source_subjects], axis=0)
    source_ids = np.concatenate([
        np.full(len(subject_labels[subject]), subject, dtype=np.int64)
        for subject in source_subjects
    ])
    target_data = subject_data[test_subject]
    target_labels = subject_labels[test_subject]
    print(
        f"[LibEER strict SEED] session={session}, "
        f"source={source_data.shape} ({len(source_subjects)} subjects), "
        f"target subject={test_subject}: {target_data.shape}")
    return (
        source_data, source_labels, target_data, target_labels, source_ids)


class CommonFeatureExtractor(nn.Module):
    def __init__(self, input_dim=310):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, inputs):
        return self.network(inputs.flatten(start_dim=1).float())


class DomainSpecificFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, inputs):
        return self.network(inputs)


class MSMDAClassifier(nn.Module):
    def __init__(self, num_classes, number_of_sources, input_dim=310):
        super().__init__()
        self.num_classes = num_classes
        self.number_of_sources = number_of_sources
        self.shared_net = CommonFeatureExtractor(input_dim)
        self.domain_extractors = nn.ModuleList(
            DomainSpecificFeatureExtractor()
            for _ in range(number_of_sources)
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(32, num_classes)
            for _ in range(number_of_sources)
        )

    @staticmethod
    def linear_mmd(source_features, target_features):
        delta = source_features - target_features
        return torch.mean(delta @ delta.T)

    def training_losses(self, source, source_labels, target, source_index):
        source_common = self.shared_net(source)
        target_common = self.shared_net(target)
        target_specific = [
            extractor(target_common) for extractor in self.domain_extractors
        ]
        source_specific = self.domain_extractors[source_index](source_common)
        classification_loss = F.cross_entropy(
            self.classifiers[source_index](source_specific), source_labels)
        mmd_loss = self.linear_mmd(
            source_specific, target_specific[source_index])

        # Preserve LibEER's branch discrepancy term.
        discrepancy_loss = source_specific.new_zeros(())
        reference = F.softmax(target_specific[source_index], dim=1)
        for branch_index, branch_features in enumerate(target_specific):
            if branch_index != source_index:
                discrepancy_loss = discrepancy_loss + torch.mean(torch.abs(
                    reference - F.softmax(branch_features, dim=1)))
        return classification_loss, mmd_loss, discrepancy_loss

    def forward(self, inputs):
        common_features = self.shared_net(inputs)
        return [
            classifier(extractor(common_features))
            for extractor, classifier in zip(
                self.domain_extractors, self.classifiers)
        ]


def _cycle(loader):
    while True:
        yield from loader


def _predict(model, data, device, batch_size):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            branch_logits = model(inputs.to(device))
            probabilities = torch.stack([
                F.softmax(logits, dim=1) for logits in branch_logits
            ]).mean(0)
            predictions.append(probabilities.argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def _build_source_domains(
        source_data, source_labels, source_subjects,
        synthetic_data, synthetic_labels):
    domains = []
    for subject in sorted(np.unique(source_subjects).tolist()):
        mask = source_subjects == subject
        domains.append((
            f"subject_{subject}",
            source_data[mask],
            source_labels[mask],
        ))
    if synthetic_data is not None and len(synthetic_data) > 0:
        domains.append((
            "target_style_synthetic",
            synthetic_data,
            synthetic_labels,
        ))
    return domains


def train_msmda_and_evaluate(
        source_data,
        source_labels,
        source_subjects,
        target_data,
        target_labels,
        device,
        synthetic_data=None,
        synthetic_labels=None,
        num_classes=3,
        class_names=None,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        verbose=True,
        val_interval=1,
        patience=30,
        selection_data=None,
        selection_labels=None,
        libeer_strict=False,
):
    """Train MS-MDA with one branch per source subject.

    Target labels are never used by the optimization losses. They are only used
    by the repository's existing target-test best-epoch selection protocol.
    """
    if selection_data is None:
        selection_data, selection_labels = target_data, target_labels
    if class_names is None:
        class_names = [str(index) for index in range(num_classes)]
    if (
        libeer_strict
        and synthetic_data is not None
        and len(synthetic_data) > 0
    ):
        # LibEER normalizes every source domain independently. Keep the
        # generator untouched and apply the same downstream-only transform to
        # the synthetic source branch.
        synthetic_data = _libeer_electrode_normalize(synthetic_data)
        print(
            "[LibEER strict MS-MDA] independently ele-normalized synthetic "
            "source domain to [0, 1]")

    domains = _build_source_domains(
        source_data, source_labels, source_subjects,
        synthetic_data, synthetic_labels)
    if not domains:
        raise ValueError("MS-MDA requires at least one source domain")

    effective_batch_size = min(
        batch_size,
        len(target_labels),
        min(len(labels) for _, _, labels in domains),
    )
    if effective_batch_size < 2:
        raise ValueError("MS-MDA requires at least two samples per domain")

    source_loaders = [
        DataLoader(
            TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(labels).long()),
            batch_size=effective_batch_size, shuffle=True, drop_last=True)
        for _, data, labels in domains
    ]
    target_loader = DataLoader(
        TensorDataset(torch.from_numpy(target_data).float()),
        batch_size=effective_batch_size, shuffle=True, drop_last=True)
    source_iterators = [_cycle(loader) for loader in source_loaders]
    target_iterator = _cycle(target_loader)

    model = MSMDAClassifier(
        num_classes=num_classes,
        number_of_sources=len(domains),
        input_dim=int(np.prod(source_data.shape[1:])),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if libeer_strict:
        # LibEER uses ceil(samples_per_source / batch_size), even though its
        # source loaders drop the final incomplete batch and are then cycled.
        steps_per_epoch = math.ceil(
            len(domains[0][2]) / effective_batch_size)
    else:
        steps_per_epoch = max(len(loader) for loader in source_loaders)
    total_steps = max(epochs * steps_per_epoch, 1)
    best_acc, best_epoch, best_state = -1.0, 0, None
    history = {}
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not verbose, desc="MS-MDA training")
    for epoch in progress_bar:
        model.train()
        epoch_losses = []
        for step_index in range(steps_per_epoch):
            schedule_step = (epoch - 1) * steps_per_epoch + step_index
            gamma = (
                2.0 / (
                    1.0 + math.exp(
                        -10.0 * schedule_step / total_steps)
                ) - 1.0
            )
            beta = gamma / 100.0
            for source_index, source_iterator in enumerate(source_iterators):
                source_inputs, source_targets = next(source_iterator)
                target_inputs = next(target_iterator)[0]
                source_inputs = source_inputs.to(device)
                source_targets = source_targets.to(device)
                target_inputs = target_inputs.to(device)

                classification_loss, mmd_loss, discrepancy_loss = (
                    model.training_losses(
                        source_inputs, source_targets,
                        target_inputs, source_index))
                loss = (
                    classification_loss
                    + gamma * mmd_loss
                    + beta * discrepancy_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if libeer_strict:
                selection_predictions = _predict(
                    model, selection_data, device, batch_size)
                selection_acc = accuracy_score(
                    selection_labels, selection_predictions)
                if selection_acc > best_acc:
                    best_acc, best_epoch = selection_acc, epoch
                    best_state = copy.deepcopy(model.state_dict())

        if libeer_strict:
            history = {
                "loss": float(np.mean(epoch_losses)),
                "selection_acc": best_acc,
                "source_domains": [name for name, _, _ in domains],
            }
            progress_bar.set_postfix(
                loss=f"{history['loss']:.3f}",
                selection_acc=f"{best_acc:.3f}")
            continue
        if epoch % val_interval != 0 and epoch != epochs:
            continue
        selection_predictions = _predict(
            model, selection_data, device, batch_size)
        selection_acc = accuracy_score(
            selection_labels, selection_predictions)
        history = {
            "loss": float(np.mean(epoch_losses)),
            "selection_acc": selection_acc,
            "source_domains": [name for name, _, _ in domains],
        }
        progress_bar.set_postfix(
            loss=f"{history['loss']:.3f}",
            selection_acc=f"{selection_acc:.3f}")
        if selection_acc > best_acc:
            best_acc, best_epoch = selection_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if patience is not None and epoch - best_epoch > patience:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs
    model.load_state_dict(best_state)
    predictions = _predict(model, target_data, device, batch_size)
    accuracy = accuracy_score(target_labels, predictions)
    f1 = f1_score(target_labels, predictions, average="macro")
    per_class = {
        class_names[class_id]: float(
            (predictions[target_labels == class_id] == class_id).mean())
        if (target_labels == class_id).any() else 0.0
        for class_id in range(num_classes)
    }
    train_count = len(source_labels)
    if synthetic_labels is not None:
        train_count += len(synthetic_labels)
    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "per_class_acc": per_class,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "model": model,
        "train_n": train_count,
        "val_n": len(selection_labels),
        "test_n": len(target_labels),
        "last_train_state": history,
    }
