"""PGCN adapter for the local SEED/SEED-IV DE evaluation protocol."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


_PGCN_ROOT = Path("/root/PGCN/PGCN")
_MODEL_PATH = _PGCN_ROOT / "model_PGCN.py"
_LOCATION_PATH = _PGCN_ROOT / "node_location.py"


def _load_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upstream_pgcn():
    if not _MODEL_PATH.is_file() or not _LOCATION_PATH.is_file():
        raise ImportError(f"PGCN source files not found under {_PGCN_ROOT}")

    # PGCN uses absolute local imports such as ``from utils import ...``.
    # Isolate those generic module names so they cannot collide with LibEER.
    local_names = ("graphpool", "layer_PGCN", "utils")
    saved = {name: sys.modules.pop(name, None) for name in local_names}
    root = str(_PGCN_ROOT)
    inserted = not sys.path or sys.path[0] != root
    if inserted:
        sys.path.insert(0, root)
    try:
        model_module = _load_file("upstream_pgcn_model", _MODEL_PATH)
        location_module = _load_file(
            "upstream_pgcn_locations", _LOCATION_PATH)
    finally:
        for name in local_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
        if inserted and sys.path and sys.path[0] == root:
            sys.path.pop(0)
    return model_module, location_module


_model_module, _location_module = _load_upstream_pgcn()


def _replace_with_buffer(module, name, value):
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, value.clone())


class PGCNClassifier(_model_module.PGCN):
    """Original PGCN with coordinates registered for device-safe execution."""

    def __init__(
        self,
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        dropout=0.5,
        leaky_relu_slope=0.01,
    ):
        if num_electrodes != 62 or in_channels != 5:
            raise ValueError("PGCN requires 62 electrodes with 5 DE bands")
        args = SimpleNamespace(
            n_class=num_classes,
            dropout=dropout,
            lr=leaky_relu_slope,
            in_feature=in_channels,
            module="",
        )
        adjacency = _location_module.convert_dis_m(
            _location_module.get_ini_dis_m(), 9)
        adjacency = nn.Parameter(
            torch.as_tensor(adjacency, dtype=torch.float32))
        coordinates = torch.as_tensor(
            _location_module.return_coordinates(), dtype=torch.float32)
        super().__init__(args, adjacency, coordinates)
        self.num_classes = num_classes

        # The upstream tensors are plain attributes, so Module.to(device)
        # otherwise leaves them on CPU.
        _replace_with_buffer(self, "coordinate", coordinates)
        _replace_with_buffer(
            self.meso_layer_1, "coordinate", coordinates)
        _replace_with_buffer(
            self.meso_layer_2, "coordinate", coordinates)

    def forward(self, x):
        logits, _, _ = super().forward(x)
        return logits


def _predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            predictions.append(
                model(xb.to(device)).argmax(1).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def train_pgcn(
    train_data,
    train_labels,
    test_data,
    test_labels,
    selection_data,
    selection_labels,
    device,
    epochs=200,
    batch_size=256,
    lr=1e-3,
    dropout=0.5,
    verbose=True,
    val_interval=1,
    patience=30,
    label_smoothing=0.05,
    num_classes=None,
    adjacency_lr=5e-5,
):
    """Train PGCN using its separate adjacency learning rate."""
    if num_classes is None:
        num_classes = int(max(
            np.max(train_labels), np.max(test_labels))) + 1
    if label_smoothing <= 0:
        label_smoothing = 0.05

    x_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_labels).long()
    x_test = torch.from_numpy(test_data).float()
    y_test = torch.from_numpy(test_labels).long()
    x_selection = torch.from_numpy(selection_data).float()
    y_selection = torch.from_numpy(selection_labels).long()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False)
    selection_loader = DataLoader(
        TensorDataset(x_selection, y_selection),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False)

    model = PGCNClassifier(
        num_classes=num_classes,
        dropout=dropout,
        leaky_relu_slope=lr,
    ).to(device)
    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: pgcn, params: {parameter_count:,}")
    print(
        "  PGCN: local/mesoscopic/global pyramid, "
        f"adjacency_lr={adjacency_lr}, label_smoothing={label_smoothing}")

    adjacency_params, local_params, other_params = [], [], []
    for name, parameter in model.named_parameters():
        if name == "adj":
            adjacency_params.append(parameter)
        elif "local" in name:
            local_params.append(parameter)
        else:
            other_params.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": adjacency_params, "lr": adjacency_lr},
        {"params": local_params, "lr": lr},
        {"params": other_params, "lr": lr},
    ], betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max(1, epochs // 3)], gamma=0.1)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing)

    best_accuracy, best_state, best_epoch = -1.0, None, 0
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not verbose, desc="Training")
    for epoch in progress_bar:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % val_interval != 0 and epoch != epochs:
            continue
        predictions, labels = _predict(
            model, selection_loader, device)
        selection_accuracy = accuracy_score(labels, predictions)
        progress_bar.set_postfix(
            selection_acc=f"{selection_accuracy:.3f}")
        if selection_accuracy > best_accuracy:
            best_accuracy, best_epoch = selection_accuracy, epoch
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()}
        if epoch - best_epoch > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    predictions, labels = _predict(model, test_loader, device)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    names = (
        ["negative", "neutral", "positive"]
        if num_classes == 3
        else ["neutral", "sad", "fear", "happy"])
    per_class = {
        names[c]: float((predictions[labels == c] == c).mean())
        if np.any(labels == c) else 0.0
        for c in range(num_classes)}
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "per_class_acc": per_class,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "model": model,
        "train_n": len(train_labels),
        "val_n": len(selection_labels),
        "test_n": len(test_labels),
    }
