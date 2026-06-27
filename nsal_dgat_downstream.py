"""NSAL-DGAT downstream adapter with its domain alignment training intact."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


_LIBEER_NSAL = Path("/root/LibEER/LibEER/models/NSAL_DGAT.py")


def _load_libeer_nsal():
    spec = importlib.util.spec_from_file_location("libeer_nsal_dgat", _LIBEER_NSAL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load NSAL-DGAT from {_LIBEER_NSAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nsal = _load_libeer_nsal()
DomainAdaptionModel = _nsal.Domain_adaption_model
Discriminator = _nsal.Discriminator
DAANLoss = _nsal.DAANLoss


class StepwiseLR:
    def __init__(self, optimizer, init_lr=1e-3, gamma=10.0, decay_rate=0.75, max_iter=200):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.max_iter = max(1, max_iter)
        self.iter_num = 0

    def step(self):
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** self.decay_rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_mult", 1.0)
        self.iter_num += 1


class NSALDGATClassifier(nn.Module):
    """Adapt LibEER NSAL-DGAT to this project's dense DE tensors.

    The LibEER encoder reshapes each sample to ``(5, 62)``. This wrapper
    explicitly transposes project tensors from ``(62, 5)`` to ``(5, 62)`` so
    channel and frequency-band axes are not scrambled.
    """

    def __init__(self, source_num, num_classes=3, channels=62, feature_dim=5,
                 layers=2, hidden_1=256, hidden_2=64, device="cpu"):
        super().__init__()
        self.hidden_2 = hidden_2
        self.num_classes = num_classes
        self.model = DomainAdaptionModel(
            channels=channels,
            feature_dim=feature_dim,
            num_of_class=num_classes,
            layers=layers,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            device=device,
            source_num=source_num,
        )

    @staticmethod
    def _format(x):
        if x.dim() == 3 and x.size(1) == 62 and x.size(2) == 5:
            return x.permute(0, 2, 1).contiguous()
        return x

    def forward(self, source, target, source_label, source_index):
        return self.model(
            self._format(source),
            self._format(target),
            source_label,
            source_index,
        )

    def target_predict(self, x):
        return self.model.target_predict(self._format(x))

    def get_init_banks(self, source, source_index):
        return self.model.get_init_banks(self._format(source), source_index)


def _predict(model, loader, device):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            probs = model.target_predict(xb.to(device))
            preds.append(probs.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def _init_source_banks(model, loader, device):
    model.eval()
    with torch.no_grad():
        for xb, source_index, _ in loader:
            model.get_init_banks(xb.to(device), source_index)


def train_nsal_dgat(train_data, train_labels, test_data, test_labels,
                    selection_data, selection_labels, device,
                    epochs=200, batch_size=256, lr=1e-3, verbose=True,
                    val_interval=1, patience=30, label_smoothing=0.0):
    X_tr = torch.from_numpy(train_data).float()
    y_tr = torch.from_numpy(train_labels).long()
    X_tgt = torch.from_numpy(test_data).float()
    X_selection = torch.from_numpy(selection_data).float()
    y_selection = torch.from_numpy(selection_labels).long()
    X_te = torch.from_numpy(test_data).float()
    y_te = torch.from_numpy(test_labels).long()

    effective_batch = min(batch_size, len(X_tr), len(X_tgt))
    if effective_batch < 1:
        raise ValueError("NSAL-DGAT needs non-empty source and target data")
    if effective_batch != batch_size:
        print(f"  [NSAL-DGAT] batch_size clipped from {batch_size} to {effective_batch}")
    batch_size = effective_batch

    source_index = torch.arange(len(X_tr)).long()
    source_loader = DataLoader(
        TensorDataset(X_tr, source_index, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    source_bank_loader = DataLoader(
        TensorDataset(X_tr, source_index, y_tr),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    target_loader = DataLoader(
        TensorDataset(X_tgt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    selection_loader = DataLoader(
        TensorDataset(X_selection, y_selection),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    if len(source_loader) == 0 or len(target_loader) == 0:
        raise ValueError("NSAL-DGAT source/target loader has zero batches")

    device_name = str(device)
    model = NSALDGATClassifier(source_num=len(X_tr), device=device_name).to(device)
    domain_discriminator = Discriminator(model.hidden_2).to(device)
    daan_loss = DAANLoss(domain_discriminator, num_class=model.num_classes).to(device)
    params = list(model.parameters()) + list(domain_discriminator.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"  Model: nsal_dgat, params: {n_params:,}")
    print("  [NSAL-DGAT] domain_alignment=DAAN global adversarial loss + target pseudo-label CE")
    print("  [NSAL-DGAT] project protocol: class-weighted CE, grad clipping, patience early stop")
    print("  [NSAL-DGAT] model selection: best epoch is selected on the target test set")

    cw = 1.0 / (np.bincount(train_labels, minlength=3).astype(np.float32) + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * 3).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.001)
    scheduler = StepwiseLR(optimizer, init_lr=lr, gamma=10.0, decay_rate=0.75, max_iter=epochs)

    _init_source_banks(model, source_bank_loader, device)
    best_acc, best_state, best_ep = -1.0, None, 0
    pbar = tqdm(range(1, epochs + 1), disable=not verbose, desc="Training")
    for ep in pbar:
        model.train()
        daan_loss.train()
        target_iter = iter(target_loader)
        for xb, src_idx, yb in source_loader:
            try:
                (target_xb,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (target_xb,) = next(target_iter)

            xb = xb.to(device)
            yb = yb.to(device)
            target_xb = target_xb.to(device)

            (src_logits, src_feature, tar_logits, tar_feature,
             _source_att, _target_att, tar_label_probs) = model(xb, target_xb, yb, src_idx)
            cls_loss = criterion(src_logits, yb)
            tar_label = torch.argmax(tar_label_probs, dim=1).to(device)
            target_loss = criterion(tar_logits, tar_label)
            noise_s = 0.005 * torch.randn_like(src_feature)
            noise_t = 0.005 * torch.randn_like(tar_feature)
            transfer_loss = daan_loss(
                src_feature + noise_s,
                tar_feature + noise_t,
                src_logits,
                tar_logits,
            )
            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1.0 * ep / 1000.0)) - 1.0)
            loss = cls_loss + transfer_loss + boost_factor * target_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()

        if ep % val_interval != 0 and ep != epochs:
            continue
        preds, trues = _predict(model, selection_loader, device)
        selection_acc = accuracy_score(trues, preds)
        pbar.set_postfix(selection_acc=f"{selection_acc:.3f}")
        if selection_acc > best_acc:
            best_acc, best_ep = selection_acc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep > patience:
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_ep = epochs
    model.load_state_dict(best_state)
    model.to(device)
    preds, trues = _predict(model, test_loader, device)

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    names = ["negative", "neutral", "positive"]
    per_class = {
        names[c]: float((preds[trues == c] == c).mean()) if (trues == c).sum() > 0 else 0.0
        for c in range(3)
    }
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "per_class_acc": per_class,
        "best_epoch": best_ep,
        "best_val_accuracy": best_acc,
        "model": model,
        "train_n": len(train_labels),
        "val_n": len(selection_labels),
        "test_n": len(test_labels),
    }
