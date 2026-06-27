"""DANN/DAN-style DGCNN downstream classifier with source-target alignment."""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from eval_de_classifier import DGCNN, laplacian


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_domains=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_domains),
            nn.Dropout(0.5),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)


class DANDGCNNClassifier(DGCNN):
    """LibEER DannDgcnn adapted for the local SEED DE evaluation loop."""

    def __init__(
        self,
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        k=2,
        dropout_rate=0.5,
        alpha=0.1,
    ):
        super().__init__(
            num_electrodes=num_electrodes,
            in_channels=in_channels,
            num_classes=num_classes,
            k=k,
            dropout_rate=dropout_rate,
        )
        self.alpha = alpha
        self.leaky_relus = nn.ModuleList([nn.LeakyReLU() for _ in range(len(self.layers))])
        self.discriminator = DomainDiscriminator(
            self.num_electrodes * self.layers[-1],
            hidden_dim=256,
            num_domains=2,
        )

    def feature_extract(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.leaky_relus[i](x)
        return x

    def classify(self, features):
        x = features.reshape(features.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        return self.fc2(x)

    def forward_with_domain(self, x, alpha=None):
        features = self.feature_extract(x)
        logits = self.classify(features)
        flat = features.reshape(features.shape[0], -1)
        domain_logits = self.discriminator(ReverseLayerF.apply(flat, self.alpha if alpha is None else alpha))
        return logits, domain_logits

    def forward(self, x):
        return self.classify(self.feature_extract(x))


def _evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues


def train_dan(
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
    label_smoothing=0.0,
    domain_weight=0.1,
):
    """Train DANN on labeled train data and unlabeled target features."""

    x_tr = torch.from_numpy(train_data).float()
    y_tr = torch.from_numpy(train_labels).long()
    x_te = torch.from_numpy(test_data).float()
    y_te = torch.from_numpy(test_labels).long()
    x_sel = torch.from_numpy(selection_data).float()
    y_sel = torch.from_numpy(selection_labels).long()

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(x_te), batch_size=batch_size, shuffle=True)
    selection_loader = DataLoader(TensorDataset(x_sel, y_sel), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    class_weight = 1.0 / (np.bincount(train_labels, minlength=3).astype(np.float32) + 1e-6)
    class_weight = torch.from_numpy(class_weight / class_weight.sum() * 3).to(device)

    model = DANDGCNNClassifier(dropout_rate=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: dan, params: {n_params:,}")
    print(f"  DAN: source-target adversarial alignment, domain_weight={domain_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()

    best_acc, best_state, best_ep = -1.0, None, 0
    pbar = tqdm(range(1, epochs + 1), disable=not verbose, desc="Training")
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    target_iter = itertools.cycle(target_loader)

    for ep in pbar:
        model.train()
        for xb, yb in train_loader:
            (xt,) = next(target_iter)
            xb, yb, xt = xb.to(device), yb.to(device), xt.to(device)

            progress = global_step / total_steps
            alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            src_logits, src_domain = model.forward_with_domain(xb, alpha=alpha)
            _, tgt_domain = model.forward_with_domain(xt, alpha=alpha)
            domain_logits = torch.cat([src_domain, tgt_domain], dim=0)
            domain_labels = torch.cat([
                torch.zeros(src_domain.size(0), dtype=torch.long, device=device),
                torch.ones(tgt_domain.size(0), dtype=torch.long, device=device),
            ])

            loss = cls_criterion(src_logits, yb)
            loss = loss + domain_weight * domain_criterion(domain_logits, domain_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        scheduler.step()

        if ep % val_interval != 0 and ep != epochs:
            continue
        preds, trues = _evaluate(model, selection_loader, device)
        selection_acc = accuracy_score(trues, preds)
        pbar.set_postfix(selection_acc=f"{selection_acc:.3f}")
        if selection_acc > best_acc:
            best_acc, best_ep = selection_acc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    preds, trues = _evaluate(model, test_loader, device)

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
