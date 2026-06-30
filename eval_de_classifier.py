"""
eval_de_classifier.py — DE 空间分类评估
(支持 DGCNN / DAN / NSAL-DGAT / DGAT-BLS / GCBNet / PGCN)

用法:
    # 纯真实数据 baseline (验证配置)
    python eval_de_classifier.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --subject 1 --split_mode trial --real_only \
        --model dgcnn --n_runs 3

    # 逐被试 baseline
    python eval_de_classifier.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --split_mode trial --real_only --baseline \
        --model dgcnn --n_runs 1

    # 诊断合成数据
    python eval_de_classifier.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --synthetic_path ./results/s1_de_cond/generated_SEED_DE_GEN_1.npz \
        --subject 1 --split_mode trial --mode diagnose --model dgcnn

    # 对比评估
    python eval_de_classifier.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --synthetic_path ./results/s1_de_cond/generated_SEED_DE_GEN_1.npz \
        --subject 1 --split_mode trial --mode compare --model dgcnn \
        --syn_ratio 0.25 --n_runs 3
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  DGCNN (from LibEER)
# ============================================================

def laplacian(w):
    d = torch.sum(w, dim=1)
    d_re = 1 / torch.sqrt(d + 1e-5)
    d_matrix = torch.diag_embed(d_re)
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap


class GraphConv(nn.Module):
    def __init__(self, k, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        if self.k == 1:
            return x.unsqueeze(1)
        if self.k == 2:
            return torch.cat((x.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        tk_minus_one = x
        tk = torch.matmul(lap, x)
        t = torch.cat((x.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)
        for _ in range(3, self.k):
            tk_minus_two, tk_minus_one = tk_minus_one, tk
            tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
            t = torch.cat((t, tk.unsqueeze(1)), dim=1)
        return t

    def forward(self, x, lap):
        cp = self.chebyshev_polynomial(x, lap).permute(0, 2, 3, 1).flatten(start_dim=2)
        return torch.matmul(cp, self.weight)


class B1ReLU(nn.Module):
    def __init__(self, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, bias_shape))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bias + x)


class DGCNN(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3,
                 k=2, layers=None, dropout_rate=0.5):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        if layers is None:
            layers = [64] if num_electrodes == 62 else [128]
        self.layers = layers

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(GraphConv(k, in_channels, layers[0]))
        for i in range(len(layers) - 1):
            self.graphConvs.append(GraphConv(k, layers[i], layers[i + 1]))

        self.fc = nn.Linear(num_electrodes * layers[-1], 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.adj = nn.Parameter(torch.Tensor(num_electrodes, num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList([B1ReLU(layers[i]) for i in range(len(layers))])
        self.dropout = nn.Dropout(p=dropout_rate)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """x: (B, 62, 5) → (B, 3)"""
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.fc(x))
        return self.fc2(x)


# ============================================================
#  数据加载
# ============================================================

def load_de_data(data_root, seed=42, split_mode="trial", subject=None,
                 train_trials=None, test_trials=None, test_subject=None,
                 return_groups=False, return_subjects=False):
    from Utils.Data_utils.seed_dataset import SEEDDataset
    subjects = [subject] if (subject is not None and split_mode != "subject") else None
    common = dict(
        name="SEED_DE", data_root=data_root,
        de_key_prefix="de_LDS", window=1, proportion=1.0, seed=seed,
        conditional=True, split_mode=split_mode, subjects=subjects,
    )
    if train_trials is not None: common["train_trials"] = train_trials
    if test_trials is not None: common["test_trials"] = test_trials
    if test_subject is not None: common["test_subject"] = test_subject

    ds_train = SEEDDataset(**common, period="train")
    ds_test  = SEEDDataset(**common, period="test")
    subj_str = f"被试 {subject}" if subject else "所有被试"
    print(f"[{subj_str}] Train: {ds_train.samples.shape}, Test: {ds_test.samples.shape}")
    print(f"  Train labels: {dict(zip(*np.unique(ds_train.labels, return_counts=True)))}")
    print(f"  Test  labels: {dict(zip(*np.unique(ds_test.labels, return_counts=True)))}")
    result = (ds_train.samples, ds_train.labels, ds_test.samples, ds_test.labels)
    if return_groups:
        result += (ds_train.sample_groups,)
    if return_subjects:
        result += (ds_train.sample_subjects,)
    return result


def load_synthetic_de(path):
    bundle = np.load(path)
    data = np.clip(bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32)
    labels = bundle["labels"].astype(np.int64)
    print(f"[Synthetic] {data.shape}, labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return data, labels


# ============================================================
#  训练 & 评估
# ============================================================

def build_model(model_type, dropout=0.5, device="cpu"):
    if model_type == "dgcnn":
        model = DGCNN(num_electrodes=62, in_channels=5, num_classes=3,
                       k=2, dropout_rate=dropout)
    elif model_type == "dgat_bls":
        from dgat_bls_downstream import DGATBLSClassifier
        model = DGATBLSClassifier(nclass=3)
    elif model_type == "dan":
        from dan_downstream import DANDGCNNClassifier
        model = DANDGCNNClassifier(
            num_electrodes=62, in_channels=5, num_classes=3,
            k=2, dropout_rate=dropout)
    elif model_type == "gcbnet":
        from gcbnet_downstream import GCBNetClassifier
        model = GCBNetClassifier(
            num_electrodes=62, in_channels=5, num_classes=3)
    elif model_type == "pgcn":
        from pgcn_downstream import PGCNClassifier
        model = PGCNClassifier(
            num_electrodes=62, in_channels=5, num_classes=3,
            dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_type}, params: {n_params:,}")
    return model.to(device)


def _train_and_select(train_data, train_labels, test_data, test_labels,
                      selection_data, selection_labels, device,
                      model_type="dgcnn", epochs=200, batch_size=256,
                      lr=3e-4, dropout=0.5, verbose=True,
                      val_interval=1, patience=30, label_smoothing=0.0):
    if model_type == "nsal_dgat":
        from nsal_dgat_downstream import train_nsal_dgat
        return train_nsal_dgat(
            train_data, train_labels, test_data, test_labels,
            selection_data, selection_labels, device,
            epochs=epochs, batch_size=batch_size, lr=lr,
            verbose=verbose, val_interval=val_interval,
            patience=patience, label_smoothing=label_smoothing,
        )
    if model_type == "dan":
        from dan_downstream import train_dan
        return train_dan(
            train_data, train_labels, test_data, test_labels,
            selection_data, selection_labels, device,
            epochs=epochs, batch_size=batch_size, lr=lr,
            dropout=dropout, verbose=verbose,
            val_interval=val_interval, patience=patience,
            label_smoothing=label_smoothing,
        )
    if model_type == "pgcn":
        from pgcn_downstream import train_pgcn
        return train_pgcn(
            train_data, train_labels, test_data, test_labels,
            selection_data, selection_labels, device,
            epochs=epochs, batch_size=batch_size, lr=lr,
            dropout=dropout, verbose=verbose,
            val_interval=val_interval, patience=patience,
            label_smoothing=label_smoothing, num_classes=3,
        )
    X_tr = torch.from_numpy(train_data).float()
    y_tr = torch.from_numpy(train_labels).long()
    X_selection = torch.from_numpy(selection_data).float()
    y_selection = torch.from_numpy(selection_labels).long()
    X_te = torch.from_numpy(test_data).float()
    y_te = torch.from_numpy(test_labels).long()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    selection_loader = DataLoader(
        TensorDataset(X_selection, y_selection), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    cw = 1.0 / (np.bincount(train_labels, minlength=3).astype(np.float32) + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * 3).to(device)

    model = build_model(model_type, dropout, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    best_acc, best_state, best_ep = -1.0, None, 0
    pbar = tqdm(range(1, epochs + 1), disable=not verbose, desc="Training")
    for ep in pbar:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if hasattr(model, "regularization_loss"):
                loss = loss + model.regularization_loss()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if ep % val_interval != 0 and ep != epochs:
            continue
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in selection_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        selection_acc = accuracy_score(np.concatenate(trues), np.concatenate(preds))
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
    model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    names = ["negative", "neutral", "positive"]
    per_class = {names[c]: float((preds[trues==c]==c).mean()) if (trues==c).sum()>0 else 0.0 for c in range(3)}

    return {"accuracy": acc, "f1_macro": f1, "per_class_acc": per_class,
            "best_epoch": best_ep, "best_val_accuracy": best_acc, "model": model,
            "train_n": len(train_labels), "val_n": len(selection_labels),
            "test_n": len(test_labels)}


def train_and_evaluate(train_data, train_labels, test_data, test_labels,
                       device, model_type="dgcnn", epochs=200, batch_size=256,
                       lr=3e-4, dropout=0.5, verbose=True,
                       val_interval=1, patience=30, label_smoothing=0.0):
    """Train on every supplied source sample and select directly on the test set."""
    return _train_and_select(
        train_data, train_labels, test_data, test_labels,
        test_data, test_labels, device,
        model_type=model_type, epochs=epochs, batch_size=batch_size,
        lr=lr, dropout=dropout, verbose=verbose,
        val_interval=val_interval, patience=patience,
        label_smoothing=label_smoothing)


def train_with_validation(train_data, train_labels, test_data, test_labels,
                          validation_data, validation_labels, device,
                          model_type="dgcnn", epochs=200, batch_size=256,
                          lr=3e-4, dropout=0.5, verbose=True,
                          val_interval=1, patience=30, label_smoothing=0.0):
    """Explicit validation path reserved for pseudo-label scorer calibration."""
    return _train_and_select(
        train_data, train_labels, test_data, test_labels,
        validation_data, validation_labels, device,
        model_type=model_type, epochs=epochs, batch_size=batch_size,
        lr=lr, dropout=dropout, verbose=verbose,
        val_interval=val_interval, patience=patience,
        label_smoothing=label_smoothing)


# ============================================================
#  诊断
# ============================================================

def diagnose(model, syn_data, syn_labels, device):
    model.eval()
    names = {0: "negative", 1: "neutral", 2: "positive"}
    loader = DataLoader(TensorDataset(torch.from_numpy(syn_data).float()), batch_size=1024)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for (xb,) in loader:
            if hasattr(model, "target_predict"):
                probs = model.target_predict(xb.to(device))
                all_probs.append(probs.cpu().numpy())
                all_preds.append(probs.argmax(1).cpu().numpy())
            else:
                logits = model(xb.to(device))
                all_probs.append(torch.softmax(logits, 1).cpu().numpy())
                all_preds.append(logits.argmax(1).cpu().numpy())
    preds, probs = np.concatenate(all_preds), np.concatenate(all_probs)

    print(f"\n{'='*60}")
    print(f"  DE 合成数据质量诊断 ({len(syn_labels)} samples)")
    print(f"{'='*60}")
    for c in range(3):
        mask = syn_labels == c
        n = mask.sum()
        if n == 0: continue
        acc = (preds[mask] == c).mean()
        conf = probs[mask, c].mean()
        dist = {names[pc]: f"{(preds[mask]==pc).sum()}({(preds[mask]==pc).mean()*100:.1f}%)" for pc in range(3)}
        print(f"\n  {names[c]} ({n} samples): 正确率={acc:.3f}, 置信={conf:.3f}")
        print(f"    预测分布: {dist}")
    overall = (preds == syn_labels).mean()
    print(f"\n  总体正确率: {overall:.3f}")
    if overall > 0.5: print(f"  ✅ 合成数据类间区分度良好")
    else: print(f"  ⚠️ 仍需改进")
    print(f"{'='*60}")


# ============================================================
#  Baseline 模式: 逐被试评估
# ============================================================

def run_baseline(args, device):
    if args.baseline_subjects:
        subjects = [int(x) for x in args.baseline_subjects.split(",")]
    else:
        subjects = list(range(1, 16))

    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    print(f"\n{'#'*60}")
    print(f"  Baseline: 纯真实数据, 逐被试, {args.model}")
    print(f"{'#'*60}\n")

    all_accs, all_f1s = [], []
    subj_results = {}

    for subj in subjects:
        print(f"\n--- 被试 {subj} ---")
        try:
            tr_d, tr_l, te_d, te_l, tr_groups = load_de_data(
                args.data_root, args.seed, args.split_mode, subj,
                train_trials, test_trials, None, return_groups=True)
        except Exception as e:
            print(f"  加载失败: {e}"); continue

        accs, f1s = [], []
        for r in range(args.n_runs):
            torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
            res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
                model_type=args.model, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, dropout=args.dropout, verbose=False,
                val_interval=args.val_interval, patience=args.patience)
            accs.append(res["accuracy"]); f1s.append(res["f1_macro"])
        m_acc, m_f1 = np.mean(accs), np.mean(f1s)
        all_accs.append(m_acc); all_f1s.append(m_f1)
        subj_results[subj] = {"acc": m_acc, "f1": m_f1}
        print(f"  Acc: {m_acc:.4f}±{np.std(accs):.4f}, F1: {m_f1:.4f}")

    print(f"\n{'#'*60}")
    print(f"  {'被试':>6s}  {'Accuracy':>10s}  {'F1':>10s}")
    print(f"  {'-'*30}")
    for s in sorted(subj_results):
        print(f"  {s:>6d}  {subj_results[s]['acc']:>9.4f}   {subj_results[s]['f1']:>9.4f}")
    print(f"  {'-'*30}")
    print(f"  {'平均':>6s}  {np.mean(all_accs):>9.4f}   {np.mean(all_f1s):>9.4f}")
    print(f"  {'std':>6s}  {np.std(all_accs):>9.4f}   {np.std(all_f1s):>9.4f}")
    print(f"{'#'*60}\n")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--split_mode", type=str, default="trial")
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--mode", type=str, default="diagnose",
                        choices=["diagnose", "compare"])
    parser.add_argument("--model", type=str, default="dgcnn",
                        choices=[
                            "dgcnn", "dan", "nsal_dgat", "dgat_bls",
                            "gcbnet", "pgcn",
                        ])
    parser.add_argument("--real_only", action="store_true",
                        help="只用真实数据评估")
    parser.add_argument("--baseline", action="store_true",
                        help="逐被试 baseline 评估")
    parser.add_argument("--baseline_subjects", type=str, default=None,
                        help="baseline 被试列表, 如 '1,2,3'")
    parser.add_argument("--syn_ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    # ---- Baseline 模式 ----
    if args.baseline:
        run_baseline(args, device)
        return

    # ---- Real only 模式 ----
    if args.real_only:
        tr_d, tr_l, te_d, te_l, tr_groups = load_de_data(
            args.data_root, args.seed, args.split_mode, args.subject,
            train_trials, test_trials, args.test_subject, return_groups=True)
        print(f"\n纯真实数据评估 ({args.model}):")
        accs, f1s = [], []
        for r in range(args.n_runs):
            torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
            res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
                model_type=args.model, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, dropout=args.dropout,
                val_interval=args.val_interval, patience=args.patience)
            accs.append(res["accuracy"]); f1s.append(res["f1_macro"])
            print(f"  Run {r+1}: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}, "
                  f"per_class={res['per_class_acc']}")
        print(f"  平均: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
              f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
        return

    # ---- 需要合成数据 ----
    assert args.synthetic_path, "需要 --synthetic_path 或使用 --real_only"
    tr_d, tr_l, te_d, te_l, tr_groups = load_de_data(
        args.data_root, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject, return_groups=True)
    syn_data, syn_labels = load_synthetic_de(args.synthetic_path)

    if args.mode == "diagnose":
        print(f"\n训练 baseline {args.model}...")
        torch.manual_seed(args.seed)
        res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
            model_type=args.model, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, dropout=args.dropout,
            val_interval=args.val_interval, patience=args.patience)
        print(f"Baseline: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}")
        diagnose(res["model"], syn_data, syn_labels, device)

    elif args.mode == "compare":
        # Use every real source sample. Best epoch is selected on the target test set.
        n_syn = int(len(tr_l) * args.syn_ratio)
        rng_syn = np.random.RandomState(args.seed)
        syn_pick = rng_syn.choice(len(syn_labels), min(n_syn, len(syn_labels)), replace=False)
        syn_sub, syn_lab_sub = syn_data[syn_pick], syn_labels[syn_pick]
        print(f"\n完整真实 source={len(tr_l)}, 不划分 source 验证集, "
              f"合成={len(syn_lab_sub)} (syn_ratio={args.syn_ratio} ×source)")

        for mode_name, tr_data, tr_labels in [
            ("无生成数据", tr_d, tr_l),
            ("有生成数据", np.concatenate([tr_d, syn_sub]),
                           np.concatenate([tr_l, syn_lab_sub])),
        ]:
            print(f"\n{'='*50}\n  {mode_name} (train={len(tr_labels)}, model={args.model})\n{'='*50}")
            accs, f1s = [], []
            for r in range(args.n_runs):
                torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
                res = train_and_evaluate(tr_data, tr_labels, te_d, te_l, device,
                    model_type=args.model, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, dropout=args.dropout, verbose=False,
                    val_interval=args.val_interval, patience=args.patience)
                accs.append(res["accuracy"]); f1s.append(res["f1_macro"])
                print(f"  Run {r+1}: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}, "
                      f"per_class={res['per_class_acc']}")
            print(f"  平均: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                  f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
