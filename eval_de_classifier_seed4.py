"""
eval_de_classifier_seed4.py — SEED-IV DE 空间分类评估 (支持 DGCNN / DE Transformer)
====================================================================================
SEED-IV: 4 类情绪 (neutral, sad, fear, happy), 24 trials/session, 15 被试, 3 sessions
数据目录: eeg_feature_smooth/1/, /2/, /3/

用法:
    # 纯真实数据 baseline (验证配置)
    python eval_de_classifier_seed4.py \
        --data_root /root/autodl-tmp/eeg_feature_smooth \
        --subject 1 --split_mode trial --real_only \
        --model dgcnn --n_runs 3

    # 逐被试 baseline
    python eval_de_classifier_seed4.py \
        --data_root /root/autodl-tmp/eeg_feature_smooth \
        --split_mode trial --real_only --baseline \
        --model dgcnn --n_runs 1

    # 诊断合成数据
    python eval_de_classifier_seed4.py \
        --data_root /root/autodl-tmp/eeg_feature_smooth \
        --synthetic_path ./results/s1_de_cond/generated_SEEDIV_DE_GEN_5.npz \
        --subject 1 --split_mode trial --mode diagnose --model dgcnn

    # 对比评估
    python eval_de_classifier_seed4.py \
        --data_root /root/autodl-tmp/eeg_feature_smooth \
        --synthetic_path ./results/s1_de_cond/generated_SEEDIV_DE_GEN_5.npz \
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

# SEED-IV: 4 类
NUM_CLASSES = 4
LABEL_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


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
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=NUM_CLASSES,
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
        """x: (B, 62, 5) → (B, NUM_CLASSES)"""
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
#  DE Transformer Classifier
# ============================================================

def _get_seed_62_coords():
    coords = torch.tensor([
        [-0.0294, 0.0839,-0.0070],[ 0.0001, 0.0882,-0.0017],[ 0.0299, 0.0849,-0.0071],
        [-0.0337, 0.0768, 0.0212],[ 0.0357, 0.0777, 0.0220],[-0.0703, 0.0425,-0.0114],
        [-0.0645, 0.0480, 0.0169],[-0.0502, 0.0531, 0.0422],[-0.0275, 0.0569, 0.0603],
        [ 0.0003, 0.0585, 0.0665],[ 0.0295, 0.0576, 0.0595],[ 0.0518, 0.0543, 0.0408],
        [ 0.0679, 0.0498, 0.0164],[ 0.0730, 0.0444,-0.0120],[-0.0808, 0.0141,-0.0111],
        [-0.0772, 0.0186, 0.0245],[-0.0602, 0.0227, 0.0555],[-0.0341, 0.0260, 0.0800],
        [ 0.0004, 0.0274, 0.0887],[ 0.0348, 0.0264, 0.0788],[ 0.0623, 0.0237, 0.0556],
        [ 0.0795, 0.0199, 0.0244],[ 0.0818, 0.0154,-0.0113],[-0.0842,-0.0160,-0.0093],
        [-0.0803,-0.0138, 0.0292],[-0.0654,-0.0116, 0.0644],[-0.0362,-0.0100, 0.0898],
        [ 0.0004,-0.0092, 0.1002],[ 0.0377,-0.0096, 0.0884],[ 0.0671,-0.0109, 0.0636],
        [ 0.0835,-0.0128, 0.0292],[ 0.0851,-0.0150,-0.0095],[-0.0848,-0.0460,-0.0071],
        [-0.0796,-0.0466, 0.0309],[-0.0636,-0.0470, 0.0656],[-0.0355,-0.0473, 0.0913],
        [ 0.0004,-0.0473, 0.0994],[ 0.0384,-0.0471, 0.0907],[ 0.0666,-0.0466, 0.0656],
        [ 0.0833,-0.0461, 0.0312],[ 0.0855,-0.0455,-0.0071],[-0.0724,-0.0735,-0.0025],
        [-0.0673,-0.0763, 0.0284],[-0.0530,-0.0788, 0.0559],[-0.0286,-0.0805, 0.0754],
        [ 0.0003,-0.0811, 0.0826],[ 0.0319,-0.0805, 0.0767],[ 0.0557,-0.0786, 0.0566],
        [ 0.0679,-0.0759, 0.0281],[ 0.0731,-0.0731,-0.0025],[-0.0548,-0.0975, 0.0028],
        [-0.0484,-0.0993, 0.0216],[-0.0365,-0.1009, 0.0372],[ 0.0002,-0.1022, 0.0506],
        [ 0.0368,-0.1008, 0.0364],[ 0.0498,-0.0994, 0.0217],[ 0.0557,-0.0976, 0.0027],
        [-0.0421,-0.1204, 0.0008],[-0.0294,-0.1124, 0.0088],[ 0.0001,-0.1149, 0.0147],
        [ 0.0298,-0.1122, 0.0088],[ 0.0428,-0.1202, 0.0008],
    ], dtype=torch.float32)
    return coords


class DETransformer(nn.Module):
    def __init__(self, n_channels=62, n_bands=5, d_model=128,
                 n_heads=4, n_layers=3, dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.band_embed = nn.Sequential(
            nn.Linear(n_bands, d_model), nn.GELU(), nn.LayerNorm(d_model))
        coords = _get_seed_62_coords()
        self.register_buffer('coords', coords)
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        h = self.band_embed(x)
        h = h + self.spatial_proj(self.coords).unsqueeze(0)
        h = self.transformer(h)
        return self.head(h.mean(dim=1))


# ============================================================
#  数据加载
# ============================================================

def load_de_data(data_root, seed=42, split_mode="trial", subject=None,
                 train_trials=None, test_trials=None, test_subject=None):
    from Utils.Data_utils.seed4_dataset import SEEDIVDataset
    subjects = [subject] if (subject is not None and split_mode != "subject") else None
    common = dict(
        name="SEEDIV_DE", data_root=data_root, data_type="de",
        de_key_prefix="de_LDS", proportion=1.0, seed=seed,
        conditional=True, split_mode=split_mode, subjects=subjects,
    )
    if train_trials is not None: common["train_trials"] = train_trials
    if test_trials is not None: common["test_trials"] = test_trials
    if test_subject is not None: common["test_subject"] = test_subject

    ds_train = SEEDIVDataset(**common, period="train")
    ds_test  = SEEDIVDataset(**common, period="test")
    subj_str = f"被试 {subject}" if subject else "所有被试"
    print(f"[{subj_str}] Train: {ds_train.samples.shape}, Test: {ds_test.samples.shape}")
    print(f"  Train labels: {dict(zip(*np.unique(ds_train.labels, return_counts=True)))}")
    print(f"  Test  labels: {dict(zip(*np.unique(ds_test.labels, return_counts=True)))}")
    return ds_train.samples, ds_train.labels, ds_test.samples, ds_test.labels


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
        model = DGCNN(num_electrodes=62, in_channels=5, num_classes=NUM_CLASSES,
                       k=2, dropout_rate=dropout)
    elif model_type == "de_transformer":
        model = DETransformer(dropout=dropout, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_type}, params: {n_params:,}")
    return model.to(device)


def train_and_evaluate(train_data, train_labels, test_data, test_labels,
                       device, model_type="dgcnn", epochs=200, batch_size=256,
                       lr=3e-4, dropout=0.5, verbose=True):
    X_tr = torch.from_numpy(train_data).float()
    y_tr = torch.from_numpy(train_labels).long()
    X_te = torch.from_numpy(test_data).float()
    y_te = torch.from_numpy(test_labels).long()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    # [FIX] minlength=NUM_CLASSES (4), 归一化乘以 NUM_CLASSES
    cw = 1.0 / (np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32) + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * NUM_CLASSES).to(device)

    model = build_model(model_type, dropout, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_acc, best_state, best_ep = 0, None, 0
    pbar = tqdm(range(1, epochs + 1), disable=not verbose, desc="Training")
    for ep in pbar:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        acc = accuracy_score(np.concatenate(trues), np.concatenate(preds))
        pbar.set_postfix(acc=f"{acc:.3f}")
        if acc > best_acc:
            best_acc, best_ep = acc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep > 30:
            break

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
    # [FIX] range(NUM_CLASSES) 遍历全部 4 类
    per_class = {
        LABEL_NAMES[c]: float((preds[trues == c] == c).mean()) if (trues == c).sum() > 0 else 0.0
        for c in range(NUM_CLASSES)
    }

    return {"accuracy": acc, "f1_macro": f1, "per_class_acc": per_class,
            "best_epoch": best_ep, "model": model,
            "train_n": len(train_labels), "test_n": len(test_labels)}


# ============================================================
#  诊断
# ============================================================

def diagnose(model, syn_data, syn_labels, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(syn_data).float()), batch_size=1024)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, 1).cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
    preds, probs = np.concatenate(all_preds), np.concatenate(all_probs)

    print(f"\n{'='*60}")
    print(f"  DE 合成数据质量诊断 ({len(syn_labels)} samples)")
    print(f"{'='*60}")
    # [FIX] range(NUM_CLASSES) 遍历全部 4 类
    for c in range(NUM_CLASSES):
        mask = syn_labels == c
        n = mask.sum()
        if n == 0: continue
        acc = (preds[mask] == c).mean()
        conf = probs[mask, c].mean()
        # [FIX] range(NUM_CLASSES) 统计全部 4 类的预测分布
        dist = {
            LABEL_NAMES[pc]: f"{(preds[mask]==pc).sum()}({(preds[mask]==pc).mean()*100:.1f}%)"
            for pc in range(NUM_CLASSES)
        }
        print(f"\n  {LABEL_NAMES[c]} ({n} samples): 正确率={acc:.3f}, 置信={conf:.3f}")
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
    print(f"  SEED-IV Baseline: 纯真实数据, 逐被试, {args.model}")
    print(f"  {NUM_CLASSES} 类: {', '.join(LABEL_NAMES[c] for c in range(NUM_CLASSES))}")
    print(f"{'#'*60}\n")

    all_accs, all_f1s = [], []
    subj_results = {}

    for subj in subjects:
        print(f"\n--- 被试 {subj} ---")
        try:
            tr_d, tr_l, te_d, te_l = load_de_data(
                args.data_root, args.seed, args.split_mode, subj,
                train_trials, test_trials, None)
        except Exception as e:
            print(f"  加载失败: {e}"); continue

        accs, f1s = [], []
        for r in range(args.n_runs):
            torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
            res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
                model_type=args.model, epochs=args.epochs, lr=args.lr,
                dropout=args.dropout, verbose=False)
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
    parser = argparse.ArgumentParser(
        description="SEED-IV DE 空间分类评估 (4类: neutral/sad/fear/happy)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="SEED-IV DE 特征目录, 如 /root/autodl-tmp/eeg_feature_smooth")
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--split_mode", type=str, default="trial")
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--mode", type=str, default="diagnose",
                        choices=["diagnose", "compare"])
    parser.add_argument("--model", type=str, default="dgcnn",
                        choices=["dgcnn", "de_transformer"])
    parser.add_argument("--real_only", action="store_true",
                        help="只用真实数据评估")
    parser.add_argument("--baseline", action="store_true",
                        help="逐被试 baseline 评估")
    parser.add_argument("--baseline_subjects", type=str, default=None,
                        help="baseline 被试列表, 如 '1,2,3'")
    parser.add_argument("--syn_ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=200)
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
        tr_d, tr_l, te_d, te_l = load_de_data(
            args.data_root, args.seed, args.split_mode, args.subject,
            train_trials, test_trials, args.test_subject)
        print(f"\n纯真实数据评估 ({args.model}, {NUM_CLASSES}类):")
        accs, f1s = [], []
        for r in range(args.n_runs):
            torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
            res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
                model_type=args.model, epochs=args.epochs, lr=args.lr,
                dropout=args.dropout)
            accs.append(res["accuracy"]); f1s.append(res["f1_macro"])
            print(f"  Run {r+1}: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}, "
                  f"per_class={res['per_class_acc']}")
        print(f"  平均: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
              f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
        return

    # ---- 需要合成数据 ----
    assert args.synthetic_path, "需要 --synthetic_path 或使用 --real_only"
    tr_d, tr_l, te_d, te_l = load_de_data(
        args.data_root, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject)
    syn_data, syn_labels = load_synthetic_de(args.synthetic_path)

    if args.mode == "diagnose":
        print(f"\n训练 baseline {args.model}...")
        torch.manual_seed(args.seed)
        res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
            model_type=args.model, epochs=args.epochs, lr=args.lr, dropout=args.dropout)
        print(f"Baseline: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}")
        diagnose(res["model"], syn_data, syn_labels, device)

    elif args.mode == "compare":
        n_target = int(len(syn_labels) * args.syn_ratio)
        idx = np.random.choice(len(syn_labels), min(n_target, len(syn_labels)), replace=False)
        syn_sub, syn_lab_sub = syn_data[idx], syn_labels[idx]
        print(f"\n使用 {len(syn_lab_sub)} 合成样本 (syn_ratio={args.syn_ratio})")

        for mode_name, tr_data, tr_labels in [
            ("无生成数据", tr_d, tr_l),
            ("有生成数据", np.concatenate([tr_d, syn_sub]), np.concatenate([tr_l, syn_lab_sub])),
        ]:
            print(f"\n{'='*50}\n  {mode_name} (train={len(tr_labels)}, model={args.model})\n{'='*50}")
            accs, f1s = [], []
            for r in range(args.n_runs):
                torch.manual_seed(args.seed + r); np.random.seed(args.seed + r)
                res = train_and_evaluate(tr_data, tr_labels, te_d, te_l, device,
                    model_type=args.model, epochs=args.epochs, lr=args.lr,
                    dropout=args.dropout, verbose=False)
                accs.append(res["accuracy"]); f1s.append(res["f1_macro"])
                print(f"  Run {r+1}: acc={res['accuracy']:.4f}, f1={res['f1_macro']:.4f}, "
                      f"per_class={res['per_class_acc']}")
            print(f"  平均: acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                  f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()