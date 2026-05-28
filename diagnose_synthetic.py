"""
诊断合成数据质量: 用真实数据训练 EEGNet, 然后给合成数据打分.
看每个类别的合成样本有多少能被正确分类、被误分成了哪类.

用法:
    python diagnose_synthetic.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/gen_1s_s1_front9_ft/generated_SEED_RAW_200.npz \
        --subject 1 --window 200 --split_mode trial
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mlp_classifier import EEGNetClassifier, load_data_by_session


def train_eegnet(train_data, train_labels, test_data, test_labels,
                 device, epochs=200, lr=1e-3, batch_size=512, dropout=0.5,
                 depthwise_mode="libeer"):
    """训练 EEGNet 并返回最佳模型 state_dict."""
    n_ch, n_t = train_data.shape[1], train_data.shape[2]
    model = EEGNetClassifier(
        n_channels=n_ch, n_times=n_t, num_classes=3,
        dropout=dropout, depthwise_mode=depthwise_mode,
    ).to(device)

    X_tr = torch.from_numpy(train_data).float()
    y_tr = torch.from_numpy(train_labels).long()
    X_te = torch.from_numpy(test_data).float()
    y_te = torch.from_numpy(test_labels).long()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    cw = 1.0 / (class_counts + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * 3.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_acc, best_state = 0.0, None
    for ep in tqdm(range(epochs), desc="Training EEGNet"):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te.to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te.numpy(), preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBaseline EEGNet test acc: {best_acc:.4f}")
    model.load_state_dict(best_state)
    return model


def diagnose(model, syn_data, syn_labels, device):
    """用训练好的模型给合成数据打分."""
    model.eval()
    names = {0: "negative", 1: "neutral", 2: "positive"}

    all_preds, all_probs = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(syn_data).float()),
                        batch_size=1024, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, 1).cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    print(f"\n{'='*65}")
    print(f"  合成数据质量诊断 ({len(syn_labels)} samples)")
    print(f"{'='*65}")

    for c in range(3):
        mask = syn_labels == c
        n = mask.sum()
        if n == 0:
            continue
        acc = (preds[mask] == c).mean()
        conf = probs[mask, c].mean()

        # 被误分成了哪些类
        pred_counts = {}
        for pc in range(3):
            cnt = (preds[mask] == pc).sum()
            pred_counts[names[pc]] = f"{cnt}({cnt/n*100:.1f}%)"

        print(f"\n  类别: {names[c]} ({n} samples)")
        print(f"    正确率:   {acc:.3f}  (被分类器识别为本类的比例)")
        print(f"    平均置信: {conf:.3f}  (分类器对本类的平均 softmax)")
        print(f"    预测分布: {pred_counts}")

        # 置信度分桶
        conf_vals = probs[mask, c]
        bins = [(0, 0.33, "低"), (0.33, 0.66, "中"), (0.66, 1.0, "高")]
        bucket_str = []
        for lo, hi, label in bins:
            cnt = ((conf_vals >= lo) & (conf_vals < hi)).sum()
            bucket_str.append(f"{label}:{cnt}({cnt/n*100:.0f}%)")
        print(f"    置信分布: {', '.join(bucket_str)}")

    # 总体
    overall_acc = (preds == syn_labels).mean()
    print(f"\n  {'─'*50}")
    print(f"  合成数据总体正确率: {overall_acc:.3f}")

    if overall_acc < 0.5:
        print(f"  ⚠️  合成数据质量较差, 超过一半样本无法被正确分类")
    elif overall_acc < 0.7:
        print(f"  ⚡ 合成数据质量一般, 建议质量过滤后再混合")
    else:
        print(f"  ✅ 合成数据质量较好")

    # 找最大问题
    worst_c = min(range(3), key=lambda c: (preds[syn_labels==c]==c).mean() if (syn_labels==c).sum()>0 else 1)
    mask_w = syn_labels == worst_c
    confused_with = np.bincount(preds[mask_w], minlength=3)
    confused_with[worst_c] = 0
    worst_target = confused_with.argmax()
    print(f"\n  最大问题: '{names[worst_c]}' 的合成样本有 "
          f"{confused_with[worst_target]}/{mask_w.sum()} 个被误判为 '{names[worst_target]}'")
    print(f"  建议: 检查条件生成对 '{names[worst_c]}' 类别的控制能力")
    print(f"{'='*65}")

    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--split_mode", type=str, default="trial")
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载真实数据
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None
    train_data, train_labels, test_data, test_labels = load_data_by_session(
        args.data_root, args.window, 42, args.split_mode, args.subject,
        train_trials, test_trials)

    # 训练 baseline EEGNet
    model = train_eegnet(train_data, train_labels, test_data, test_labels,
                         device, epochs=args.epochs, lr=args.lr)
    model.to(device)

    # 加载合成数据
    bundle = np.load(args.synthetic_path)
    syn_data = np.clip(bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32)
    syn_labels = bundle["labels"].astype(np.int64)
    print(f"\nSynthetic: {syn_data.shape}, labels: {dict(zip(*np.unique(syn_labels, return_counts=True)))}")

    # 诊断
    diagnose(model, syn_data, syn_labels, device)


if __name__ == "__main__":
    main()