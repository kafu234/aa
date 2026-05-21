"""
eval_mlp_classifier.py — 纯 MLP 分类器评估生成数据质量
=========================================================

核心思想:
  用 "生成数据 + 部分原始训练数据" 训练 MLP，然后在 "剩余原始测试数据" 上评估。
  通过对比 "有生成数据" 与 "无生成数据" 的分类精度，检验生成数据的质量。

  - 如果生成数据质量高 → 加入生成数据后精度持平或提高
  - 如果生成数据质量差 → 加入生成数据后精度下降

用法:
    # 只用原始数据 (baseline)
    python eval_mlp_classifier.py \
        --data_root /path/to/Preprocessed_EEG \
        --no_synthetic

    # 用原始数据 + 生成数据
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/result/generated_SEED_RAW_200.npz

    # 自动对比两种模式 (推荐)
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/result/generated_SEED_RAW_200.npz \
        --compare

    # 调整原始训练数据比例 (默认 0.6, 即 60% 训练 40% 测试)
    python eval_mlp_classifier.py \
        --data_root /path/to/Preprocessed_EEG \
        --synthetic_path ./results/SEED_RAW/generated_SEED_RAW_200.npz \
        --compare --train_ratio 0.5

    python train_seed.py --config ./Config/seed_raw.yaml --gpu 0 --conditional --num_samples 30000    --results_dir /root/autodl-tmp/result
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  数据加载: 复用 SEEDDataset 的逻辑，但只提取 numpy 数组
# ============================================================
def load_original_data(data_root, window=200, seed=42):
    """
    加载全部原始 SEED 数据 (不做 train/test 划分，统一返回).

    Returns:
        data:   np.ndarray, shape (N, 62, window), 归一化到 [-1, 1]
        labels: np.ndarray, shape (N,), values in {0, 1, 2}
    """
    from Utils.Data_utils.seed_dataset import SEEDDataset

    # 用 proportion=1.0 把所有数据都放到 "train" 里
    ds = SEEDDataset(
        name="SEED_RAW",
        data_root=data_root,
        data_type="raw",
        window=window,
        proportion=1.0,  # 全部数据
        seed=seed,
        period="train",
        conditional=True,
        sfreq=200,
        bandpass_low=0.5,
        bandpass_high=50.0,
        notch_freq=50.0,
        notch_width=2.0,
        baseline_correction=True,
    )
    return ds.samples, ds.labels  # (N, 62, 200), (N,)


def load_synthetic_data(synthetic_path):
    """
    加载生成数据 (.npz 文件).

    Returns:
        data:   np.ndarray, shape (N, 62, window)
        labels: np.ndarray, shape (N,)
    """
    bundle = np.load(synthetic_path)
    data = bundle["data"]      # (N, 62, 200)
    labels = bundle["labels"]  # (N,)

    # 生成数据已经恢复到 z-score 尺度 (×5)，需要重新映射到 [-1, 1]
    # train_seed.py 中: samples = samples * CLIP_STD (5.0)
    # 所以这里除以 5 再 clip 回 [-1, 1]
    CLIP_STD = 5.0
    data = np.clip(data / CLIP_STD, -1.0, 1.0)

    print(f"[Synthetic] Loaded {data.shape[0]} samples from {synthetic_path}")
    print(f"[Synthetic] Shape: {data.shape}, Labels: {np.unique(labels)}")
    return data.astype(np.float32), labels.astype(np.int64)


# ============================================================
#  数据划分
# ============================================================
def split_data(data, labels, train_ratio=0.6, seed=42):
    """
    将原始数据按比例划分为 训练集 和 测试集.
    分层采样，保证各类比例一致.
    """
    np.random.seed(seed)
    n = len(labels)
    indices = np.arange(n)

    train_idx = []
    test_idx = []

    for c in np.unique(labels):
        c_idx = indices[labels == c]
        np.random.shuffle(c_idx)
        n_train = int(len(c_idx) * train_ratio)
        train_idx.extend(c_idx[:n_train])
        test_idx.extend(c_idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return (
        data[train_idx], labels[train_idx],
        data[test_idx], labels[test_idx],
    )


# ============================================================
#  MLP 分类器
# ============================================================
class EEGMLPClassifier(nn.Module):
    """
    纯 MLP 分类器: 输入 (batch, 62, 200) → 展平 → MLP → 3 类

    架构:
      Input (62×200 = 12400)
        → Linear(12400, 1024) + BN + ReLU + Dropout
        → Linear(1024, 512)   + BN + ReLU + Dropout
        → Linear(512, 256)    + BN + ReLU + Dropout
        → Linear(256, 128)    + BN + ReLU + Dropout
        → Linear(128, 3)
    """

    def __init__(self, input_channels=62, seq_len=200, num_classes=3,
                 hidden_dims=None, dropout=0.3):
        super().__init__()
        self.input_dim = input_channels * seq_len

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]

        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 62, 200) → (batch, 12400)
        x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================
#  训练 & 评估
# ============================================================
def train_and_evaluate(
    train_data, train_labels,
    test_data, test_labels,
    epochs=100, batch_size=128, lr=1e-3,
    weight_decay=1e-4, device="cpu",
    verbose=True, run_name="",
):
    """
    训练 MLP 分类器并在测试集上评估.

    Returns:
        results: dict 包含 accuracy, f1_macro, f1_weighted,
                 per_class_acc, confusion_matrix, best_epoch
    """
    # 转为 tensor
    X_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_labels).long()
    X_test = torch.from_numpy(test_data).float()
    y_test = torch.from_numpy(test_labels).long()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 类别权重 (处理不平衡)
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3.0
    class_weights = torch.from_numpy(class_weights).to(device)

    # 模型
    model = EEGMLPClassifier(
        input_channels=train_data.shape[1],
        seq_len=train_data.shape[2],
        num_classes=3,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    patience = 20
    no_improve = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"Training {run_name}",
                disable=not verbose)
    for epoch in pbar:
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # --- Eval ---
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        test_acc = accuracy_score(all_true, all_preds)

        pbar.set_postfix(loss=f"{train_loss:.4f}",
                         train_acc=f"{train_acc:.3f}",
                         test_acc=f"{test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # --- 最终评估 (用 best model) ---
    model.load_state_dict(best_state)
    model.to(device).eval()

    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    acc = accuracy_score(all_true, all_preds)
    f1_mac = f1_score(all_true, all_preds, average="macro")
    f1_wt = f1_score(all_true, all_preds, average="weighted")
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])

    per_class_acc = {}
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    for c in range(3):
        mask = all_true == c
        if mask.sum() > 0:
            per_class_acc[label_names[c]] = (all_preds[mask] == c).mean()
        else:
            per_class_acc[label_names[c]] = 0.0

    report = classification_report(
        all_true, all_preds,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    )

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "report": report,
        "best_epoch": best_epoch,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
    }


# ============================================================
#  打印结果
# ============================================================
def print_results(results, title="Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  训练样本: {results['train_samples']}")
    print(f"  测试样本: {results['test_samples']}")
    print(f"  最佳 Epoch: {results['best_epoch']}")
    print(f"  ---")
    print(f"  Accuracy:    {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1 (macro):  {results['f1_macro']:.4f}")
    print(f"  F1 (weight): {results['f1_weighted']:.4f}")
    print(f"  ---")
    print(f"  各类别准确率:")
    for cls, acc in results["per_class_acc"].items():
        print(f"    {cls:>10s}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  分类报告:")
    for line in results["report"].split("\n"):
        print(f"    {line}")
    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    cm = results["confusion_matrix"]
    header = "           negative  neutral  positive"
    print(f"    {header}")
    for i, name in enumerate(["negative", "neutral", "positive"]):
        row = "    ".join(f"{cm[i,j]:>8d}" for j in range(3))
        print(f"    {name:>10s}  {row}")
    print(f"{'='*60}\n")


def print_comparison(res_no_syn, res_with_syn):
    """对比打印两组结果."""
    print(f"\n{'#'*60}")
    print(f"  对比总结: 生成数据对分类性能的影响")
    print(f"{'#'*60}")

    metrics = [
        ("Accuracy", "accuracy"),
        ("F1 (macro)", "f1_macro"),
        ("F1 (weighted)", "f1_weighted"),
    ]

    print(f"\n  {'指标':<16s} {'无生成数据':>12s} {'有生成数据':>12s} {'差值':>10s} {'变化':>8s}")
    print(f"  {'-'*58}")

    for name, key in metrics:
        v1 = res_no_syn[key]
        v2 = res_with_syn[key]
        diff = v2 - v1
        sign = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
        print(f"  {name:<16s} {v1:>11.4f}  {v2:>11.4f}  {diff:>+9.4f}  {sign:>6s}")

    print(f"\n  各类别准确率:")
    print(f"  {'类别':<12s} {'无生成数据':>12s} {'有生成数据':>12s} {'差值':>10s}")
    print(f"  {'-'*48}")
    for cls in ["negative", "neutral", "positive"]:
        v1 = res_no_syn["per_class_acc"][cls]
        v2 = res_with_syn["per_class_acc"][cls]
        diff = v2 - v1
        print(f"  {cls:<12s} {v1:>11.4f}  {v2:>11.4f}  {diff:>+9.4f}")

    print(f"\n  训练集大小: {res_no_syn['train_samples']} → {res_with_syn['train_samples']}"
          f" (+{res_with_syn['train_samples'] - res_no_syn['train_samples']} 生成样本)")
    print(f"  测试集大小: {res_no_syn['test_samples']} (相同)")

    # 结论
    acc_diff = res_with_syn["accuracy"] - res_no_syn["accuracy"]
    f1_diff = res_with_syn["f1_macro"] - res_no_syn["f1_macro"]

    print(f"\n  📊 结论: ", end="")
    if acc_diff > 0.01 and f1_diff > 0.01:
        print("生成数据显著提升了分类性能 ✅ → 质量良好")
    elif acc_diff > 0.0 or f1_diff > 0.0:
        print("生成数据略微提升了分类性能 → 质量可接受")
    elif abs(acc_diff) < 0.01 and abs(f1_diff) < 0.01:
        print("生成数据对分类性能无明显影响 → 质量一般，无害但无益")
    else:
        print("生成数据降低了分类性能 ⚠️ → 质量不足，引入了噪声")
    print(f"{'#'*60}\n")


# ============================================================
#  主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="MLP 分类器评估生成数据质量 (SEED-RAW)"
    )

    # 数据路径
    parser.add_argument("--data_root", type=str, required=True,
                        help="原始 SEED EEG 数据目录 (包含 .mat 文件)")
    parser.add_argument("--synthetic_path", type=str, default=None,
                        help="生成数据 .npz 路径 (由 train_seed.py 产生)")

    # 模式选择
    parser.add_argument("--no_synthetic", action="store_true",
                        help="只用原始数据训练 (baseline)")
    parser.add_argument("--compare", action="store_true",
                        help="自动运行两组实验并对比")

    # 数据划分
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="原始数据中用于训练的比例 (默认 0.6)")
    parser.add_argument("--window", type=int, default=200,
                        help="EEG 窗口长度 (默认 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")

    # 生成数据混合比例
    parser.add_argument("--syn_ratio", type=float, default=1.0,
                        help="使用多少比例的生成数据 (1.0=全部, 0.5=一半)")

    # 设备
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")

    # 重复实验
    parser.add_argument("--n_runs", type=int, default=3,
                        help="重复实验次数 (取平均)")

    args = parser.parse_args()

    # 设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. 加载原始数据 ----
    print("\n" + "=" * 60)
    print("  加载原始 SEED 数据...")
    print("=" * 60)
    orig_data, orig_labels = load_original_data(
        args.data_root, window=args.window, seed=args.seed
    )
    print(f"原始数据: {orig_data.shape}, 标签分布: "
          f"{dict(zip(*np.unique(orig_labels, return_counts=True)))}")

    # ---- 2. 加载生成数据 (如果有) ----
    syn_data, syn_labels = None, None
    if args.synthetic_path and not args.no_synthetic:
        print("\n" + "=" * 60)
        print("  加载生成数据...")
        print("=" * 60)
        syn_data, syn_labels = load_synthetic_data(args.synthetic_path)

        # 按比例采样
        if args.syn_ratio < 1.0:
            n_use = int(len(syn_labels) * args.syn_ratio)
            idx = np.random.choice(len(syn_labels), n_use, replace=False)
            syn_data = syn_data[idx]
            syn_labels = syn_labels[idx]
            print(f"使用 {args.syn_ratio*100:.0f}% 生成数据: {syn_data.shape[0]} 样本")

    # ---- 3. 决定运行模式 ----
    if args.compare:
        if syn_data is None:
            print("ERROR: --compare 模式需要 --synthetic_path")
            return
        modes = ["no_synthetic", "with_synthetic"]
    elif args.no_synthetic or syn_data is None:
        modes = ["no_synthetic"]
    else:
        modes = ["with_synthetic"]

    all_results = {}

    for mode in modes:
        print(f"\n{'*'*60}")
        print(f"  实验模式: {mode}")
        print(f"{'*'*60}")

        run_accs = []
        run_f1s = []
        run_results = []

        for run_i in range(args.n_runs):
            run_seed = args.seed + run_i

            # 划分原始数据
            train_orig, train_orig_labels, test_data, test_labels = split_data(
                orig_data, orig_labels,
                train_ratio=args.train_ratio,
                seed=run_seed,
            )

            if mode == "with_synthetic":
                # 合并: 原始训练 + 生成数据
                train_data = np.concatenate([train_orig, syn_data], axis=0)
                train_labels = np.concatenate([train_orig_labels, syn_labels], axis=0)
                run_name = f"With Syn (run {run_i+1}/{args.n_runs})"
            else:
                # 只用原始训练
                train_data = train_orig
                train_labels = train_orig_labels
                run_name = f"No Syn (run {run_i+1}/{args.n_runs})"

            print(f"\n  Run {run_i+1}: 训练={len(train_labels)}, 测试={len(test_labels)}")
            print(f"    训练标签分布: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

            # 固定 torch 种子
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            results = train_and_evaluate(
                train_data, train_labels,
                test_data, test_labels,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                verbose=True,
                run_name=run_name,
            )

            run_accs.append(results["accuracy"])
            run_f1s.append(results["f1_macro"])
            run_results.append(results)

            print_results(results, title=f"{run_name}")

        # 汇总多次运行
        if args.n_runs > 1:
            print(f"\n  === {mode} 汇总 ({args.n_runs} 次运行) ===")
            print(f"  Accuracy: {np.mean(run_accs):.4f} ± {np.std(run_accs):.4f}")
            print(f"  F1 macro: {np.mean(run_f1s):.4f} ± {np.std(run_f1s):.4f}")

        # 保存最后一次的结果用于对比 (或取平均)
        avg_result = run_results[-1].copy()
        avg_result["accuracy"] = np.mean(run_accs)
        avg_result["f1_macro"] = np.mean(run_f1s)
        avg_result["f1_weighted"] = np.mean([r["f1_weighted"] for r in run_results])

        # 各类平均
        avg_pca = {}
        for cls in ["negative", "neutral", "positive"]:
            avg_pca[cls] = np.mean([r["per_class_acc"][cls] for r in run_results])
        avg_result["per_class_acc"] = avg_pca

        all_results[mode] = avg_result

    # ---- 4. 对比输出 ----
    if args.compare and len(all_results) == 2:
        print_comparison(all_results["no_synthetic"], all_results["with_synthetic"])

    print("Done!")


if __name__ == "__main__":
    main()