"""
diagnose_seed4.py — SEED-IV 负提升诊断工具
=============================================
用法:
  python diagnose_seed4.py \
      --data_root /root/autodl-tmp/eeg_feature_smooth \
      --synthetic_path ./results/seed4_trial/s1/generated_*.npz \
      --subject 1

会依次检查:
  1. 标签正确性 (最可能的问题来源)
  2. 真实数据基线准确率
  3. 合成数据质量
  4. 增强后是否有提升
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.Data_utils.seed4_dataset import (
    SEEDIVDataset, SEEDIV_SESSION_LABELS, NUM_CLASSES, LABEL_NAMES
)


def check_labels(data_root):
    """
    诊断 1: 检查标签是否正确.
    读取 label.mat 的原始内容, 和硬编码标签对比.
    """
    print("\n" + "=" * 60)
    print("  诊断 1: 标签正确性检查")
    print("=" * 60)

    # 读取 label.mat
    label_paths = [
        os.path.join(data_root, "session1_label.mat"),
        os.path.join(data_root, "..", "session1_label.mat"),
    ]
    for lp in label_paths:
        if os.path.exists(lp):
            print(f"\n  找到 label.mat: {os.path.abspath(lp)}")
            label_data = sio.loadmat(lp)
            for key in label_data:
                if key.startswith("_"):
                    continue
                arr = label_data[key]
                if isinstance(arr, np.ndarray):
                    print(f"    变量 '{key}': shape={arr.shape}, dtype={arr.dtype}")
                    print(f"    unique values: {sorted(np.unique(arr).tolist())}")
                    if arr.ndim == 2:
                        for row_idx in range(min(arr.shape[0], 3)):
                            row = arr[row_idx].flatten().astype(int).tolist()
                            print(f"    row {row_idx} (session {row_idx+1}): {row}")
            break
    else:
        print("  未找到 label.mat")

    # 打印硬编码标签
    print(f"\n  硬编码标签 (来自 BCMI 官方):")
    for sess in [1, 2, 3]:
        print(f"    Session {sess}: {SEEDIV_SESSION_LABELS[sess]}")

    print(f"\n  ⚠ 如果 label.mat 的值和硬编码不一致,")
    print(f"    说明标签可能是错的, 这是负提升的最大嫌疑!")


def check_baseline(data_root, subject, train_trials, test_trials):
    """
    诊断 2: 不加增强, 纯真实数据的基线准确率.
    如果基线就很低 (<40%), 说明标签可能有问题.
    """
    print("\n" + "=" * 60)
    print(f"  诊断 2: 被试 {subject} 基线准确率 (无增强)")
    print("=" * 60)

    ds_train = SEEDIVDataset(
        data_root=data_root, subjects=[subject],
        split_mode="trial", train_trials=train_trials, test_trials=test_trials,
        period="train", conditional=True,
    )
    ds_test = SEEDIVDataset(
        data_root=data_root, subjects=[subject],
        split_mode="trial", train_trials=train_trials, test_trials=test_trials,
        period="test", conditional=True,
    )

    X_train, y_train = ds_train.samples, ds_train.labels
    X_test, y_test = ds_test.samples, ds_test.labels

    # reshape (N, 62, 5) → (N, 310)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"\n  训练集: {X_train.shape[0]} samples, 标签分布: {dict(Counter(y_train.tolist()))}")
    print(f"  测试集: {X_test.shape[0]} samples, 标签分布: {dict(Counter(y_test.tolist()))}")

    # LDA 快速分类
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_flat, y_train)
    y_pred = lda.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  LDA 基线准确率: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  随机猜测准确率: {1/NUM_CLASSES:.4f} ({100/NUM_CLASSES:.1f}%)")

    if acc < 0.35:
        print(f"\n  ⚠ 基线准确率极低 (<35%), 标签很可能是错的!")
        print(f"    → 请对照诊断 1 的 label.mat 内容修正标签映射")
    elif acc < 0.50:
        print(f"\n  ⚠ 基线准确率偏低 (<50%), 可能是标签问题或数据本身难度大")
    else:
        print(f"\n  ✓ 基线准确率合理, 标签大概率是正确的")

    print(f"\n  分类报告:")
    target_names = [f"{LABEL_NAMES[i]}({i})" for i in range(NUM_CLASSES)]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return acc


def check_synthetic(synthetic_path, data_root, subject, train_trials, test_trials):
    """
    诊断 3: 合成数据质量检查.
    """
    print("\n" + "=" * 60)
    print(f"  诊断 3: 合成数据质量分析")
    print("=" * 60)

    if not os.path.exists(synthetic_path):
        print(f"  ✗ 未找到合成数据: {synthetic_path}")
        return

    bundle = np.load(synthetic_path)
    syn_samples = bundle["samples"]
    syn_labels = bundle["labels"]

    print(f"\n  合成数据: {syn_samples.shape}")
    print(f"  标签分布: {dict(Counter(syn_labels.tolist()))}")
    print(f"  值范围: [{syn_samples.min():.4f}, {syn_samples.max():.4f}]")
    print(f"  均值: {syn_samples.mean():.4f}, 标准差: {syn_samples.std():.4f}")

    # 加载真实数据对比
    ds_train = SEEDIVDataset(
        data_root=data_root, subjects=[subject],
        split_mode="trial", train_trials=train_trials, test_trials=test_trials,
        period="train", conditional=True,
    )
    real = ds_train.samples
    real_labels = ds_train.labels

    print(f"\n  真实数据: {real.shape}")
    print(f"  值范围: [{real.min():.4f}, {real.max():.4f}]")
    print(f"  均值: {real.mean():.4f}, 标准差: {real.std():.4f}")

    # 逐类对比
    print(f"\n  逐类均值/标准差对比:")
    print(f"  {'类别':>10s}  {'真实均值':>10s}  {'合成均值':>10s}  {'真实std':>10s}  {'合成std':>10s}  {'差异':>8s}")
    for c in range(NUM_CLASSES):
        r = real[real_labels == c]
        s = syn_samples[syn_labels == c]
        if len(r) == 0 or len(s) == 0:
            print(f"  {LABEL_NAMES[c]:>10s}  {'N/A':>10s}  {'N/A':>10s}")
            continue
        r_mean, s_mean = r.mean(), s.mean()
        r_std, s_std = r.std(), s.std()
        diff = abs(r_mean - s_mean)
        status = "✓" if diff < 0.1 else "⚠" if diff < 0.3 else "✗"
        print(f"  {LABEL_NAMES[c]:>10s}  {r_mean:>10.4f}  {s_mean:>10.4f}  "
              f"{r_std:>10.4f}  {s_std:>10.4f}  {status} {diff:.4f}")

    # 用 LDA 区分真假 (越难区分越好)
    n_real, n_syn = min(len(real), len(syn_samples)), min(len(real), len(syn_samples))
    combined_X = np.concatenate([
        real[:n_real].reshape(n_real, -1),
        syn_samples[:n_syn].reshape(n_syn, -1),
    ])
    combined_y = np.array([0]*n_real + [1]*n_syn)

    from sklearn.model_selection import cross_val_score
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, combined_X, combined_y, cv=5)
    disc_acc = scores.mean()

    print(f"\n  真/假判别准确率 (LDA 5-fold): {disc_acc:.4f}")
    if disc_acc > 0.85:
        print(f"  ⚠ 判别准确率 >85%, 合成数据和真实数据差异太大!")
        print(f"    → 生成质量不够, 需要更多训练 epoch 或调参")
    elif disc_acc > 0.70:
        print(f"  ⚠ 判别准确率 70-85%, 合成数据质量一般")
    else:
        print(f"  ✓ 判别准确率 <70%, 合成数据质量较好")


def check_augmentation_effect(data_root, synthetic_path, subject, train_trials, test_trials,
                               syn_ratios=[0.1, 0.25, 0.5]):
    """
    诊断 4: 不同增强比例的效果.
    """
    print("\n" + "=" * 60)
    print(f"  诊断 4: 增强效果 (被试 {subject})")
    print("=" * 60)

    if not os.path.exists(synthetic_path):
        print(f"  ✗ 未找到合成数据: {synthetic_path}")
        return

    ds_train = SEEDIVDataset(
        data_root=data_root, subjects=[subject],
        split_mode="trial", train_trials=train_trials, test_trials=test_trials,
        period="train", conditional=True,
    )
    ds_test = SEEDIVDataset(
        data_root=data_root, subjects=[subject],
        split_mode="trial", train_trials=train_trials, test_trials=test_trials,
        period="test", conditional=True,
    )

    X_train, y_train = ds_train.samples, ds_train.labels
    X_test, y_test = ds_test.samples, ds_test.labels

    bundle = np.load(synthetic_path)
    syn_samples, syn_labels = bundle["samples"], bundle["labels"]

    # 基线
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train.reshape(len(X_train), -1), y_train)
    baseline_acc = accuracy_score(y_test, lda.predict(X_test.reshape(len(X_test), -1)))

    print(f"\n  {'比例':>8s}  {'准确率':>8s}  {'vs基线':>8s}")
    print(f"  {'baseline':>8s}  {baseline_acc:>8.4f}  {'---':>8s}")

    for ratio in syn_ratios:
        n_add = int(len(X_train) * ratio)
        if n_add > len(syn_samples):
            n_add = len(syn_samples)

        # 按类别平衡采样
        indices = []
        per_class = n_add // NUM_CLASSES
        for c in range(NUM_CLASSES):
            c_idx = np.where(syn_labels == c)[0]
            if len(c_idx) > 0:
                chosen = np.random.choice(c_idx, min(per_class, len(c_idx)), replace=True)
                indices.extend(chosen.tolist())
        indices = np.array(indices)

        X_aug = np.concatenate([X_train, syn_samples[indices]])
        y_aug = np.concatenate([y_train, syn_labels[indices]])

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_aug.reshape(len(X_aug), -1), y_aug)
        aug_acc = accuracy_score(y_test, lda.predict(X_test.reshape(len(X_test), -1)))

        delta = aug_acc - baseline_acc
        status = "✓" if delta > 0 else "✗"
        print(f"  {ratio:>8.2f}  {aug_acc:>8.4f}  {status} {delta:+.4f}")

    if all_negative := baseline_acc > baseline_acc:  # placeholder
        pass
    print(f"\n  如果所有比例都是负提升:")
    print(f"    → 最可能原因: 标签错误 (回看诊断 1)")
    print(f"    → 次可能原因: 合成数据质量太差 (回看诊断 3)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--train_trials", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--test_trials", type=str, default="15,16,17,18,19,20,21,22,23")
    args = parser.parse_args()

    train_trials = [int(x) for x in args.train_trials.split(",")]
    test_trials = [int(x) for x in args.test_trials.split(",")]

    # 诊断 1: 标签检查
    check_labels(args.data_root)

    # 诊断 2: 基线准确率
    baseline = check_baseline(args.data_root, args.subject, train_trials, test_trials)

    # 诊断 3 & 4: 合成数据 (如果提供了)
    if args.synthetic_path:
        import glob
        syn_path = glob.glob(args.synthetic_path)
        if syn_path:
            check_synthetic(syn_path[0], args.data_root, args.subject, train_trials, test_trials)
            check_augmentation_effect(
                args.data_root, syn_path[0], args.subject, train_trials, test_trials,
            )
        else:
            print(f"\n  未找到合成数据文件: {args.synthetic_path}")

    print("\n" + "=" * 60)
    print("  诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()