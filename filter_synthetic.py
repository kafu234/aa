"""
filter_synthetic.py — 质量过滤: 按类别平衡保留高置信合成样本

原理: 用真实数据训练 DGCNN, 给合成样本打分,
      每个类别只保留被正确分类的样本, 按置信度排序, 三类取相同数量.

用法:
    python filter_synthetic.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --synthetic_path /root/autodl-tmp/results/de_ft/s2/generated_SEED_DE_GEN_1.npz \
        --output /root/autodl-tmp/results/de_ft/s2/generated_filtered.npz \
        --subject 2 --split_mode trial \
        --train_trials 0,1,2,3,4,5,6,7,8 \
        --test_trials 9,10,11,12,13,14

    # 过滤后评估
    python eval_de_classifier.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --synthetic_path /root/autodl-tmp/results/de_ft/s2/generated_filtered.npz \
        --subject 2 --split_mode trial \
        --train_trials 0,1,2,3,4,5,6,7,8 \
        --test_trials 9,10,11,12,13,14 \
        --mode compare --model dgcnn --syn_ratio 0.5 --n_runs 3
"""

import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_de_classifier import DGCNN, load_de_data, train_and_evaluate, load_synthetic_de


def main():
    parser = argparse.ArgumentParser(description="质量过滤合成数据")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--split_mode", type=str, default="trial")
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    # ---- 1. 训练 baseline DGCNN ----
    print("训练 baseline DGCNN...")
    tr_d, tr_l, te_d, te_l = load_de_data(
        args.data_root, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject)

    torch.manual_seed(args.seed)
    res = train_and_evaluate(tr_d, tr_l, te_d, te_l, device,
        model_type="dgcnn", epochs=args.epochs)
    model = res["model"]
    print(f"Baseline acc: {res['accuracy']:.4f}\n")

    # ---- 2. 给合成数据打分 ----
    print("给合成数据打分...")
    syn_data, syn_labels = load_synthetic_de(args.synthetic_path)

    model.eval()
    all_probs, all_preds = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(syn_data).float()), batch_size=1024)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, 1).cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)

    names = {0: "negative", 1: "neutral", 2: "positive"}

    # ---- 3. 按类别统计 ----
    print(f"\n{'='*50}")
    print(f"  过滤前统计")
    print(f"{'='*50}")
    per_class_correct = {}
    for c in range(3):
        mask = syn_labels == c
        n_total = mask.sum()
        correct = (preds[mask] == c)
        n_correct = correct.sum()
        avg_conf = probs[mask, c][correct].mean() if n_correct > 0 else 0
        per_class_correct[c] = n_correct
        print(f"  {names[c]}: {n_correct}/{n_total} 正确 ({n_correct/n_total*100:.1f}%), "
              f"平均置信={avg_conf:.3f}")

    # ---- 4. 平衡过滤: 每类取相同数量的高置信正确样本 ----
    min_keep = min(per_class_correct.values())
    if min_keep == 0:
        print("\n⚠️ 某个类别没有任何正确分类的样本, 无法过滤")
        print("建议: 重新训练该被试的生成模型")
        return

    print(f"\n每类保留: {min_keep} 样本 (按置信度 top-{min_keep})")

    filtered_data, filtered_labels = [], []
    for c in range(3):
        class_mask = syn_labels == c
        class_data = syn_data[class_mask]
        class_probs = probs[class_mask, c]
        class_preds = preds[class_mask]

        # 正确分类的样本
        correct = class_preds == c
        correct_indices = np.where(correct)[0]
        correct_conf = class_probs[correct]

        # 按置信度排序, 取 top
        top_order = np.argsort(-correct_conf)[:min_keep]
        top_idx = correct_indices[top_order]
        top_conf = correct_conf[top_order]

        filtered_data.append(class_data[top_idx])
        filtered_labels.append(np.full(min_keep, c, dtype=np.int64))
        print(f"  {names[c]}: 保留 {min_keep} 样本, "
              f"置信范围 [{top_conf.min():.3f}, {top_conf.max():.3f}], "
              f"平均={top_conf.mean():.3f}")

    filtered_data = np.concatenate(filtered_data)
    filtered_labels = np.concatenate(filtered_labels)

    print(f"\n过滤后: {filtered_data.shape}, 共 {len(filtered_labels)} 样本 (每类 {min_keep})")

    # ---- 5. 保存 (×5 还原到保存格式) ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output,
             data=(filtered_data * 5.0).astype(np.float32),
             labels=filtered_labels)
    print(f"保存至: {args.output}")

    # ---- 6. 快速验证: 用过滤后数据测试几个比例 ----
    print(f"\n{'='*50}")
    print(f"  快速验证 (过滤后)")
    print(f"{'='*50}")
    print(f"  Baseline: {res['accuracy']:.4f}")

    for ratio in [0.1, 0.25, 0.5, 1.0]:
        n = int(len(filtered_labels) * ratio)
        if n == 0:
            continue
        idx = np.random.choice(len(filtered_labels), min(n, len(filtered_labels)), replace=False)
        aug_d = np.concatenate([tr_d, filtered_data[idx]])
        aug_l = np.concatenate([tr_l, filtered_labels[idx]])

        accs = []
        for r in range(3):
            torch.manual_seed(args.seed + r)
            np.random.seed(args.seed + r)
            res2 = train_and_evaluate(aug_d, aug_l, te_d, te_l, device,
                model_type="dgcnn", epochs=args.epochs, verbose=False)
            accs.append(res2["accuracy"])
        diff = np.mean(accs) - res["accuracy"]
        sign = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
        print(f"  ratio={ratio:.2f}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} ({sign}{abs(diff):.4f})")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()