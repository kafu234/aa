"""
第一步: 量化合成 DE 数据 vs 真实 DE 数据的"分布保真度", 判断
"diagnose 高但下游无增益"的主因是哪一个。

复用 eval_de_classifier 的 load_de_data / load_synthetic_de, 保证归一化口径
与下游评估完全一致 (都是 per-file z-score → clip±5 → /5 → [-1,1])。

三个指标:
  1) 类内方差比 (syn/real):  < 1 明显 → mode collapse / 塌缩
  2) 通道相关矩阵距离:        大 → 通道间相关结构没保住 (DGCNN 读的就是它)
  3) real-vs-syn 域分类器:    ~100% → 存在 covariate shift, 增广天然帮不上

用法 (参数和你跑 eval_de_classifier 时保持一致):
    python diagnose_distribution.py \
        --data_root /root/autodl-tmp/ExtractedFeatures \
        --synthetic_path ./results/s1_de_cond/generated_SEED_DE_GEN_1.npz \
        --subject 1 --split_mode trial \
        --train_trials 0,1,2,3,4,5,6,7,8 --test_trials 9,10,11,12,13,14
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_de_classifier import load_de_data, load_synthetic_de

CLASS_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


# ------------------------------------------------------------------
# 指标 1: 类内方差比
# ------------------------------------------------------------------
def intra_class_variance_ratio(real, real_lab, syn, syn_lab, n_classes=3):
    """
    对每个类, 比较合成 vs 真实在 310 个特征 (62ch x 5band) 上的方差。
    返回 syn_var / real_var 的逐特征比值的统计量。
    < 1  → 合成样本更"挤", 多样性不足 (塌缩)。
    """
    print("\n" + "=" * 64)
    print("  指标 1 / 类内方差比 (syn_var / real_var)")
    print("  解读:  ≈1 健康   <0.7 多样性不足   <0.4 严重塌缩")
    print("=" * 64)

    ratios_all = []
    for c in range(n_classes):
        r = real[real_lab == c].reshape((real_lab == c).sum(), -1)  # (Nr, 310)
        s = syn[syn_lab == c].reshape((syn_lab == c).sum(), -1)     # (Ns, 310)
        if len(r) < 5 or len(s) < 5:
            print(f"  [{CLASS_NAMES[c]}] 样本不足, 跳过 (real={len(r)}, syn={len(s)})")
            continue
        var_r = r.var(axis=0) + 1e-8
        var_s = s.var(axis=0)
        ratio = var_s / var_r            # 逐特征
        ratios_all.append(ratio)
        print(f"  [{CLASS_NAMES[c]:>8s}] real_n={len(r):4d} syn_n={len(s):4d}  "
              f"方差比 中位数={np.median(ratio):.3f}  "
              f"均值={ratio.mean():.3f}  "
              f"<0.5的特征占比={np.mean(ratio < 0.5) * 100:.0f}%")

    if ratios_all:
        allr = np.concatenate(ratios_all)
        med = np.median(allr)
        print(f"\n  总体方差比中位数 = {med:.3f}")
        if med < 0.4:
            print("  >> 严重塌缩: 优先调 guidance_scale↓ / classifier_weight↓ / anchored 采样")
        elif med < 0.7:
            print("  >> 多样性偏低: 建议降 guidance_scale, 试 anchored 采样")
        else:
            print("  >> 多样性尚可, 主因可能不在塌缩, 看指标 2/3")
    return ratios_all


# ------------------------------------------------------------------
# 指标 2: 通道相关矩阵距离 (DGCNN 依赖的结构)
# ------------------------------------------------------------------
def channel_corr_distance(real, real_lab, syn, syn_lab, n_classes=3):
    """
    对每个类、每个频带, 用 N 个样本作为观测, 算 62x62 通道间 Pearson 相关矩阵,
    比较合成 vs 真实的 Frobenius 距离 (按矩阵规模归一)。
    距离大 → 边缘分布可能对了, 但通道联合结构错位, 图网络会被误导。
    """
    print("\n" + "=" * 64)
    print("  指标 2 / 通道相关矩阵距离 (per class, per band)")
    print("  解读:  <0.1 很好   0.1~0.25 一般   >0.25 结构明显错位")
    print("=" * 64)

    def corr_per_band(X):  # X: (N, 62, 5) -> list of 62x62
        mats = []
        for b in range(X.shape[2]):
            Xb = X[:, :, b]                      # (N, 62)
            # np.corrcoef 对 (62, N) 求各通道两两相关
            mats.append(np.corrcoef(Xb.T))
        return mats

    dists_all = []
    for c in range(n_classes):
        r = real[real_lab == c]
        s = syn[syn_lab == c]
        if len(r) < 30 or len(s) < 30:
            print(f"  [{CLASS_NAMES[c]}] 样本不足 (real={len(r)}, syn={len(s)}), 相关估计不稳, 跳过")
            continue
        cr = corr_per_band(r)
        cs = corr_per_band(s)
        band_d = []
        for b in range(5):
            diff = np.nan_to_num(cr[b] - cs[b])
            # 按矩阵元素个数归一的 Frobenius 距离 (RMS of 相关差)
            d = np.sqrt((diff ** 2).mean())
            band_d.append(d)
        dists_all.extend(band_d)
        band_str = "  ".join(f"{BAND_NAMES[b]}={band_d[b]:.3f}" for b in range(5))
        print(f"  [{CLASS_NAMES[c]:>8s}]  {band_str}   类均值={np.mean(band_d):.3f}")

    if dists_all:
        overall = np.mean(dists_all)
        print(f"\n  总体相关距离 = {overall:.3f}")
        if overall > 0.25:
            print("  >> 通道相关结构错位是主因: 加 batch 协方差/相关损失, 或用 anchored 采样")
        elif overall > 0.1:
            print("  >> 相关结构有偏差, 值得加协方差损失")
        else:
            print("  >> 相关结构保持良好, 主因不在这里")
    return dists_all


# ------------------------------------------------------------------
# 指标 3: real-vs-syn 域分类器 (covariate shift)
# ------------------------------------------------------------------
def domain_classifier_test(real, syn, seed=42):
    """
    训一个二分类器区分 real(0) vs syn(1)。
    test 准确率 ~0.5 → 两者不可区分 (理想);  ~1.0 → 强分布偏移。
    线性(LogReg) + 非线性(RandomForest) 各跑一个, 后者能抓非线性差异。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("\n" + "=" * 64)
    print("  指标 3 / real-vs-syn 域分类器")
    print("  解读:  acc≈0.5 不可区分(好)   0.7~0.9 有偏移   >0.95 强 covariate shift")
    print("=" * 64)

    Xr = real.reshape(len(real), -1)
    Xs = syn.reshape(len(syn), -1)
    # 类别平衡: 取两边较小的数量, 避免分类器靠先验作弊
    n = min(len(Xr), len(Xs))
    rng = np.random.RandomState(seed)
    Xr = Xr[rng.choice(len(Xr), n, replace=False)]
    Xs = Xs[rng.choice(len(Xs), n, replace=False)]

    X = np.concatenate([Xr, Xs], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    for name, clf in [
        ("LogReg     ", LogisticRegression(max_iter=2000, C=1.0)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed)),
    ]:
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        try:
            auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        except Exception:
            auc = float("nan")
        print(f"  {name}:  test_acc={acc:.3f}  auc={auc:.3f}")

    print("\n  (若两个都 >0.95, 合成与真实存在系统性偏移, 先解决偏移再谈增广)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--synthetic_path", type=str, required=True)
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--split_mode", type=str, default="trial")
    p.add_argument("--train_trials", type=str, default=None)
    p.add_argument("--test_trials", type=str, default=None)
    p.add_argument("--test_subject", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    # 真实数据: 用 train split (生成器就是学它的分布)
    real, real_lab, _, _ = load_de_data(
        args.data_root, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject)
    syn, syn_lab = load_synthetic_de(args.synthetic_path)

    print(f"\nReal: {real.shape}  Synthetic: {syn.shape}")
    print(f"Real range [{real.min():.2f}, {real.max():.2f}]  "
          f"Syn range [{syn.min():.2f}, {syn.max():.2f}]")

    intra_class_variance_ratio(real, real_lab, syn, syn_lab)
    channel_corr_distance(real, real_lab, syn, syn_lab)
    domain_classifier_test(real, syn, seed=args.seed)

    print("\n" + "=" * 64)
    print("  小结: 哪个指标最差, 就先解决哪条 (见对话里第二/四步)")
    print("=" * 64)


if __name__ == "__main__":
    main()
