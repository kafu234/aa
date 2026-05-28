"""
align_spectral.py — 频域对齐: 让生成数据的频谱精确匹配真实数据

问题: FM-TS 生成的 EEG 频谱形状不对 (gamma 偏高, 类间排序错乱)
方法: 对每个类别、每个通道, 在频域上逐频点缩放, 使 PSD 匹配真实数据

用法:
    python align_spectral.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path ./results/s1_merged/generated_SEED_RAW_200.npz \
        --output ./results/s1_merged/generated_spectral_aligned.npz \
        --subject 1 --window 200 --split_mode trial \
        --train_trials 0,1,2,3,4,5,6,7,8 \
        --test_trials 9,10,11,12,13,14
"""

import os
import sys
import argparse
import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def spectral_align(real_data, syn_data, sfreq=200, smoothing=3):
    """
    频域对齐: 让 syn_data 的 PSD 匹配 real_data.

    方法:
      1. 计算真实/生成数据的平均 PSD (per channel)
      2. 算比值 ratio = sqrt(real_psd / gen_psd)
      3. 在频域上乘以 ratio: FFT → ×ratio → IFFT

    Args:
        real_data: (N1, 62, T) 真实数据
        syn_data:  (N2, 62, T) 生成数据
        sfreq:     采样率
        smoothing: ratio 平滑窗口大小, 避免单频点噪声放大

    Returns:
        aligned: (N2, 62, T) 对齐后的生成数据
    """
    N, C, T = syn_data.shape

    # 计算平均 PSD (per channel)
    # welch 输出: (N, C, n_freqs)
    _, real_psd = welch(real_data, fs=sfreq, axis=-1)  # (N1, 62, n_freqs)
    _, syn_psd = welch(syn_data, fs=sfreq, axis=-1)    # (N2, 62, n_freqs)

    real_mean_psd = real_psd.mean(axis=0)  # (62, n_freqs)
    syn_mean_psd = syn_psd.mean(axis=0)    # (62, n_freqs)

    # 比值 (幅度域, 所以 sqrt)
    ratio = np.sqrt(real_mean_psd / (syn_mean_psd + 1e-12))  # (62, n_freqs)

    # 平滑 ratio 避免单频点噪声
    if smoothing > 1:
        from scipy.ndimage import uniform_filter1d
        ratio = uniform_filter1d(ratio, size=smoothing, axis=-1)

    # 限制 ratio 范围, 防止极端缩放
    ratio = np.clip(ratio, 0.2, 5.0)

    # 在 FFT 域对齐
    # welch 的频率分辨率和 rfft 不同, 需要插值到 rfft 的频率网格
    welch_freqs = np.linspace(0, sfreq / 2, ratio.shape[-1])
    rfft_freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)

    # 插值 ratio 到 rfft 频率网格
    ratio_interp = np.zeros((C, len(rfft_freqs)), dtype=np.float32)
    for ch in range(C):
        ratio_interp[ch] = np.interp(rfft_freqs, welch_freqs, ratio[ch])

    # 批量处理: FFT → ×ratio → IFFT
    aligned = np.zeros_like(syn_data)
    batch_size = 500
    for i in range(0, N, batch_size):
        batch = syn_data[i:i + batch_size]  # (B, 62, T)
        fft_data = np.fft.rfft(batch, axis=-1)  # (B, 62, T//2+1)
        fft_aligned = fft_data * ratio_interp[np.newaxis, :, :]  # 广播
        aligned[i:i + batch_size] = np.fft.irfft(fft_aligned, n=T, axis=-1)

    return aligned.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="频域对齐生成数据")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--split_mode", type=str, default="trial")
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    args = parser.parse_args()

    from Utils.Data_utils.seed_dataset import SEEDDataset

    # ---- 加载真实数据 ----
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None

    ds = SEEDDataset(
        name="SEED_RAW", data_root=args.data_root, data_type="raw",
        window=args.window, proportion=1.0, seed=42,
        conditional=True, sfreq=200,
        bandpass_low=0.5, bandpass_high=50.0,
        notch_freq=50.0, notch_width=2.0, baseline_correction=True,
        split_mode=args.split_mode, subjects=[args.subject],
        train_trials=train_trials, test_trials=test_trials,
        period="train",
    )
    real, real_labels = ds.samples, ds.labels

    # ---- 加载生成数据 ----
    CLIP_STD = 5.0
    bundle = np.load(args.synthetic_path)
    syn_data = np.clip(bundle["data"] / CLIP_STD, -1.0, 1.0).astype(np.float32)
    syn_labels = bundle["labels"].astype(np.int64)

    names = {0: "negative", 1: "neutral", 2: "positive"}
    bands = [(1, 4, "delta"), (4, 8, "theta"), (8, 13, "alpha"),
             (13, 30, "beta"), (30, 50, "gamma")]

    # ---- 对齐前 ----
    print("=== 对齐前 ===")
    for c in range(3):
        data = syn_data[syn_labels == c][:100]
        f, psd = welch(data, fs=200, axis=-1)
        powers = [f"{bn}={psd[:, :, (f >= lo) & (f < hi)].mean():.6f}"
                  for lo, hi, bn in bands]
        print(f"  gen {names[c]}: {', '.join(powers)}")

    # ---- 逐类别频域对齐 ----
    print("\n正在对齐...")
    for c in range(3):
        real_c = real[real_labels == c]
        syn_mask = syn_labels == c
        syn_c = syn_data[syn_mask]

        if len(real_c) == 0 or len(syn_c) == 0:
            continue

        aligned_c = spectral_align(real_c, syn_c, sfreq=200)
        syn_data[syn_mask] = aligned_c
        print(f"  {names[c]}: {len(syn_c)} samples aligned")

    syn_data = np.clip(syn_data, -1.0, 1.0).astype(np.float32)

    # ---- 对齐后 ----
    print("\n=== 对齐后 ===")
    print("真实数据:")
    for c in range(3):
        data = real[real_labels == c][:100]
        f, psd = welch(data, fs=200, axis=-1)
        powers = [f"{bn}={psd[:, :, (f >= lo) & (f < hi)].mean():.6f}"
                  for lo, hi, bn in bands]
        print(f"  real {names[c]}: {', '.join(powers)}")

    print("生成数据 (对齐后):")
    for c in range(3):
        data = syn_data[syn_labels == c][:100]
        f, psd = welch(data, fs=200, axis=-1)
        powers = [f"{bn}={psd[:, :, (f >= lo) & (f < hi)].mean():.6f}"
                  for lo, hi, bn in bands]
        print(f"  gen {names[c]}: {', '.join(powers)}")

    # ---- 保存 ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, data=(syn_data * CLIP_STD).astype(np.float32), labels=syn_labels)
    print(f"\n保存至: {args.output}")


if __name__ == "__main__":
    main()