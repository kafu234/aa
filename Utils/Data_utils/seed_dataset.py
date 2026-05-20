"""
SEED Dataset Loader for FM-TS
==============================
- 每个文件单独归一化
- 数据格式 (62, 200): 62通道在前, 200时间点在后
- 支持条件生成 (按情绪标签)
"""

import os
import re
import glob
import torch
import numpy as np
from scipy import io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

# SEED 标准标签 (15 trials)
SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}  # negative=0, neutral=1, positive=2
NUM_CLASSES = 3


class SEEDDataset(Dataset):
    """
    SEED 原始 EEG 数据集加载器。

    输出格式: 每个样本 shape = (62, window)
        - 62 = EEG 通道数
        - window = 时间点数 (默认 200)

    归一化策略: 每个 .mat 文件单独做 MinMaxScaler，
    将每个通道的数据缩放到 [0,1]，再映射到 [-1,1]。
    """

    def __init__(
        self,
        name="SEED",
        data_root="./Data/datasets/SEED/Preprocessed_EEG",
        data_type="raw",
        de_key_prefix="de_LDS",
        raw_key_suffix="_eeg",
        freq_bands=None,
        subjects=None,
        sessions=None,
        window=200,
        stride=None,
        proportion=0.8,
        neg_one_to_one=True,
        scaler_type="minmax",
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        conditional=True,
        target_label=None,
        **kwargs,
    ):
        super(SEEDDataset, self).__init__()
        assert period in ["train", "test"]

        self.name = name
        self.window = window
        self.stride = stride if stride is not None else window
        self.period = period
        self.auto_norm = neg_one_to_one
        self.conditional = conditional

        # seq_length=62, feature_size=window (FM-TS 需要)
        self.var_num = window

        # ---- 1. 加载标签 ----
        seed_labels = self._load_labels(data_root)

        # ---- 2. 按文件加载 + 每个文件单独归一化 + 滑动窗口 ----
        all_samples, all_labels = self._load_and_process(
            data_root, seed_labels, raw_key_suffix,
            subjects, sessions, window, self.stride, neg_one_to_one
        )

        # ---- 3. 按标签过滤 ----
        if target_label is not None:
            mask = all_labels == target_label
            all_samples = all_samples[mask]
            all_labels = all_labels[mask]

        # ---- 4. 训练/测试划分 ----
        train_data, train_labels, test_data, test_labels = self._split(
            all_samples, all_labels, proportion, seed
        )

        if period == "train":
            self.samples = train_data
            self.labels = train_labels
        else:
            self.samples = test_data
            self.labels = test_labels

        self.sample_num = self.samples.shape[0]

        print(
            f"[SEEDDataset] period={period}, samples={self.sample_num}, "
            f"shape=({self.samples.shape[1]}, {self.samples.shape[2]}), "
            f"classes={np.unique(self.labels).tolist()}"
        )

    # ================================================================
    #  加载标签
    # ================================================================
    def _load_labels(self, data_root):
        search_paths = [
            os.path.join(data_root, "label.mat"),
            os.path.join(data_root, "..", "label.mat"),
            os.path.join(data_root, "..", "..", "label.mat"),
        ]
        for lp in search_paths:
            if os.path.exists(lp):
                label_data = sio.loadmat(os.path.abspath(lp))
                for key in label_data:
                    if not key.startswith("_"):
                        labels = label_data[key].flatten().astype(int).tolist()
                        print(f"[SEEDDataset] Labels loaded from: {os.path.abspath(lp)}")
                        print(f"[SEEDDataset] Labels: {labels}")
                        return labels
        print(f"[SEEDDataset] WARNING: label.mat not found, using hardcoded labels")
        return SEED_LABELS

    # ================================================================
    #  按文件加载 + 每个文件单独归一化 + 滑动窗口
    # ================================================================
    def _load_and_process(
        self, data_root, seed_labels, raw_key_suffix,
        subjects, sessions, window, stride, neg_one_to_one
    ):
        mat_files = sorted(glob.glob(os.path.join(data_root, "*.mat")))
        mat_files = [f for f in mat_files if "label" not in os.path.basename(f).lower()]

        if len(mat_files) == 0:
            raise FileNotFoundError(f"No .mat files found in {data_root}")

        # 按被试分组
        file_groups = self._group_files(mat_files)

        all_samples = []
        all_labels = []
        first_file = True

        for subj_idx, subj_sessions in file_groups.items():
            if subjects is not None and subj_idx not in subjects:
                continue

            for sess_idx, fpath in enumerate(subj_sessions):
                if sessions is not None and sess_idx not in sessions:
                    continue

                mat_data = sio.loadmat(fpath)

                # --- 提取该文件所有 trial: 每个 trial shape = (62, T) ---
                trials, trial_labels = self._extract_trials(
                    mat_data, raw_key_suffix, seed_labels
                )
                if len(trials) == 0:
                    continue

                # 打印首个文件的 trial-label 对应
                if first_file:
                    label_names = {0: "neg", 1: "neu", 2: "pos"}
                    print(f"[SEEDDataset] First file: {os.path.basename(fpath)}, {len(trials)} trials")
                    for i, (t, l) in enumerate(zip(trials, trial_labels)):
                        print(f"  trial {i+1}: shape={t.shape}, label={l} ({label_names.get(l, '?')})")
                    first_file = False

                # --- 拼接该文件所有 trial: (62, T_total) ---
                file_data = np.concatenate(trials, axis=1)  # (62, T_total)

                # --- 每个文件单独归一化: 按通道 MinMaxScale 到 [0,1] ---
                file_normed = self._normalize_per_file(file_data)

                # --- 映射到 [-1, 1] ---
                if neg_one_to_one:
                    file_normed = file_normed * 2.0 - 1.0

                # --- 滑动窗口切片 + 分配标签 ---
                offset = 0
                for trial_data, label in zip(trials, trial_labels):
                    T = trial_data.shape[1]  # 该 trial 的时间长度
                    # 在归一化后的数据中取对应段
                    trial_normed = file_normed[:, offset:offset + T]  # (62, T)
                    offset += T

                    if T < window:
                        # padding
                        padded = np.zeros((62, window), dtype=np.float32)
                        padded[:, :T] = trial_normed
                        all_samples.append(padded)
                        all_labels.append(label)
                        continue

                    num_windows = (T - window) // stride + 1
                    for i in range(num_windows):
                        start = i * stride
                        end = start + window
                        seg = trial_normed[:, start:end]  # (62, window)
                        all_samples.append(seg)
                        all_labels.append(label)

        samples = np.stack(all_samples, axis=0).astype(np.float32)  # (N, 62, window)
        labels = np.array(all_labels, dtype=np.int64)
        print(f"[SEEDDataset] Total samples: {samples.shape[0]}, shape per sample: ({samples.shape[1]}, {samples.shape[2]})")
        return samples, labels

    # ================================================================
    #  每个文件单独归一化: 按通道 MinMaxScale 到 [0,1]
    # ================================================================
    def _normalize_per_file(self, data):
        """
        对 (62, T) 数据按通道做 MinMax 归一化到 [0, 1]。
        每个通道独立缩放。
        """
        normed = np.zeros_like(data, dtype=np.float32)
        for ch in range(data.shape[0]):
            ch_data = data[ch, :]
            ch_min = ch_data.min()
            ch_max = ch_data.max()
            if ch_max - ch_min > 1e-8:
                normed[ch, :] = (ch_data - ch_min) / (ch_max - ch_min)
            else:
                normed[ch, :] = 0.0
        return normed

    # ================================================================
    #  提取 trial (按数字顺序)
    # ================================================================
    def _extract_trials(self, mat_data, raw_key_suffix, seed_labels):
        """返回 trials: list of (62, T), labels: list of int"""
        eeg_key_map = {}
        for key in mat_data.keys():
            if key.startswith("_"):
                continue
            if raw_key_suffix not in key and "eeg" not in key.lower():
                continue
            match = re.search(r'(\d+)$', key)
            if match:
                trial_num = int(match.group(1))
                eeg_key_map[trial_num] = key

        trials = []
        labels = []
        if len(eeg_key_map) > 0:
            for trial_idx in range(1, max(eeg_key_map.keys()) + 1):
                if trial_idx not in eeg_key_map:
                    continue
                if trial_idx - 1 >= len(seed_labels):
                    break

                eeg_data = mat_data[eeg_key_map[trial_idx]]
                if not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
                    continue

                # 确保 shape = (62, T)
                if eeg_data.shape[0] != 62 and eeg_data.shape[1] == 62:
                    eeg_data = eeg_data.T
                elif eeg_data.shape[0] != 62:
                    continue

                raw_label = seed_labels[trial_idx - 1]
                mapped_label = LABEL_MAP.get(raw_label, raw_label)

                trials.append(eeg_data.astype(np.float32))
                labels.append(mapped_label)

        return trials, labels

    # ================================================================
    #  工具方法
    # ================================================================
    def _group_files(self, mat_files):
        groups = {}
        for f in mat_files:
            basename = os.path.basename(f).replace(".mat", "")
            parts = basename.split("_")
            try:
                subj_id = int(parts[0])
            except (ValueError, IndexError):
                subj_id = len(groups) + 1
            if subj_id not in groups:
                groups[subj_id] = []
            groups[subj_id].append(f)
        for subj_id in groups:
            groups[subj_id].sort()
        return groups

    def _split(self, data, labels, proportion, seed):
        n = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_size = int(np.ceil(n * proportion))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        np.random.set_state(st0)
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        test_data = data[test_idx] if len(test_idx) > 0 else data[:1]
        test_labels = labels[test_idx] if len(test_idx) > 0 else labels[:1]
        return train_data, train_labels, test_data, test_labels

    def _save_ground_truth(self, *args):
        pass

    # ================================================================
    #  __getitem__ / __len__
    # ================================================================
    def __getitem__(self, ind):
        x = self.samples[ind]  # (62, window)
        x_tensor = torch.from_numpy(x).float()

        if self.conditional:
            label = torch.tensor(self.labels[ind], dtype=torch.long)
            return x_tensor, label
        return x_tensor

    def __len__(self):
        return self.sample_num


def print_dataset_stats(dataset):
    labels = dataset.labels
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset.name} ({dataset.period})")
    print(f"Total samples: {len(dataset)}")
    print(f"Shape per sample: (62, {dataset.window})")
    print(f"Label distribution:")
    for c in range(NUM_CLASSES):
        count = (labels == c).sum()
        pct = count / len(labels) * 100
        name = {0: "negative", 1: "neutral", 2: "positive"}[c]
        print(f"  {name} (label={c}): {count} ({pct:.1f}%)")
    print(f"{'='*50}\n")