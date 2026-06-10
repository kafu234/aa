"""
SEED-IV Dataset Loader for FM-TS (DE 特征专用)
================================================
SEED-IV 与 SEED 的关键差异:
  - 4 类情绪: neutral(0), sad(1), fear(2), happy(3)
  - 24 trials / session (SEED 是 15)
  - 标签按 session 不同 (SEED 所有 session 共享同一组标签)
  - 目录结构: eeg_feature_smooth/1/, /2/, /3/ 三个子目录

数据格式:
  eeg_feature_smooth/
    1/                          # session 1
      1_20160518.mat            # 被试1
      2_20150915.mat            # 被试2
      ...
    2/                          # session 2
      ...
    3/                          # session 3
      ...

  每个 .mat 文件含 de_LDS1 ~ de_LDS24, shape = (62, T, 5)
"""

import os
import re
import glob
import torch
import numpy as np
from scipy import io as sio
from torch.utils.data import Dataset
from Utils.Data_utils.group_split import make_recording_group_ids


# ================================================================
#  SEED-IV 标签: 每个 session 的 24 trial 标签不同
# ================================================================
# 0=neutral, 1=sad, 2=fear, 3=happy
SEEDIV_SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

NUM_CLASSES = 4
NUM_TRIALS = 24
LABEL_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


class SEEDIVDataset(Dataset):
    """
    SEED-IV DE 特征加载器.

    输出格式: 每个样本 shape = (62, 5)
        - 62 = EEG 通道数
        - 5 = 频带数 (delta, theta, alpha, beta, gamma)

    归一化策略: 每个 .mat 文件单独做 z-score → clip → [-1, 1].
    """

    def __init__(
        self,
        name="SEEDIV_DE_GEN",
        data_root="./Data/SEED-IV/eeg_feature_smooth",
        data_type="de",
        de_key_prefix="de_LDS",
        subjects=None,
        sessions=None,
        proportion=0.8,
        neg_one_to_one=True,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        conditional=True,
        target_label=None,
        split_mode="session",
        train_sessions=(1, 2),
        test_sessions=(3,),
        train_trials=None,
        test_trials=None,
        test_subject=None,
        **kwargs,
    ):
        super().__init__()
        assert period in ["train", "test"]

        self.name = name
        self.period = period
        self.conditional = conditional
        self.var_num = 5  # DE 特征维度
        self.window = 5   # 兼容接口
        self.split_mode = split_mode
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions
        self.test_subject = test_subject
        # SEED-IV: 24 trials, 默认前 16 训练后 8 测试
        self.train_trials = train_trials if train_trials is not None else list(range(16))
        self.test_trials = test_trials if test_trials is not None else list(range(16, 24))

        # ---- 1. 加载 DE 数据 ----
        all_samples, all_labels, all_sessions, all_trials, all_subjects = \
            self._load_de_data(data_root, de_key_prefix, subjects, sessions)

        # ---- 2. 按标签过滤 ----
        if target_label is not None:
            mask = all_labels == target_label
            all_samples = all_samples[mask]
            all_labels = all_labels[mask]
            all_sessions = all_sessions[mask]
            all_trials = all_trials[mask]
            all_subjects = all_subjects[mask]

        # ---- 3. 训练/测试划分 ----
        if split_mode == "session":
            train_data, train_labels, test_data, test_labels = self._split_by_session(
                all_samples, all_labels, all_sessions,
                train_sessions, test_sessions,
            )
        elif split_mode == "trial":
            train_data, train_labels, test_data, test_labels = self._split_by_trial(
                all_samples, all_labels, all_trials,
                self.train_trials, self.test_trials,
            )
        elif split_mode == "subject":
            train_data, train_labels, test_data, test_labels = self._split_by_subject(
                all_samples, all_labels, all_subjects,
                self.test_subject,
            )
        else:
            train_data, train_labels, test_data, test_labels = self._split_random(
                all_samples, all_labels, proportion, seed
            )

        period_mask = self._get_period_mask(
            split_mode, all_samples.shape[0], all_sessions, all_trials,
            all_subjects, train_sessions, test_sessions, proportion, seed, period,
        )
        if period == "train":
            self.samples = train_data
            self.labels = train_labels
        else:
            self.samples = test_data
            self.labels = test_labels

        self.sample_sessions = all_sessions[period_mask]
        self.sample_trials = all_trials[period_mask]
        self.sample_subjects = all_subjects[period_mask]
        self.sample_groups = make_recording_group_ids(
            self.sample_subjects, self.sample_sessions, self.sample_trials)
        self.sample_num = self.samples.shape[0]

        print(
            f"[SEEDIVDataset] period={period}, samples={self.sample_num}, "
            f"shape=({self.samples.shape[1]}, {self.samples.shape[2]}), "
            f"classes={sorted(np.unique(self.labels).tolist())}"
        )

    def _get_period_mask(self, split_mode, n_samples, sessions, trials, subjects,
                         train_sessions, test_sessions, proportion, seed, period):
        if split_mode == "session":
            selected = train_sessions if period == "train" else test_sessions
            return np.isin(sessions, list(selected))
        if split_mode == "trial":
            selected = self.train_trials if period == "train" else self.test_trials
            return np.isin(trials, list(selected))
        if split_mode == "subject":
            return subjects != self.test_subject if period == "train" else subjects == self.test_subject

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        train_size = int(np.ceil(n_samples * proportion))
        selected = indices[:train_size] if period == "train" else indices[train_size:]
        return selected

    # ================================================================
    #  DE 特征加载
    # ================================================================
    def _load_de_data(self, data_root, de_key_prefix, subjects, sessions):
        """
        加载 SEED-IV DE 特征.

        目录结构:
          data_root/1/  → session 1 的所有被试 .mat 文件
          data_root/2/  → session 2
          data_root/3/  → session 3
        """
        all_samples = []
        all_labels = []
        all_sessions = []
        all_trials = []
        all_subjects = []
        first_file = True

        for sess_id in [1, 2, 3]:
            if sessions is not None and sess_id not in sessions:
                continue

            sess_dir = os.path.join(data_root, str(sess_id))
            if not os.path.isdir(sess_dir):
                # 也尝试不带子目录的结构 (所有文件在同一目录)
                sess_dir = data_root
                if sess_id > 1:
                    continue

            mat_files = sorted(glob.glob(os.path.join(sess_dir, "*.mat")))
            mat_files = [f for f in mat_files
                         if "label" not in os.path.basename(f).lower()
                         and "readme" not in os.path.basename(f).lower()]

            if not mat_files:
                print(f"[SEEDIVDataset] WARNING: No .mat files in {sess_dir}")
                continue

            # 获取该 session 的标签
            session_labels = self._get_session_labels(data_root, sess_id)

            for file_idx, fpath in enumerate(mat_files):
                # 提取被试编号
                subj_id = self._extract_subject_id(fpath, file_idx)

                if subjects is not None and subj_id not in subjects:
                    continue

                try:
                    # 只加载 de_LDS1 ~ de_LDS24, 不读其他特征 (psd, dasm 等)
                    # 否则整个 .mat 全读进内存, 非常慢
                    de_keys = [f"{de_key_prefix}{i+1}" for i in range(NUM_TRIALS)]
                    mat_data = sio.loadmat(fpath, variable_names=de_keys)
                except Exception as e:
                    print(f"[SEEDIVDataset] Error loading {fpath}: {e}")
                    continue

                file_de_list = []
                file_labels = []
                file_trial_ids = []

                for trial_idx in range(NUM_TRIALS):
                    key = f"{de_key_prefix}{trial_idx + 1}"
                    if key not in mat_data:
                        continue

                    de_trial = mat_data[key]  # 期望 (62, T, 5)
                    if de_trial.ndim != 3:
                        continue

                    n_ch, n_t, n_bands = de_trial.shape
                    if n_ch != 62:
                        # 可能是 (T, 62, 5) 或其他排列
                        if de_trial.shape[1] == 62:
                            de_trial = de_trial.transpose(1, 0, 2)
                            n_ch, n_t, n_bands = de_trial.shape
                        else:
                            continue

                    # (62, T, 5) → (T, 62, 5)
                    samples_trial = de_trial.transpose(1, 0, 2).astype(np.float32)
                    file_de_list.append(samples_trial)

                    label = session_labels[trial_idx] if trial_idx < len(session_labels) else 0
                    file_labels.extend([label] * n_t)
                    file_trial_ids.extend([trial_idx] * n_t)

                if not file_de_list:
                    continue

                file_data = np.concatenate(file_de_list, axis=0)  # (N_file, 62, 5)

                if first_file:
                    print(f"[SEEDIVDataset-DE] First file: session{sess_id}/{os.path.basename(fpath)}, "
                          f"{len(file_de_list)} trials, {file_data.shape[0]} samples, "
                          f"shape per sample: {file_data.shape[1:]}")
                    first_file = False

                # 归一化: per-file, per-channel, per-band
                file_data = self._normalize_de_file(file_data)

                all_samples.append(file_data)
                all_labels.extend(file_labels)
                all_sessions.extend([sess_id] * len(file_labels))
                all_trials.extend(file_trial_ids)
                all_subjects.extend([subj_id] * len(file_labels))

        if not all_samples:
            raise FileNotFoundError(
                f"No DE data loaded. Check data_root: {data_root}\n"
                f"Expected structure: data_root/1/*.mat, data_root/2/*.mat, data_root/3/*.mat"
            )

        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_sessions = np.array(all_sessions, dtype=np.int64)
        all_trials = np.array(all_trials, dtype=np.int64)
        all_subjects = np.array(all_subjects, dtype=np.int64)

        # ---- 标签范围检查 & 自动修正 ----
        unique_labels = sorted(np.unique(all_labels).tolist())
        label_min, label_max = unique_labels[0], unique_labels[-1]
        print(f"[SEEDIVDataset-DE] Raw label values: {unique_labels} "
              f"(min={label_min}, max={label_max})")

        # 过滤掉 -1 (rest/baseline/transition 时段, 不是有效情绪类别)
        if -1 in unique_labels:
            n_before = len(all_labels)
            valid_mask = (all_labels >= 0)
            all_samples = all_samples[valid_mask]
            all_labels = all_labels[valid_mask]
            all_sessions = all_sessions[valid_mask]
            all_trials = all_trials[valid_mask]
            all_subjects = all_subjects[valid_mask]
            n_removed = n_before - len(all_labels)
            print(f"[SEEDIVDataset-DE] Filtered out {n_removed} samples with label=-1 "
                  f"(rest/baseline)")
            unique_labels = sorted(np.unique(all_labels).tolist())
            label_min, label_max = unique_labels[0], unique_labels[-1]

        if label_min == 1 and label_max == 4:
            # 1-indexed (1,2,3,4) → 转为 0-indexed (0,1,2,3)
            print(f"[SEEDIVDataset-DE] ⚠ Detected 1-indexed labels (1-4), "
                  f"converting to 0-indexed (0-3)")
            all_labels = all_labels - 1
        elif label_min != 0 or label_max >= NUM_CLASSES:
            # 其他异常范围: 强制 remap 到 0 ~ NUM_CLASSES-1
            label_map = {old: new for new, old in enumerate(unique_labels)}
            print(f"[SEEDIVDataset-DE] ⚠ Unexpected labels {unique_labels}, "
                  f"remapping: {label_map}")
            all_labels = np.array([label_map[l] for l in all_labels], dtype=np.int64)

        # 最终验证
        final_labels = sorted(np.unique(all_labels).tolist())
        assert all(0 <= l < NUM_CLASSES for l in final_labels), \
            f"Labels {final_labels} out of range [0, {NUM_CLASSES}). " \
            f"Check your SEED-IV label.mat!"

        print(f"[SEEDIVDataset-DE] Total: {all_samples.shape[0]} samples, "
              f"shape: {all_samples.shape}, "
              f"classes: {final_labels}, "
              f"subjects: {sorted(np.unique(all_subjects).tolist())}")

        return all_samples, all_labels, all_sessions, all_trials, all_subjects

    # ================================================================
    #  标签获取
    # ================================================================
    def _get_session_labels(self, data_root, sess_id):
        """
        获取指定 session 的标签.

        直接使用硬编码标签 (来自 BCMI 官方文档):
          0=neutral, 1=sad, 2=fear, 3=happy

        不从 label.mat 读取, 因为不同版本的 label.mat 编码不统一:
          - 有的用 [-1, 0, 1, 2] (类似 SEED 的 -1 编码)
          - 有的用 [0, 1, 2, 3]
          - 有的用 [1, 2, 3, 4]
        直接用硬编码最安全.
        """
        if sess_id in SEEDIV_SESSION_LABELS:
            return SEEDIV_SESSION_LABELS[sess_id]

        print(f"[SEEDIVDataset] WARNING: No labels for session {sess_id}, using zeros")
        return [0] * NUM_TRIALS

    # ================================================================
    #  工具方法
    # ================================================================
    @staticmethod
    def _extract_subject_id(fpath, fallback_idx):
        """从文件名提取被试编号, 如 '1_20160518.mat' → 1"""
        basename = os.path.basename(fpath).replace(".mat", "")
        match = re.match(r'^(\d+)', basename)
        if match:
            return int(match.group(1))
        return fallback_idx + 1

    @staticmethod
    def _normalize_de_file(data, clip_std=5.0):
        """
        DE 特征归一化: per-channel per-band z-score → clip → [-1, 1].
        Input:  (N, 62, 5)
        Output: (N, 62, 5)
        """
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        normed = (data - mean) / std
        normed = np.clip(normed, -clip_std, clip_std)
        normed = normed / clip_std
        return normed.astype(np.float32)

    def _split_by_session(self, data, labels, sessions,
                          train_sessions, test_sessions):
        train_mask = np.isin(sessions, list(train_sessions))
        test_mask = np.isin(sessions, list(test_sessions))
        print(f"[Session Split] train sessions={list(train_sessions)}: {train_mask.sum()} samples, "
              f"test sessions={list(test_sessions)}: {test_mask.sum()} samples")
        return data[train_mask], labels[train_mask], data[test_mask], labels[test_mask]

    def _split_by_trial(self, data, labels, trials,
                        train_trials, test_trials):
        """
        按 trial 划分: SEED-IV 每个 session 有 24 个 trial (trial index 0-23).
        默认前 16 个 trial 训练, 后 8 个 trial 测试.
        """
        train_mask = np.isin(trials, list(train_trials))
        test_mask = np.isin(trials, list(test_trials))

        train_data = data[train_mask]
        train_labels = labels[train_mask]
        test_data = data[test_mask]
        test_labels = labels[test_mask]

        print(f"[Trial Split] train trials={list(train_trials)}: {train_mask.sum()} samples, "
              f"test trials={list(test_trials)}: {test_mask.sum()} samples")
        print(f"  Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        print(f"  Test  labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
        return train_data, train_labels, test_data, test_labels

    def _split_by_subject(self, data, labels, subjects, test_subject):
        test_mask = (subjects == test_subject)
        train_mask = ~test_mask
        train_subjs = sorted(np.unique(subjects[train_mask]).tolist())
        print(f"[Subject Split] train: {len(train_subjs)} subjects ({train_mask.sum()} samples), "
              f"test: subject {test_subject} ({test_mask.sum()} samples)")
        return data[train_mask], labels[train_mask], data[test_mask], labels[test_mask]

    def _split_random(self, data, labels, proportion, seed):
        n = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_size = int(np.ceil(n * proportion))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        np.random.set_state(st0)
        return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]

    # ================================================================
    #  __getitem__ / __len__
    # ================================================================
    def __getitem__(self, ind):
        x = self.samples[ind]
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
    print(f"Shape per sample: (62, 5)")
    print(f"Label distribution:")
    for c in range(NUM_CLASSES):
        count = (labels == c).sum()
        pct = count / len(labels) * 100
        print(f"  {LABEL_NAMES[c]} (label={c}): {count} ({pct:.1f}%)")
    print(f"{'='*50}\n")