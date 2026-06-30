"""SEED LDS-smoothed differential-entropy feature loader for FM-TS."""

import os
import glob
import torch
import numpy as np
from scipy import io as sio
from torch.utils.data import Dataset
from Utils.Data_utils.group_split import make_recording_group_ids

# SEED 标准标签 (15 trials)
SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}  # negative=0, neutral=1, positive=2
NUM_CLASSES = 3


class SEEDDataset(Dataset):
    """Load official SEED DE features with sample shape ``(62, 5)``."""

    def __init__(
        self,
        name="SEED_DE",
        data_root="./Data/datasets/SEED/ExtractedFeatures",
        de_key_prefix="de_LDS",
        subjects=None,
        sessions=None,
        window=1,
        proportion=0.8,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        conditional=True,
        target_label=None,
        split_mode="random",        # ← "random", "session", "trial", 或 "subject"
        train_sessions=(0, 1),      # ← session 划分用
        test_sessions=(2,),         # ← session 划分用
        train_trials=None,          # ← trial 划分用, 如 list(range(9))
        test_trials=None,           # ← trial 划分用, 如 list(range(9,15))
        test_subject=None,          # ← subject 划分用, 如 15 (LOSO)
    ):
        super(SEEDDataset, self).__init__()
        assert period in ["train", "test"]

        self.name = name
        self.window = int(window)
        self.period = period
        self.conditional = conditional
        self.var_num = 5
        self.split_mode = split_mode
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions
        self.train_trials = train_trials if train_trials is not None else list(range(9))
        self.test_trials = test_trials if test_trials is not None else list(range(9, 15))
        self.test_subject = test_subject

        # ---- 1. 加载标签 ----
        seed_labels = self._load_labels(data_root)

        # ---- 2. 加载 DE 特征 ----
        all_samples, all_labels, all_sessions, all_trials, all_subjects = (
            self._load_de_data(
                data_root, seed_labels, de_key_prefix, subjects, sessions
            )
        )

        # ---- 3. 按标签过滤 ----
        if target_label is not None:
            mask = all_labels == target_label
            all_samples = all_samples[mask]
            all_labels = all_labels[mask]
            all_sessions = all_sessions[mask]
            all_trials = all_trials[mask]
            all_subjects = all_subjects[mask]

        # ---- 4. 训练/测试划分 ----
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
            train_data, train_labels, test_data, test_labels = self._split(
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

        loaded_subjects = sorted(np.unique(self.sample_subjects).astype(int).tolist())
        print(
            f"[SEEDDataset-DE] split_mode={self.split_mode}, "
            f"subjects={loaded_subjects}, period={period}"
        )
        print(
            f"[SEEDDataset] samples={self.sample_num}, "
            f"shape=({self.samples.shape[1]}, {self.samples.shape[2]}), "
            f"classes={np.unique(self.labels).tolist()}"
        )

    def _get_period_mask(self, split_mode, n_samples, sessions, trials, subjects,
                         train_sessions, test_sessions, proportion, seed, period):
        if split_mode == "session":
            selected = train_sessions if period == "train" else test_sessions
            return np.isin(sessions, selected)
        if split_mode == "trial":
            selected = self.train_trials if period == "train" else self.test_trials
            return np.isin(trials, selected)
        if split_mode == "subject":
            return subjects != self.test_subject if period == "train" else subjects == self.test_subject

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        train_size = int(np.ceil(n_samples * proportion))
        selected = indices[:train_size] if period == "train" else indices[train_size:]
        if len(selected) == 0:
            selected = np.array([0])
        return selected

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
    #  DE 特征加载 (从 SEED ExtractedFeatures 目录)
    # ================================================================
    def _load_de_data(self, data_root, seed_labels, de_key_prefix, subjects, sessions):
        """
        加载 SEED 官方 DE 特征 (含 LDS 平滑).

        ExtractedFeatures/*.mat 中:
          de_LDS1 ~ de_LDS15: 每个 trial 的 DE 特征, shape = (62, T, 5)
          62 通道, T 个时间窗, 5 个频带 (delta, theta, alpha, beta, gamma)

        每个时间窗 (62, 5) 作为一个样本.
        归一化: 每个文件单独做 z-score → clip → 映射到 [-1, 1].
        """
        import glob
        from scipy import io as sio

        mat_files = sorted(glob.glob(os.path.join(data_root, "*.mat")))
        mat_files = [f for f in mat_files if "label" not in os.path.basename(f).lower()]
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {data_root}")

        file_groups = self._group_files(mat_files)

        all_samples = []
        all_labels = []
        all_sessions = []
        all_trials = []
        all_subjects = []
        first_file = True

        for subj_idx, subj_sessions in file_groups.items():
            if subjects is not None and subj_idx not in subjects:
                continue

            for sess_idx, fpath in enumerate(subj_sessions):
                if sessions is not None and sess_idx not in sessions:
                    continue

                mat_data = sio.loadmat(fpath)

                file_de_list = []
                file_labels = []
                file_trial_ids = []

                for trial_idx in range(15):
                    key = f"{de_key_prefix}{trial_idx + 1}"
                    if key not in mat_data:
                        continue

                    de_trial = mat_data[key]  # (62, T, 5)
                    if de_trial.ndim != 3:
                        continue

                    n_ch, n_t, n_bands = de_trial.shape
                    samples_trial = de_trial.transpose(1, 0, 2).astype(np.float32)  # (T, 62, 5)
                    file_de_list.append(samples_trial)

                    label = seed_labels[trial_idx] if trial_idx < len(seed_labels) else 0
                    label = LABEL_MAP.get(label, label)  # -1→0, 0→1, 1→2
                    file_labels.extend([label] * n_t)
                    file_trial_ids.extend([trial_idx] * n_t)

                if not file_de_list:
                    continue

                file_data = np.concatenate(file_de_list, axis=0)  # (N_file, 62, 5)

                if first_file:
                    print(f"[SEEDDataset-DE] First file: {os.path.basename(fpath)}, "
                          f"{len(file_de_list)} trials, {file_data.shape[0]} samples, "
                          f"shape per sample: {file_data.shape[1:]}")
                    first_file = False

                # 归一化: per-file, per-channel, per-band
                file_data = self._normalize_de_file(file_data)

                all_samples.append(file_data)
                all_labels.extend(file_labels)
                all_sessions.extend([sess_idx] * len(file_labels))
                all_trials.extend(file_trial_ids)
                all_subjects.extend([subj_idx] * len(file_labels))

        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_sessions = np.array(all_sessions, dtype=np.int64)
        all_trials = np.array(all_trials, dtype=np.int64)
        all_subjects = np.array(all_subjects, dtype=np.int64)

        print(f"[SEEDDataset-DE] Total: {all_samples.shape[0]} samples, "
              f"shape: {all_samples.shape}, classes: {sorted(np.unique(all_labels).tolist())}")

        return all_samples, all_labels, all_sessions, all_trials, all_subjects

    @staticmethod
    def _normalize_de_file(data, clip_std=5.0):
        """
        DE 特征归一化: per-channel per-band z-score → clip → [-1, 1].
        Input:  (N, 62, 5)
        Output: (N, 62, 5)
        """
        mean = data.mean(axis=0, keepdims=True)   # (1, 62, 5)
        std = data.std(axis=0, keepdims=True)     # (1, 62, 5)
        std[std < 1e-8] = 1.0
        normed = (data - mean) / std
        normed = np.clip(normed, -clip_std, clip_std)
        normed = normed / clip_std
        return normed.astype(np.float32)

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

    def _split_by_session(self, data, labels, sessions,
                          train_sessions=(0, 1), test_sessions=(2,)):
        """
        按 session 划分: 前两个 session 训练, 第三个 session 测试.
        这是 SEED 数据集的标准评估协议 (within-subject cross-session).
        """
        train_mask = np.isin(sessions, train_sessions)
        test_mask = np.isin(sessions, test_sessions)

        train_data = data[train_mask]
        train_labels = labels[train_mask]
        test_data = data[test_mask]
        test_labels = labels[test_mask]

        print(f"[Session Split] train sessions={train_sessions}: {train_mask.sum()} samples, "
              f"test sessions={test_sessions}: {test_mask.sum()} samples")
        return train_data, train_labels, test_data, test_labels

    def _split_by_trial(self, data, labels, trials,
                        train_trials=list(range(9)), test_trials=list(range(9, 15))):
        """
        按 trial 划分: 每个 session 的前 9 个 trial 训练, 后 6 个 trial 测试.
        SEED 每个 session 有 15 个 trial (trial index 0-14).
        """
        train_mask = np.isin(trials, train_trials)
        test_mask = np.isin(trials, test_trials)

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
        """
        跨被试划分 (LOSO): 指定被试作为测试集, 其余被试作为训练集.
        test_subject: int, 测试被试编号 (如 15).
        """
        test_mask = (subjects == test_subject)
        train_mask = ~test_mask

        train_data = data[train_mask]
        train_labels = labels[train_mask]
        test_data = data[test_mask]
        test_labels = labels[test_mask]

        train_subjs = sorted(np.unique(subjects[train_mask]).tolist())
        print(f"[Subject Split] train: {len(train_subjs)} subjects ({train_mask.sum()} samples), "
              f"test: subject {test_subject} ({test_mask.sum()} samples)")
        print(f"  Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        print(f"  Test  labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
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
    print(f"Shape per sample: {tuple(dataset.samples.shape[1:])}")
    print(f"Split mode: {dataset.split_mode}")
    print(f"Label distribution:")
    for c in range(NUM_CLASSES):
        count = (labels == c).sum()
        pct = count / len(labels) * 100
        name = {0: "negative", 1: "neutral", 2: "positive"}[c]
        print(f"  {name} (label={c}): {count} ({pct:.1f}%)")
    print(f"{'='*50}\n")
