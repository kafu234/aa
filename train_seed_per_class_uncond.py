"""
train_seed_per_class_uncond.py
================================
为 SEED 的每个情绪类别分别训练一个无条件 FM-TS 生成模型，避免单一条件模型发生
label condition collapse。

典型用法：

1) subject 1, front-9 trial，从零训练三个类别无条件模型，并合并生成数据：
python train_seed_per_class_uncond.py \
    --config ./Config/seed_raw.yaml \
    --gpu 0 \
    --split_mode trial \
    --subject 1 \
    --train_trials 0,1,2,3,4,5,6,7,8 \
    --test_trials 9,10,11,12,13,14 \
    --spectral_weight 0.1 \
    --max_epochs 1000 \
    --num_samples_per_class 1000 \
    --results_dir ./results/gen_1s_s1_front9_per_class_uncond

2) 所有被试 front-9 先训练三个类别全局无条件模型：
python train_seed_per_class_uncond.py \
    --config ./Config/seed_raw.yaml \
    --gpu 0 \
    --split_mode trial \
    --train_trials 0,1,2,3,4,5,6,7,8 \
    --test_trials 9,10,11,12,13,14 \
    --spectral_weight 0.1 \
    --max_epochs 10000 \
    --num_samples_per_class 1 \
    --results_dir ./results/gen_1s_global_front9_per_class_uncond

3) 从全局三个类别模型微调到 subject 1 front-9：
python train_seed_per_class_uncond.py \
    --config ./Config/seed_raw.yaml \
    --gpu 0 \
    --split_mode trial \
    --subject 1 \
    --train_trials 0,1,2,3,4,5,6,7,8 \
    --test_trials 9,10,11,12,13,14 \
    --spectral_weight 0.1 \
    --max_epochs 1000 \
    --num_samples_per_class 1000 \
    --finetune \
    --checkpoint_root ./results/gen_1s_global_front9_per_class_uncond \
    --results_dir ./results/gen_1s_s1_front9_per_class_uncond_ft
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Models.interpretable_diffusion.FMTS import FM_TS
from Utils.Data_utils.seed_dataset import SEEDDataset, print_dataset_stats


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
CLIP_STD = 5.0


class SimpleEMA:
    """轻量 EMA，不依赖 ema-pytorch。"""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.shadow.load_state_dict(state_dict, strict=strict)


class SEEDTrainer:
    """无条件 per-class 训练器。"""

    def __init__(self, model, train_loader, config, args):
        self.model = model
        self.train_loader = train_loader
        self.device = next(model.parameters()).device
        self.args = args

        solver_cfg = config.get("solver", {})
        self.max_epochs = solver_cfg.get("max_epochs", 5000)
        if args.max_epochs is not None:
            self.max_epochs = args.max_epochs
        self.grad_accum = solver_cfg.get("gradient_accumulate_every", 1)
        self.save_cycle = solver_cfg.get("save_cycle", 500)

        lr = float(os.environ.get("hucfg_lr", solver_cfg.get("base_lr", 3e-4)))
        self.optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.96))
        try:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=500, verbose=True
            )
        except TypeError:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=500
            )

        ema_cfg = solver_cfg.get("ema", {})
        self.ema = SimpleEMA(model, decay=ema_cfg.get("decay", 0.995))
        self.ema_update_every = ema_cfg.get("update_interval", 10)

        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_loss = float("inf")

    def _infinite_loader(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def train(self):
        if len(self.train_loader) == 0:
            raise ValueError(
                "DataLoader 为空。请减小 batch_size，或设置 drop_last=False。"
            )

        print(f"\n{'='*60}")
        print("  Training per-class unconditional FM-TS")
        print(f"  Device: {self.device}")
        print(f"  Steps : {self.max_epochs}")
        print(f"  Output: {self.results_dir}")
        print(f"{'='*60}\n")

        loader = self._infinite_loader()
        self.model.train()
        tic = time.time()

        with tqdm(total=self.max_epochs, desc="Training") as pbar:
            while self.step < self.max_epochs:
                total_loss = 0.0
                for _ in range(self.grad_accum):
                    batch = next(loader)
                    if isinstance(batch, (list, tuple)):
                        data = batch[0].to(self.device)
                    else:
                        data = batch.to(self.device)

                    # 无条件模型：不传 labels
                    loss = self.model(data, labels=None)
                    loss = loss / self.grad_accum
                    loss.backward()
                    total_loss += loss.item()

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step(total_loss)

                self.step += 1
                if self.step % self.ema_update_every == 0:
                    self.ema.update(self.model)

                pbar.set_description(f"loss: {total_loss:.6f}")
                pbar.update(1)

                if self.step % self.save_cycle == 0 and total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self.save("checkpoint-best.pt")

        # Preserve the final state for resuming, but generate from the best EMA.
        self.save("checkpoint-last.pt")
        best_path = self.results_dir / "checkpoint-best.pt"
        if not best_path.exists():
            self.best_loss = total_loss
            self.save("checkpoint-best.pt")
        self.load(best_path)
        elapsed = time.time() - tic
        print(f"Training complete. Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    def save(self, filename):
        path = self.results_dir / filename
        torch.save(
            {
                "step": self.step,
                "model": self.model.generator_state_dict(),
                "ema": self.ema.shadow.generator_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )

    def load(self, path, finetune=False):
        data = torch.load(path, map_location=self.device)
        self.model.load_generator_state_dict(data["model"])
        self.ema.shadow.load_generator_state_dict(data["ema"])
        if not finetune:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except Exception:
                print("  Warning: optimizer state mismatch, using fresh optimizer")
            self.step = data.get("step", 0)
            self.best_loss = data.get("best_loss", float("inf"))
        else:
            self.step = 0
            self.best_loss = float("inf")
        print(f"Loaded checkpoint from {path}, finetune={finetune}, step={self.step}")

    @torch.no_grad()
    def generate(self, num_samples, batch_size=64):
        self.ema.shadow.eval()
        all_samples = []
        total_batches = (num_samples + batch_size - 1) // batch_size
        for start in tqdm(range(0, num_samples, batch_size), total=total_batches, desc="Generating"):
            bs = min(batch_size, num_samples - start)
            samples = self.ema.shadow.generate_mts(batch_size=bs, labels=None)
            all_samples.append(samples.cpu().numpy())
        return np.concatenate(all_samples, axis=0)[:num_samples]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_int_list(s):
    if s is None or s == "":
        return None
    return [int(x) for x in s.split(",")]


def apply_common_seed_dataset_args(ds_cfg, args, target_label):
    ds_cfg = deepcopy(ds_cfg)
    ds_cfg["period"] = "train"
    ds_cfg["conditional"] = False  # 每类无条件模型，dataset 也不用返回 label
    ds_cfg["target_label"] = target_label
    ds_cfg["split_mode"] = args.split_mode
    if args.subject is not None and args.split_mode != "subject":
        ds_cfg["subjects"] = [args.subject]
    if args.test_subject is not None:
        ds_cfg["test_subject"] = args.test_subject
    train_trials = parse_int_list(args.train_trials)
    test_trials = parse_int_list(args.test_trials)
    if train_trials is not None:
        ds_cfg["train_trials"] = train_trials
    if test_trials is not None:
        ds_cfg["test_trials"] = test_trials
    return ds_cfg


def build_unconditional_model(config, args):
    model_cfg = deepcopy(config["model"]["params"])
    model_cfg["num_classes"] = 0
    # spectral_weight 仍然对无条件模型有效；classifier/cfg/guidance 不使用
    model_cfg["classifier_weight"] = 0.0
    model_cfg["cfg_dropout"] = 0.0
    model_cfg["guidance_scale"] = 1.0
    model_cfg["spectral_weight"] = args.spectral_weight
    return FM_TS(**model_cfg)


def train_or_load_one_class(label, config, args, device):
    label_name = LABEL_NAMES[label]
    class_dir = Path(args.results_dir) / f"class_{label}_{label_name}"
    class_dir.mkdir(parents=True, exist_ok=True)

    ds_base = config["dataloader"]["train_dataset"]["params"]
    ds_cfg = apply_common_seed_dataset_args(ds_base, args, target_label=label)

    print(f"\n{'#'*70}")
    print(f"# Class {label}: {label_name} — unconditional model")
    print(f"# Output: {class_dir}")
    print(f"{'#'*70}")

    train_dataset = SEEDDataset(**ds_cfg)
    print_dataset_stats(train_dataset)
    if len(train_dataset) == 0:
        raise ValueError(f"类别 {label} ({label_name}) 没有训练样本，请检查 trial/subject/split 设置。")

    batch_size = config["dataloader"].get("batch_size", 64)
    # 单类数据可能较少，drop_last=False 防止 loader 为空
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config["dataloader"].get("shuffle", True),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_unconditional_model(config, args).to(device)
    trainer_args = SimpleNamespace(
        results_dir=str(class_dir),
        max_epochs=args.max_epochs,
    )
    trainer = SEEDTrainer(model, train_loader, config, trainer_args)

    ckpt_path = class_dir / "checkpoint-best.pt"
    if args.checkpoint_root is not None:
        src_ckpt = Path(args.checkpoint_root) / f"class_{label}_{label_name}" / "checkpoint-best.pt"
        if not src_ckpt.exists():
            raise FileNotFoundError(f"找不到类别 {label} 的 checkpoint: {src_ckpt}")
        trainer.load(str(src_ckpt), finetune=args.finetune)
    elif args.sample_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"sample_only=True 但找不到 checkpoint: {ckpt_path}")
        trainer.load(str(ckpt_path), finetune=False)

    if not args.sample_only:
        trainer.train()
    elif ckpt_path.exists() and args.checkpoint_root is None:
        trainer.load(str(ckpt_path), finetune=False)

    return trainer, train_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train one unconditional FM-TS model per SEED class and merge generated samples."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=None, help="每个类别模型训练 step 数")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--split_mode", type=str, default="trial", choices=["random", "session", "trial", "subject"])
    parser.add_argument("--subject", type=int, default=None, help="指定被试编号；不指定则使用所有被试")
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    parser.add_argument("--spectral_weight", type=float, default=0.1)
    parser.add_argument("--num_samples_per_class", type=int, default=1000)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--sample_only", action="store_true", help="只从每个 class 子目录 checkpoint 生成")
    parser.add_argument("--checkpoint_root", type=str, default=None,
                        help="用于 per-class 微调/生成的根目录，里面应有 class_0_negative 等子目录")
    parser.add_argument("--finetune", action="store_true",
                        help="从 checkpoint_root 加载每类模型后重置 step/optimizer 进行微调")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    os.environ.setdefault("hucfg_num_steps", "100")
    os.environ.setdefault("hucfg_t_sampling", "logitnorm")
    os.environ.setdefault("hucfg_attention_rope_use", "-1")
    os.environ.setdefault("hucfg_Kscale", "0.03")
    os.environ.setdefault("hucfg_lr", str(config["solver"].get("base_lr", 3e-4)))

    out_root = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    generated_data = []
    generated_labels = []
    window = config["dataloader"]["train_dataset"]["params"].get("window", config["model"]["params"]["feature_size"])
    dataset_name = config["dataloader"]["train_dataset"]["params"].get("name", "SEED_RAW")

    for label in [0, 1, 2]:
        trainer, _ = train_or_load_one_class(label, config, args, device)
        n = args.num_samples_per_class
        if n > 0:
            samples = trainer.generate(n, batch_size=args.sample_batch_size)
            # 模型输出 [-1, 1]，保存时恢复到 z-score 空间，与原 train_seed.py 保持一致
            samples = samples * CLIP_STD
            labels = np.full(n, label, dtype=np.int64)
            generated_data.append(samples)
            generated_labels.append(labels)

    if args.num_samples_per_class > 0:
        data = np.concatenate(generated_data, axis=0).astype(np.float32)
        labels = np.concatenate(generated_labels, axis=0).astype(np.int64)

        # 打乱保存，避免类别块状排列影响后续 loader
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(labels))
        data = data[idx]
        labels = labels[idx]

        save_path = out_root / f"generated_{dataset_name}_{window}_per_class_uncond.npz"
        np.savez(save_path, data=data, labels=labels)

        print(f"\n{'='*60}")
        print("  Per-class unconditional generated results")
        print(f"  Shape: {data.shape}")
        print(f"  File : {save_path}")
        for c in [0, 1, 2]:
            print(f"    {LABEL_NAMES[c]} (label={c}): {(labels == c).sum()} samples")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
