"""
train_seed.py — SEED 数据集的 Flow Matching 训练脚本 (自包含)
=============================================================
新增: --unlabeled_finetune 无标签微调, 用测试 trial 适应分布

用法:
    # 正常条件训练
    python train_seed.py --config ./Config/seed_de.yaml --gpu 0 --conditional

    # 无标签微调 (适应测试分布)
    python train_seed.py --config ./Config/seed_de_gen.yaml --gpu 0 \
        --conditional \
        --split_mode trial --subject 2 \
        --train_trials 9,10,11,12,13,14 --test_trials 0,1,2,3,4,5,6,7,8 \
        --checkpoint ./results/s2/checkpoint-best.pt \
        --finetune --max_epochs 1000 \
        --unlabeled_finetune \
        --results_dir ./results/s2_adapted
"""

import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Models.interpretable_diffusion.FMTS import FM_TS
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.Data_utils.seed_dataset import SEEDDataset, print_dataset_stats


class SimpleEMA:
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

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.shadow.load_state_dict(state_dict, strict=strict)


class SEEDTrainer:
    def __init__(self, model, train_loader, config, args):
        self.model = model
        self.train_loader = train_loader
        self.device = next(model.parameters()).device
        self.args = args

        solver_cfg = config.get("solver", {})
        self.max_epochs = solver_cfg.get("max_epochs", 5000)
        if hasattr(args, 'max_epochs') and args.max_epochs is not None:
            self.max_epochs = args.max_epochs
        self.grad_accum = solver_cfg.get("gradient_accumulate_every", 2)
        self.save_cycle = solver_cfg.get("save_cycle", 500)

        lr = float(os.environ.get("hucfg_lr", solver_cfg.get("base_lr", 3e-4)))
        self.optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.96))
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, verbose=True)

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
        print(f"\n{'='*60}")
        print(f"  Training FM-TS on SEED")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Unlabeled finetune: {getattr(self.args, 'unlabeled_finetune', False)}")
        print(f"{'='*60}\n")

        loader = self._infinite_loader()
        self.model.train()
        tic = time.time()
        is_unlabeled = getattr(self.args, 'unlabeled_finetune', False)

        with tqdm(total=self.max_epochs, desc="Training") as pbar:
            while self.step < self.max_epochs:
                total_loss = 0.0
                for _ in range(self.grad_accum):
                    batch = next(loader)

                    if isinstance(batch, (list, tuple)):
                        data = batch[0].to(self.device)
                        # === 改动1: unlabeled_finetune 时不传标签 ===
                        if self.args.conditional and not is_unlabeled:
                            labels = batch[1].to(self.device)
                        else:
                            labels = None
                    else:
                        data = batch.to(self.device)
                        labels = None

                    loss = self.model(data, labels=labels)
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

                if self.step % self.save_cycle == 0:
                    if total_loss < self.best_loss:
                        self.best_loss = total_loss
                        self.save("checkpoint-best.pt")

        self.save("checkpoint-last.pt")
        best_path = self.results_dir / "checkpoint-best.pt"
        if not best_path.exists():
            self.best_loss = total_loss
            self.save("checkpoint-best.pt")
        elapsed = time.time() - tic
        print(f"\nTraining complete. Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    def save(self, filename):
        path = self.results_dir / filename
        torch.save({
            "step": self.step, "model": self.model.generator_state_dict(),
            "ema": self.ema.shadow.generator_state_dict(), "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_generator_state_dict(data["model"])
        self.ema.shadow.load_generator_state_dict(data["ema"])
        try:
            self.optimizer.load_state_dict(data["optimizer"])
        except Exception:
            print(f"  Warning: optimizer state mismatch, using fresh optimizer")
        self.step = data["step"]
        self.best_loss = data.get("best_loss", float("inf"))
        print(f"Loaded checkpoint from {path}, step={self.step}")
        if unexpected:
            print(f"  Ignored {len(unexpected)} unexpected keys (e.g. guidance_classifier)")
        if missing:
            print(f"  Missing {len(missing)} keys (new modules will be randomly initialized)")

    def _sample_anchor_batch(self, batch_labels, batch_size):
        dataset = self.train_loader.dataset
        real_samples = np.asarray(dataset.samples)
        real_labels = np.asarray(dataset.labels).astype(int)
        if real_samples.shape[0] == 0:
            raise ValueError("Cannot use anchored sampling: training dataset is empty")

        if batch_labels is None:
            indices = np.random.randint(0, real_samples.shape[0], size=batch_size)
            anchor_labels = real_labels[indices]
        else:
            if torch.is_tensor(batch_labels):
                batch_label_np = batch_labels.detach().cpu().numpy().astype(int)
            else:
                batch_label_np = np.asarray(batch_labels, dtype=int)
            indices = []
            for label in batch_label_np:
                candidates = np.flatnonzero(real_labels == label)
                if candidates.size == 0:
                    candidates = np.arange(real_samples.shape[0])
                indices.append(np.random.choice(candidates))
            indices = np.asarray(indices, dtype=int)
            anchor_labels = batch_label_np

        anchors = torch.from_numpy(real_samples[indices]).float()
        anchor_labels = torch.as_tensor(anchor_labels, dtype=torch.long)
        return anchors, anchor_labels

    @torch.no_grad()
    def generate(self, num_samples, batch_size=64, labels=None,
                 sample_mode="full", anchor_t_start=0.75):
        if sample_mode not in {"full", "anchored"}:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

        self.ema.shadow.eval()
        all_samples, all_labels = [], []
        total_batches = (num_samples + batch_size - 1) // batch_size
        for start in tqdm(range(0, num_samples, batch_size), total=total_batches, desc="Generating"):
            bs = min(batch_size, num_samples - start)
            batch_labels = None
            record_labels = [0] * bs

            if labels is None:
                pass
            elif isinstance(labels, int):
                batch_labels = torch.full((bs,), labels, dtype=torch.long)
                record_labels = [labels] * bs
            else:
                batch_labels = torch.tensor(labels[start:start+bs], dtype=torch.long)
                record_labels = batch_labels.tolist()

            anchors = None
            if sample_mode == "anchored":
                anchors, anchor_labels = self._sample_anchor_batch(batch_labels, bs)
                if batch_labels is None:
                    record_labels = anchor_labels.tolist()
                    if self.ema.shadow.label_embedding is not None:
                        batch_labels = anchor_labels

            samples = self.ema.shadow.generate_mts(
                batch_size=bs, labels=batch_labels, anchors=anchors,
                t_start=anchor_t_start)
            all_samples.append(samples.cpu().numpy())
            all_labels.extend(record_labels)
        return np.concatenate(all_samples, axis=0)[:num_samples], np.array(all_labels[:num_samples])


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train FM-TS on SEED dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--sample_mode", type=str, default="full",
                        choices=["full", "anchored"],
                        help="生成方式: full=纯噪声起步, anchored=同类真实样本附近起步")
    parser.add_argument("--anchor_t_start", type=float, default=0.75,
                        help="anchored 生成的起始时间, 越大越像真实 anchor, 越小变化越大")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--target_label", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--classifier_weight", type=float, default=0.02)
    parser.add_argument("--cls_epochs", type=int, default=50)
    parser.add_argument("--cfg_dropout", type=float, default=0.15)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--spectral_weight", type=float, default=0.1)
    parser.add_argument("--condition_margin_weight", type=float, default=0.0)
    parser.add_argument("--condition_margin", type=float, default=0.02)
    parser.add_argument("--condition_margin_max_t", type=float, default=0.8)
    parser.add_argument("--condition_margin_batch", type=int, default=64)
    parser.add_argument("--split_mode", type=str, default="session",
                        choices=["random", "session", "trial", "subject"])
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--test_subject", type=int, default=None)
    parser.add_argument("--train_trials", type=str, default=None)
    parser.add_argument("--test_trials", type=str, default=None)
    # === 改动2: 新增参数 ===
    parser.add_argument("--unlabeled_finetune", action="store_true",
                        help="无标签微调: 加载条件模型但训练时不传标签, "
                             "用于在测试数据上适应分布")
    parser.add_argument("--use_test_period", action="store_true",
                        help="加载测试集数据 (配合 --unlabeled_finetune 使用, "
                             "跨session时加载session3, 跨subject时加载留出被试)")
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

    ds_cfg = config["dataloader"]["train_dataset"]["params"]
    ds_cfg["output_dir"] = args.results_dir or f"./results/{ds_cfg.get('name', 'SEED')}"

    if not args.conditional:
        ds_cfg["conditional"] = False
    else:
        ds_cfg["conditional"] = True

    ds_cfg["period"] = "test" if args.use_test_period else "train"
    ds_cfg["split_mode"] = args.split_mode
    if args.subject is not None and args.split_mode != "subject":
        ds_cfg["subjects"] = [args.subject]
        print(f"[被试 {args.subject}] 只使用该被试的数据")
    if args.test_subject is not None:
        ds_cfg["test_subject"] = args.test_subject
    if args.train_trials is not None:
        ds_cfg["train_trials"] = [int(x) for x in args.train_trials.split(",")]
    if args.test_trials is not None:
        ds_cfg["test_trials"] = [int(x) for x in args.test_trials.split(",")]
    if args.target_label is not None:
        ds_cfg["target_label"] = args.target_label
        label_name = {0: "negative", 1: "neutral", 2: "positive"}[args.target_label]
        print(f"[分类别训练] 只使用 '{label_name}' (label={args.target_label}) 的数据")
    train_dataset = SEEDDataset(**ds_cfg)
    print_dataset_stats(train_dataset)

    batch_size = config["dataloader"].get("batch_size", 64)
    drop_last = len(train_dataset) >= batch_size
    if not drop_last:
        print(f"[DataLoader] dataset size {len(train_dataset)} < batch_size {batch_size}; drop_last=False")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config["dataloader"].get("shuffle", True),
        num_workers=4, pin_memory=True, drop_last=drop_last,
    )

    model_cfg = config["model"]["params"]
    if args.conditional:
        model_cfg["num_classes"] = 3
        model_cfg["classifier_weight"] = args.classifier_weight
        model_cfg["cfg_dropout"] = args.cfg_dropout
        model_cfg["spectral_weight"] = args.spectral_weight
        model_cfg["guidance_scale"] = args.guidance_scale
        model_cfg["condition_margin_weight"] = args.condition_margin_weight
        model_cfg["condition_margin"] = args.condition_margin
        model_cfg["condition_margin_max_t"] = args.condition_margin_max_t
        model_cfg["condition_margin_batch"] = args.condition_margin_batch
        print(f"\n[Conditional mode] num_classes=3, classifier_weight={args.classifier_weight}, "
              f"cfg_dropout={args.cfg_dropout}, spectral_weight={args.spectral_weight}, "
              f"guidance_scale={args.guidance_scale}, condition_margin_weight={args.condition_margin_weight}, "
              f"condition_margin={args.condition_margin}, condition_margin_max_t={args.condition_margin_max_t}, "
              f"condition_margin_batch={args.condition_margin_batch}")
    else:
        model_cfg["num_classes"] = 0

    print(f"Model config: seq_length={model_cfg['seq_length']}, feature_size={model_cfg['feature_size']}")
    model = FM_TS(**model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")

    if args.results_dir is None:
        args.results_dir = f"./results/{ds_cfg.get('name', 'SEED')}"

    trainer = SEEDTrainer(model, train_loader, config, args)

    if args.checkpoint is not None:
        trainer.load(args.checkpoint)
        if args.finetune:
            trainer.step = 0
            trainer.best_loss = float("inf")
            print(f"[Finetune] 重置训练步数: step=0, max_epochs={trainer.max_epochs}")

    # === Guidance Classifier: 必须在 checkpoint 加载之后, 避免被覆盖 ===
    if args.conditional and args.classifier_weight > 0 and not args.unlabeled_finetune:
        os.makedirs(args.results_dir, exist_ok=True)
        classifier_path = os.path.join(args.results_dir, "guidance_classifier.pt")
        if os.path.exists(classifier_path):
            model.load_classifier(classifier_path, device=device)
        else:
            real_data = train_dataset.samples
            real_labels = train_dataset.labels
            model.pretrain_classifier(
                real_data, real_labels, epochs=args.cls_epochs, device=device)
            model.save_classifier(classifier_path)

    if not args.sample_only:
        trainer.train()
        # 加载最佳 checkpoint 用于生成 (训练结束时的模型不一定是最优的)
        best_ckpt = os.path.join(args.results_dir, "checkpoint-best.pt")
        if os.path.exists(best_ckpt):
            print(f"[Generate] 加载最佳 checkpoint: {best_ckpt}")
            trainer.load(best_ckpt)

    if args.conditional:
        sensitivity = trainer.ema.shadow.condition_sensitivity()
        print(f"[Condition sensitivity] pairwise_rmse={sensitivity['pairwise_rmse']:.6f}, "
              f"relative_rmse={sensitivity['relative_rmse']:.4f}")

    # 生成样本
    num_gen = args.num_samples if args.num_samples > 0 else len(train_dataset)
    print(f"\nGenerating {num_gen} samples...")
    print(f"  → sample_mode={args.sample_mode}, anchor_t_start={args.anchor_t_start}")

    if args.conditional:
        if args.target_label is not None:
            label_name = {0: "negative", 1: "neutral", 2: "positive"}[args.target_label]
            print(f"  → Generating class: {label_name} (label={args.target_label})")
            samples, labels = trainer.generate(
                num_gen, labels=args.target_label, sample_mode=args.sample_mode,
                anchor_t_start=args.anchor_t_start)
        else:
            print(f"  → Generating all 3 classes equally")
            gen_labels = np.concatenate([
                np.full(num_gen // 3, 0),
                np.full(num_gen // 3, 1),
                np.full(num_gen - 2 * (num_gen // 3), 2),
            ]).astype(int)
            np.random.shuffle(gen_labels)
            samples, labels = trainer.generate(
                num_gen, labels=gen_labels, sample_mode=args.sample_mode,
                anchor_t_start=args.anchor_t_start)
    else:
        samples, labels = trainer.generate(
            num_gen, sample_mode=args.sample_mode, anchor_t_start=args.anchor_t_start)

    CLIP_STD = 5.0
    samples = samples * CLIP_STD

    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    name = ds_cfg.get('name', 'SEED')
    win = train_dataset.window
    bundle_path = os.path.join(save_dir, f"generated_{name}_{win}.npz")
    np.savez(bundle_path, data=samples, labels=labels)

    print(f"\n{'='*50}")
    print(f"  Generated: {samples.shape}, saved to {bundle_path}")
    for c in range(3):
        count = int((labels == c).sum())
        label_name = {0: "negative", 1: "neutral", 2: "positive"}[c]
        print(f"    {label_name}: {count}")
    print(f"{'='*50}\nDone!")


if __name__ == "__main__":
    main()