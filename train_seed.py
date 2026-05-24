"""
train_seed.py — SEED 数据集的 Flow Matching 训练脚本 (自包含)
=============================================================
用法:
    # DE 特征 (无条件生成)
    python train_seed.py --config ./Config/seed_de.yaml --gpu 0

    # DE 特征 (条件生成，需要情绪标签)
    python train_seed.py --config ./Config/seed_de.yaml --gpu 0 --conditional

    # 原始 EEG
    python train_seed.py --config ./Config/seed_raw.yaml --gpu 0

    # 只生成样本 (加载已训练模型)
    python train_seed.py --config ./Config/seed_de.yaml --gpu 0 --sample_only --checkpoint ./results/seed_de/checkpoint-best.pt

不依赖 Diffusion-TS 的 engine 模块，直接使用 FM-TS 模型。
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


# ============================================================
#  EMA (指数移动平均)
# ============================================================
class SimpleEMA:
    """轻量 EMA，不依赖 ema-pytorch 库。"""

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

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ============================================================
#  训练器
# ============================================================
class SEEDTrainer:
    def __init__(self, model, train_loader, config, args):
        self.model = model
        self.train_loader = train_loader
        self.device = next(model.parameters()).device
        self.args = args

        # 从 config 读取超参数
        solver_cfg = config.get("solver", {})
        self.max_epochs = solver_cfg.get("max_epochs", 5000)
        if hasattr(args, 'max_epochs') and args.max_epochs is not None:
            self.max_epochs = args.max_epochs
        self.grad_accum = solver_cfg.get("gradient_accumulate_every", 2)
        self.save_cycle = solver_cfg.get("save_cycle", 500)

        # 优化器
        lr = float(os.environ.get("hucfg_lr", solver_cfg.get("base_lr", 3e-4)))
        self.optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.96))
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )

        # EMA
        ema_cfg = solver_cfg.get("ema", {})
        self.ema = SimpleEMA(model, decay=ema_cfg.get("decay", 0.995))
        self.ema_update_every = ema_cfg.get("update_interval", 10)

        # 输出目录
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_loss = float("inf")

    def _infinite_loader(self):
        """无限循环 DataLoader。"""
        while True:
            for batch in self.train_loader:
                yield batch

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training FM-TS on SEED")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Grad accumulation: {self.grad_accum}")
        print(f"  Results dir: {self.results_dir}")
        print(f"{'='*60}\n")

        loader = self._infinite_loader()
        self.model.train()
        tic = time.time()

        with tqdm(total=self.max_epochs, desc="Training") as pbar:
            while self.step < self.max_epochs:
                total_loss = 0.0

                for _ in range(self.grad_accum):
                    batch = next(loader)

                    # 处理 conditional 和 unconditional 两种情况
                    if isinstance(batch, (list, tuple)):
                        data = batch[0].to(self.device)
                        labels = batch[1].to(self.device) if self.args.conditional else None
                    else:
                        data = batch.to(self.device)
                        labels = None

                    # FM-TS forward: 输入 (batch, seq_length, feature_size)
                    loss = self.model(data, labels=labels)
                    loss = loss / self.grad_accum
                    loss.backward()
                    total_loss += loss.item()

                # 梯度裁剪 + 更新
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step(total_loss)

                # EMA 更新
                self.step += 1
                if self.step % self.ema_update_every == 0:
                    self.ema.update(self.model)

                # 日志
                pbar.set_description(f"loss: {total_loss:.6f}")
                pbar.update(1)

                # 保存 checkpoint (只保留 best)
                if self.step % self.save_cycle == 0:
                    if total_loss < self.best_loss:
                        self.best_loss = total_loss
                        self.save("checkpoint-best.pt")

        # 训练结束，覆盖保存最终模型
        self.save("checkpoint-best.pt")
        elapsed = time.time() - tic
        print(f"\nTraining complete. Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    def save(self, filename):
        path = self.results_dir / filename
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.step = data["step"]
        self.best_loss = data.get("best_loss", float("inf"))
        print(f"Loaded checkpoint from {path}, step={self.step}")

    @torch.no_grad()
    def generate(self, num_samples, batch_size=64, labels=None):
        """
        用 EMA 模型生成样本。
        labels: None=无条件, int=指定类别, ndarray=每个样本的类别
        """
        self.ema.shadow.eval()
        all_samples = []
        all_labels = []

        total_batches = (num_samples + batch_size - 1) // batch_size
        for start in tqdm(range(0, num_samples, batch_size), total=total_batches, desc="Generating"):
            bs = min(batch_size, num_samples - start)

            if labels is None:
                # 无条件生成
                samples = self.ema.shadow.generate_mts(batch_size=bs, labels=None)
                all_labels.extend([0] * bs)
            elif isinstance(labels, int):
                # 生成指定类别
                samples = self.ema.shadow.generate_mts(batch_size=bs, labels=labels)
                all_labels.extend([labels] * bs)
            else:
                # 按每个样本的标签生成
                batch_labels = torch.tensor(labels[start:start+bs], dtype=torch.long)
                samples = self.ema.shadow.generate_mts(batch_size=bs, labels=batch_labels)
                all_labels.extend(batch_labels.tolist())

            all_samples.append(samples.cpu().numpy())

        return np.concatenate(all_samples, axis=0)[:num_samples], np.array(all_labels[:num_samples])


# ============================================================
#  配置加载
# ============================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
#  主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train FM-TS on SEED dataset")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="覆盖 config 里的 max_epochs (微调时用, 如 500)")
    parser.add_argument("--finetune", action="store_true",
                        help="微调模式: 加载 checkpoint 权重但重置训练步数")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--conditional", action="store_true", help="Enable conditional generation")
    parser.add_argument("--sample_only", action="store_true", help="Only generate samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to generate (0=same as training set)")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--target_label", type=int, default=None, choices=[0, 1, 2],
                        help="Generate only this emotion class: 0=negative, 1=neutral, 2=positive. "
                             "Default: generate all three classes equally.")
    # Classifier Guidance
    parser.add_argument("--classifier_weight", type=float, default=0.02,
                        help="分类器引导损失权重 (0=不用, 默认 0.02)")
    parser.add_argument("--cls_epochs", type=int, default=50,
                        help="引导分类器预训练轮数")
    # Classifier-Free Guidance + 频谱损失
    parser.add_argument("--cfg_dropout", type=float, default=0.15,
                        help="CFG 训练时随机丢弃标签概率 (默认 0.15)")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="CFG 推理时引导强度 (默认 2.0)")
    parser.add_argument("--spectral_weight", type=float, default=0.1,
                        help="频谱一致性损失权重 (默认 0.1)")
    parser.add_argument("--split_mode", type=str, default="session",
                        choices=["random", "session", "trial"],
                        help="session=前2session训/第3session测, trial=每session前9trial训/后6trial测")
    parser.add_argument("--subject", type=int, default=None,
                        help="指定被试编号 (1-15), 不指定则使用所有被试")
    parser.add_argument("--train_trials", type=str, default=None,
                        help="训练用 trial 编号, 如 '0,1,2,3,4,5,6,7,8'")
    parser.add_argument("--test_trials", type=str, default=None,
                        help="测试用 trial 编号, 如 '9,10,11,12,13,14'")
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载配置
    config = load_config(args.config)

    # 设置环境变量 (FM-TS 通过 os.environ 读取部分参数)
    os.environ.setdefault("hucfg_num_steps", "100")
    os.environ.setdefault("hucfg_t_sampling", "logitnorm")
    os.environ.setdefault("hucfg_attention_rope_use", "-1")
    os.environ.setdefault("hucfg_Kscale", "0.03")
    os.environ.setdefault("hucfg_lr", str(config["solver"].get("base_lr", 3e-4)))

    # ---- 创建数据集 ----
    ds_cfg = config["dataloader"]["train_dataset"]["params"]
    ds_cfg["output_dir"] = args.results_dir or f"./results/{ds_cfg.get('name', 'SEED')}"

    # 如果命令行没指定 conditional，从 config 读取
    if not args.conditional:
        ds_cfg["conditional"] = ds_cfg.get("conditional", False)
        # 无条件模式: 确保 conditional=False
        ds_cfg["conditional"] = False
    else:
        ds_cfg["conditional"] = True

    ds_cfg["period"] = "train"  # 确保是训练集
    ds_cfg["split_mode"] = args.split_mode
    if args.subject is not None:
        ds_cfg["subjects"] = [args.subject]
        print(f"[被试 {args.subject}] 只使用该被试的数据")
    if args.train_trials is not None:
        ds_cfg["train_trials"] = [int(x) for x in args.train_trials.split(",")]
    if args.test_trials is not None:
        ds_cfg["test_trials"] = [int(x) for x in args.test_trials.split(",")]
    train_dataset = SEEDDataset(**ds_cfg)
    print_dataset_stats(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"].get("batch_size", 64),
        shuffle=config["dataloader"].get("shuffle", True),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 创建模型 ----
    model_cfg = config["model"]["params"]

    # 条件生成: 设置 num_classes
    if args.conditional:
        model_cfg["num_classes"] = 3  # SEED: negative/neutral/positive
        model_cfg["classifier_weight"] = args.classifier_weight
        model_cfg["cfg_dropout"] = args.cfg_dropout
        model_cfg["spectral_weight"] = args.spectral_weight
        model_cfg["guidance_scale"] = args.guidance_scale
        print(f"\n[Conditional mode] num_classes=3, classifier_weight={args.classifier_weight}, "
              f"cfg_dropout={args.cfg_dropout}, spectral_weight={args.spectral_weight}, "
              f"guidance_scale={args.guidance_scale}")
    else:
        model_cfg["num_classes"] = 0
    print(f"Model config: seq_length={model_cfg['seq_length']}, feature_size={model_cfg['feature_size']}")

    model = FM_TS(**model_cfg).to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")

    # ---- Classifier Guidance: 预训练/加载引导分类器 ----
    if args.conditional and args.classifier_weight > 0:
        if args.results_dir is None:
            args.results_dir = f"./results/{ds_cfg.get('name', 'SEED')}"
        os.makedirs(args.results_dir, exist_ok=True)
        classifier_path = os.path.join(args.results_dir, "guidance_classifier.pt")

        if os.path.exists(classifier_path):
            model.load_classifier(classifier_path, device=device)
        else:
            real_data = train_dataset.samples   # (N, 62, 200)
            real_labels = train_dataset.labels  # (N,)
            model.pretrain_classifier(
                real_data, real_labels,
                epochs=args.cls_epochs, device=device,
            )
            model.save_classifier(classifier_path)

    # ---- 训练器 ----
    if args.results_dir is None:
        args.results_dir = f"./results/{ds_cfg.get('name', 'SEED')}"

    trainer = SEEDTrainer(model, train_loader, config, args)

    # 加载 checkpoint
    if args.checkpoint is not None:
        trainer.load(args.checkpoint)
        if args.finetune:
            trainer.step = 0
            trainer.best_loss = float("inf")
            print(f"[Finetune] 重置训练步数: step=0, max_epochs={trainer.max_epochs}")

    # ---- 训练或生成 ----
    if not args.sample_only:
        trainer.train()

    # ---- 生成样本 ----
    num_gen = args.num_samples if args.num_samples > 0 else len(train_dataset)
    print(f"\nGenerating {num_gen} samples...")

    if args.conditional:
        if args.target_label is not None:
            # 只生成指定类别
            label_name = {0: "negative", 1: "neutral", 2: "positive"}[args.target_label]
            print(f"  → Generating class: {label_name} (label={args.target_label})")
            samples, labels = trainer.generate(num_gen, labels=args.target_label)
        else:
            # 三类均匀生成
            print(f"  → Generating all 3 classes equally")
            gen_labels = np.concatenate([
                np.full(num_gen // 3, 0),     # negative
                np.full(num_gen // 3, 1),     # neutral
                np.full(num_gen - 2 * (num_gen // 3), 2),  # positive
            ]).astype(int)
            np.random.shuffle(gen_labels)
            samples, labels = trainer.generate(num_gen, labels=gen_labels)
    else:
        # 无条件生成
        samples, labels = trainer.generate(num_gen)

    # 恢复到 z-score 尺度: 模型输出 [-1,1] → ×5 → z-score 空间
    CLIP_STD = 5.0
    samples = samples * CLIP_STD

    # 保存: 只输出一个打包文件
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)

    name = ds_cfg.get('name', 'SEED')
    win = train_dataset.window
    bundle_path = os.path.join(save_dir, f"generated_{name}_{win}.npz")
    np.savez(bundle_path, data=samples, labels=labels)

    print(f"\n{'='*50}")
    print(f"  Generated Results")
    print(f"{'='*50}")
    print(f"  Shape : {samples.shape}")
    print(f"  File  : {bundle_path}")
    print(f"  Label distribution:")
    for c in range(3):
        count = int((labels == c).sum())
        label_name = {0: "negative", 1: "neutral", 2: "positive"}[c]
        print(f"    {label_name} (label={c}): {count} samples")
    print(f"{'='*50}")
    print(f"\n  Usage:")
    print(f"    bundle = np.load('{bundle_path}')")
    print(f"    data, labels = bundle['data'], bundle['labels']")
    print(f"\nDone!")


if __name__ == "__main__":
    main()