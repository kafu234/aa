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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Models.interpretable_diffusion.FMTS import FM_TS
from Models.interpretable_diffusion.model_utils import (
    unnormalize_to_zero_to_one,
    build_data_driven_graph,
)
from Utils.Data_utils.seed_dataset import SEEDDataset, print_dataset_stats
from Utils.Data_utils.group_split import group_holdout, stratified_group_holdout


class SimpleEMA:
    def __init__(self, model, decay=0.995, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        if step <= self.warmup_steps:
            self.shadow.load_generator_state_dict(model.generator_state_dict())
            return
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
        self.ema = SimpleEMA(
            model,
            decay=ema_cfg.get("decay", 0.995),
            warmup_steps=ema_cfg.get("warmup_steps", 0),
        )
        self.ema_update_every = ema_cfg.get("update_interval", 10)

        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_loss = float("inf")

        # === FIX #1: checkpoint 选择改用 EMA 影子在「真·留出验证集」上的 FM loss ===
        # 验证样本从数据中划出并【从训练里剔除】, 避免 in-sample 偏置 / 奖励记忆。
        self.best_val = float("inf")
        self.val_ema = None
        self._use_labels = (getattr(args, "conditional", False)
                            and not getattr(args, "unlabeled_finetune", False))
        ds = self.train_loader.dataset
        x_all = np.asarray(ds.samples, dtype=np.float32)
        y_all = (np.asarray(ds.labels).astype(np.int64)
                 if getattr(ds, "labels", None) is not None else None)
        if getattr(args, "no_validation", False):
            self.val_x = None
            self.val_y = None
            self._train_samples = x_all
            self._train_labels = y_all
            print(f"[Trainer] 无验证集，固定训练步数，训练样本={len(x_all)}")
            return
        validation_level = getattr(args, "validation_level", "trial")
        if validation_level == "subject" and not getattr(args, "unlabeled_finetune", False):
            groups = np.asarray(ds.sample_subjects, dtype=np.int64)
            train_idx, val_idx = group_holdout(
                groups, val_ratio=args.validation_ratio, seed=args.seed)
        else:
            groups = np.asarray(ds.sample_groups, dtype=np.int64)
            if getattr(args, "unlabeled_finetune", False):
                train_idx, val_idx = group_holdout(
                    groups, val_ratio=args.validation_ratio, seed=args.seed)
            else:
                train_idx, val_idx = stratified_group_holdout(
                    y_all, groups, val_ratio=args.validation_ratio, seed=args.seed)
        n_val = len(val_idx)

        self.val_x = torch.from_numpy(x_all[val_idx]).float().to(self.device)
        self.val_y = (torch.from_numpy(y_all[val_idx]).long().to(self.device)
                      if (self._use_labels and y_all is not None) else None)

        # 训练集 + anchored 采样都只用 train_idx (已剔除验证样本)
        self._train_samples = x_all[train_idx]
        self._train_labels = y_all[train_idx] if y_all is not None else None
        bs = self.train_loader.batch_size
        nw = getattr(self.train_loader, "num_workers", 0)
        pm = getattr(self.train_loader, "pin_memory", False)
        self.train_loader = DataLoader(
            Subset(ds, train_idx.tolist()), batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pm, drop_last=(len(train_idx) >= bs))
        print(f"[Trainer] {validation_level} 级留出验证集 n_val={n_val}, "
              f"groups={len(np.unique(groups[val_idx]))} (已从训练中剔除), "
              f"训练样本={len(train_idx)}")

    @torch.no_grad()
    def _eval_ema_fm_loss(self, n_draws=4):
        """在固定验证批上用 EMA 影子算纯 FM MSE。

        - 评估的就是生成时实际使用的那套 EMA 权重 (而非在线模型);
        - 只算 FM 项, 不含 spectral/classifier/margin, 也不依赖 guidance classifier;
        - (t, z0) 用固定种子的 generator 采样, 使该指标在不同 step 之间可比、低方差。
        """
        shadow = self.ema.shadow
        shadow.eval()
        x1 = self.val_x
        label_emb = None
        if self.val_y is not None and shadow.label_embedding is not None:
            label_emb = shadow.label_embedding(self.val_y)
        gen = torch.Generator(device=self.device).manual_seed(2024)
        total = 0.0
        for _ in range(n_draws):
            z0 = torch.randn(x1.shape, generator=gen, device=self.device)
            t = torch.rand(x1.shape[0], 1, 1, generator=gen, device=self.device)
            z_t = t * x1 + (1.0 - t) * z0
            target = x1 - z0
            out = shadow.output(
                z_t, t.squeeze(-1).squeeze(-1) * shadow.time_scalar,
                None, label_emb=label_emb)
            total += F.mse_loss(out, target).item()
        return total / n_draws

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
                    self.ema.update(self.model, self.step)

                pbar.set_description(f"loss: {total_loss:.6f}")
                pbar.update(1)

                if self.step % self.save_cycle == 0:
                    if getattr(self.args, "no_validation", False):
                        self.best_loss = total_loss
                        pbar.set_postfix(train_loss=f"{total_loss:.5f}")
                        continue
                    # === FIX #1: 用 EMA 影子在固定验证批上的 FM loss 选 best ===
                    val = self._eval_ema_fm_loss()
                    # 轻度平滑, 抑制单点抖动
                    self.val_ema = val if self.val_ema is None else \
                        0.6 * self.val_ema + 0.4 * val
                    self.best_loss = total_loss  # 仅作日志记录, 不再用于选择
                    if self.val_ema < self.best_val:
                        self.best_val = self.val_ema
                        self.save("checkpoint-best.pt")
                    pbar.set_postfix(val_fm=f"{self.val_ema:.5f}",
                                     best=f"{self.best_val:.5f}")

        self.save("checkpoint-last.pt")
        best_path = self.results_dir / "checkpoint-best.pt"
        if getattr(self.args, "no_validation", False):
            # Fixed-step training has no model-selection criterion. Keep the
            # compatibility filename synchronized with this run's final state
            # instead of silently reusing a stale checkpoint from an old run.
            self.best_loss = total_loss
            self.save("checkpoint-best.pt")
        elif not best_path.exists():
            self.best_loss = total_loss
            self.save("checkpoint-best.pt")
        elapsed = time.time() - tic
        print(f"\nTraining complete. Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    def save(self, filename):
        path = self.results_dir / filename
        torch.save({
            "step": self.step, "model": self.model.generator_state_dict(),
            "ema": self.ema.shadow.generator_state_dict(), "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss, "best_val": self.best_val,
            # === FIX #2: 存训练期的生成相关配置 ===
            # 采样时是否启用 CFG 取决于 cfg_dropout (是否训过无条件路径),
            # 必须由 checkpoint 决定, 不能依赖 --sample_only 时命令行传入的值。
            "cfg_dropout": float(self.model.cfg_dropout),
            "guidance_scale": float(self.model.guidance_scale),
            "num_classes": int(self.model.num_classes),
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
        self.best_val = data.get("best_val", float("inf"))
        # === FIX #2: 用 checkpoint 里训练期的 cfg_dropout 覆盖当前模型 ===
        # 否则 sample_only 时若命令行 cfg_dropout 默认 >0, 会对一个没训过无条件
        # 路径的模型错误启用 CFG, 导致 v_uncond 走未训练分支、生成质量崩坏。
        # guidance_scale 故意不覆盖 —— 它是采样时可自由扫的旋钮。
        if "cfg_dropout" in data:
            for m in (self.model, self.ema.shadow):
                m.cfg_dropout = data["cfg_dropout"]
            print(f"  Restored training-time cfg_dropout={data['cfg_dropout']} "
                  f"(决定采样时 CFG 是否启用; guidance_scale 仍由命令行控制)")
        print(f"Loaded checkpoint from {path}, step={self.step}")
        if unexpected:
            print(f"  Ignored {len(unexpected)} unexpected keys (e.g. guidance_classifier)")
        if missing:
            print(f"  Missing {len(missing)} keys (new modules will be randomly initialized)")

    def _sample_anchor_batch(self, batch_labels, batch_size):
        # FIX #1: 只用训练部分做锚点 (排除留出验证样本)
        real_samples = self._train_samples
        real_labels = (self._train_labels.astype(int)
                       if self._train_labels is not None
                       else np.zeros(real_samples.shape[0], dtype=int))
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


GRAPH_DEFAULTS = {
    "use_graph_bias": False,
    "graph_topk": 6,
    "graph_dropedge": 0.1,
    "graph_bias_layers": 2,
    "graph_gate_init": -4.0,
    "use_dual_graph": True,
    "target_graph_weight": 0.3,
    "use_confidence_weighted_target_graph": True,
}


def _resolve_graph_config(config, args):
    graph_cfg = dict(GRAPH_DEFAULTS)
    graph_cfg.update(config.get("graph", {}))
    for key in GRAPH_DEFAULTS:
        if key in config:
            graph_cfg[key] = config[key]

    for key in ("use_graph_bias", "use_dual_graph", "use_confidence_weighted_target_graph"):
        value = getattr(args, key)
        if value is not None:
            graph_cfg[key] = bool(value)
        else:
            graph_cfg[key] = bool(graph_cfg[key])

    for key in ("graph_topk", "graph_bias_layers"):
        value = getattr(args, key)
        if value is not None:
            graph_cfg[key] = int(value)
        else:
            graph_cfg[key] = int(graph_cfg[key])

    for key in ("graph_dropedge", "graph_gate_init", "target_graph_weight"):
        value = getattr(args, key)
        if value is not None:
            graph_cfg[key] = float(value)
        else:
            graph_cfg[key] = float(graph_cfg[key])

    graph_cfg["graph_topk"] = max(0, graph_cfg["graph_topk"])
    graph_cfg["graph_bias_layers"] = max(0, graph_cfg["graph_bias_layers"])
    graph_cfg["graph_dropedge"] = float(np.clip(graph_cfg["graph_dropedge"], 0.0, 1.0))
    graph_cfg["target_graph_weight"] = float(np.clip(graph_cfg["target_graph_weight"], 0.0, 1.0))
    return graph_cfg


def _graph_available(A):
    if A is None:
        return False
    if not torch.is_tensor(A):
        A = torch.as_tensor(A)
    return bool(torch.count_nonzero(A).item() > 0)


def _graph_stats(A):
    if A is None or not _graph_available(A):
        return 100.0, 0.0
    A = A.detach().float().cpu()
    n = A.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool)
    edges = A[off_diag]
    nonzero = edges[edges > 0]
    density = nonzero.numel() / max(edges.numel(), 1)
    sparsity = 100.0 * (1.0 - density)
    mean_weight = float(nonzero.mean().item()) if nonzero.numel() > 0 else 0.0
    return sparsity, mean_weight


def _set_graph_on_trainer(trainer, method_name, *args, **kwargs):
    getattr(trainer.model, method_name)(*args, **kwargs)
    getattr(trainer.ema.shadow, method_name)(*args, **kwargs)


def _load_anchor_graph_inputs(anchor_bundle, use_confidence_weighted):
    if anchor_bundle is None or "data" not in anchor_bundle.files:
        return None, None
    data = np.clip(anchor_bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32)
    sample_weight = None
    if use_confidence_weighted and "confidence" in anchor_bundle.files and "agreement" in anchor_bundle.files:
        sample_weight = (
            anchor_bundle["confidence"].astype(np.float32)
            * anchor_bundle["agreement"].astype(np.float32)
        )
    return data, sample_weight


def _build_source_train_samples(ds_cfg):
    source_cfg = deepcopy(ds_cfg)
    source_cfg["period"] = "train"
    source_dataset = SEEDDataset(**source_cfg)
    return np.asarray(source_dataset.samples, dtype=np.float32)


def configure_graph_bias(trainer, args, ds_cfg, graph_cfg, anchor_bundle=None):
    source_graph = None
    target_graph = None
    graph_source = "disabled"
    source_available = False
    target_available = False

    if graph_cfg["use_graph_bias"]:
        if _graph_available(trainer.model.source_graph_matrix):
            source_graph = trainer.model.source_graph_matrix.detach().cpu()
            source_available = True
        else:
            try:
                if getattr(args, "use_test_period", False):
                    source_samples = _build_source_train_samples(ds_cfg)
                else:
                    source_samples = np.asarray(trainer._train_samples, dtype=np.float32)
                source_graph = build_data_driven_graph(
                    source_samples,
                    topk=graph_cfg["graph_topk"],
                    abs_corr=True,
                )
                source_available = _graph_available(source_graph)
            except Exception as exc:
                print(f"[Graph] Warning: failed to build source graph: {exc}")
                source_graph = None

        if (getattr(args, "finetune", False)
                and graph_cfg["use_dual_graph"]
                and anchor_bundle is not None):
            try:
                anchor_x, sample_weight = _load_anchor_graph_inputs(
                    anchor_bundle,
                    graph_cfg["use_confidence_weighted_target_graph"],
                )
                if anchor_x is not None:
                    target_graph = build_data_driven_graph(
                        anchor_x,
                        topk=graph_cfg["graph_topk"],
                        abs_corr=True,
                        sample_weight=sample_weight,
                    )
                    target_available = _graph_available(target_graph)
            except Exception as exc:
                print(f"[Graph] Warning: failed to build target anchor graph: {exc}")
                target_graph = None

        if source_available and target_available and graph_cfg["use_dual_graph"]:
            _set_graph_on_trainer(
                trainer,
                "set_mixed_graph",
                source_graph,
                target_graph,
                target_graph_weight=graph_cfg["target_graph_weight"],
            )
            graph_source = "source_target_mix"
        elif source_graph is not None:
            _set_graph_on_trainer(trainer, "set_source_graph", source_graph)
            if getattr(args, "finetune", False) and anchor_bundle is not None:
                graph_source = "source_train_fallback"
            else:
                graph_source = "source_train"

    active_graph = trainer.model.graph_bias_matrix if graph_cfg["use_graph_bias"] else None
    sparsity, mean_weight = _graph_stats(active_graph)
    gate_sigmoid = float(torch.sigmoid(torch.tensor(graph_cfg["graph_gate_init"])).item())

    print(f"Graph bias enabled: {graph_cfg['use_graph_bias']}")
    print(f"Use dual graph: {graph_cfg['use_dual_graph']}")
    print(f"Graph top-k: {graph_cfg['graph_topk']}")
    print(f"Graph DropEdge: {graph_cfg['graph_dropedge']}")
    print(f"Graph bias layers: {graph_cfg['graph_bias_layers']}")
    print(f"Graph gate init: {graph_cfg['graph_gate_init']}")
    print(f"Source graph available: {source_available}")
    print(f"Target graph available: {target_available}")
    print(f"Graph source: {graph_source}")
    print(f"Target graph weight: {graph_cfg['target_graph_weight']}")
    print(f"Confidence weighted target graph: {graph_cfg['use_confidence_weighted_target_graph']}")
    print(f"Graph sparsity: {sparsity:.2f}%")
    print(f"Graph mean edge weight: {mean_weight:.6f}")
    print(f"Graph gate initial sigmoid: {gate_sigmoid:.6f}")


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
    parser.add_argument("--validation_level", choices=["trial", "subject"], default="trial",
                        help="checkpoint validation grouping; use subject for source-domain LOSO training")
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--no_validation", action="store_true",
                        help="use all available training data and fixed epochs")
    parser.add_argument("--skip_generation", action="store_true",
                        help="train/checkpoint only; do not generate samples")
    parser.add_argument("--anchor_bundle", type=str, default=None,
                        help="pseudo-labeled target anchors (.npz); labels must not be target ground truth")
    parser.add_argument("--use_graph_bias", action="store_true", default=None,
                        help="enable source/target graph-guided attention bias")
    parser.add_argument("--graph_topk", type=int, default=None)
    parser.add_argument("--graph_dropedge", type=float, default=None)
    parser.add_argument("--graph_bias_layers", type=int, default=None)
    parser.add_argument("--graph_gate_init", type=float, default=None)
    parser.add_argument("--use_dual_graph", action="store_true", default=None)
    parser.add_argument("--target_graph_weight", type=float, default=None)
    parser.add_argument("--use_confidence_weighted_target_graph", action="store_true", default=None)
    # === 改动2: 新增参数 ===
    parser.add_argument("--unlabeled_finetune", action="store_true",
                        help="无标签微调: 加载条件模型但训练时不传标签, "
                             "用于在测试数据上适应分布")
    parser.add_argument("--use_test_period", action="store_true",
                        help="加载测试集数据，仅用于显式声明的传导式域适配")
    parser.add_argument("--allow_transductive_test_adaptation", action="store_true",
                        help="明确允许使用无标签测试数据适配；普通泛化实验禁止启用")
    args = parser.parse_args()

    if args.use_test_period:
        if not args.unlabeled_finetune or not args.allow_transductive_test_adaptation:
            parser.error(
                "--use_test_period 会使用测试域数据，必须同时指定 "
                "--unlabeled_finetune 和 --allow_transductive_test_adaptation")
        if args.checkpoint is None or not args.finetune:
            parser.error("测试域适配必须通过 --checkpoint 加载模型并指定 --finetune")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    if args.target_graph_weight is not None and not (0.0 <= args.target_graph_weight <= 1.0):
        parser.error("--target_graph_weight must be in [0, 1]")
    graph_cfg = _resolve_graph_config(config, args)

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
    model_cfg["use_graph_bias"] = graph_cfg["use_graph_bias"]
    model_cfg["graph_dropedge"] = graph_cfg["graph_dropedge"]
    model_cfg["graph_bias_layers"] = graph_cfg["graph_bias_layers"]
    model_cfg["graph_gate_init"] = graph_cfg["graph_gate_init"]

    print(f"Model config: seq_length={model_cfg['seq_length']}, feature_size={model_cfg['feature_size']}")
    model = FM_TS(**model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")

    if args.results_dir is None:
        args.results_dir = f"./results/{ds_cfg.get('name', 'SEED')}"

    trainer = SEEDTrainer(model, train_loader, config, args)
    anchor_bundle = None
    if args.anchor_bundle is not None:
        anchor_bundle = np.load(args.anchor_bundle)
        trainer._train_samples = np.clip(
            anchor_bundle["data"] / 5.0, -1.0, 1.0).astype(np.float32)
        trainer._train_labels = anchor_bundle["labels"].astype(np.int64)
        missing_classes = sorted(set(range(model.num_classes)) - set(np.unique(trainer._train_labels)))
        if missing_classes:
            raise ValueError(f"anchor bundle is missing pseudo-label classes: {missing_classes}")
        print(f"[Pseudo anchors] loaded {len(trainer._train_labels)} samples from {args.anchor_bundle}")

    if args.checkpoint is not None:
        trainer.load(args.checkpoint)
        if args.finetune:
            trainer.step = 0
            trainer.best_loss = float("inf")
            trainer.best_val = float("inf")   # FIX #1: 微调时同步重置验证选择状态
            trainer.val_ema = None
            print(f"[Finetune] 重置训练步数: step=0, max_epochs={trainer.max_epochs}")

    configure_graph_bias(trainer, args, ds_cfg, graph_cfg, anchor_bundle=anchor_bundle)

    # === Guidance Classifier: 必须在 checkpoint 加载之后, 避免被覆盖 ===
    # FIX #4: sample_only 时不需要 guidance classifier (采样路径用的是 CFG, 不碰它),
    #         不再为采样白白预训练 50 个 epoch。
    if (args.conditional and args.classifier_weight > 0
            and not args.unlabeled_finetune and not args.sample_only):
        os.makedirs(args.results_dir, exist_ok=True)
        classifier_name = "guidance_classifier_all_source.pt" if args.no_validation else "guidance_classifier_fit_only.pt"
        classifier_path = os.path.join(args.results_dir, classifier_name)
        if os.path.exists(classifier_path):
            model.load_classifier(classifier_path, device=device)
        else:
            model.pretrain_classifier(
                trainer._train_samples, trainer._train_labels,
                epochs=args.cls_epochs, device=device)
            model.save_classifier(classifier_path)

    if not args.sample_only:
        trainer.train()
        checkpoint_name = "checkpoint-last.pt" if args.no_validation else "checkpoint-best.pt"
        generation_ckpt = os.path.join(args.results_dir, checkpoint_name)
        if os.path.exists(generation_ckpt):
            print(f"[Generate] 加载 checkpoint: {generation_ckpt}")
            trainer.load(generation_ckpt)

    if args.conditional:
        sensitivity = trainer.ema.shadow.condition_sensitivity()
        print(f"[Condition sensitivity] pairwise_rmse={sensitivity['pairwise_rmse']:.6f}, "
              f"relative_rmse={sensitivity['relative_rmse']:.4f}")

    if args.skip_generation:
        print("[Generate] skipped by --skip_generation")
        return

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
