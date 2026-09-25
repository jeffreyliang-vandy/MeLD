"""LightningDiT latent training: one script for one process or many.

    python train.py --config cfg.yaml                    # single process
    accelerate launch train.py --config cfg.yaml         # multi-GPU

An `Accelerator()` is always constructed and is a no-op at one process, so there is a
single code path. This replaces `train_single.py` and `train_multigpu.py`, which had
drifted apart (only one honoured grad accumulation, only one seeded, cos_loss was
dropped by both, ...).

Batching: `train.global_batch_size` is the batch per *optimizer step*. Every process builds
its own loader with `global_batch_size / (grad_accum_steps * num_processes)` rows, and
Accelerate (default `split_batches=False`) deals those whole batches to the ranks
round-robin. Each rank therefore collates only its own rows, so the text conditions `y`
stay aligned with `x`. The global batch must divide evenly by both factors.

Config keys read (see configs/meld_default.yaml, `dit:`): data.{data_path,cond_path,seq_len,
num_workers,num_classes}, model.*, train.{output_dir,exp_name,max_steps,global_batch_size,
log_every,ckpt_every,mixed_precision,grad_accum_steps,seed,resume,weight_init}, optimizer.*,
transport.*.
"""
import argparse
import json
import logging
import os
import random
import re
from copy import deepcopy
from time import time

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.condition2text import generate_text_conditions
from datasets.real_dataset import LatentDataset
from models.lightning1dit import LightningDiT_models
from models.text_encoder import FrozenCLIPTextEncoder
from transport import create_transport

_CKPT_RE = re.compile(r"^(\d+)\.pt$")
_MIXED_PRECISION_ALIASES = {"": "no", "none": "no", "false": "no", "true": "fp16"}


# ------------------------------ utils -------------------------------- #
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_logger(save_dir, is_main, append=False):
    """File + console logger on the main process; a silent one everywhere else."""
    logger = logging.getLogger("lightning_dit.train")
    logger.propagate = False
    logger.handlers.clear()
    if not is_main:
        logger.addHandler(logging.NullHandler())
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(os.path.join(save_dir, "log.txt"),
                                        mode="a" if append else "w")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def newest_checkpoint(ckpt_dir):
    """Newest `{step:07d}.pt`, ordered numerically. Mirrors model/dit_adapter.py, which
    cannot be imported here: this script runs in the DiT environment."""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = [(int(m.group(1)), name) for name in os.listdir(ckpt_dir)
             if (m := _CKPT_RE.match(name))]
    return os.path.join(ckpt_dir, max(steps)[1]) if steps else None


def resolve_mixed_precision(value):
    """Accelerate wants "no" | "fp16" | "bf16"; accept the old bool spelling too."""
    if value is None or isinstance(value, bool):
        return "fp16" if value else "no"
    mp = str(value).lower()
    mp = _MIXED_PRECISION_ALIASES.get(mp, mp)
    if mp not in ("no", "fp16", "bf16"):
        raise ValueError(f"train.mixed_precision must be no|fp16|bf16 (or a bool), got {value!r}")
    return mp


def build_model(cfg):
    m, d = cfg["model"], cfg["data"]
    return LightningDiT_models[m["model_type"]](
        input_size=d["seq_len"] // m.get("patch_size", 1),
        in_channels=m["in_chans"],
        seq_len=d["seq_len"],
        class_dropout_prob=m.get("class_dropout_prob", 0.1),
        learn_sigma=m.get("learn_sigma", False),
        use_qknorm=m.get("use_qknorm", False),
        use_swiglu=m.get("use_swiglu", False),
        use_rope=m.get("use_rope", False),
        use_rmsnorm=m.get("use_rmsnorm", False),
        wo_shift=m.get("wo_shift", False),
        use_checkpoint=m.get("use_checkpoint", False),
        num_classes=d.get("num_classes", 0),
        cond_dim=m.get("cond_dim", 512),
    )


def per_process_batch(global_batch_size, grad_accum_steps, num_processes):
    """Rows each process loads per micro-step; the global batch must divide evenly."""
    ways = grad_accum_steps * num_processes
    if global_batch_size % ways:
        raise ValueError(
            f"train.global_batch_size ({global_batch_size}) must be divisible by "
            f"grad_accum_steps ({grad_accum_steps}) x number of processes "
            f"({num_processes}) = {ways}")
    return global_batch_size // ways


# ------------------------- main training loop ------------------------ #
def train(cfg):
    t, d = cfg["train"], cfg["data"]

    accelerator = Accelerator(
        mixed_precision=resolve_mixed_precision(t.get("mixed_precision", "no")),
        gradient_accumulation_steps=int(t.get("grad_accum_steps", 1)),
        # Loaders are per process (see the module docstring), so never split a batch again.
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        # With no conditions every sample is unconditional, so y_embedder.projection
        # never receives a gradient; with conditions, null_embed gets none on a batch
        # where CFG dropout drops nothing (likely for small per-process batches). DDP
        # raises on either unless told to expect it.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.split_batches:   # would divide the per-process batch a second time
        raise RuntimeError("split_batches must be off: the loaders are already per process")
    is_main = accelerator.is_main_process

    seed = t.get("seed")
    if seed is not None:
        set_seed(int(seed))
        # generate_text_conditions draws prompt dropout from these two RNGs. Left as
        # seeded above, every rank would make identical draws at every step.
        random.seed(int(seed) + accelerator.process_index)
        np.random.seed((int(seed) + accelerator.process_index) % 2**32)

    exp_dir = os.path.join(t["output_dir"], t.get("exp_name") or "exp")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger = create_logger(exp_dir, is_main, append=bool(t.get("resume", False)))
    writer = SummaryWriter(os.path.join(exp_dir, "tb")) if is_main else None
    logger.info(json.dumps(cfg, indent=2))

    # model / EMA ------------------------------------------------------ #
    model = build_model(cfg)
    ema = deepcopy(model)
    requires_grad(ema, False)
    ema.eval()

    if "weight_init" in t:
        state = torch.load(t["weight_init"], map_location="cpu")["model"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        ema.load_state_dict(model.state_dict())
        logger.info(f"Loaded weights from {t['weight_init']} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=(0.9, cfg["optimizer"]["beta2"]),
        weight_decay=0.0,
    )

    # resume: restore *before* prepare, so keys carry no DDP "module." prefix and the
    # optimizer state is moved to the device along with the model.
    step = 0
    if t.get("resume", False):
        latest = newest_checkpoint(ckpt_dir)
        if latest is None:
            logger.info("resume: no checkpoint found; starting from scratch.")
        else:
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            opt.load_state_dict(ckpt["opt"])
            step = int(ckpt["step"])
            logger.info(f"Resumed from {latest} at step {step}")

    # data ------------------------------------------------------------- #
    dataset = LatentDataset(
        data_dir=d["data_path"],
        cond_dir=d.get("cond_path", None),
        dtype=torch.float32,
        channel_first=True,
    )
    global_bs = int(t["global_batch_size"])
    world = accelerator.num_processes
    per_proc_bs = per_process_batch(global_bs, accelerator.gradient_accumulation_steps, world)
    if len(dataset) < per_proc_bs * world:
        # Otherwise `while step < max_steps: for ... in loader` spins forever, or some
        # rank would train only on batches wrapped around from the start.
        raise ValueError(
            f"the dataset has {len(dataset)} rows, fewer than one micro-step's "
            f"{per_proc_bs} x {world} process(es); lower dit.train.global_batch_size")
    loader = DataLoader(
        dataset,
        batch_size=per_proc_bs,
        shuffle=True,
        num_workers=int(d.get("num_workers", 0)),
        pin_memory=False,   # the dataset is already an in-RAM tensor
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(dataset)} | global batch: {global_bs} = "
                f"{per_proc_bs} rows x {world} process(es) x "
                f"{accelerator.gradient_accumulation_steps} accumulation step(s)")

    # Frozen, so it stays out of the model: no EMA copy, no optimizer, no DDP, and not
    # in the checkpoints. Loaded only when there is something to condition on.
    text_encoder = None
    if d.get("cond_path"):
        text_encoder = FrozenCLIPTextEncoder().to(accelerator.device)
        text_encoder.check_dim(cfg["model"].get("cond_dim", 512))

    transport = create_transport(**cfg["transport"])

    model, opt, loader = accelerator.prepare(model, opt, loader)
    ema.to(accelerator.device)

    # training loop ---------------------------------------------------- #
    max_steps = int(t["max_steps"])
    log_every = int(t["log_every"])
    ckpt_every = int(t["ckpt_every"])
    grad_clip = cfg["optimizer"].get("max_grad_norm", None)

    running_loss = torch.zeros((), device=accelerator.device)
    log_steps = 0
    start_time = time()
    model.train()

    while step < max_steps:
        for x, y in loader:
            if isinstance(y, torch.Tensor):     # no conditions: y is just the row id
                model_kwargs = {}
            else:
                emb, uncond = text_encoder.encode(
                    generate_text_conditions([list(attrs) for attrs in zip(*y)]),
                    accelerator.device)
                model_kwargs = dict(y=emb, uncond=uncond)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_dict = transport.training_losses(model, x, model_kwargs)
                    loss = loss_dict["loss"].mean()
                    if "cos_loss" in loss_dict:
                        loss = loss + loss_dict["cos_loss"].mean()

                accelerator.backward(loss)
                if grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue    # mid-accumulation: no optimizer step happened

            update_ema(ema, accelerator.unwrap_model(model))
            running_loss += loss.detach().float()
            log_steps += 1
            step += 1

            if step % log_every == 0:
                # One collective per log interval rather than one per step.
                avg_loss = accelerator.gather(
                    (running_loss / log_steps).reshape(1)).mean().item()
                steps_per_sec = log_steps / (time() - start_time)
                logger.info(f"[{step}/{max_steps}] loss={avg_loss:.4f}  "
                            f"{steps_per_sec:.2f} it/s")
                if writer:
                    writer.add_scalar("loss/train", avg_loss, step)
                running_loss.zero_()
                log_steps = 0
                start_time = time()

            finished = step >= max_steps
            # Always leave a final checkpoint: without it, a max_steps that is not a
            # multiple of ckpt_every produces nothing for the sampling stage to load.
            if step % ckpt_every == 0 or finished:
                accelerator.wait_for_everyone()
                if is_main:
                    path = os.path.join(ckpt_dir, f"{step:07d}.pt")
                    torch.save({
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "cfg": cfg,
                    }, path)
                    logger.info(f"Checkpoint saved to {path}")

            if finished:
                break

    accelerator.wait_for_everyone()
    logger.info("Training finished.")
    if writer:
        writer.close()


# ------------------------------ entry -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(load_config(args.config))
