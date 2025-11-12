"""
LightningDiT time-series / image-latent training (multi-GPU via Accelerate)
---------------------------------------------------------------------------
* No manual DDP; no raw torch.distributed calls in the training loop.
* Semantics aligned with train_1gpu.py: autocast + EMA + AdamW + checkpoints.
* Works with ImgLatentDataset (image latents) by default; trivial to adapt.

Author: adapted from Maple (HUST-VL) & your single-core script
"""

import os, math, json, yaml, argparse, logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from models.lightning1dit import LightningDiT_models
from transport import create_transport
from datasets.img_latent_dataset import ImgLatentDataset   # or your LatentDataset

# ------------------------------ utils -------------------------------- #
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_logger(save_dir, is_main):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        level = logging.INFO if is_main else logging.WARNING
        logger.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if is_main:
            fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    # robust to wrapping: always unwrap the source
    model_unwrapped = model
    # 'model' can be an Accelerate wrapper; unwrap safely:
    try:
        from accelerate.utils import extract_model_from_parallel
        model_unwrapped = extract_model_from_parallel(model)
    except Exception:
        # fallback: accelerator.unwrap_model will be used by caller if needed
        pass

    for ema_p, p in zip(ema_model.parameters(), model_unwrapped.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def pick_latest_by_step(ckpt_dir):
    files = glob(os.path.join(ckpt_dir, "*.pt"))
    if not files: 
        return None
    # Expect filenames like 0000500.pt; extract integer prefix
    def step_of(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0])
        except Exception:
            return -1
    files.sort(key=step_of)
    return files[-1]

# ------------------------- main training loop ------------------------ #
def train(cfg):
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,   # <-- key change
        static_graph=False             # safer when parameter usage can vary
    )
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    # experiment folders
    exp_root = cfg["train"]["output_dir"]
    os.makedirs(exp_root, exist_ok=True)
    exp_name = cfg["train"].get("exp_name") or "exp"
    exp_dir = os.path.join(exp_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(exp_dir, is_main)
    writer = SummaryWriter(os.path.join(exp_dir, "tb")) if is_main else None

    if is_main:
        logger.info(json.dumps(cfg, indent=2))

    # model ------------------------------------------------------------ #
    # If you train on image latents:
    if "vae" in cfg and "downsample_ratio" in cfg["vae"]:
        downsample_ratio = cfg["vae"]["downsample_ratio"]
    else:
        downsample_ratio = 16
    if "image_size" in cfg["data"]:
        assert cfg["data"]["image_size"] % downsample_ratio == 0, "image_size must be divisible by VAE downsample ratio."
        tokens = cfg["data"]["image_size"] // downsample_ratio
    else:
        # time-series path (mirror single-core): tokens = seq_len // patch
        patch = cfg["model"].get("patch_size", 1)
        tokens = cfg["data"]["seq_len"] // patch

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=tokens,
        in_channels=cfg["model"].get("in_chans", 4),
        seq_len=cfg["data"].get("seq_len"),  # harmless for image-latent case
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        use_checkpoint=cfg["model"].get("use_checkpoint", False),
        num_classes=cfg["data"].get("num_classes", 0),
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # optional pretrained weights
    if "weight_init" in cfg["train"]:
        ckpt = torch.load(cfg["train"]["weight_init"], map_location="cpu")
        # strip 'module.' if present
        state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        ema.load_state_dict(model.state_dict(), strict=False)
        if is_main:
            logger.info(f"Loaded weights from {cfg['train']['weight_init']} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")

    if is_main:
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # optimizer -------------------------------------------------------- #
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=(0.9, cfg["optimizer"]["beta2"]),
        weight_decay=0.0,
    )

    # transport (diffusion) ------------------------------------------- #
    transport = create_transport(**cfg["transport"])

    # data ------------------------------------------------------------- #
    # If image latents:
    if "image_size" in cfg["data"]:
        dataset = ImgLatentDataset(
            data_dir=cfg["data"]["data_path"],
            latent_norm=cfg["data"].get("latent_norm", False),
            latent_multiplier=cfg["data"].get("latent_multiplier", 0.18215),
        )
    else:
        # Or your time-series LatentDataset
        from datasets.real_dataset import LatentDataset
        dataset = LatentDataset(cfg["data"]["data_path"], channel_first=True)

    # Let Accelerate shard Sampler & DataLoader when needed:
    per_device_batch = cfg["train"]["global_batch_size"] // max(1, accelerator.num_processes)
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )
    if is_main:
        logger.info(f"Dataset size: {len(dataset)} | per-device batch: {per_device_batch} "
                    f"| global batch: {per_device_batch * accelerator.num_processes}")

    # prepare with accelerate (wraps model/opt/loader as needed) ------- #
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # training state --------------------------------------------------- #
    step, running_loss, log_steps = 0, 0.0, 0
    start_time = time()

    model.train()
    ema.eval()

    max_steps = cfg["train"]["max_steps"]
    log_every = cfg["train"]["log_every"]
    ckpt_every = cfg["train"]["ckpt_every"]
    grad_clip = cfg["optimizer"].get("max_grad_norm", None)
    mixed_precision = cfg["train"].get("mixed_precision", False)

    # resume ----------------------------------------------------------- #
    if cfg["train"].get("resume", False):
        latest = pick_latest_by_step(ckpt_dir)
        if latest is not None:
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            opt.load_state_dict(ckpt["opt"])
            step = ckpt.get("step", int(os.path.splitext(os.path.basename(latest))[0]))
            if is_main:
                logger.info(f"Resumed from {latest} at step {step}")
        else:
            if is_main:
                logger.info("No checkpoint found; starting from scratch.")

    # one-time EMA sync with current weights
    update_ema(ema, accelerator.unwrap_model(model), decay=0.0)

    # -------------------------- training loop ------------------------ #
    while step < max_steps:
        for x, *_ in loader:
            x = x.to(device, non_blocking=True)
            model_kwargs = {}

            with accelerator.autocast() if mixed_precision else torch.cuda.amp.autocast(enabled=False):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                if "cos_loss" in loss_dict:
                    # keep behavior if enabled in transport
                    loss = loss + loss_dict["cos_loss"].mean()

            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            if grad_clip is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            update_ema(ema, accelerator.unwrap_model(model))

            # logging ------------------------------------------------- #
            running_loss += loss.item()
            log_steps += 1
            step += 1

            if step % log_every == 0:
                # synchronize timing across processes for fair throughput
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                steps_per_sec = log_steps / (time() - start_time)
                # average loss across processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.gather(avg_loss).mean().item()
                if is_main:
                    logger.info(f"[{step}/{max_steps}] loss={avg_loss:.4f}  {steps_per_sec:.2f} it/s")
                    if writer:
                        writer.add_scalar("loss/train", avg_loss, step)
                running_loss, log_steps = 0.0, 0
                start_time = time()

            # checkpoint --------------------------------------------- #
            if step % ckpt_every == 0:
                if is_main:
                    save_obj = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema":   ema.state_dict(),
                        "opt":   opt.state_dict(),
                        "step":  step,
                        "config": cfg,
                    }
                    path = os.path.join(ckpt_dir, f"{step:07d}.pt")
                    torch.save(save_obj, path)
                    logger.info(f"Checkpoint saved to {path}")
                accelerator.wait_for_everyone()

            if step >= max_steps:
                break

    if is_main:
        logger.info("Training finished.")

# ------------------------------ entry -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
