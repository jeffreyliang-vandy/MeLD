# train_1gpu.py
"""
LightningDiT time‑series training (single‑GPU version)
------------------------------------------------------

* No accelerate / DDP.
* Works with LatentDataset that holds a tensor (N, seq_len, C).
* Mixed precision via torch.cuda.amp.autocast if requested.

Author: adapted from Maple (HUST‑VL) & ChatGPT
"""

import os, math, json, yaml, argparse, logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict
from datasets.condition2text import generate_text_conditions

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --------------------------------------------------------------------- #
#                   import project‑local modules                        #
# --------------------------------------------------------------------- #
from models.lightning1dit import LightningDiT_models
from transport import create_transport
from datasets.real_dataset import LatentDataset     # <- your loader


# ------------------------------ utils -------------------------------- #
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.remove(os.path.join(save_dir, "log.txt")) if os.path.exists(os.path.join(save_dir, "log.txt")) else None
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(save_dir, "log.txt")),
        ],
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ------------------------- main training loop ------------------------ #
def train(cfg):
    # device ----------------------------------------------------------- #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mixed_precision = cfg["train"].get("mixed_precision", False)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # experiment folder ------------------------------------------------ #
    exp_root = cfg["train"]["output_dir"]
    os.makedirs(exp_root, exist_ok=True)
    exp_name = cfg["train"].get("exp_name") or "exp"
    exp_dir = os.path.join(exp_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(exp_dir)
    writer = SummaryWriter(os.path.join(exp_dir, "tb"))

    logger.info(json.dumps(cfg, indent=2))

    # model ------------------------------------------------------------ #
    seq_len = cfg["data"]["seq_len"]              # L
    patch    = cfg["model"].get("patch_size", 1)
    tokens   = seq_len // patch                  # T  (passed as input_size)

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=tokens,
        in_channels=cfg["model"]["in_chans"],
        seq_len = cfg["data"]["seq_len"],
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        use_checkpoint=cfg["model"].get("use_checkpoint", False),
        num_classes=cfg["data"].get("num_classes", 0),
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # optional pretrained weights ------------------------------------- #
    if "weight_init" in cfg["train"]:
        ckpt = torch.load(cfg["train"]["weight_init"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        ema.load_state_dict(ckpt["model"], strict=False)
        logger.info(f"Loaded weights from {cfg['train']['weight_init']}")

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # optimizer -------------------------------------------------------- #
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=(0.9, cfg["optimizer"]["beta2"]),
        weight_decay=0.0,
    )

    # data ------------------------------------------------------------- #
    logger.info("Loading dataset...")
    dataset = LatentDataset(data_dir = cfg["data"]["data_path"], 
                            cond_dir = cfg["data"].get("cond_path", None),
                            dtype=torch.float32,
                            channel_first=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["global_batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(dataset)}  |  Batch: {cfg['train']['global_batch_size']}")

    # transport (diffusion) ------------------------------------------- #
    transport = create_transport(**cfg["transport"])

    # training state --------------------------------------------------- #
    step, running_loss, log_steps = 0, 0.0, 0
    start_time = time()

    model.train()
    ema.eval()

    max_steps = cfg["train"]["max_steps"]
    log_every = cfg["train"]["log_every"]
    ckpt_every = cfg["train"]["ckpt_every"]
    grad_clip = cfg["optimizer"].get("max_grad_norm", None)

    while step < max_steps:
        for x, y in loader:
            x = x.to(device)
            if isinstance(y, torch.Tensor):
                model_kwargs = {}
            else:
                y = generate_text_conditions([list(sample_attrs) for sample_attrs in zip(*y)])
                model_kwargs = dict(y=y)

            # noise scheduling etc. is inside transport.training_losses
            # with torch.cuda.amp.autocast(enabled=mixed_precision,dtype=torch.bfloat16):
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            update_ema(ema, model)

            # logging ------------------------------------------------- #
            running_loss += loss.item()
            log_steps += 1
            step += 1

            if step % log_every == 0:
                steps_per_sec = log_steps / (time() - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"[{step}/{max_steps}] loss={avg_loss:.4f}  {steps_per_sec:.2f} it/s")
                writer.add_scalar("loss/train", avg_loss, step)
                running_loss, log_steps = 0.0, 0
                start_time = time()

            # checkpoint --------------------------------------------- #
            if step % ckpt_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "ema":   ema.state_dict(),
                    "opt":   opt.state_dict(),
                    "step":  step,
                }
                ckpt_path = os.path.join(ckpt_dir, f"{step:07d}.pt")
                torch.save(ckpt, ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")

            if step >= max_steps:
                break

    logger.info("Training finished.")


# ------------------------------ entry -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
