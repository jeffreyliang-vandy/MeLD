# train_accelerate.py

import os, json, yaml, argparse, logging
from time import time
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets.condition2text import generate_text_conditions
from models.lightning1dit import LightningDiT_models
from transport import create_transport
from datasets.real_dataset import LatentDataset


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


def train(cfg):
    # mixed precision: "no" | "fp16" | "bf16"
    mp = cfg["train"].get("mixed_precision", "no")
    if mp is True:
        mp = "bf16"
    if mp is False:
        mp = "no"

    accelerator = Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=cfg["train"].get("grad_accum_steps", 1),
    )

    # seed (optional)
    seed = cfg["train"].get("seed", None)
    if seed is not None:
        set_seed(seed)

    # experiment dirs
    exp_root = cfg["train"]["output_dir"]
    exp_name = cfg["train"].get("exp_name") or "exp"
    exp_dir  = os.path.join(exp_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        writer = SummaryWriter(os.path.join(exp_dir, "tb"))
        logger.info(json.dumps(cfg, indent=2))
    else:
        logger = logging.getLogger(__name__)
        writer = None

    # model
    seq_len = cfg["data"]["seq_len"]
    patch   = cfg["model"].get("patch_size", 1)
    tokens  = seq_len // patch

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=tokens,
        in_channels=cfg["model"]["in_chans"],
        seq_len=cfg["data"]["seq_len"],
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        use_checkpoint=cfg["model"].get("use_checkpoint", False),
        num_classes=cfg["data"].get("num_classes", 0),
    )

    ema = deepcopy(model)
    requires_grad(ema, False)
    ema.eval()

    # optional weight init
    if "weight_init" in cfg["train"]:
        ckpt = torch.load(cfg["train"]["weight_init"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        ema.load_state_dict(ckpt["model"], strict=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded weights from {cfg['train']['weight_init']}")

    # optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=(0.9, cfg["optimizer"]["beta2"]),
        weight_decay=0.0,
    )

    # dataset
    if accelerator.is_main_process:
        logger.info("Loading dataset...")

    dataset = LatentDataset(
        data_dir=cfg["data"]["data_path"],
        cond_dir=cfg["data"].get("cond_path", None),
        dtype=torch.float32,
        channel_first=True,
    )

    # ------------------------------------------------------------
    # IMPORTANT: global batch semantics
    # ------------------------------------------------------------
    world_size = accelerator.num_processes
    global_bs  = cfg["train"]["global_batch_size"]
    per_gpu_bs = global_bs // world_size

    loader = DataLoader(
        dataset,
        batch_size=per_gpu_bs,      # <-- per-GPU batch derived from global
        shuffle=True,
        # num_workers=cfg["data"].get("num_workers", 0),
        # pin_memory=True,
        num_workers=0,
        pin_memory=False, ## changed to False to resolve bus error
        drop_last=True,
    )
    total_params = sum(p.numel() for p in model.parameters())        
    if accelerator.is_main_process:
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"{total_params/1e6:.1f}M parameters")
        logger.info(f"Global batch size: {global_bs}")
        logger.info(f"World size (GPUs): {world_size}")
        logger.info(f"Per-GPU batch size: {per_gpu_bs}")

    # transport
    transport = create_transport(**cfg["transport"])

    # accelerate prepare
    model, opt, loader = accelerator.prepare(model, opt, loader)
    ema.to(accelerator.device)

    # training state
    step, running_loss, log_steps = 0, 0.0, 0
    start_time = time()

    max_steps   = cfg["train"]["max_steps"]
    log_every   = cfg["train"]["log_every"]
    ckpt_every  = cfg["train"]["ckpt_every"]
    grad_clip   = cfg["optimizer"].get("max_grad_norm", None)

    model.train()

    while step < max_steps:
        for x, y in loader:
            x = x.to(accelerator.device, non_blocking=True)

            if isinstance(y, torch.Tensor):
                model_kwargs = {}
            else:
                y_txt = generate_text_conditions([list(sample_attrs) for sample_attrs in zip(*y)])
                model_kwargs = dict(y=y_txt)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_dict = transport.training_losses(model, x, model_kwargs)
                    loss = loss_dict["loss"].mean()

                accelerator.backward(loss)

                if grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                opt.step()
                opt.zero_grad(set_to_none=True)

                update_ema(ema, accelerator.unwrap_model(model))

            # global mean loss for logging
            gathered_loss = accelerator.gather(loss.detach())
            loss_global_mean = gathered_loss.mean().item()

            running_loss += loss_global_mean
            log_steps += 1
            step += 1

            if accelerator.is_main_process and (step % log_every == 0):
                steps_per_sec = log_steps / (time() - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"[{step}/{max_steps}] loss={avg_loss:.4f}  {steps_per_sec:.2f} it/s")
                writer.add_scalar("loss/train", avg_loss, step)
                running_loss, log_steps = 0.0, 0
                start_time = time()

            if (step % ckpt_every == 0) and accelerator.is_main_process:
                ckpt = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema":   ema.state_dict(),
                    "opt":   opt.state_dict(),
                    "step":  step,
                    "cfg":   cfg,
                }
                ckpt_path = os.path.join(ckpt_dir, f"{step:07d}.pt")
                torch.save(ckpt, ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")

            if step >= max_steps:
                break

    if accelerator.is_main_process:
        logger.info("Training finished.")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
