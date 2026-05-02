# sample_accelerate.py
"""
LightningDiT 1-D sequence sampling with Accelerate + batch splitting
====================================================================

Usage
-----
Single process:
    python sample_accelerate.py --config configs/sample.yaml

With Accelerate:
    accelerate launch sample_accelerate.py --config configs/sample.yaml

Required YAML fields
--------------------
ckpt_path : path to EMA checkpoint (*.pt)

data:
    seq_len       : length L
    in_chans      : number of variables C

model:
    model_type    : key in LightningDiT_models
    in_chans      : number of input channels

sample:
    total                 : total number of samples across all processes
    batch_size            : logical per-process batch size
    micro_batch_size      : micro-batch size used for batch splitting
    cfg_scale             : 0 (=disabled) or >1 for classifier-free guidance
    num_sampling_steps    : diffusion steps for ODE sampler
    cond_path             : optional condition dataset path

Notes
-----
- `batch_size` is the logical batch size per process.
- `micro_batch_size` is the chunk size used inside each process to reduce memory.
- Final outputs are merged on the main process into:
      samples.pt
      conditions.csv.gz   (if conditional sampling is enabled)
"""

import os
import math
import yaml
import argparse
from time import strftime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from accelerate import Accelerator

from models.lightning1dit import LightningDiT_models
from transport import create_transport, Sampler
from datasets.real_dataset import ConditionDataset
from datasets.condition2text import generate_text_conditions


# ------------------------------------------------------------------ #
#                        helper utils                                #
# ------------------------------------------------------------------ #
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def log(accelerator, msg):
    t = strftime("%Y-%m-%d %H:%M:%S")
    accelerator.print(f"\033[34m[LightningDiT-Sample {t}]\033[0m {msg}", flush=True)


def split_evenly(total, world_size, rank):
    """
    Split `total` items across ranks as evenly as possible.
    Returns the number assigned to this rank.
    """
    base = total // world_size
    rem = total % world_size
    return base + (1 if rank < rem else 0)


def build_output_dir(cfg):
    return os.path.join(
        cfg["output_dir"],
        f"samples-{cfg['sample']['total']}-cfg{cfg['sample']['cfg_scale']}"
    )


def save_local_shard(out_dir, rank, samples, condition_rows=None):
    samples_path = os.path.join(out_dir, f"samples_rank{rank:03d}.pt")
    torch.save(samples, samples_path)

    if condition_rows is not None and len(condition_rows) > 0:
        cond_path = os.path.join(out_dir, f"conditions_rank{rank:03d}.csv.gz")
        df = pd.DataFrame(
            [dict(item.split(": ", 1) for item in row) for row in condition_rows]
        )
        df.to_csv(cond_path, compression="gzip", index=False)


def merge_shards(out_dir, world_size, save_conditions):
    sample_shards = []
    cond_shards = []

    for rank in range(world_size):
        sample_path = os.path.join(out_dir, f"samples_rank{rank:03d}.pt")
        if os.path.exists(sample_path):
            sample_shards.append(torch.load(sample_path, map_location="cpu"))

        if save_conditions:
            cond_path = os.path.join(out_dir, f"conditions_rank{rank:03d}.csv.gz")
            if os.path.exists(cond_path):
                cond_shards.append(pd.read_csv(cond_path, compression="gzip"))

    if len(sample_shards) == 0:
        raise RuntimeError("No sample shards were found to merge.")

    merged_samples = torch.cat(sample_shards, dim=0)
    torch.save(merged_samples, os.path.join(out_dir, "samples.pt"))

    if save_conditions and len(cond_shards) > 0:
        merged_conditions = pd.concat(cond_shards, ignore_index=True)
        merged_conditions.to_csv(
            os.path.join(out_dir, "conditions.csv.gz"),
            compression="gzip",
            index=False,
        )

    # Optional cleanup of shard files
    for rank in range(world_size):
        sample_path = os.path.join(out_dir, f"samples_rank{rank:03d}.pt")
        if os.path.exists(sample_path):
            os.remove(sample_path)

        cond_path = os.path.join(out_dir, f"conditions_rank{rank:03d}.csv.gz")
        if os.path.exists(cond_path):
            os.remove(cond_path)


# ------------------------------------------------------------------ #
#                     micro-batched sampling                         #
# ------------------------------------------------------------------ #
@torch.no_grad()
def run_sampling_microbatched(
    *,
    model,
    sample_fn,
    seq_len,
    in_chans,
    total_batch,
    micro_batch_size,
    use_cfg,
    cfg_scale,
    dataset,
    rng,
    device,
):
    """
    Generate `total_batch` samples on this process, split into micro-batches.

    Returns
    -------
    samples_cpu : Tensor of shape [total_batch, seq_len, in_chans] on CPU
    used_conditions : list[str]
        Raw condition rows used for conditional sampling.
    """
    out_chunks = []
    used_conditions = []

    remaining = total_batch
    while remaining > 0:
        m = min(micro_batch_size, remaining)

        z = torch.randn(m, in_chans, seq_len, device=device)

        if use_cfg:
            if dataset is None:
                raise ValueError("cfg_scale > 1 but no conditional dataset was provided.")

            z = torch.cat([z, z], dim=0)

            indices = rng.choice(len(dataset), size=m, replace=True)
            local_conditions = [dataset[i] for i in indices]
            used_conditions.extend(dataset.cond_data[i] for i in indices)

            y = generate_text_conditions(local_conditions, dropout_rate=0.0)
            y_null = [""] * m
            y = y + y_null

            model_kwargs = dict(
                y=y,
                cfg_scale=cfg_scale,
                cfg_interval=False,
                cfg_interval_start=0.0,
            )
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = {}
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]

        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        out_chunks.append(samples.cpu().permute(0, 2, 1))
        remaining -= m

    return torch.cat(out_chunks, dim=0), used_conditions


# ------------------------------------------------------------------ #
#                     main sampling routine                          #
# ------------------------------------------------------------------ #
@torch.no_grad()
def sample(cfg):
    accelerator = Accelerator()
    device = accelerator.device

    log(
        accelerator,
        f"Sampling on device={device}, rank={accelerator.process_index}/{accelerator.num_processes}, "
        f"mixed_precision={accelerator.mixed_precision}",
    )

    # --------------------------- model ------------------------------ #
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
    state = ckpt["ema"] if "ema" in ckpt else ckpt

    seq_len = cfg["data"]["seq_len"]
    patch = cfg["model"].get("patch_size", 1)
    tokens = seq_len // patch

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=tokens,
        in_channels=cfg["model"]["in_chans"],
        seq_len=cfg["data"]["seq_len"],
        num_classes=cfg["data"].get("num_classes", 0),
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        learn_sigma=cfg["model"].get("learn_sigma", False),
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    model = model.to(device)

    # Accelerator wraps the model for distributed execution
    model = accelerator.prepare(model)

    # Use the underlying module for custom methods like forward_with_cfg
    unwrapped_model = accelerator.unwrap_model(model)

    log(accelerator, "Model loaded.")

    # --------------------------- transport / sampler --------------- #
    transport = create_transport(**cfg["transport"])
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=cfg["sample"].get("sampling_method", "heun"),
        num_steps=cfg["sample"]["num_sampling_steps"],
        atol=cfg["sample"].get("atol", 1e-5),
        rtol=cfg["sample"].get("rtol", 1e-5),
        reverse=cfg["sample"].get("reverse", False),
        timestep_shift=cfg["sample"].get("timestep_shift", 0.0),
    )

    # --------------------------- output dir ------------------------ #
    out_dir = build_output_dir(cfg)
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    log(accelerator, f"Saving tensors to {out_dir}")

    # --------------------------- conditional data ------------------ #
    dataset = None
    condition_path = cfg["sample"].get("cond_path", None)
    if condition_path is not None:
        log(accelerator, f"Loading conditional dataset from {condition_path}")
        dataset = ConditionDataset(data_dir=condition_path)

    # --------------------------- generate config ------------------- #
    total = int(cfg["sample"]["total"])
    batch_size = int(cfg["sample"]["batch_size"])
    micro_batch_size = batch_size // accelerator.num_processes
    cfg_scale = float(cfg["sample"]["cfg_scale"])
    use_cfg = cfg_scale >= 1.0

    if micro_batch_size <= 0:
        raise ValueError("sample.micro_batch_size must be > 0")
    if micro_batch_size > batch_size:
        micro_batch_size = batch_size

    local_total = split_evenly(total, accelerator.num_processes, accelerator.process_index)
    local_steps = math.ceil(local_total / batch_size) if local_total > 0 else 0

    log(
        accelerator,
        f"Global total={total}, local total={local_total}, "
        f"batch_size={batch_size}, micro_batch_size={micro_batch_size}, use_cfg={use_cfg}",
    )

    # Rank-specific RNG for reproducible but distinct conditional draws
    base_seed = int(cfg["sample"].get("seed", 42))
    rng = np.random.default_rng(base_seed + accelerator.process_index)

    local_sample_chunks = []
    local_condition_rows = []

    produced = 0
    iterator = range(local_steps)
    if accelerator.is_local_main_process:
        iterator = tqdm(iterator, total=local_steps)

    for _ in iterator:
        current_batch = min(batch_size, local_total - produced)

        samples_cpu, used_conditions = run_sampling_microbatched(
            model=unwrapped_model,
            sample_fn=sample_fn,
            seq_len=seq_len,
            in_chans=cfg["model"]["in_chans"],
            total_batch=current_batch,
            micro_batch_size=micro_batch_size,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            dataset=dataset,
            rng=rng,
            device=device,
        )

        local_sample_chunks.append(samples_cpu)
        if use_cfg:
            local_condition_rows.extend(used_conditions)

        produced += current_batch

    if len(local_sample_chunks) == 0:
        local_samples = torch.empty(0, seq_len, cfg["model"]["in_chans"])
    else:
        local_samples = torch.cat(local_sample_chunks, dim=0)[:local_total]

    log(accelerator, f"Rank {accelerator.process_index}: sampling done, saving local shard.")

    save_local_shard(
        out_dir=out_dir,
        rank=accelerator.process_index,
        samples=local_samples,
        condition_rows=local_condition_rows if use_cfg else None,
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merge_shards(
            out_dir=out_dir,
            world_size=accelerator.num_processes,
            save_conditions=use_cfg,
        )
        log(accelerator, f"Saved {total} merged samples to {out_dir}")


# ------------------------------------------------------------------ #
#                            entry                                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sample(cfg)