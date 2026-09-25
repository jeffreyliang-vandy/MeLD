"""LightningDiT 1-D sequence sampling: one script for one process or many.

    python inference.py --config cfg.yaml                    # single process
    accelerate launch inference.py --config cfg.yaml         # multi-GPU

Config keys read (see configs/meld_default.yaml, `dit:`):
    ckpt_path, output_dir            set by model/dit_adapter.py
    data.seq_len, model.*            architecture, as in training
    sample.total                     number of samples to generate
    sample.batch_size                GLOBAL batch per step; each process handles
                                     batch_size / num_processes of it (divisibility checked)
    sample.cfg_scale                 above 1.0 enables classifier-free guidance (a condition
                                     table is then required); otherwise unconditional
    sample.cond_path                 condition table, used only when guidance is on
    sample.seed                      seeds the noise, prompts and condition order
    sample.{sampling_method,num_sampling_steps,atol,rtol,reverse,timestep_shift}

Sharding and ordering. Sample `g` (g = 0..total-1) is one "slot". A per-process DataLoader
walks the slots; Accelerate (`split_batches=False`) deals whole batches to the ranks
round-robin, so at step s rank r holds batch s*W + r. `gather_for_metrics` concatenates in
rank order and drops the rows padded onto a short last step, which restores slot order
exactly. This is asserted before saving rather than assumed.

Conditions. With guidance on, slot g uses condition row `perm[g % N]`, where `perm` is a
permutation of the N table rows seeded by `sample.seed`: every row is used before any is
reused. `conditions.csv.gz` row j therefore describes `samples.pt` row j, which
4_sample_synthetic_data.py relies on (it numbers patients by sample position).

Outputs, in `<output_dir>/samples-<total>-cfg<cfg_scale>/`:
    samples.pt           (total, seq_len, in_chans)
    conditions.csv.gz    (total rows; only when guidance is on)

Reproducibility: noise, prompt part order and condition order are seeded from
`sample.seed` (+ rank), so a run repeats exactly for the same process count.
"""
import argparse
import os
import random
from time import strftime

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.condition2text import generate_text_conditions
from datasets.real_dataset import ConditionDataset
from models.lightning1dit import LightningDiT_models
from models.text_encoder import FrozenCLIPTextEncoder
from transport import Sampler, create_transport


# ------------------------------ utils -------------------------------- #
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SampleSlots(Dataset):
    """Slot g -> the tensor g. Slots stand for samples still to be generated."""

    def __init__(self, total):
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, g):
        return torch.tensor(g, dtype=torch.long)


def per_process_batch(batch_size, num_processes):
    if batch_size % num_processes:
        raise ValueError(f"sample.batch_size ({batch_size}) must be divisible by the "
                         f"number of processes ({num_processes})")
    return batch_size // num_processes


def condition_order(n_rows, seed):
    """Seeded permutation of the condition rows; slot g uses order[g % n_rows]."""
    return np.random.default_rng(seed).permutation(n_rows)


def condition_frame(rows):
    """Rows of "column: value" strings -> a DataFrame, as the trainer's dataset builds them."""
    return pd.DataFrame([dict(item.split(": ", 1) for item in row) for row in rows])


# ------------------------- main sampling routine --------------------- #
@torch.no_grad()
def sample(cfg):
    s = cfg["sample"]
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=False))
    if accelerator.split_batches:   # would divide the per-process batch a second time
        raise RuntimeError("split_batches must be off: the loaders are already per process")
    device, is_main = accelerator.device, accelerator.is_main_process
    world, rank = accelerator.num_processes, accelerator.process_index

    def log(msg):
        accelerator.print(f"\033[34m[LightningDiT-Sample {strftime('%Y-%m-%d %H:%M:%S')}]"
                          f"\033[0m {msg}", flush=True)

    total = int(s["total"])
    batch_size = int(s["batch_size"])
    per_proc = per_process_batch(batch_size, world)
    cfg_scale = float(s["cfg_scale"])
    use_cfg = cfg_scale > 1.0
    seed = int(s.get("seed", 42))
    in_chans = cfg["model"]["in_chans"]
    seq_len = cfg["data"]["seq_len"]

    # Different streams per rank, identical from run to run.
    random.seed(seed + rank)
    np.random.seed((seed + rank) % 2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank)

    dataset = order = None
    if use_cfg:
        cond_path = s.get("cond_path")
        if cond_path is None:
            raise ValueError(f"sample.cfg_scale is {cfg_scale} (> 1.0 enables guidance), "
                             f"which needs sample.cond_path")
        dataset = ConditionDataset(data_dir=cond_path)
        if len(dataset) == 0:
            raise ValueError(f"the condition table at {cond_path} is empty")
        order = condition_order(len(dataset), seed)

    def condition_row(g):
        return dataset.cond_data[int(order[g % len(dataset)])]

    # model ------------------------------------------------------------ #
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
    state = ckpt["ema"] if "ema" in ckpt else ckpt
    m = cfg["model"]
    model = LightningDiT_models[m["model_type"]](
        input_size=seq_len // m.get("patch_size", 1),
        in_channels=in_chans,
        seq_len=seq_len,
        num_classes=cfg["data"].get("num_classes", 0),
        use_qknorm=m.get("use_qknorm", False),
        use_swiglu=m.get("use_swiglu", False),
        use_rope=m.get("use_rope", False),
        use_rmsnorm=m.get("use_rmsnorm", False),
        wo_shift=m.get("wo_shift", False),
        learn_sigma=m.get("learn_sigma", False),
        cond_dim=m.get("cond_dim", 512),
    )
    # Strict: a checkpoint from before CLIP moved out of the DiT carries
    # y_embedder.text_encoder.* keys and was trained on a re-initialised CLIP.
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()   # no gradients, so no DDP wrapper either
    log(f"Model loaded from {cfg['ckpt_path']}")

    text_encoder = None
    if use_cfg:
        text_encoder = FrozenCLIPTextEncoder().to(device)
        text_encoder.check_dim(m.get("cond_dim", 512))

    transport = create_transport(**cfg["transport"])
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=s.get("sampling_method", "heun"),
        num_steps=s["num_sampling_steps"],
        atol=s.get("atol", 1e-5),
        rtol=s.get("rtol", 1e-5),
        reverse=s.get("reverse", False),
        timestep_shift=s.get("timestep_shift", 0.0),
    )

    out_dir = os.path.join(cfg["output_dir"], f"samples-{s['total']}-cfg{s['cfg_scale']}")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    loader = accelerator.prepare(
        DataLoader(SampleSlots(total), batch_size=per_proc, shuffle=False, drop_last=False))
    log(f"Sampling {total} on {world} process(es): global batch {batch_size} = "
        f"{per_proc} rows x {world}; guidance={'on' if use_cfg else 'off'}; out={out_dir}")

    samples_out, slots_out = [], []
    for slots in tqdm(loader, total=len(loader), disable=not accelerator.is_local_main_process):
        n = slots.shape[0]
        z = torch.randn(n, in_chans, seq_len, device=device, generator=generator)

        if use_cfg:
            rows = [condition_row(g) for g in slots.tolist()]
            # Encoded once per batch, not once per ODE step.
            emb, uncond = text_encoder.encode(
                generate_text_conditions(rows, dropout_rate=0.0), device)
            y = torch.cat([emb, torch.zeros_like(emb)], dim=0)
            uncond = torch.cat([uncond, torch.ones_like(uncond)], dim=0)
            z = torch.cat([z, z], dim=0)
            model_kwargs = dict(y=y, uncond=uncond, cfg_scale=cfg_scale,
                                cfg_interval=False, cfg_interval_start=0.0)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs, model_fn = {}, model.forward

        out = sample_fn(z, model_fn, **model_kwargs)[-1]
        if use_cfg:
            out, _ = out.chunk(2, dim=0)   # drop the unconditional half

        # Rank order within a step + drop of wrapped padding rows == slot order.
        out = accelerator.gather_for_metrics(out.permute(0, 2, 1).contiguous())
        slots = accelerator.gather_for_metrics(slots)
        if is_main:
            samples_out.append(out.cpu())
            slots_out.append(slots.cpu())

    accelerator.wait_for_everyone()
    if is_main:
        samples = torch.cat(samples_out)
        if not torch.equal(torch.cat(slots_out), torch.arange(total)):
            raise RuntimeError("gathered samples are not in slot order; refusing to save "
                               "samples that would not match their conditions")
        torch.save(samples, os.path.join(out_dir, "samples.pt"))
        if use_cfg:
            condition_frame([condition_row(g) for g in range(total)]).to_csv(
                os.path.join(out_dir, "conditions.csv.gz"), compression="gzip", index=False)
        log(f"Saved {total} samples to {out_dir}")


# ------------------------------ entry -------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config")
    args = parser.parse_args()
    sample(load_yaml(args.config))
