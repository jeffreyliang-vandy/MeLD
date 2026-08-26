"""Stage 2: encode a dataset into VAE latents for the diffusion model.

    python 2_sample_latent.py --config configs/my_run.yaml

Reads `vae.encode`. Set `vae.encode.h5_path` to encode a different split from the one
used for training, and `vae.encode.condition_path` to emit a row-aligned condition
table for classifier-free guidance.
"""
import argparse
import os

import pandas as pd
import torch

from model import meld_config, timeautoencoder as tae, vae_common as vc


def resolve_weights(cfg):
    """Path to the checkpoint named by vae.encode.checkpoint (None -> best)."""
    checkpoint = cfg.vae.encode.checkpoint
    if checkpoint in (None, ""):
        return cfg.best_path()
    return cfg.checkpoint_path(checkpoint)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the MeLD config file")
    args = ap.parse_args()

    cfg = meld_config.load(args.config)
    tae.set_matmul_precision(cfg.vae.runtime.matmul_precision)
    device = vc.resolve_device(cfg.vae.runtime.device)
    torch.manual_seed(int(cfg.run.seed))

    weights = resolve_weights(cfg)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"VAE weights not found: {weights}")
    print(f"Device: {device}\nWeights: {weights}")

    # The checkpoint carries its own architecture, so there is no separate parameter
    # file to keep in sync with it.
    model, _ = tae.DeapStack.from_checkpoint(weights, map_location=device)
    model = model.to(device)
    print("Model loaded.")

    h5_path = str(cfg.vae.encode.h5_path)
    print(f"Data: {h5_path}")
    dataset = vc.open_dataset(h5_path)

    sample_size = cfg.vae.encode.sample_size
    latents = vc.encode_latents(
        model, dataset, cfg, device,
        batch_size=cfg.vae.encode.batch_size,
        sample_size=None if sample_size is None else int(sample_size))

    latent_path = cfg.latent_path()
    torch.save(latents, latent_path)
    print(f"Latents {tuple(latents.shape)} -> {latent_path}")

    condition_path = cfg.vae.encode.condition_path
    if condition_path:
        conditions = pd.read_parquet(str(condition_path)).reset_index(drop=True)
        if len(conditions) < latents.shape[0]:
            raise ValueError(
                f"condition table has {len(conditions)} rows but there are "
                f"{latents.shape[0]} latents; they must be row-aligned")
        # encode_latents restores the dataset's original row order, so the cohort
        # table lines up positionally.
        conditions = conditions.iloc[: latents.shape[0]].reset_index(drop=True)
        out = cfg.condition_path()
        conditions.to_parquet(out)
        print(f"Conditions {conditions.shape} -> {out}")


if __name__ == "__main__":
    main()
