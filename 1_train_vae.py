"""Stage 1: train the sequence VAE, then encode the training latents.

    python 1_train_vae.py --config configs/my_run.yaml                # single process
    accelerate launch 1_train_vae.py --config configs/my_run.yaml     # multi-GPU

Same script either way -- an `Accelerator()` is always constructed, and it is a no-op
at one process. Every setting lives in the config file; see configs/meld_default.yaml
for the full annotated schema. `vae.stages` selects what runs: `[train, encode]` by
default, `[encode]` alone to re-encode with existing weights without retraining.
"""
import argparse
import os

import torch
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import meld_config, timeautoencoder as tae, vae_common as vc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the MeLD config file")
    args = ap.parse_args()

    cfg = meld_config.load(args.config)
    tae.set_matmul_precision(cfg.vae.runtime.matmul_precision)

    # cfg.vae.runtime.device == "cpu" forces CPU even under a plain `python` launch;
    # "auto"/"cuda" defer to Accelerate/the launcher (e.g. `accelerate launch --cpu`).
    accelerator = Accelerator(split_batches=True,
                              cpu=(str(cfg.vae.runtime.device) == "cpu"))
    is_main = accelerator.is_main_process
    device = accelerator.device

    out_dir = str(cfg.vae.output.dir)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger = vc.setup_logger(out_dir, str(cfg.vae.output.log_name), is_main_process=is_main)
    def log(msg):
        if is_main:
            logger.info(msg)

    log(f"Config: {cfg.config_path}")
    log(f"Output: {cfg.vae.output.dir}")
    torch.manual_seed(int(cfg.run.seed))
    log(f"Device: {device} | processes: {accelerator.num_processes}")

    h5_path = str(cfg.vae.data.h5_path)
    shapes = vc.read_h5_shapes(h5_path)
    log(f"Data: N={shapes['n_samples']} seq_len={shapes['seq_len']} "
        f"feature_size={shapes['feature_size']}")

    model, kwargs = vc.build_model(cfg, shapes)
    if is_main:
        vc.write_legacy_params(cfg.params_path(), kwargs)

    dataset = vc.open_dataset(h5_path)
    stages = set(cfg.vae.stages)
    t = cfg.vae.train

    if "train" in stages:
        log("Training VAE...")
        train_ds, val_ds = vc.split_dataset(cfg, dataset)
        log(f"Train/val split: {len(train_ds)}/{len(val_ds)} "
            f"(seed {cfg.vae.data.split_seed})")

        # split_batches=True means the configured batch size is the global one.
        per_proc = max(1, int(cfg.vae.loader.batch_size) // accelerator.num_processes)
        opts = vc.loader_kwargs(cfg, per_proc)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **opts)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **opts)

        optimizer = Adam(model.parameters(), lr=float(t.lr),
                         weight_decay=float(t.weight_decay))
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)
        log(f"{sum(p.numel() for p in model.parameters())} parameters | "
            f"lr {t.lr} | weight_decay {t.weight_decay}")

        past_epoch, best_loss = 0, float("inf")
        if bool(t.resume):
            past_epoch, best_loss = tae.load_checkpoint(
                accelerator.unwrap_model(model), optimizer, out_dir,
                checkpoint_path=t.resume_from, map_location="cpu",
                strict=bool(t.load_strict))

        vc.run_training(
            cfg, model, optimizer, train_loader, val_loader, device, out_dir, log,
            past_epoch=past_epoch, best_loss=best_loss,
            backward=accelerator.backward,
            reduce_fn=lambda x: accelerator.gather(x).mean().item(),
            is_main=is_main, barrier=accelerator.wait_for_everyone,
            unwrap=accelerator.unwrap_model)
    else:
        log("Skipping training (vae.stages has no 'train').")
        model = accelerator.prepare(model)

    if "encode" in stages:
        best = cfg.best_path()
        accelerator.wait_for_everyone()
        if os.path.exists(best):
            log(f"Loading best weights from {best}")
            rebuilt, _ = tae.DeapStack.from_checkpoint(best, map_location="cpu")
            model = accelerator.prepare(rebuilt)
        else:
            log(f"No best checkpoint at {best}; encoding current weights.")

        latents = vc.encode_latents(
            model, dataset, cfg, device,
            batch_size=cfg.vae.loader.batch_size,
            gather_fn=accelerator.gather_for_metrics, is_main_process=is_main)

        if is_main:
            out = cfg.latent_path()
            torch.save(latents, out)
            log(f"Latents {tuple(latents.shape)} -> {out}")


if __name__ == "__main__":
    main()
