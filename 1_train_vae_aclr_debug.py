"""
Diagnostic version of 1_train_vae_aclr.py (accelerate / multi-GPU path).

Same instrumentation as 1_train_vae_debug.py (see that file's docstring), wired
through `accelerate`, plus it records - as MEASURED FACTS, not theory - the
single-GPU vs multi-GPU asymmetries that could explain why the NaN differs
between the two paths:

  * `accelerator.num_processes`, `split_batches`, mixed-precision dtype
  * the DataLoader batch size vs the ACTUAL per-step tensor batch size
    (`1_train_vae_aclr.py` does `per_gpu = batch_size // num_processes` AND uses
    `Accelerator(split_batches=True)`, which can divide the batch twice)
  * whether gradients are actually all-reduced: the original calls
    `model.module.get_loss(...)`, bypassing the DDP wrapper's gradient sync

Per-rank dumps land in `<diag_dir>` for rank 0 and `<diag_dir>/rank<k>` otherwise
(a NaN may surface on one rank only). Copy the whole tree off the server.

Fixes from NAN_DIAGNOSIS_REPORT.md are opt-in flags (defaults reproduce the NaN):
  * --grad_clip <max_norm>   fix 1: accelerator.clip_grad_norm_ before .step()
  * --sincos_num_terms <n>   fix 2: frequency terms in compute_sine_cosine (6-8)

    accelerate launch MeLD/1_train_vae_aclr_debug.py -DP data.h5 -MP runs/ --seed 0
    accelerate launch MeLD/1_train_vae_aclr_debug.py -DP data.h5 -MP runs/ --seed 0 \\
        --grad_clip 1.0 --sincos_num_terms 8
"""

import os
import argparse
import logging
import time
from argparse import BooleanOptionalAction

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam
from accelerate import Accelerator

from MeLD.model import timeautoencoder as tae
from MeLD.model import nan_diagnostics as nd
from MeLD.model.data_loader import HDF5Dataset


class IndexedDataset(Dataset):
    """Return (*sample, original_row_index), unwrapping nested Subsets."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def _orig_row(self, idx):
        b, cur = self.base_dataset, idx
        while hasattr(b, "indices") and hasattr(b, "dataset"):
            cur = b.indices[cur]
            b = b.dataset
        return cur

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        cur = self._orig_row(idx)
        if isinstance(item, (tuple, list)):
            return (*item, cur)
        return item, cur

    def __getitems__(self, indices):
        b = self.base_dataset
        if hasattr(b, "__getitems__"):
            items = b.__getitems__(list(indices))
        else:
            items = [b[i] for i in indices]
        out = []
        for pos, i in enumerate(indices):
            cur = self._orig_row(i)
            it = items[pos]
            out.append((*it, cur) if isinstance(it, (tuple, list)) else (it, cur))
        return out


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def main():
    accelerator = Accelerator(split_batches=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_model", "-VM", default=None)
    parser.add_argument("--data_path", "-DP", required=True)
    parser.add_argument("--model_path", "-MP", required=True)
    parser.add_argument("--id", "-I", type=str, default="patient")

    parser.add_argument("--epochs", "-EP", type=int, default=5000)
    parser.add_argument("--warmup", "-WU", type=int, default=50)
    parser.add_argument("--batch_size", "-BS", type=int, default=128)
    parser.add_argument("--lr", "-LR", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-WD", type=float, default=1e-6)
    parser.add_argument("--patience", "-PT", type=int, default=20)

    parser.add_argument("--min_beta", type=float, default=1e-5)
    parser.add_argument("--max_beta", type=float, default=1e-2)
    parser.add_argument("--min_kl", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lat_dim", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--bidirectional", action="store_true", default=False)

    # --- fixes from NAN_DIAGNOSIS_REPORT.md (opt-in; defaults reproduce the NaN) ---
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="fix 1: clip_grad_norm_ max_norm before .step() "
                             "(0 = disabled). Try 1.0-5.0.")
    parser.add_argument("--sincos_num_terms", type=int, default=16,
                        help="fix 2: frequency terms in compute_sine_cosine "
                             "(16 -> top freq 2**15; use 6-8).")
    parser.add_argument("--fourier_layernorm", action=BooleanOptionalAction,
                        default=True,
                        help="LayerNorm the mlp_nums (Fourier) embedding.")
    parser.add_argument("--logvar_min", type=float, default=-6.0)
    parser.add_argument("--logvar_max", type=float, default=2.0,
                        help="fc_logvar squashed to (logvar_min, logvar_max) "
                             "via sigmoid (replaces the dead Hardtanh).")
    parser.add_argument("--mu_clip", type=float, default=5.0,
                        help="soft bound mu := mu_clip*tanh(mu/mu_clip); <=0 off.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diag_dir", type=str, default=None)
    parser.add_argument("--buffer_steps", type=int, default=200)
    parser.add_argument("--snapshot_steps", type=int, default=2)
    parser.add_argument("--heartbeat_every", type=int, default=50)
    parser.add_argument("--metrics_every", type=int, default=25)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--anomaly", action="store_true", default=False)
    parser.add_argument("--inject_nan_at_step", type=int, default=-1)
    parser.add_argument("--inject_nan_phase", choices=["pre_forward", "pre_step"],
                        default="pre_step")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_module_bypass", action=BooleanOptionalAction,
                        default=True,
                        help="call model.module.get_loss like the original "
                             "(bypasses DDP gradient sync); on by default so the "
                             "diagnostic matches the path under test. "
                             "--no-use_module_bypass to route through the DDP wrapper.")
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(
        args.model_path, f"vae_{args.vae_model}" if args.vae_model else "vae"
    )
    base_diag = args.diag_dir or os.path.join(args.checkpoint_dir, "nan_diag")
    rank = accelerator.process_index
    diag_dir = base_diag if rank == 0 else os.path.join(base_diag, f"rank{rank}")

    logger = logging.getLogger("vae_aclr_debug")
    logger.setLevel(logging.INFO)
    if accelerator.is_main_process and not logger.handlers:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(os.path.join(args.checkpoint_dir, "training.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    nd.set_global_seed(args.seed, deterministic=not args.nondeterministic)

    with h5py.File(args.data_path, "r") as f:
        n_bins = int(f.attrs["n_bins"])
        n_cats = int(f.attrs["n_cats"])
        n_nums = int(f.attrs["n_nums"])
        cards = f.attrs["cards"].tolist()
        N, seq_len, feature_size = f["processed_data"].shape
        time_dim = f["time_info"].shape[2]
        assert n_nums == f["missing"].shape[2]
        assert sum([n_bins, n_cats, n_nums]) == feature_size

    model_config = {
        "channels": args.channels, "batch_size": args.batch_size,
        "seq_len": seq_len, "n_bins": n_bins, "n_cats": n_cats, "n_nums": n_nums,
        "cards": cards, "feature_size": feature_size,
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "bidirectional": args.bidirectional, "emb_dim": args.emb_dim,
        "time_dim": time_dim, "lat_dim": args.lat_dim,
        "num_terms": args.sincos_num_terms,          # fix 2
        "fourier_layernorm": args.fourier_layernorm,  # §8.3
        "logvar_min": args.logvar_min,                # §8.2
        "logvar_max": args.logvar_max,                # §8.2
        "mu_clip": args.mu_clip,                      # §8.2
    }
    if accelerator.is_main_process:
        torch.save(model_config, os.path.join(args.checkpoint_dir, "vae_params.pth"))
    accelerator.wait_for_everyone()

    dataset = HDF5Dataset(args.data_path)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_gen
    )

    # mirrors the original's (double-dividing) batch math so the bug is under test
    per_gpu_batch_size = max(1, args.batch_size // accelerator.num_processes)
    loader_gen = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        IndexedDataset(train_dataset), shuffle=True, batch_size=per_gpu_batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        IndexedDataset(val_dataset), shuffle=False, batch_size=per_gpu_batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    ae = tae.DeapStack(**model_config)
    optimizer_ae = Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ae, optimizer_ae, train_loader, val_loader = accelerator.prepare(
        ae, optimizer_ae, train_loader, val_loader
    )
    unwrapped = accelerator.unwrap_model(ae)
    past_epoch, best_loss = tae.load_checkpoint(
        unwrapped, optimizer_ae, args.checkpoint_dir
    )

    diag = nd.NaNDiagnostics(
        unwrapped, optimizer_ae, diag_dir,
        buffer_steps=args.buffer_steps, snapshot_steps=args.snapshot_steps,
        metrics_every=args.metrics_every, anomaly=args.anomaly,
        inject_nan_at_step=args.inject_nan_at_step,
        inject_nan_phase=args.inject_nan_phase,
        is_main=True,   # every rank writes its own dir
    )
    diag.attach_forward_hooks()
    diag.patch_sine_cosine(tae)

    grads_synced = not (accelerator.num_processes > 1 and args.use_module_bypass)
    clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    ctx = {
        "rank": rank,
        "num_processes": accelerator.num_processes,
        "split_batches": True,
        "mixed_precision": str(accelerator.mixed_precision),
        "dataloader_batch_size": per_gpu_batch_size,
        "grads_synced": grads_synced,
        "module_bypass": bool(args.use_module_bypass),
        "grad_clip": clip,
        "num_terms": args.sincos_num_terms,
    }
    diag.log("=" * 78)
    diag.log(f"RUN start (accelerate)  rank={rank}  seed={args.seed}")
    diag.log(f"context={ctx}")
    diag.log(f"model_config={model_config}")
    diag.log(f"resume: past_epoch={past_epoch}  best_loss={best_loss}")
    diag.log(f"fixes: grad_clip={clip or 'OFF'}  "
             f"sincos_num_terms={args.sincos_num_terms}")
    diag.log(f"latent bounds: mu_clip={args.mu_clip}  "
             f"logvar in ({args.logvar_min},{args.logvar_max}) via sigmoid  "
             f"fourier_layernorm={args.fourier_layernorm}")

    max_beta = args.max_beta
    beta = max_beta
    beta_sched = frange_cycle_linear(
        n_iter=args.epochs, start=0.0, stop=max_beta,
        n_cycle=max(1, int(args.epochs / 5)), ratio=0.8,
    )
    delta = torch.tensor(args.min_kl, dtype=torch.float32, device=accelerator.device)
    patience = 0
    gstep = 0

    def get_loss(*a):
        # original 1_train_vae_aclr.py calls model.module.get_loss(...), which
        # bypasses the DDP wrapper's gradient all-reduce. Keep that by default so
        # the diagnostic exercises the path actually under test.
        target = getattr(ae, "module", ae) if args.use_module_bypass else unwrapped
        return target.get_loss(*a)

    try:
        for epoch in range(past_epoch, args.epochs):
            ae.train()
            t0 = time.time()
            for data, time_info, missing, masking, idx in train_loader:
                batch = {"data": data, "time_info": time_info,
                         "missing": missing, "masking": masking}
                diag.begin_step()
                diag.maybe_inject(gstep, "pre_forward")
                diag.check_inputs(batch)

                optimizer_ae.zero_grad()
                RE, KL = get_loss(data, time_info, missing, masking)
                diag.check_loss(RE, KL)
                loss = RE + beta * torch.maximum(KL, delta)
                accelerator.backward(loss)

                diag.check_grads()             # logs the TRUE pre-clip grad norm
                diag.optimizer_state_stats()
                diag.snapshot(gstep, batch, idx)
                diag.maybe_inject(gstep, "pre_step")
                pre_clip_norm = None
                if clip is not None:           # fix 1: gradient clipping
                    pre_clip_norm = float(
                        accelerator.clip_grad_norm_(ae.parameters(), clip)
                    )
                optimizer_ae.step()
                diag.check_params()

                scalars = {
                    "split": "train",
                    "loss": nd.finite_float(loss)[0],
                    "RE": nd.finite_float(RE)[0],
                    "KL": nd.finite_float(KL)[0],
                    "beta": float(beta),
                    "actual_batch": int(data.shape[0]),
                    "pre_clip_grad_norm": pre_clip_norm,
                    **ctx,
                }
                diag.heartbeat(gstep, epoch, every=args.heartbeat_every)
                diag.finish_step(gstep, epoch, scalars, batch, idx)

                gstep += 1
                if args.max_steps > 0 and gstep >= args.max_steps:
                    diag.log(f"reached --max_steps={args.max_steps}; stopping.")
                    diag.restore()
                    return

            # finiteness-only validation (also tracks mean val RE for the schedule)
            ae.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for data, time_info, missing, masking, idx in val_loader:
                    batch = {"data": data, "time_info": time_info,
                             "missing": missing, "masking": masking}
                    diag.begin_step()
                    diag.check_inputs(batch)
                    RE, KL = get_loss(data, time_info, missing, masking)
                    diag.check_loss(RE, KL)
                    diag.finish_step(gstep, epoch,
                                     {"split": "val", **ctx}, batch, idx,
                                     record=False)
                    rv = nd.finite_float(RE)[0]
                    if rv is not None:
                        vs += rv
                        vn += 1
            v_re = vs / max(1, vn)

            diag.log(f"epoch {epoch}/{args.epochs} done | val_RE={v_re:.6f} "
                     f"beta={beta:.6g} | {time.time() - t0:.1f}s")
            if epoch > args.warmup:
                beta = float(beta_sched[epoch])
                if v_re < best_loss:
                    best_loss = v_re
                    patience = 0
                else:
                    patience += 1
                    if patience > args.patience and max_beta > args.min_beta:
                        max_beta = max(max_beta * 0.5, args.min_beta)
                        beta_sched = frange_cycle_linear(
                            n_iter=args.epochs, start=0.0, stop=max_beta,
                            n_cycle=max(1, int(args.epochs / 5)), ratio=0.8,
                        )
                        patience = 0
                        diag.log(f"patience>{args.patience}: max_beta -> {max_beta:.6g}")
    except nd.NaNTripped as e:
        diag.log(f"STOP (rank {rank}): {e}")
        diag.restore()
        raise SystemExit(3)

    diag.restore()
    diag.log("run finished without a NaN trip.")


if __name__ == "__main__":
    main()
