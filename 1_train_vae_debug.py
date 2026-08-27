"""
Diagnostic version of 1_train_vae.py (single-GPU, plain Adam).

Same model, data, optimizer and beta schedule as 1_train_vae.py, plus:
  * fixed global seed + deterministic 90/10 split (a caught NaN is replayable)
  * every sample carries its original h5 row index
  * per-step tripwires in causal order
        inputs -> forward activations -> RE -> KL -> grads -> params
    reporting the EARLIEST stage that went non-finite
  * a rolling replay buffer: on the first NaN it dumps the pre-update model +
    optimizer state, the offending batch, and the last N per-step records
  * one consolidated `diagnosis.log` (plus JSON / .pt dumps) under
        <model_path>/vae[_<vae_model>]/nan_diag/
    -- copy that directory off the server to analyse the failure offline.

Two mitigations from NAN_DIAGNOSIS_REPORT.md are wired in as opt-in flags so the
SAME instrumented harness can validate them:
  * --grad_clip <max_norm>   fix 1: clip_grad_norm_ right before optimizer.step()
                             (applied AFTER the true pre-clip grad norm is logged)
  * --sincos_num_terms <n>   fix 2: frequency terms in compute_sine_cosine
                             (default 16 -> top freq 2**15; use 6-8 to tame it)
Both default to OFF/unchanged, so a bare run still reproduces the NaN.

Example:
    # fast self-test: NaN poked into fc_mu.weight before optimizer.step() at
    # step 5 -> expect a clean trip at stage 'params' with SystemExit(3).
    python MeLD/1_train_vae_debug.py -DP data.h5 -MP runs/ \\
        --inject_nan_at_step 5 --max_steps 10
    # reproduce the NaN (baseline)
    python MeLD/1_train_vae_debug.py -DP data.h5 -MP runs/ --seed 0
    # validate the fix with the same instrumentation
    python MeLD/1_train_vae_debug.py -DP data.h5 -MP runs/ --seed 0 \\
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

from MeLD.model import timeautoencoder as tae
from MeLD.model import nan_diagnostics as nd
from MeLD.model.data_loader import HDF5Dataset


# ==========================================
# Helpers
# ==========================================
class IndexedDataset(Dataset):
    """Return (*sample, original_row_index).

    Unwraps nested torch Subsets (from random_split) so the reported index is
    the true h5 row, which is what the NaN dumps need to be replayable.
    """

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
        # preserve HDF5Dataset's fast batched read path (see data_loader.py)
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


def setup_logger(checkpoint_dir):
    logger = logging.getLogger("vae_debug")
    logger.setLevel(logging.INFO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training.log")
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


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


# ==========================================
# Instrumented epochs
# ==========================================
def train_epoch(epoch, loader, model, optimizer, device, args, beta, diag, gstep):
    model.train()
    delta = torch.tensor(args.min_kl, dtype=torch.float32, device=device)
    tot_loss = tot_re = tot_kl = 0.0
    nb = len(loader)
    clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None

    for data, time_info, missing, masking, idx in loader:
        data = data.to(device)
        time_info = time_info.to(device)
        missing = missing.to(device)
        masking = masking.to(device)
        batch = {"data": data, "time_info": time_info,
                 "missing": missing, "masking": masking}

        diag.begin_step()
        diag.maybe_inject(gstep, "pre_forward")
        diag.check_inputs(batch)

        optimizer.zero_grad()
        RE, KL = model.get_loss(data, time_info, missing, masking)
        diag.check_loss(RE, KL)

        loss = RE + beta * torch.maximum(KL, delta)
        loss.backward()

        diag.check_grads()               # logs the TRUE pre-clip gradient norm
        diag.optimizer_state_stats()
        diag.snapshot(gstep, batch, idx)   # BEFORE the update
        diag.maybe_inject(gstep, "pre_step")
        pre_clip_norm = None
        if clip is not None:               # fix 1: gradient clipping
            # clip_grad_norm_ returns the total norm BEFORE clipping
            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            )
        optimizer.step()
        diag.check_params()

        re_v, _ = nd.finite_float(RE)
        kl_v, _ = nd.finite_float(KL)
        loss_v, _ = nd.finite_float(loss)
        scalars = {"split": "train", "loss": loss_v, "RE": re_v, "KL": kl_v,
                   "beta": float(beta), "grad_clip": clip,
                   "pre_clip_grad_norm": pre_clip_norm,
                   "num_terms": args.sincos_num_terms}
        diag.heartbeat(gstep, epoch, every=args.heartbeat_every)
        diag.finish_step(gstep, epoch, scalars, batch, idx)   # records or trips

        tot_loss += (loss_v if loss_v is not None else float("nan"))
        tot_re += (re_v if re_v is not None else float("nan"))
        tot_kl += (kl_v if kl_v is not None else float("nan"))
        gstep += 1
        if args.max_steps > 0 and gstep >= args.max_steps:
            diag.log(f"reached --max_steps={args.max_steps}; stopping.")
            return tot_loss / nb, tot_re / nb, tot_kl / nb, gstep, True

    return tot_loss / nb, tot_re / nb, tot_kl / nb, gstep, False


def val_epoch(epoch, loader, model, device, args, beta, diag, gstep):
    model.eval()
    delta = torch.tensor(args.min_kl, dtype=torch.float32, device=device)
    tot_re = tot_kl = 0.0
    nb = len(loader)
    with torch.no_grad():
        for data, time_info, missing, masking, idx in loader:
            data = data.to(device)
            time_info = time_info.to(device)
            missing = missing.to(device)
            masking = masking.to(device)
            batch = {"data": data, "time_info": time_info,
                     "missing": missing, "masking": masking}
            diag.begin_step()
            diag.check_inputs(batch)
            RE, KL = model.get_loss(data, time_info, missing, masking)
            diag.check_loss(RE, KL)
            # finiteness-only check, do not pollute the training trajectory buffer
            diag.finish_step(gstep, epoch,
                             {"split": "val", "RE": nd.finite_float(RE)[0],
                              "KL": nd.finite_float(KL)[0]},
                             batch, idx, record=False)
            tot_re += nd.finite_float(RE)[0] or float("nan")
            tot_kl += nd.finite_float(KL)[0] or float("nan")
    return tot_re / nb, tot_kl / nb


# ==========================================
# Main
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--early_stop_patience", "-ESP", type=int, default=100)
    parser.add_argument("--save_every", "-SE", type=int, default=50)

    parser.add_argument("--min_beta", type=float, default=1e-5)
    parser.add_argument("--max_beta", type=float, default=1e-2)
    parser.add_argument("--min_kl", type=float, default=0.0)

    parser.add_argument("--lat_dim", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--bidirectional", action="store_true", default=False)

    # --- fixes from NAN_DIAGNOSIS_REPORT.md (opt-in; defaults reproduce the NaN) ---
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="fix 1: clip_grad_norm_ max_norm before optimizer.step "
                             "(0 = disabled). Try 1.0-5.0.")
    parser.add_argument("--sincos_num_terms", type=int, default=16,
                        help="fix 2: frequency terms in compute_sine_cosine "
                             "(16 -> top freq 2**15, the gradient amplifier; "
                             "use 6-8).")
    # --- latent / Fourier bounds (§8.2 / §8.3); ENABLED by default here ---
    parser.add_argument("--fourier_layernorm", action=BooleanOptionalAction,
                        default=True,
                        help="LayerNorm the mlp_nums (Fourier) embedding before "
                             "it joins x_emb_sum.")
    parser.add_argument("--logvar_min", type=float, default=-6.0)
    parser.add_argument("--logvar_max", type=float, default=2.0,
                        help="fc_logvar squashed to (logvar_min, logvar_max) via "
                             "a smooth sigmoid (replaces the dead Hardtanh).")
    parser.add_argument("--mu_clip", type=float, default=5.0,
                        help="soft bound mu := mu_clip*tanh(mu/mu_clip); "
                             "<=0 disables.")

    # --- diagnostic-only args ---
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diag_dir", type=str, default=None)
    parser.add_argument("--buffer_steps", type=int, default=200,
                        help="full per-step records kept for metrics_tail.jsonl")
    parser.add_argument("--snapshot_steps", type=int, default=2)
    parser.add_argument("--heartbeat_every", type=int, default=50,
                        help="compact one-liner into diagnosis.log every N steps")
    parser.add_argument("--metrics_every", type=int, default=25,
                        help="coarse record into metrics.jsonl every N steps "
                             "(the ring buffer keeps every step regardless)")
    parser.add_argument("--nondeterministic", action="store_true", default=False,
                        help="skip cuDNN-deterministic / benchmark-off; use with "
                             "--num_workers 8 to stay close to 1_train_vae.py "
                             "numerics if the deterministic run does not trip.")
    parser.add_argument("--anomaly", action="store_true", default=False,
                        help="torch.autograd.set_detect_anomaly(True): 10-30x "
                             "slower, only catches NaN in backward, points at an "
                             "op downstream of the cause. Secondary tool.")
    parser.add_argument("--inject_nan_at_step", type=int, default=-1,
                        help="harness self-test: poke a NaN into fc_mu.weight at "
                             "this global step; expect a clean trip + dumps.")
    parser.add_argument("--inject_nan_phase", choices=["pre_forward", "pre_step"],
                        default="pre_step",
                        help="pre_step -> trip at stage 'params'; "
                             "pre_forward -> trip at stage 'forward'.")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = deterministic + clean tracebacks. Use 8 to "
                             "match 1_train_vae.py's dataloader.")
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(
        args.model_path, f"vae_{args.vae_model}" if args.vae_model else "vae"
    )
    if args.diag_dir is None:
        args.diag_dir = os.path.join(args.checkpoint_dir, "nan_diag")

    logger = setup_logger(args.checkpoint_dir)
    logger.info(f"device={device}  seed={args.seed}  diag_dir={args.diag_dir}")

    nd.set_global_seed(args.seed, deterministic=not args.nondeterministic)

    # --- data metadata / model config ---
    with h5py.File(args.data_path, "r") as f:
        n_bins = int(f.attrs["n_bins"])
        n_cats = int(f.attrs["n_cats"])
        n_nums = int(f.attrs["n_nums"])
        cards = f.attrs["cards"].tolist()
        N, seq_len, feature_size = f["processed_data"].shape
        time_dim = f["time_info"].shape[2]
        missing_feat_dim = f["missing"].shape[2]
        assert n_nums == missing_feat_dim
        assert sum([n_bins, n_cats, n_nums]) == feature_size

    logger.info(f"Dataset: N={N}, seq_len={seq_len}, feature_size={feature_size}")

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
    torch.save(model_config, os.path.join(args.checkpoint_dir, "vae_params.pth"))

    # --- deterministic split + indexed datasets ---
    dataset = HDF5Dataset(args.data_path)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_gen
    )

    loader_gen = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        IndexedDataset(train_dataset), shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        IndexedDataset(val_dataset), shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- model / optimizer ---
    ae = tae.DeapStack(**model_config).to(device)
    optimizer_ae = Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"{sum(p.numel() for p in ae.parameters()) / 1e6:.1f}M params")

    past_epoch, best_loss = tae.load_checkpoint(ae, optimizer_ae, args.checkpoint_dir)

    # --- diagnostics ---
    diag = nd.NaNDiagnostics(
        ae, optimizer_ae, args.diag_dir,
        buffer_steps=args.buffer_steps, snapshot_steps=args.snapshot_steps,
        metrics_every=args.metrics_every, anomaly=args.anomaly,
        inject_nan_at_step=args.inject_nan_at_step,
        inject_nan_phase=args.inject_nan_phase,
    )
    diag.attach_forward_hooks()
    diag.patch_sine_cosine(tae)
    diag.log("=" * 78)
    diag.log(f"RUN start  seed={args.seed}  device={device}  "
             f"batch_size={args.batch_size}  lr={args.lr}  wd={args.weight_decay}")
    diag.log(f"model_config={model_config}")
    diag.log(f"resume: past_epoch={past_epoch}  best_loss={best_loss}")
    diag.log(f"beta: max_beta={args.max_beta}  min_kl={args.min_kl}  "
             f"warmup={args.warmup}")
    _clip_s = f"{args.grad_clip}" if args.grad_clip and args.grad_clip > 0 else "OFF"
    diag.log(f"fixes: grad_clip={_clip_s}  sincos_num_terms={args.sincos_num_terms}")
    diag.log(f"latent bounds: mu_clip={args.mu_clip}  "
             f"logvar in ({args.logvar_min},{args.logvar_max}) via sigmoid  "
             f"fourier_layernorm={args.fourier_layernorm}")

    # beta trajectory kept identical to 1_train_vae.py (incl. the patience-driven
    # max_beta halving) so the run reproduces the same optimisation path.
    max_beta = args.max_beta
    beta = max_beta
    beta_sched = frange_cycle_linear(
        n_iter=args.epochs, start=0.0, stop=max_beta,
        n_cycle=max(1, int(args.epochs / 5)), ratio=0.8,
    )
    patience = 0
    gstep = 0
    stop = False

    try:
        for epoch in range(past_epoch, args.epochs):
            t0 = time.time()
            tr_loss, tr_re, tr_kl, gstep, stop = train_epoch(
                epoch, train_loader, ae, optimizer_ae, device, args, beta, diag, gstep
            )
            if stop:
                break
            v_re, v_kl = val_epoch(
                epoch, val_loader, ae, device, args, beta, diag, gstep
            )
            diag.log(
                f"epoch {epoch}/{args.epochs} | tr_RE={tr_re:.6f} val_RE={v_re:.6f} "
                f"KL={tr_kl:.4f} beta={beta:.6g} | {time.time() - t0:.1f}s"
            )

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
        diag.log(f"STOP: {e}")
        diag.restore()
        logger.error(f"NaN reproduced. See {args.diag_dir}/diagnosis.log")
        raise SystemExit(3)

    diag.restore()
    diag.log("run finished without a NaN trip.")


if __name__ == "__main__":
    main()
