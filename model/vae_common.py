"""Shared machinery for the VAE stages.

`1_train_vae.py` runs both plain (`python 1_train_vae.py`) and multi-GPU
(`accelerate launch 1_train_vae.py`) via an unconditional `Accelerator()`, which is a
no-op at one process. The two used to be separate ~350-line scripts and had already
drifted (only the Accelerate copy honoured --num_workers, and then only in two of its
three dataloaders; only the plain copy wrote the loss table). `run_training` here is
the entire epoch loop -- beta schedule, checkpointing, patience/early-stop -- so there
is one copy to change and no way for the two paths to diverge again.
"""
import logging
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from . import timeautoencoder as tae
from .data_loader import HDF5Dataset


class IndexedDataset(Dataset):
    """Wraps a dataset so each item carries its index.

    Encoding runs with shuffle=False, but HDF5Dataset.__getitems__ internally sorts
    indices for disk locality, so the indices are needed to restore the original row
    order afterwards.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        batch_data = self.base_dataset[idx]
        if isinstance(batch_data, (tuple, list)):
            return (*batch_data, idx)
        return batch_data, idx


def setup_logger(checkpoint_dir, log_name="training.log", is_main_process=True):
    """Log to console and to a file inside the run directory."""
    logger = logging.getLogger(f"meld.vae.{checkpoint_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not is_main_process:
        logger.addHandler(logging.NullHandler())
        return logger

    os.makedirs(checkpoint_dir, exist_ok=True)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(os.path.join(checkpoint_dir, log_name))
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    """Cyclical linear schedule for the KL weight."""
    if n_cycle < 1:
        raise ValueError(f"n_cycle must be >= 1, got {n_cycle}")
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


def resolve_device(spec="auto"):
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(spec))


def read_h5_shapes(h5_path):
    """Read the data-derived DeapStack arguments from the HDF5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 data not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for name in ("processed_data", "time_info", "missing", "masking"):
            if name not in f:
                raise KeyError(f"{h5_path} is missing the '{name}' dataset")
        n_samples, seq_len, feature_size = f["processed_data"].shape
        shapes = {
            "seq_len": int(seq_len),
            "feature_size": int(feature_size),
            "time_dim": int(f["time_info"].shape[2]),
            "n_bins": int(f.attrs["n_bins"]),
            "n_cats": int(f.attrs["n_cats"]),
            "n_nums": int(f.attrs["n_nums"]),
            "cards": [int(c) for c in f.attrs["cards"]],
        }
        missing_dim = int(f["missing"].shape[2])

    total = shapes["n_bins"] + shapes["n_cats"] + shapes["n_nums"]
    if total != shapes["feature_size"]:
        raise ValueError(
            f"n_bins+n_cats+n_nums ({total}) != feature_size ({shapes['feature_size']})")
    if shapes["n_nums"] != missing_dim:
        raise ValueError(
            f"n_nums ({shapes['n_nums']}) != missing.shape[2] ({missing_dim})")
    shapes["n_samples"] = int(n_samples)
    return shapes


def loader_kwargs(cfg, batch_size):
    """DataLoader options from config, valid for num_workers == 0.

    prefetch_factor and persistent_workers must be absent, not just falsy, when there
    are no workers, or torch raises.
    """
    rt = cfg.vae.runtime
    workers = int(rt.num_workers)
    kwargs = dict(batch_size=int(batch_size), num_workers=workers,
                  pin_memory=bool(rt.pin_memory))
    if workers > 0:
        if rt.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(rt.prefetch_factor)
        kwargs["persistent_workers"] = bool(rt.persistent_workers)
    return kwargs


def split_dataset(cfg, dataset):
    """Deterministic train/validation split.

    Seeded on purpose: an unseeded split reshuffles on every resume, so the recorded
    best validation loss refers to a different set than the one being compared to it.
    """
    val_size = int(round(float(cfg.vae.data.val_fraction) * len(dataset)))
    val_size = max(1, min(val_size, len(dataset) - 1))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(cfg.vae.data.split_seed))
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_model(cfg, shapes):
    """Instantiate DeapStack from config plus the data-derived shapes."""
    kwargs = cfg.deapstack_kwargs(shapes)
    return tae.DeapStack(**kwargs), kwargs


LEGACY_PARAM_KEYS = (
    "channels", "batch_size", "seq_len", "n_bins", "n_cats", "n_nums", "cards",
    "feature_size", "hidden_size", "num_layers", "bidirectional", "emb_dim",
    "time_dim", "lat_dim",
)


def write_legacy_params(path, kwargs):
    """Write the 14-key vae_params.pth.

    Nothing in this pipeline reads it any more -- checkpoints are self-describing.
    It exists solely for baseline/MeLD-Transformer, whose TimeLDM takes exactly these
    fourteen arguments and no **kwargs, so the file must never grow.
    """
    payload = {k: kwargs[k] for k in LEGACY_PARAM_KEYS}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return payload


def process_epoch(epoch, data_loader, model, optimizer, device, min_kl, beta,
                  is_train=True, log_every_batch=False, backward=None,
                  reduce_fn=None, sync=None):
    """Run one epoch. `backward`/`reduce_fn`/`sync` let Accelerate hook in."""
    total_loss = total_re = total_kl = 0.0
    data_time_total = compute_time_total = 0.0

    model.train() if is_train else model.eval()
    num_batches = len(data_loader)
    batch_start_time = time.time()

    for batch_idx, (data, time_info, missing, masking) in enumerate(data_loader):
        data = data.to(device)
        time_info = time_info.to(device)
        missing = missing.to(device)
        masking = masking.to(device)

        per_batch_data_time = time.time() - batch_start_time
        data_time_total += per_batch_data_time
        compute_start_time = time.time()

        if is_train:
            optimizer.zero_grad()

        target = model.module if hasattr(model, "module") else model
        RE, KL = target.get_loss(data, time_info, missing, masking)
        delta = torch.tensor(min_kl, dtype=KL.dtype, device=KL.device)
        loss = RE + beta * torch.maximum(KL, delta)

        if is_train:
            if backward is None:
                loss.backward()
            else:
                backward(loss)
            optimizer.step()

        if reduce_fn is None:
            avg_loss, avg_RE, avg_KL = loss.item(), RE.item(), KL.item()
        else:
            avg_loss, avg_RE, avg_KL = (reduce_fn(loss), reduce_fn(RE), reduce_fn(KL))

        total_loss += avg_loss
        total_re += avg_RE
        total_kl += avg_KL

        if sync is not None:
            sync()
        compute_time_total += time.time() - compute_start_time

        if log_every_batch:
            print(f"\nEpoch {epoch} - Batch {batch_idx}/{num_batches}| "
                  f"Loss: {avg_loss:.6f} | RE: {avg_RE:.6f} | KL: {avg_KL:.2f} | "
                  f"Beta: {beta:.6f} | Data Loading Time: {per_batch_data_time:.2f}s | "
                  f"GPU Compute Time: {time.time() - compute_start_time:.2f}s")

        batch_start_time = time.time()

    return (total_loss / num_batches, total_re / num_batches, total_kl / num_batches,
            data_time_total / num_batches, compute_time_total / num_batches)


def encode_latents(model, dataset, cfg, device, batch_size=None, sample_size=None,
                   gather_fn=None, is_main_process=True):
    """Encode every sequence, restoring the dataset's original row order."""
    from tqdm import tqdm

    target = model.module if hasattr(model, "module") else model
    target.eval()

    kwargs = loader_kwargs(cfg, batch_size or cfg.vae.encode.batch_size)
    loader = DataLoader(IndexedDataset(dataset), shuffle=False, drop_last=False, **kwargs)

    emb_list, idxs_list, seen = [], [], 0
    with torch.no_grad():
        for data, time_info, missing, masking, idxs in tqdm(loader, desc="Encoding"):
            _, emb_batch, _, _ = target(
                data.to(device), time_info.to(device),
                missing.to(device), masking.to(device))
            if gather_fn is not None:
                emb_batch, idxs = gather_fn(emb_batch), gather_fn(idxs)
            if is_main_process:
                emb_list.append(emb_batch.cpu())
                idxs_list.append(idxs.cpu())
            seen += data.shape[0]
            # Stop once enough rows are collected; the old code compared batch counts
            # after appending, so it always ran one batch past the limit and then
            # never trimmed the result.
            if sample_size is not None and seen >= int(sample_size):
                break

    if not is_main_process:
        return None

    all_embs = torch.cat(emb_list)
    all_indices = torch.cat(idxs_list).numpy()
    sorted_embs = all_embs[all_indices.argsort()]
    if sample_size is not None:
        sorted_embs = sorted_embs[: int(sample_size)]
    return sorted_embs


def open_dataset(h5_path):
    return HDF5Dataset(h5_path)


def run_training(cfg, model, optimizer, train_loader, val_loader, device, out_dir, log,
                 past_epoch=0, best_loss=float("inf"), backward=None, reduce_fn=None,
                 is_main=True, barrier=None, unwrap=None):
    """The epoch loop: beta schedule, best/periodic checkpointing, patience/early-stop.

    `backward`/`reduce_fn`/`barrier`/`unwrap` let Accelerate hook in; a single-process
    caller leaves them at their defaults (no-ops / identity).
    """
    unwrap = unwrap or (lambda m: m)
    t = cfg.vae.train
    beta = max_beta = float(t.beta.max)
    n_cycle = int(t.beta.n_cycle)
    beta_sched = frange_cycle_linear(
        n_iter=int(t.epochs), start=float(t.beta.schedule_start),
        stop=max_beta, n_cycle=n_cycle, ratio=float(t.beta.ratio))
    patience = 0

    for epoch in range(past_epoch, int(t.epochs)):
        _, tr_re, tr_kl, tr_wait, tr_comp = process_epoch(
            epoch, train_loader, model, optimizer, device,
            float(t.beta.min_kl), beta, is_train=True,
            backward=backward, reduce_fn=reduce_fn)
        with torch.no_grad():
            _, val_re, val_kl, _, _ = process_epoch(
                epoch, val_loader, model, optimizer, device,
                float(t.beta.min_kl), beta, is_train=False, reduce_fn=reduce_fn)

        if epoch % int(t.log_every) == 0:
            log(f"Epoch: {epoch}/{t.epochs} | Tr RE: {tr_re:.6f} | "
                f"Val RE: {val_re:.6f} | KL: {tr_kl:.2f} | Beta: {beta:.6f} | "
                f"Tr Wait (CPU): {tr_wait:.2f}s | Tr Compute: {tr_comp:.2f}s")

        if epoch <= int(t.warmup):
            continue

        beta = beta_sched[epoch]

        if val_re < best_loss:
            best_loss = val_re
            patience = 0
            log(f"New best at epoch {epoch}: train {tr_re:.6f} / "
                f"val {val_re:.6f}. Saving...")
            if barrier is not None:
                barrier()
            if is_main:
                unwrap(model).save(cfg.best_path(), epoch=epoch, loss=best_loss)
        else:
            patience += 1
            if patience > int(t.patience) and max_beta > float(t.beta.min):
                max_beta = max(max_beta * float(t.beta.decay_factor), float(t.beta.min))
                log(f"Patience > {t.patience}. Reducing max_beta to {max_beta:.6f}")
                beta_sched = frange_cycle_linear(
                    n_iter=int(t.epochs), start=float(t.beta.schedule_start),
                    stop=max_beta, n_cycle=n_cycle, ratio=float(t.beta.ratio))
                patience = 0
            if patience > int(t.early_stop_patience):
                log(f"Patience > {t.early_stop_patience}. Early stopping.")
                break

        if epoch % int(t.save_every) == 0:
            if barrier is not None:
                barrier()
            if is_main:
                # `epoch` is already absolute -- the loop starts at past_epoch. The old
                # code added past_epoch again, inflating the number in the filename that
                # resume then sorts on.
                tae.save_checkpoint(unwrap(model), optimizer, epoch, tr_re,
                                    out_dir, cfg.checkpoint_path(epoch))
                with open(os.path.join(out_dir, str(cfg.vae.output.loss_table_name)),
                         "a") as f:
                    f.write(f"\nepoch {epoch:4d} | train_re: {tr_re:.6f} | "
                            f"train_kl: {tr_kl:.2f} | val_re: {val_re:.6f} | "
                            f"val_kl: {val_kl:.2f} | beta: {max_beta}")

    return best_loss
