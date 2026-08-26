"""Stage 4: decode sampled latents back into a tabular synthetic dataset.

    python 4_sample_synthetic_data.py --config configs/my_run.yaml

Reads `vae.decode`. By default it picks up the DiT's `samples.pt` and un-normalises it
using the latents the DiT was trained on; set `vae.decode.scaled: true` if the latents
are already in the VAE's own scale.
"""
import argparse
import gc
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from model import DP as dp
from model import meld_config, process_edited as pce, timeautoencoder as tae
from model import vae_common as vc

_OUTPUT_KEYS = ("bins", "nums", "times", "eos", "missings")


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"latents not found: {path}")
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    if path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu").float()
    raise ValueError(f"expected a .pt/.pth/.npy latent file, got {path}")


def decode(model, samples, device, batch_size, emit_missing):
    collected = {k: [] for k in _OUTPUT_KEYS}
    collected["cats"] = []
    n_batches = (samples.shape[0] + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), total=n_batches, desc="Decoding"):
            batch = samples[i * batch_size:(i + 1) * batch_size].to(device)
            out = model.decoder(batch)
            for key in _OUTPUT_KEYS:
                if key in out:
                    collected[key].append(out[key].cpu())
            if "cats" in out:
                collected["cats"].append([c.cpu() for c in out["cats"]])

    gen = {k: torch.cat(v, dim=0) for k, v in collected.items()
           if k != "cats" and v}
    if collected["cats"]:
        n_cats = len(collected["cats"][0])
        gen["cats"] = [torch.cat([b[i] for b in collected["cats"]], dim=0)
                       for i in range(n_cats)]
    if not emit_missing:
        gen.pop("missings", None)
    return gen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the MeLD config file")
    args = ap.parse_args()

    cfg = meld_config.load(args.config)
    d = cfg.vae.decode
    tae.set_matmul_precision(cfg.vae.runtime.matmul_precision)
    device = vc.resolve_device(cfg.vae.runtime.device)
    seed = int(cfg.run.seed)
    set_seeds(seed)

    samples = load_samples(str(d.latents_path))
    print(f"Latents {tuple(samples.shape)} <- {d.latents_path}")

    if not bool(d.scaled):
        # The DiT trains on latents normalised to (-1, 1) per channel and never stores
        # the bounds, so they are recomputed from the same file it trained on.
        reference = load_samples(str(d.norm_source))
        # dp.normalize returns (normalized, min, max); the two names below are swapped
        # relative to that, but they are passed on positionally in the same swapped
        # order, so the errors cancel. Do not fix one half of this.
        _, max_val, min_val = dp.normalize(reference)
        samples = dp.inverse_normalize(samples, max_val, min_val)
        print(f"Un-normalised against {d.norm_source}")

    weights = cfg.best_path()
    if not os.path.exists(weights):
        raise FileNotFoundError(f"VAE weights not found: {weights}")
    model, _ = tae.DeapStack.from_checkpoint(weights, map_location=device)
    model = model.to(device).eval()
    print(f"VAE loaded from {weights}")

    gen = decode(model, samples, device, int(d.batch_size), bool(d.emit_missing))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    with open(str(cfg.vae.data.parser_path), "rb") as f:
        parser = pickle.load(f)

    data_size, seq_len, _ = samples.shape
    synth, _, syn_eos = pce.convert_to_tensor(parser, gen, data_size, seq_len)

    with torch.no_grad():
        # Same truncation the decoder uses, from the same helper, so the configured
        # eos_threshold applies here too instead of a second hard-coded 0.5.
        eos_mask = tae.eos_truncation_mask(
            syn_eos, float(cfg.vae.arch.eos_threshold)).view(-1).cpu().numpy()
        syn_id = torch.ones_like(syn_eos[:, :, 0]).cumsum(dim=0).view(-1).cpu().numpy()

    table, _ = pce.convert_to_table(parser, synth)
    table[str(d.date_column)] = np.ones_like(syn_id)
    table[str(d.id_column)] = syn_id
    table = table[eos_mask > 0]

    os.makedirs(str(d.output_dir), exist_ok=True)
    out = os.path.join(str(d.output_dir), str(d.output_template).format(
        name=str(cfg.run.name), seed=seed))
    table.to_csv(out, index=False, compression="gzip")
    print(f"Synthetic data {table.shape} -> {out}")


if __name__ == "__main__":
    main()
