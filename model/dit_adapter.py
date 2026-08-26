"""Translate a MeLD config into the config the vendored LightningD1T expects.

`model/LightningD1T/` is third-party code with its own YAML schema and a
`--config`-only CLI. Rather than edit it, this module emits a config in exactly that
schema and launches the vendored entry point as a subprocess.

The subprocess matters for more than tidiness: the DiT needs its own conda env
(torch 2.2 + transformers) while the VAE half runs in `ddpm`, so a single process
could never import both. `dit.runtime.python` selects that interpreter.

Deliberately imports nothing heavier than the standard library plus yaml, so it can
run in either environment.
"""
import os
import re
import subprocess
import sys

import yaml

# Emitted verbatim; these are the sections the vendored scripts read.
_TRAIN_SECTIONS = ("data", "model", "train", "optimizer", "transport")
_SAMPLE_SECTIONS = ("data", "model", "train", "optimizer", "transport", "sample")

_CKPT_RE = re.compile(r"^(\d+)\.pt$")


class DitAdapterError(Exception):
    pass


def _prune_nulls(node, _in_transport=False):
    """Drop null-valued keys, because the vendored code branches on key presence.

    `train_single.py` does `if "weight_init" in cfg["train"]` and then
    `torch.load(cfg["train"]["weight_init"])`, so emitting an explicit null turns a
    skipped branch into `torch.load(None)`. Same shape of problem for `data.image_size`
    and a top-level `vae` key, which switch entire code paths.

    `transport` is exempt: it is splatted into create_transport(), whose defaults for
    loss_weight/train_eps/sample_eps are None and meaningful.
    """
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out[key] = _prune_nulls(value, _in_transport or key == "transport")
        elif value is not None or _in_transport:
            out[key] = value
    return out


def _encode_mixed_precision(value, entry):
    """The two trainers disagree about this key's type.

    train_multigpu.py hands it to Accelerate and wants "no"/"fp16"/"bf16";
    train_single.py hands it to GradScaler(enabled=...) and wants a bool -- where any
    non-empty string, "no" included, is truthy and would silently enable the scaler.
    """
    if entry == "train_single.py":
        if isinstance(value, str):
            return value.lower() not in ("", "no", "none", "false")
        return bool(value)
    if isinstance(value, bool):
        return "bf16" if value else "no"
    return str(value)


def _latent_shape(path):
    """(N, seq_len, channels) of the step-2 latent file, or None if absent."""
    if not path or not os.path.exists(path):
        return None
    import torch  # local: keeps this module importable without torch
    tensor = torch.load(path, map_location="cpu")
    if tensor.dim() != 3:
        raise DitAdapterError(
            f"expected a 3-D (N, seq_len, channels) latent tensor in {path}, "
            f"got shape {tuple(tensor.shape)}")
    return tuple(tensor.shape)


def newest_checkpoint(ckpt_dir):
    """Newest `{step:07d}.pt`, ordered numerically rather than lexicographically."""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for name in os.listdir(ckpt_dir):
        m = _CKPT_RE.match(name)
        if m:
            steps.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
    return max(steps)[1] if steps else None


def to_vendored(cfg, phase):
    """Build the vendored-schema dict for `phase` in {"train", "sample"}."""
    if phase not in ("train", "sample"):
        raise DitAdapterError(f"phase must be 'train' or 'sample', got {phase!r}")

    from omegaconf import OmegaConf
    dit = OmegaConf.to_container(cfg.dit, resolve=True)

    sections = _TRAIN_SECTIONS if phase == "train" else _SAMPLE_SECTIONS
    out = {k: dict(dit[k]) for k in sections if k in dit}

    # seq_len and in_chans describe the latent tensor, so take them from the tensor
    # itself rather than trusting three files to agree. The vendored LatentDataset
    # never cross-checks them, so a mismatch would otherwise surface as an opaque
    # shape error somewhere inside the model.
    latent_shape = _latent_shape(out["data"].get("data_path"))
    if latent_shape is not None:
        _, seq_len, channels = latent_shape
        if out["data"].get("seq_len") is None:
            out["data"]["seq_len"] = seq_len
        elif int(out["data"]["seq_len"]) != seq_len:
            raise DitAdapterError(
                f"dit.data.seq_len is {out['data']['seq_len']} but the latents in "
                f"{out['data']['data_path']} have seq_len {seq_len}")
        if int(out["model"]["in_chans"]) != channels:
            raise DitAdapterError(
                f"dit.model.in_chans is {out['model']['in_chans']} but the latents in "
                f"{out['data']['data_path']} have {channels} channels "
                f"(check vae.arch.lat_dim)")
    elif out["data"].get("seq_len") is None:
        raise DitAdapterError(
            f"dit.data.seq_len could not be derived: no latent file at "
            f"{out['data'].get('data_path')}. Run step 2 first, or set it explicitly.")

    entry = dit["runtime"]["entry_train" if phase == "train" else "entry_sample"]
    if "mixed_precision" in out.get("train", {}):
        out["train"]["mixed_precision"] = _encode_mixed_precision(
            out["train"]["mixed_precision"], dit["runtime"]["entry_train"])
    # Only train_multigpu.py reads train.seed; harmless elsewhere but keep it honest.
    if entry != "train_multigpu.py":
        out.get("train", {}).pop("seed", None)

    if phase == "sample":
        ckpt_step = dit.get("inference", {}).get("ckpt_step")
        if ckpt_step is None:
            ckpt = newest_checkpoint(cfg.dit_ckpt_dir())
            if ckpt is None:
                raise DitAdapterError(
                    f"no checkpoints found in {cfg.dit_ckpt_dir()} -- train the DiT "
                    f"first, or set dit.inference.ckpt_step")
        else:
            ckpt = os.path.join(cfg.dit_ckpt_dir(), f"{int(ckpt_step):07d}.pt")
            if not os.path.exists(ckpt):
                raise DitAdapterError(f"checkpoint not found: {ckpt}")
        # The vendored samplers read these from the top level, and never derive
        # ckpt_path from exp_name -- which is why the tracked configs restate the
        # same directory three times.
        out["ckpt_path"] = ckpt
        out["output_dir"] = cfg.dit_exp_dir()

    return _prune_nulls(out)


def materialize(cfg, phase):
    """Write the vendored config to disk and return its absolute path."""
    payload = to_vendored(cfg, phase)
    base = str(cfg.dit.runtime.materialized_config)
    root, ext = os.path.splitext(base)
    path = f"{root}.{phase}{ext or '.yaml'}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def run(cfg, phase, dry_run=False):
    """Materialize the config and invoke the vendored entry point."""
    dit_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "LightningD1T")
    if not os.path.isdir(dit_dir):
        raise DitAdapterError(f"vendored DiT not found at {dit_dir}")

    entry = str(cfg.dit.runtime.entry_train if phase == "train"
                else cfg.dit.runtime.entry_sample)
    if not os.path.exists(os.path.join(dit_dir, entry)):
        raise DitAdapterError(f"entry point not found: {os.path.join(dit_dir, entry)}")

    config_path = materialize(cfg, phase)

    launcher = cfg.dit.runtime.launcher
    launcher = list(launcher) if launcher else []
    python = str(cfg.dit.runtime.python or sys.executable)
    cmd = ([python] if not launcher else launcher) + [entry, "--config", config_path]

    env = dict(os.environ)
    clip = cfg.dit.runtime.clip_model_path
    if clip:
        # Consumed by CLIPTextEmbedder, which upstream hardcodes to an absolute path
        # on the original author's machine.
        env["MELD_CLIP_PATH"] = str(clip)
    # The vendored modules use flat imports (`from models...`), so they must resolve
    # against their own directory.
    env["PYTHONPATH"] = os.pathsep.join(
        [dit_dir] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    print(f"[dit_adapter] config : {config_path}")
    print(f"[dit_adapter] cwd    : {dit_dir}")
    print(f"[dit_adapter] command: {' '.join(cmd)}")
    if dry_run:
        return 0

    return subprocess.run(cmd, cwd=dit_dir, env=env, check=True).returncode
