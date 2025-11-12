# sample_1gpu.py
"""
LightningDiT 1‑D sequence sampling (single GPU)
==============================================

Usage
-----
python sample_1gpu.py --config configs/sample.yaml

The YAML must contain:
  ckpt_path         : path to EMA checkpoint (*.pt)
  data:
      seq_len       : length L
      in_chans      : number of variables C
      mean_std_file : torch file with dict {"mean": (1,C), "std": (1,C)}
  sample:
      total         : number of samples to generate
      batch_size    : batch size per forward
      cfg_scale     : 0 (=disabled) or >1 for classifier‑free guidance
      num_sampling_steps     : diffusion steps for ODE sampler
"""

import os, math, yaml, argparse, json, logging, torch
from time import strftime
from tqdm import tqdm

from models.lightning1dit import LightningDiT_models
from transport import create_transport, Sampler


# ------------------------------------------------------------------ #
#                        helper utils                                #
# ------------------------------------------------------------------ #
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def log(msg):
    t = strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[34m[LightningDiT‑Sample {t}]\033[0m {msg}", flush=True)


# ------------------------------------------------------------------ #
#                     main sampling routine                          #
# ------------------------------------------------------------------ #
@torch.no_grad()
def sample(cfg):

    # --------------------------- device ----------------------------- #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Sampling on {device} (mixed precision OFF)")

    # --------------------------- model ------------------------------ #
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
    state = ckpt["ema"] if "ema" in ckpt else ckpt

    seq_len  = cfg["data"]["seq_len"]
    patch    = cfg["model"].get("patch_size", 1)
    tokens   = seq_len // patch

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=tokens,
        in_channels=cfg["model"]["in_chans"],
        seq_len = cfg["data"]["seq_len"],
        num_classes=cfg["data"].get("num_classes",0),
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        learn_sigma=cfg["model"].get("learn_sigma", False),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    log("Model loaded.")

    # --------------------------- transport / sampler --------------- #
    transport = create_transport(**cfg["transport"])
    sampler   = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method = cfg["sample"].get("sampling_method", "heun"),
        num_steps       = cfg["sample"]["num_sampling_steps"],
        atol            = cfg["sample"].get("atol", 1e-5),
        rtol            = cfg["sample"].get("rtol", 1e-5),
        reverse         = cfg["sample"].get("reverse", False),
        timestep_shift  = cfg["sample"].get("timestep_shift", 0.0),
    )

    # --------------------------- output dir ------------------------ #
    out_dir = os.path.join(
        cfg["output_dir"],
        f"samples-{cfg['sample']['total']}-cfg{cfg['sample']['cfg_scale']}"
    )
    os.makedirs(out_dir, exist_ok=True)
    log(f"Saving tensors to {out_dir}")

    # --------------------------- generate -------------------------- #
    total      = cfg["sample"]["total"]
    batch_size = cfg["sample"]["batch_size"]
    cfg_scale  = cfg["sample"]["cfg_scale"]
    use_cfg    = cfg_scale > 1.0

    steps = math.ceil(total / batch_size)
    idx   = 0
    sample_list = list()

    for _ in tqdm(range(steps), total=steps):
        m = min(batch_size, total - idx)

        # latent noise
        z = torch.randn(m, cfg["model"]["in_chans"], seq_len, device=device)

        # classifier‑free guidance
        if use_cfg:
            z = torch.cat([z, z], 0)
            # no conditioning labels → just duplicate; y ignored inside model
            model_kwargs = dict(cfg_scale=cfg_scale,
                                cfg_interval=False,
                                cfg_interval_start=0.0)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = {}
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if use_cfg:
            samples, _ = samples.chunk(2, 0)

        ## save
        sample_list.append(samples.cpu().permute(0, 2, 1))

    log("Sampling done.")
    torch.save(torch.concat(sample_list,dim=0),os.path.join(out_dir, "samples.pt"))


# ------------------------------------------------------------------ #
#                            entry                                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sample(cfg)
