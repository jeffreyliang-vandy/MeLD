"""Record / compare a DeapStack architecture + loss fingerprint.

Used to prove that exposing hard-coded values as configuration keys does not change
the model at its default settings. Run `record` on a known-good tree, `check` after
refactoring, and the two must agree exactly.

    python tools/arch_fingerprint.py record golden.json
    python tools/arch_fingerprint.py check  golden.json

Deliberately builds the model from an explicit dict of shape arguments and runs on
synthetic tensors, so it needs neither an HDF5 file nor the real cohort data.
"""
import argparse
import json
import sys

import torch

from model import timeautoencoder as tae

# A small but structurally complete spec: binary, categorical and numeric columns are
# all non-empty, and time_dim is an even number of sin/cos pairs.
SHAPES = {
    "channels": 8,
    "batch_size": 4,
    "seq_len": 8,
    "n_bins": 3,
    "n_cats": 2,
    "n_nums": 6,
    "cards": [4, 5],
    "feature_size": 11,          # n_bins + n_cats + n_nums
    "hidden_size": 16,
    "num_layers": 2,
    "bidirectional": False,
    "emb_dim": 12,
    "time_dim": 8,
    "lat_dim": 4,
}

SEED = 20260818


def _batch(spec, device):
    """Synthetic inputs matching the DP.partition_multi_seq output contract."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    B, L = spec["batch_size"], spec["seq_len"]
    n_bins, n_nums, cards = spec["n_bins"], spec["n_nums"], spec["cards"]

    bins = torch.randint(0, 2, (B, L, n_bins), generator=g).float()
    cats = torch.stack(
        [torch.randint(0, c, (B, L), generator=g) for c in cards], dim=-1
    ).float()
    nums = torch.rand(B, L, n_nums, generator=g)
    x = torch.cat([bins, cats, nums], dim=-1)

    time_info = torch.rand(B, L, spec["time_dim"], generator=g) * 2 - 1
    missing = torch.randint(0, 2, (B, L, n_nums), generator=g).float()
    # masking is (end_of_sequence, padding_code); keep at least one real step per row.
    eos = torch.zeros(B, L)
    eos[:, -1] = 1.0
    padding = torch.zeros(B, L)
    masking = torch.stack([eos, padding], dim=-1)

    return tuple(t.to(device) for t in (x, time_info, missing, masking))


def fingerprint(spec=None):
    spec = dict(spec or SHAPES)
    device = torch.device("cpu")

    torch.manual_seed(SEED)
    model = tae.DeapStack(**spec).to(device)

    sd = model.state_dict()
    params = sorted((k, list(v.shape)) for k, v in sd.items())
    n_params = sum(p.numel() for p in model.parameters())

    x, time_info, missing, masking = _batch(spec, device)
    torch.manual_seed(SEED)          # pin the reparametrisation noise
    model.train()
    loss_re, loss_kl = model.get_loss(x, time_info, missing, masking)
    (loss_re + loss_kl).backward()

    grad_norm = torch.norm(
        torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    )

    return {
        "shapes": spec,
        "n_params": int(n_params),
        "n_tensors": len(params),
        "state_dict": params,
        # repr() of the float keeps every bit; == on these strings is a bit-exact test
        "loss_RE": repr(float(loss_re)),
        "loss_KL": repr(float(loss_kl)),
        "grad_norm": repr(float(grad_norm)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("path")
    args = ap.parse_args()

    current = fingerprint()

    if args.mode == "record":
        with open(args.path, "w") as f:
            json.dump(current, f, indent=2)
        print(f"recorded {current['n_tensors']} tensors, {current['n_params']} params")
        print(f"  RE={current['loss_RE']}  KL={current['loss_KL']}")
        print(f"-> {args.path}")
        return 0

    with open(args.path) as f:
        golden = json.load(f)

    diffs = []
    if golden["state_dict"] != current["state_dict"]:
        g = {k: tuple(v) for k, v in golden["state_dict"]}
        c = {k: tuple(v) for k, v in current["state_dict"]}
        for k in sorted(set(g) - set(c)):
            diffs.append(f"  missing tensor: {k} {g[k]}")
        for k in sorted(set(c) - set(g)):
            diffs.append(f"  new tensor:     {k} {c[k]}")
        for k in sorted(set(g) & set(c)):
            if g[k] != c[k]:
                diffs.append(f"  shape changed:  {k} {g[k]} -> {c[k]}")
    for key in ("n_params", "loss_RE", "loss_KL", "grad_norm"):
        if golden[key] != current[key]:
            diffs.append(f"  {key}: {golden[key]} -> {current[key]}")

    if diffs:
        print("FINGERPRINT MISMATCH")
        print("\n".join(diffs))
        return 1

    print(f"OK - identical: {current['n_tensors']} tensors, {current['n_params']} params")
    print(f"  RE={current['loss_RE']}  KL={current['loss_KL']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
