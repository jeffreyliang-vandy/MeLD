"""Convert pre-config VAE checkpoints to the self-describing format.

Checkpoints now embed the architecture that produced them, so steps 2 and 4 rebuild
the model from the checkpoint alone. Older checkpoints kept that information in a
separate `vae_params.pth`; this merges the two, once, so no legacy branch has to
survive in the pipeline.

    python tools/migrate_vae_ckpt.py <vae_dir>            # report what would change
    python tools/migrate_vae_ckpt.py <vae_dir> --apply

`vae_params.pth` itself is left alone: baseline/MeLD-Transformer still reads it, and
its TimeLDM takes exactly those fourteen arguments with no **kwargs.
"""
import argparse
import os
import shutil
import sys

import torch

from model import timeautoencoder as tae


def find_checkpoints(vae_dir):
    names = [n for n in sorted(os.listdir(vae_dir))
             if n.endswith((".pth", ".pt")) and n != "vae_params.pth"]
    return [os.path.join(vae_dir, n) for n in names]


def migrate_one(path, params, apply, backup):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        return "skip", "not a checkpoint dict"
    if blob.get("config") is not None:
        return "skip", "already self-describing"
    if "model_state_dict" not in blob:
        return "skip", "no model_state_dict"

    # DeapStack's keyword defaults reproduce the previously hard-coded behaviour, so
    # the 14 persisted arguments plus those defaults describe the original model
    # exactly.
    model = tae.DeapStack(**dict(params))
    # Take the model's own resolved config rather than just the 14 persisted keys, so
    # a migrated checkpoint is as complete as a freshly trained one.
    config = dict(model.config)
    saved, expected = blob["model_state_dict"], model.state_dict()
    if set(saved) != set(expected):
        return "fail", (f"state_dict mismatch: {len(set(expected) - set(saved))} missing, "
                        f"{len(set(saved) - set(expected))} unexpected")
    bad = [k for k in expected if tuple(saved[k].shape) != tuple(expected[k].shape)]
    if bad:
        return "fail", f"shape mismatch on {bad[:3]}"

    if apply:
        if backup:
            shutil.copy2(path, path + ".bak")
        blob["config"] = config
        blob["schema"] = tae.CHECKPOINT_SCHEMA
        torch.save(blob, path)
    return "ok", f"{len(config)} config entries"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vae_dir", help="directory holding vae.pth and vae_params.pth")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="skip writing .bak copies")
    args = ap.parse_args()

    params_path = os.path.join(args.vae_dir, "vae_params.pth")
    if not os.path.exists(params_path):
        sys.exit(f"no vae_params.pth in {args.vae_dir}; nothing to migrate from")
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    print(f"vae_params.pth: {len(params)} keys")

    checkpoints = find_checkpoints(args.vae_dir)
    if not checkpoints:
        sys.exit(f"no checkpoints found in {args.vae_dir}")

    failures = 0
    for path in checkpoints:
        status, detail = migrate_one(path, params, args.apply, not args.no_backup)
        mark = {"ok": "migrated" if args.apply else "would migrate",
                "skip": "skipped", "fail": "FAILED"}[status]
        print(f"  {mark:15} {os.path.basename(path):40} {detail}")
        failures += status == "fail"

    if not args.apply:
        print("\nDry run. Re-run with --apply to write the changes.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
