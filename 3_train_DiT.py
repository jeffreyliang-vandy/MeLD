"""Stage 3: train the latent diffusion transformer, and sample from it.

    python 3_train_DiT.py --config configs/my_run.yaml            # train, then sample
    python 3_train_DiT.py --config configs/my_run.yaml --stage train
    python 3_train_DiT.py --config configs/my_run.yaml --dry-run  # show the command

`model/LightningD1T/` is vendored third-party code with its own YAML schema and its
own conda environment, so this does not import it. `model/dit_adapter.py` translates
the MeLD config into that schema, writes it out, and launches the vendored entry point
as a subprocess -- with `dit.runtime.python` selecting the interpreter.
"""
import argparse
import sys

from model import dit_adapter, meld_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the MeLD config file")
    ap.add_argument("--stage", choices=["train", "sample", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the vendored config and print the command, then stop")
    args = ap.parse_args()

    cfg = meld_config.load(args.config)
    stages = ["train", "sample"] if args.stage == "both" else [args.stage]

    for stage in stages:
        print(f"\n=== DiT: {stage} ===")
        dit_adapter.run(cfg, stage, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
