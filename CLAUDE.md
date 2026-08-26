# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the MeLD paper (Generating Synthetic Multi-national Longitudinal Cohorts for
Clinically Grounded HIV Research). It is a **latent diffusion pipeline for longitudinal patient
records**: a sequence VAE compresses per-patient visit sequences into a latent tensor, a 1-D
Diffusion Transformer (LightningDiT, vendored as `model/LightningD1T/`) is trained on those
latents, and the sampled latents are decoded back into a tabular CSV of synthetic patients.

There is no test suite and no linter config. Verification is the synthetic smoke run described
under "Verifying a change". Do not invent build/test commands.

## Operating restrictions

`AGENTS.md` at the repo root carries hard restrictions — read it before any git or remote-server
work. In short: start a new branch per session; never pull to main without escalation; never
delete anything on the remote server; never read raw data there, only summarised results. Code
moves one way (local edit → push → server pulls a detached checkout at a sha, runs outside the
repo, exports scrubbed summaries back).

## Everything is one config file

All four stages take `--config <path>` and nothing else. There are no other CLI flags.

```bash
python 1_train_vae.py             --config configs/hiv_lite1.yaml   # train VAE + encode latents
python 2_sample_latent.py         --config configs/hiv_lite1.yaml   # (re-)encode a split
python 3_train_DiT.py             --config configs/hiv_lite1.yaml   # train DiT, then sample
python 4_sample_synthetic_data.py --config configs/hiv_lite1.yaml   # decode to CSV
```

- `configs/meld_default.yaml` is **the schema** — every key, annotated, at its historical
  default. A user config supplies only overrides; any key not in this file is rejected at load
  time with a "did you mean" suggestion.
- `configs/hiv_lite1.yaml` is the paper run.
- `model/meld_config.py` loads, merges, derives and validates. `model/dit_adapter.py` translates
  the `dit:` half into the vendored schema.

Put new configs in repo-root `configs/`. **Not** in `model/LightningD1T/configs/`, which
`.gitignore:5` silently ignores.

### Derivation and validation

Nulls mean "derive": `vae.output.dir` from `run.root`/`run.name`, `dit.data.data_path` from the
step-2 latent path, `vae.decode.latents_path` from the DiT's sample directory, and so on. Paths
are absolutised at load, because the DiT reads its config with `cwd=model/LightningD1T/`.

Three checks are worth knowing because they will stop you:

- **`deapstack_kwargs` cross-checks the config against `DeapStack.__init__`.** Add a constructor
  argument without adding a config key (or vice versa) and every stage fails at startup naming
  the drift. This is what keeps "every parameter is exposed" true over time.
- `dit.model.in_chans` must equal `vae.arch.lat_dim`, and both are checked against the actual
  latent tensor's shape.
- `dit.transport` keys are whitelisted, because `create_transport(**cfg["transport"])` is
  splatted wholesale and rejects extras with an opaque `TypeError`.

### Environments

Two conda envs; a single process cannot import both halves.

- `meld.yml` (env name is **`ddpm`**) — VAE side: steps 1, 2, 4 and `evaluation/`.
- `model/LightningD1T/requirements.txt` — DiT side, pinned to `torch==2.2.0` + cu121.

`3_train_DiT.py` therefore launches the vendored entry point as a **subprocess**;
`dit.runtime.python` selects that interpreter. Nothing imports across the boundary.

## Pipeline and the data contract

**0. Preprocess** (`0_data_preprocess.ipynb`) — align rows into visit-level records with an ID
column and a column literally named `date`, then `dp.partition_multi_seq(...)`, which returns
`((processed_data, time_info, missing, masking), parser)` — a **2-tuple**; the notebook's cell
unpacks four values and does not run as written. You must write the HDF5 yourself:

- datasets `processed_data` `(N,T,F)`, `time_info` `(N,T,8)`, `missing` `(N,T,n_nums)`,
  `masking` `(N,T,2)` (EOS flag + padding flag)
- attrs `n_bins`, `n_cats`, `n_nums`, `cards`

`tools/make_smoke_data.py` does exactly this for synthetic data and is the reference
implementation. Read `parser.datatype_info()` **after** `partition_multi_seq`, which drops the
partition column and re-slices `missing`; reading it before is off by one.

**1. Train VAE** — writes `vae.pth` (best), `vae_checkpoint_epoch_<n>.pth`, `latent_feature.pt`,
logs, and `vae_params.pth`. Then falls through to encoding unless `vae.stages` says otherwise
(`[encode]` alone re-encodes without retraining).

**2. Encode latents** — `latent_feature<suffix>.pt` plus, with `vae.encode.condition_path`, a
row-aligned `condition_features<suffix>.parquet` for CFG. Steps 2 and 4 derive that suffix from
the same `cfg.latent_path()`, so pinning `vae.encode.checkpoint` can no longer produce latents
the decoder cannot find.

**3. Train/sample DiT** — `model/dit_adapter.py` emits a vendored-schema YAML into the experiment
directory and runs `train_single.py` / `inference_single.py`. Conditioning is text-based CFG:
condition rows are rendered to prompts by `datasets/condition2text.py` and embedded by a frozen
`CLIPTextEmbedder`, which is built **unconditionally** — so a CLIP checkpoint is required even at
`cfg_scale: 0`. Set `dit.runtime.clip_model_path` or `$MELD_CLIP_PATH`.

**4. Decode** — un-normalises through the training latents' per-channel min/max, decodes, inverts
the parser, truncates at EOS, writes `syn_<name>_<seed>.csv.gz`.

### Checkpoints are self-describing

`DeapStack` records its resolved arguments on `self.config`, and every checkpoint embeds them:
`{schema, config, model_state_dict, optimizer_state_dict, epoch, loss}`. Steps 2 and 4 call
`DeapStack.from_checkpoint(path)` and need no side-car file.

`from_checkpoint(..., overrides=...)` accepts *behavioural* changes (thresholds, loss weights,
activations) but rejects anything in `STRUCTURAL_KEYS`, since those determine the saved weights'
shapes.

`vae_params.pth` is still written and **must stay at exactly 14 keys**: nothing in this pipeline
reads it, but `baseline/MeLD-Transformer`'s `TimeLDM.__init__` takes those fourteen arguments
with no `**kwargs`, so adding keys breaks that ablation.

Pre-config checkpoints: `python tools/migrate_vae_ckpt.py <vae_dir> --apply` merges an old
`vae_params.pth` + `vae.pth` into the new format (dry run by default, keeps `.bak` copies).

## Layout

- `model/meld_config.py` — config loading, derivation, validation. Lives under `model/` so it is
  importable from every entry point.
- `model/vae_common.py` — everything `1_train_vae.py` uses: dataloaders, the epoch loop
  (`run_training`), the beta schedule, HDF5 shape reading, latent encoding. `1_train_vae.py`
  always constructs an `Accelerator()`; the same script runs single-process (`python`) or
  multi-GPU (`accelerate launch`) with no separate twin to drift out of sync.
- `model/dit_adapter.py` — vendored-config translation and subprocess launch.
- `model/timeautoencoder.py` — the VAE. `DeapStack` = `Embedding_data` → `Encoder`/`Decoder`
  (bi-GRU + `Transformer_Block`). `auto_loss` computes per-type reconstruction loss.
  `eos_truncation_mask` is shared with step 4.
- `model/DP.py` — sequence partitioning, cyclical date encoding, latent min-max normalise.
- `model/process_edited.py` — `DataFrameParser` plus `convert_to_tensor`/`convert_to_table`.
- `model/data_loader.py` — `HDF5Dataset`, with `__getitems__` for bulk reads (sorts indices for
  disk locality, then unsorts). Preserve that.
- `evaluation/5_compute_metric_*.py` — paper metrics. The CLI here is **not** uniform and has not
  been migrated to the config; check `add_argument` before invoking. `5_compute_metric_alpha_beta.py`
  uses `-S` for the *synthetic* path, `5_evaluate_privacy.py` takes `-L <HALO-pickled dir>`, and
  `5_compute_metric_aug_times.py` is a notebook dump that is not runnable as a script.
- `baseline/` — vendored baselines plus the `MeLD-DDPM` / `MeLD-Transformer` ablations, which
  consume the same step-2 latents.

## Verifying a change

No real data is needed, and per `AGENTS.md` none may be read.

```bash
python tools/make_smoke_data.py --out-dir /tmp/smoke        # tiny synthetic HDF5 + parser
python tools/arch_fingerprint.py record golden.json         # before touching the model
python tools/arch_fingerprint.py check  golden.json         # after
```

`arch_fingerprint` compares the full `state_dict` shape list, parameter count, and a seeded
forward/backward's RE, KL and gradient norm. **Any change to `model/timeautoencoder.py` that is
meant to be behaviour-preserving must keep this bit-identical.** That is how the current defaults
were shown to reproduce the historical hard-coded values exactly.

Then run all four stages against a small config (see the smoke settings: `epochs: 6`,
`LightningDiT-mini/1`, `max_steps: 4`, `sample.total: 16`, `runtime.device: cpu`). `1_train_vae.py`
should produce the same epoch-0 loss run plain (`python 1_train_vae.py`) and under Accelerate at
one process (`accelerate launch --num_processes 1 --cpu 1_train_vae.py`).

## Conventions and gotchas

- `.gitignore` excludes `*.csv*`, `*.pt*` and `model/LightningD1T/configs/*.yaml`. **Never commit
  patient data, generated CSVs, or checkpoints** — this project handles real HIV cohort data.
- `model/LightningD1T/` is vendored. The only local change is `lightning1dit.py`'s CLIP path,
  which now reads `$MELD_CLIP_PATH` with the original literal as fallback. Keep further edits out
  of that tree; put translation logic in `dit_adapter.py`.
- In `4_sample_synthetic_data.py`, `_, max_val, min_val = dp.normalize(...)` has its names swapped
  relative to `DP.normalize`'s `(normalized, min, max)`, but they are passed on positionally in
  the same swapped order, so the errors cancel. **Fixing one half inverts the latents.**
- `vae.loss.first_visit_scale` has two preserved quirks: values `<= 1.0` disable it, and the
  applied weight is `(1 + scale)`. Both are kept bit-for-bit because changing either rescales the
  loss of every existing run.
- `vae.arch.channels: 0` disables the transformer block entirely. There is deliberately no
  separate `use_transformer` flag.
