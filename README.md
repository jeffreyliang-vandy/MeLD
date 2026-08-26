# MeLD — Medical Longitudinal Latent Diffusion

This is the implementation of the MeLD model described in the paper:
**Generating Synthetic Multi-national Longitudinal Cohorts for Clinically Grounded HIV Research**
by Zhuohui J Liang, Zhuohang Li, Nicholas Jackson, Yanink Caro-Vega, Ronaldo I.
Moreiraa, Fabio Paredes, Jordany Bernadin, Diana Varela, Carina Cesar, Alessandro
Blasimme, Jessica M. Perkins, Amir Asiaee, Stephany N. Duda, Bradley A.
Malin, Bryan E. Shepherd, and Chao Yan

\[[Medrxiv](https://www.medrxiv.org/content/10.1101/2025.11.14.25340245v1)\]

A collection of scripts and model implementations for learning latent representations from longitudinal medical data and generating synthetic patient time series using a VAE + DiT (Diffusion Transformer) pipeline.

**Update (2026-06):** 
- MeLD is now updated to support more general longitudinal data. 
- Now support CFG (Classifier-Free Guidance) generation, with CLIP as encoder to encode condtions as text embeddings.


## Repository layout (important files)
- `configs/meld_default.yaml` — the configuration schema: every key, annotated, at its default
- `configs/hiv_lite1.yaml` — the paper's run configuration
- `model/meld_config.py` — loads/derives/validates the config used by every stage
- `model/timeautoencoder.py` — VAE model definitions and utilities
- `model/vae_common.py` — training/encoding logic shared by `1_train_vae.py`
- `model/dit_adapter.py` — bridges the config to the vendored LightningDiT
- `model/LightningD1T/` — Diffusion Transformer implementation (vendored)
- `1_train_vae.py` — train the VAE (plain `python`, or multi-GPU via `accelerate launch`)
- `2_sample_latent.py` — encode data into latent representations
- `3_train_DiT.py` — train the Diffusion Transformer and sample from it
- `4_sample_synthetic_data.py` — decode generated latents back into the original data space
- `0_data_preprocess.ipynb` — example preprocessing; `tools/make_smoke_data.py` is a runnable
  reference that also writes the HDF5 file the pipeline expects
- `evaluation/` — metrics and evaluation scripts used in the paper
- `baseline/` — bundled baseline models

## Quickstart

### Prerequisites

Two environments, one per half of the pipeline:

```bash
conda env create -f meld.yml                                  # VAE side (env name: ddpm)
conda create -n meld_dit python=3.10 && \
  pip install -r model/LightningD1T/requirements.txt          # DiT side
```

### Configure once, run four stages

Every stage reads the same file and takes no other arguments. Copy
`configs/hiv_lite1.yaml`, point it at your data, and run:

```bash
python 1_train_vae.py             --config configs/my_run.yaml
python 2_sample_latent.py         --config configs/my_run.yaml
python 3_train_DiT.py             --config configs/my_run.yaml
python 4_sample_synthetic_data.py --config configs/my_run.yaml
```

`configs/meld_default.yaml` documents every available key and its default; your config only
needs the ones you change, and an unrecognised key is reported at startup.

Two settings are machine-specific and can come from the environment instead of the file, so a
config can be shared unchanged:

- `run.root` — where relative paths resolve (`$MELD_RUN_ROOT`)
- `dit.runtime.clip_model_path` — local CLIP checkpoint for text conditioning (`$MELD_CLIP_PATH`)

Multi-GPU VAE training uses the same script and config:

```bash
accelerate launch 1_train_vae.py --config configs/my_run.yaml
```

`3_train_DiT.py` runs the DiT in its own environment as a subprocess; set `dit.runtime.python`
to that interpreter. Use `--stage train|sample` to run one half, or `--dry-run` to inspect the
generated LightningDiT config without launching anything.

### Trying it without real data

```bash
python tools/make_smoke_data.py --out-dir /tmp/smoke
```

writes a small synthetic HDF5 dataset, a fitted parser and a cohort table, which is enough to
exercise all four stages on CPU.

## Evaluation

The `evaluation/` folder contains scripts used in the paper to compute metrics such as distributional similarity, predictive performance, and privacy analyses. Example scripts:

- `5_compute_metric_dist.py`
- `5_compute_metric_pred.py`
- `5_evaluate_privacy.py`

evaluation can be done with

```bash
conda run -n ddpm \
python evaluation/5_compute_metric_xxxx.py -R path/to/real_data -T path/to/synthetic_data -M your_synthetic_dataset_name -S path/to/save_results
```

Run evaluation examples with the synthetic dataset path produced by the pipeline. See individual scripts for usage flags.


## Acknowledgements

Thanks to the code contributions of:
- TimeAutoDiff: https://github.com/namjoonsuh/TimeAutoDiff
- LightningDiT: https://github.com/hustvl/LightningDiT
- synthetic data benchmarking: https://github.com/yy6linda/synthetic-ehr-benchmarking
- HALO-Inpatient: https://github.com/btheodorou99/HALO_Inpatient

## Citation
Please cite the paper if you use this code or model in your research:

```
@article {Liang2025.11.14.25340245,
	author = {Liang, Zhuohui J. and Li, Zhuohang and Jackson, Nicholas J. and Caro-Vega, Yanink and Moreira, Ronaldo I. and Paredes, Fabio and Bernadin, Jordany and Varela, Diana and Cesar, Carina and Blasimme, Alessandro and Perkins, Jessica M. and Asiaee, Amir and Duda, Stephany N. and Malin, Bradley A. and Shepherd, Bryan E. and Yan, Chao},
	title = {Generating Synthetic Multi-national Longitudinal Cohorts for Clinically Grounded HIV Research},
	elocation-id = {2025.11.14.25340245},
	year = {2025},
	doi = {10.1101/2025.11.14.25340245},
	URL = {https://www.medrxiv.org/content/early/2025/11/17/2025.11.14.25340245},
	eprint = {https://www.medrxiv.org/content/early/2025/11/17/2025.11.14.25340245.full.pdf},
	journal = {medRxiv}
}
```