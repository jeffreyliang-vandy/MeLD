# MeLD — Medical Longitudinal Latent Diffusion Models

This is the implementation of the MeLD model described in the paper:
**Generating Faithful Synthetic Longitudinal HIV Cohorts for Clinically Grounded Data Analysis**
<!-- by Zhuohui J Liang, Zhuohang Li, Nicholas Jackson, Yanink Caro-Vega, Ronaldo I.
Moreiraa, Fabio Paredes, Jordany Bernadin, Diana Varela, Carina Cesar, Alessandro
Blasimme, Jessica M. Perkins, Amir Asiaee, Stephany N. Duda, Bradley A.
Malin, Bryan E. Shepherd, and Chao Yan -->

A collection of scripts and model implementations for learning latent representations from longitudinal medical data and generating synthetic patient time series using a VAE + DiT (Diffusion Transformer) pipeline.

## Repository layout (important files)
- `model` - VAE model definitions and utilities
- `LightningDiT/` — Diffusion Transformer implementation and training utilities (see `LightningDiT/README.md` for details)
- `1_train_vae.py` — Train a VAE on your dataset and save model checkpoints
- `2_sample_latent.py` — Encode data into latent representations using a trained VAE
- `3_train_DiT.sh` — Wrapper to train LightningDiT on latent samples (requires a config in `LightningDiT/configs/`)
- `4_sample_synthetic_data.py` — Convert generated latent samples back into original data space
- `evaluation/` — Metrics and evaluation scripts used in the paper
- `baseline/` — bundled baseline models used by the pipeline

## Quickstart — run the main pipeline

### Prerequisites

- Python 3.8+ (conda or venv recommended)
- PyTorch compatible with your CUDA / CPU setup (see `LightningDiT/requirements.txt`)

Install dependencies (example using a virtualenv):

```bash
# create and activate a venv
conda create -n meld_vae -f model/environment.yaml
conda create -n meld_dit -f LightningDiT/requirements.txt
```

### Pipeline (order matters)

1) Train the VAE

```bash
conda run -n meld_vae \
python 1_train_vae.py -VM your_vae_name
```

2) Extract latent representations

```bash
conda run -n meld_vae \
python 2_sample_latent.py -MV your_vae_name
```

3) Train DiT on latent samples and generate latent synthetic samples

```bash
# Make sure you create or edit a config under `LightningDiT/configs/`
conda run -n meld_dit bash 3_train_DiT.sh
```

4) Decode latent samples back into original data space

```bash
conda run -n meld_vae \
python 4_sample_synthetic_data.py \
    -VM your_vae_name \
    -DP path/to/dit_samples \
    -S path/to/output/synthetic_data \
    -M my_synthetic_dataset_name
```

## Configuration notes

- `LightningDiT/configs/` contains example configs used for training the DiT model. Copy and adapt those for your dataset and compute budget.
- Data preprocessing can be found in `0_data_preprocessing.py`, and model-specific data logic can be found under `HALO_Inpatient/`, `SynTEG/`, `TimeDiff` and `TimeAutoDiff/` — review those scripts when preparing your dataset.

## Evaluation

The `evaluation/` folder contains scripts used in the paper to compute metrics such as distributional similarity, predictive performance, and privacy analyses. Example scripts:

- `5_compute_metric_dist.py`
- `5_compute_metric_pred.py`
- `5_evaluate_privacy.py`

evaluation can be done with

```bash
conda run -n meld_vae \
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
