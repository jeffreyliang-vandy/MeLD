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
- `model/timeautoencoder.py` - VAE model definitions and utilities
- `model/LightningD1T/` — Diffusion Transformer implementation and training utilities
- `1_train_vae.py` — Train a VAE on your dataset and save model checkpoints
- `2_sample_latent.py` — Encode data into latent representations using a trained VAE
- `3_train_DiT.sh` — Wrapper to train LightningDiT on latent samples (requires a config in `LightningD1T/configs/`)
- `4_sample_synthetic_data.py` — Convert generated latent samples back into original data space
- `evaluation/` — Metrics and evaluation scripts used in the paper
- `baseline/` — Bundled baseline models used by the pipeline
- `0_data_preprocess.ipynb` - Example preprocess code and Pseudo data

## Quickstart — run the main pipeline

### Prerequisites

- Python 3.8+ (conda or venv recommended)
- PyTorch compatible with your CUDA / CPU setup (see `LightningD1T/requirements.txt`)

Install dependencies (example using a virtualenv):

```bash
# create and activate a venv
conda create -n meld_vae -f model/environment.yaml
conda create -n meld_dit -f model/LightningD1T/requirements.txt
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
python 2_sample_latent.py -VM your_vae_name
```

3) Train DiT on latent samples and generate latent synthetic samples

```bash
# Make sure you create or edit a config under `LightningD1T/configs/`
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

- `model/LightningD1T/configs/` contains example configs used for training the DiT model. Copy and adapt those for your dataset and compute budget.
- Data preprocessing can be found in `0_data_preprocessing.py`, and model-specific data logic can be found under `HALO_Inpatient/`, `SynTEG/`, `TimeDiff` and `TimeAutoDiff/` — review those scripts when preparing your dataset.
- We provide baseline models code from 
    - LLM based model `HALO_Inpatient/`, 
    - GAN based model `SynTEG/`, `
    - diffusion based model `TimeDiff` and 
    - latent diffusion based model `TimeAutoDiff/` 

    in `baseline/`, detailed describhe of the baselines can be found in the paper.or the original paper.


- In paper, MeLD variants are implemented with training the diffusion components of `MeLD-DDPM`, `MeLD-DiT`, `TimeDiff` and `TimeAutoDiff` on latent sample from `2_sample_latent.py`.

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