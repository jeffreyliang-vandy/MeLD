import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
import random
import warnings
import argparse
from tqdm import tqdm
import pickle, os

import evaluation_metric.Metrics as mt
import evaluation_metric.predictive_metrics as pdm
import evaluation_metric.correl as correl
from model import DP as dp

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--real_path","-R",type=str,required=True)
args_parser.add_argument("--test_path","-T",type=str,required=True)
args_parser.add_argument("--save_path","-S",type=str,default="./eval/score/")
args_parser.add_argument("--seed",type=int,default=0)
args_parser.add_argument("--model_name","-M",type=str,required=True)
args = args_parser.parse_args()

warnings.simplefilter(action='ignore', category=RuntimeWarning)
device = 'cuda'

assert os.path.exists(args.real_path), "Real data path not exist"
assert os.path.exists(args.test_path), "Test data path not exist"
assert os.path.exists(args.save_path), "Saving path not exist"

real_df = pd.read_csv(args.real_path).fillna(0)
real_df['date'] = real_df.index
processed_data,_,_,_ = dp.partition_multi_seq(real_df,1,'patient_id',20)

_synth_data = pd.read_csv(args.test_path).fillna(0)
_synth_data['date'] = _synth_data.index
synth_data, _, _, _ = dp.partition_multi_seq(_synth_data,1,'patient_id',20)

# Define batch size for evaluation
eval_batch_size = 5000  # Adjust as needed

# Convert data into TensorDataset
real_dataset = TensorDataset(torch.tensor(processed_data, dtype=torch.float32))
synth_dataset = TensorDataset(torch.tensor(synth_data, dtype=torch.float32))

# Create DataLoaders
real_dataloader = DataLoader(real_dataset, batch_size=eval_batch_size, shuffle=True, drop_last=True)
synth_dataloader = DataLoader(synth_dataset, batch_size=eval_batch_size, shuffle=True, drop_last=True)

real_dataloader = cycle(real_dataloader)
synth_dataloader = cycle(synth_dataloader)

# Sample collection for metrics
device = "cuda"
iterations = 2000
result_disc = []
result_pred = []
result_tmp = []

# Iterate over batches
for _ in tqdm(range(10), desc="Evaluating Metrics"):
    # Get a new batch from each dataloader
    real_batch = next(real_dataloader)[0]  # Extract tensor from dataset
    synth_batch = next(synth_dataloader)[0]
    print(f"real.shape {real_batch.shape}")
    print(f"syn.shape {synth_batch.shape}")

    # Ensure correct shapes for processing
    real_batch = real_batch.to(device)  # Convert to NumPy if needed
    synth_batch = synth_batch.to(device)
    
    torch.cuda.empty_cache()  # Clear CUDA memory

    # Compute evaluation metrics
    a = mt.discriminative_score_metrics(real_batch, synth_batch, iterations)
    b = pdm.predictive_score_metrics(real_batch, synth_batch, 5)
    c = mt.temp_disc_score(real_batch, synth_batch, iterations)

    # Store results
    result_disc.append(a)
    result_pred.append(b)
    result_tmp.append(c)

# Evaluation is now fully dynamic with batches!

"""
1.Discriminative Score measures the fidelity of synthetic time series data to original
data, by training a classification model (optimizing a 2-layer LSTM) to distinguish between
sequences from the original and generated datasets

2.Predictive Score measures the utility of generated sequences by training a posthoc
sequence prediction model (optimizing a 2-layer LSTM) to predict next-step temporal
vectors under a Train-on-Synthetic-Test-on-Real (TSTR) framework.

3.Temporal Discriminative Score measures the similarity of distributions of inter-row dif-
ferences between generated and original sequential data

4.Feature Correlation Score measures the averaged L2-distance of correlation matrices
computed on real and synthetic data
"""
print(f"Discriminative score abs(0.5 - acc): {np.mean(result_disc):2f}(sd {np.std(result_disc):2f})")
print(f"Predictive score (syn pred real's MAE): {np.mean(result_pred):2f}(sd {np.std(result_pred):2f})")
print(f"Temperal Discriminative score abs(0.5 - acc): {np.mean(result_tmp):2f}(sd {np.std(result_tmp):2f})")

pickle.dump({"Discriminative Score":[np.mean(result_disc),np.std(result_disc)],
             "Temperal Discriminative Score":[np.mean(result_tmp),np.std(result_tmp)],
             "Predictive Score":[np.mean(result_pred),np.std(result_pred)],
             },
             open(os.path.join(args.save_path,f"{args.model_name}.pkl"),"wb"))