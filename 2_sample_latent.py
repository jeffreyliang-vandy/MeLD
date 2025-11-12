import os
# Verify CUDA device visibility in PyTorch
import torch
import torch.nn.functional as F
print("Available CUDA devices:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())

from model import timeautoencoder as tae
from model import DP as dp
import pandas as pd
import numpy as np
import gc
from model import process_edited as pce
import random
from tqdm import tqdm
import argparse

args_parser = argparse.ArgumentParser()
# Add argument for conditional flag
args_parser.add_argument("--vae_model", "-VM", default=None)
args_parser.add_argument("--checkpoint", "-V", type=str, default="")
args = args_parser.parse_args()

vae_model = f"vae_{args.vae_model}"
ckpt = f"_checkpoint_epoch_{args.checkpoint}" if len(args.checkpoint) > 0 else ""
print(f"Using checkpoint {ckpt}")

##################################################################################################################

processed_data = torch.load("../Dataset/save/processed_data.pt").cpu()
time_info = torch.load("../Dataset/save/time_info.pt").cpu()
missing = torch.load("../Dataset/save/missing.pt").cpu()
masking = torch.load("../Dataset/save/masking.pt").cpu()

device = 'cuda'
print("Load VAE model...")

PARAMS_PATH = f"../Dataset/save/{vae_model}/vae_params.pth"

# Load saved model parameters
if os.path.exists(PARAMS_PATH):
    model_params = torch.load(PARAMS_PATH, map_location=device)
    print(f"✅ Model parameters loaded successfully from {PARAMS_PATH}")
else:
    raise FileNotFoundError(f"❌ Parameters file not found at {PARAMS_PATH}")

ae = tae.DeapStack(**model_params).to(device)
print("Loading Weights...")
# Load trained model weights
CHECKPOINT_PATH = f"../Dataset/save/{vae_model}/vae{ckpt}.pth"

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    ae.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model weights loaded successfully from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"❌ Model checkpoint not found at {CHECKPOINT_PATH}")

# ae = torch.compile(ae)
ae.eval()

print("Getting Latent Class...")

batch_size = 128  # Start small, increase if memory allows

num_batches = processed_data.shape[0] // batch_size + int(processed_data.shape[0] % batch_size > 0)

latent_features_list = []
mu_list = []
logvar_list = []

with torch.no_grad():
    for i in tqdm(range(num_batches),total = num_batches):
        start = i * batch_size
        end = min(start + batch_size, processed_data.shape[0])

        batch_data = processed_data[start:end].to(device)
        batch_time_info = time_info[start:end].to(device)
        batch_masking = masking[start:end].to(device)
        batch_missing = missing[start:end].to(device)

        _, latent_features_batch, mu_batch, logvar_batch = ae(batch_data, batch_time_info, batch_missing, batch_masking)

        latent_features_list.append(latent_features_batch.cpu())
        mu_list.append(mu_batch.cpu())
        logvar_list.append(logvar_batch.cpu())


# Concatenate results back
latent_features = torch.cat(latent_features_list, dim=0).cpu()
latent_features_shape = latent_features.shape
SAVE_PATH = f"../Dataset/save/{vae_model}/latent_feature{args.checkpoint}.pt"
torch.save(latent_features,SAVE_PATH)
print(f"saving latent.shape:{latent_features.shape} at {SAVE_PATH}")

## save mu and logvar
mu = torch.cat(mu_list,dim = 0)
SAVE_PATH = f"../Dataset/save/{vae_model}/mu{args.checkpoint}.pt"
print(f"saving mu.shape:{mu.shape} at {SAVE_PATH}")
# torch.save(mu,SAVE_PATH)
del mu, mu_list, mu_batch

logvar = torch.cat(logvar_list,dim = 0)
SAVE_PATH = f"../Dataset/save/{vae_model}/logvar{args.checkpoint}.pt"
# torch.save(logvar,SAVE_PATH)
del logvar, logvar_list, logvar_batch

print("✅ Model execution completed successfully with batched inference.")

## Using Testing data to see generalizebitly

test_data = pd.read_csv("../Dataset/hiv_test.csv.gz").fillna(0)
processed_data,time_info,missing,masking = dp.partition_multi_seq(test_data,threshold=1,column_to_partition='patient_id',max_len=120)
processed_data = processed_data.cpu()
time_info = time_info.cpu()
missing = missing.cpu()
masking = masking.cpu()

num_batches = processed_data.shape[0] // batch_size + int(processed_data.shape[0] % batch_size > 0)

latent_features_list = []
mu_list = []
logvar_list = []

with torch.no_grad():
    for i in tqdm(range(num_batches),total = num_batches):
        start = i * batch_size
        end = min(start + batch_size, processed_data.shape[0])

        batch_data = processed_data[start:end].to(device)
        batch_time_info = time_info[start:end].to(device)
        batch_masking = masking[start:end].to(device)
        batch_missing = missing[start:end].to(device)

        _, latent_features_batch, mu_batch, logvar_batch = ae(batch_data, batch_time_info, batch_missing, batch_masking)

        latent_features_list.append(latent_features_batch.cpu())
        mu_list.append(mu_batch.cpu())
        logvar_list.append(logvar_batch.cpu())


# Concatenate results back
latent_features = torch.cat(latent_features_list, dim=0).cpu()
latent_features_shape = latent_features.shape

batch_size = 128  # Start small, increase if memory allows
num_batches = latent_features.shape[0] // batch_size + int(latent_features.shape[0] % batch_size > 0)
output_dict = {'bins': [], 'cats': [], 'nums': [],'times': [], 'eos': [],'missings': []}
with torch.no_grad():
    for i in tqdm(range(num_batches), total=num_batches):
        start = i * batch_size
        end = min(start + batch_size, latent_features.shape[0])

        batch_data = latent_features[start:end].to(device)
        # batch_time_info = time_info[start:end].to(device)  # Not used, but ensuring consistency

        output = ae.decoder(batch_data)

        # Collect outputs in the correct lists
        if 'bins' in output:
            output_dict['bins'].append(output['bins'].cpu())

        if 'cats' in output:
            # `cats` is a list of tensors, so process each one
            cats_list = [cat.cpu() for cat in output['cats']]
            output_dict['cats'].append(cats_list)  # Append list directly

        if 'nums' in output:
            output_dict['nums'].append(output['nums'].cpu())

        if 'times' in output:
            output_dict['times'].append(output['times'].cpu())

        if 'eos' in output:
            output_dict['eos'].append(output['eos'].cpu())

        if 'missings' in output:
            output_dict['missings'].append(output['missings'].cpu())

# Concatenating collected results
gen_output = {}

if output_dict['bins']:
    gen_output['bins'] = torch.cat(output_dict['bins'], dim=0)

if output_dict['cats']:
    # `cats` is a list of lists of tensors, so we need to process it separately
    num_categories = len(output_dict['cats'][0])  # Get number of categorical features
    gen_output['cats'] = [torch.cat([batch[i] for batch in output_dict['cats']], dim=0) for i in range(num_categories)]

if output_dict['nums']:
    gen_output['nums'] = torch.cat(output_dict['nums'], dim=0)

if output_dict['times']:
    gen_output['times'] = torch.cat(output_dict['times'], dim=0)

if output_dict['eos']:
    gen_output['eos'] = torch.cat(output_dict['eos'], dim=0)

if output_dict['missings']:
    gen_output['missings'] = torch.cat(output_dict['missings'], dim=0)



del ae, output_dict, latent_features
torch.cuda.empty_cache()
gc.collect()

print("Transforming tensor back to tabular...")
# data_size, seq_len, _ = latent_features.shape
data_size, seq_len, _ = processed_data.shape

# Read dataframe
print("Loading Data...")
filename = f'../Dataset/hiv_train.csv.gz'
print(filename)
real_df = pd.read_csv(filename).fillna(0).drop_duplicates()
real_df1 = real_df.drop(columns=['date','patient_id'], axis=1)

### Output to tensor:
threshold = 1
synth_data, synth_time, syn_eos = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)

del gen_output
synth_data = synth_data.cpu()
np.save(f"../Dataset/save/{vae_model}/dummy{args.checkpoint}.npy",synth_data.numpy())
synth_time = synth_time.cpu()
syn_eos = syn_eos.cpu()

### End of sequence:
eos_probabilities = F.softmax(syn_eos, dim=-1)[...,1]

eos_predictions = (eos_probabilities > 0.5).int()  # Binary EOS predictions (B, L)
# Compute the cumulative mask: set to 0 after the first EOS in each sequence
eos_cumsum = eos_predictions.cumsum(dim=1).cumsum(dim=1)  # Cumulative sum along the sequence length
eos_mask = (eos_cumsum <= 1).float().unsqueeze(-1)  # Mask: 1 before and at first EOS, 0 after
eos_mask = eos_mask.view(-1).cpu().numpy()
eos_cumsum

syn_id = torch.ones_like(syn_eos[:,:,0])
syn_id = syn_id.cumsum(dim=0)
syn_id = syn_id.view(-1).cpu().numpy()

### Tensor to tabular:
_synth_data, _ = pce.convert_to_table(real_df1, synth_data, threshold)
_synth_data['date'] = None
_synth_data['patient_id'] = syn_id
_synth_data = _synth_data[eos_mask > 0]

_synth_data.to_csv(f"../Dataset/save/{vae_model}/dummy{args.checkpoint}.csv.gz",index=False, compression="gzip")