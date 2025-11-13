import os

# Set the visible devices to only GPU 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Verify CUDA device visibility in PyTorch
import torch
print("Available CUDA devices:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())

import numpy as np
import TimeLDM as tae
import DP as dp
import pandas as pd
import numpy as np
import time
import process_edited as pce
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from itertools import cycle

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--conditional","-C",action="store_true")
args_parser.add_argument("--n","-N",type=int,required=False)
args_parser.add_argument("--save_dir","-S",type=str,default="/data/7TB/jeff/results/timeautodiff/")
args_parser.add_argument("--model_name","-M",type=str,required=True)
args_parser.add_argument("--vae_checkpoint","-V",type=str,default=None,required=False)
args_parser.add_argument("--vae_model","-VM",type=str,default=None,required=False)
args_parser.add_argument("--ddpm_latents_path","-DP",type=str,default=None,required=False)
args_parser.add_argument("--scaled","-SL",action="store_true",default=False)
args = args_parser.parse_args()

##################################################################################################################
# Pre-processing Data
print("Loading Data...")

# filename = f'../Dataset/Multi-Sequence Data/nasdaq100_2019.csv'
filename = f'../Dataset/hiv_train.csv.gz'
# Read dataframe
print(filename)
real_df = pd.read_csv(filename).fillna(0)

real_df1 = real_df.drop(['patient_id','date'], axis=1)

has_na = real_df.isna().any().any()
print("Are there any missing values in the DataFrame? ", has_na)
print(f"real_df.shape: {real_df.shape}")

os.makedirs(args.save_dir, exist_ok=True)
##################################################################################################################
print("Preprocess Data...")
device = 'cuda'

processed_data = torch.load("../Dataset/save/processed_data.pt")
time_info = torch.load("../Dataset/save/shift_time_info.pt")

if args.vae_checkpoint is not None:
    ckpt = f"_checkpoint_epoch_{args.vae_checkpoint}"
else: ckpt = ""

if args.vae_model is not None:
    vae_model = f"_{args.vae_model}"
else: vae_model = ""

VAE_CHECKPOINT = f"../Dataset/save/vae{vae_model}/latent_feature.pt"
latent_features = torch.load(VAE_CHECKPOINT)

print(f"preprocessed_data.shape: {processed_data.shape}")
print(f"time_info.shape: {time_info.shape}")

### Load a ddpm results
print(f"Loading Latents at {args.ddpm_latents_path}...")
if os.path.exists(args.ddpm_latents_path):
    if args.ddpm_latents_path.endswith('.pt'):
        samples = torch.load(args.ddpm_latents_path)
    elif args.ddpm_latents_path.endswith('.npy'):
        samples = torch.tensor(np.load(args.ddpm_latents_path))

else:
    raise FileNotFoundError(f"❌ latents file not found at {args.ddpm_latents_path}")

if args.scaled:
    pass
else:
    # Inverse Noralized latent sample
    latent_features = torch.load(VAE_CHECKPOINT)
    _, max_val, min_val = dp.normalize(latent_features)
    samples = dp.inverse_normalize(samples,max_val,min_val)

del latent_features

# Manually clear cache
torch.cuda.empty_cache()
import gc
gc.collect()

##################################################################################################################
# Post-process the generated data 
threshold = 1

print("Load VAE model...")

PARAMS_PATH = f"../Dataset/save/vae{vae_model}/vae_params.pth"
# Load saved model parameters
if os.path.exists(PARAMS_PATH):
    model_params = torch.load(PARAMS_PATH, map_location=device)
    print(f"✅ Model parameters loaded successfully from {PARAMS_PATH}")
else:
    raise FileNotFoundError(f"❌ Parameters file not found at {PARAMS_PATH}")

ae = tae.TimeLDM(**model_params).to(device)
print("Loading Weights...")
# Load trained model weights
if args.vae_checkpoint is not None:
    CHECKPOINT_PATH = f"../Dataset/save/vae{vae_model}/vae_checkpoint_epoch_{args.vae_checkpoint}.pth"
else:
    CHECKPOINT_PATH = f"../Dataset/save/vae{vae_model}/vae.pth"

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    ae.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model weights loaded successfully from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"❌ Model checkpoint not found at {CHECKPOINT_PATH}")

# ae = torch.compile(ae)
ae.eval()

print("Post-process the generated data...")

batch_size = 128  # Start small, increase if memory allows
num_batches = samples.shape[0] // batch_size + int(samples.shape[0] % batch_size > 0)
output_dict = {'bins': [], 'cats': [], 'nums': [],'times': [], 'eos': [], 'missings': []}
with torch.no_grad():
    for i in tqdm(range(num_batches), total=num_batches):
        start = i * batch_size
        end = min(start + batch_size, samples.shape[0])

        batch_data = samples[start:end].to(device)
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

del ae
samples = samples.cpu()
torch.cuda.empty_cache()
gc.collect()

print("Transforming tensor back to tabular...")
# data_size, seq_len, _ = latent_features.shape
data_size, seq_len, _ = samples.shape

### Output to tensor:
synth_data, synth_time, syn_eos = pce.convert_to_tensor(real_df1, gen_output, threshold, data_size, seq_len)

### End of sequence:
with torch.no_grad():
    eos_probabilities = F.softmax(syn_eos, dim=-1)[...,1]
    eos_predictions = (eos_probabilities > 0.5).int()  # Binary EOS predictions (B, L)
    # Compute the cumulative mask: set to 0 after the first EOS in each sequence
    eos_cumsum = eos_predictions.cumsum(dim=1).cumsum(dim=1)  # Cumulative sum along the sequence length
    eos_mask = (eos_cumsum <= 1).float().unsqueeze(-1)  # Mask: 1 before and at first EOS, 0 after
    eos_mask = eos_mask.view(-1).cpu().numpy()

    ### Get unique ID
    syn_id = torch.ones_like(syn_eos[:,:,0])
    syn_id = syn_id.cumsum(dim=0)
    syn_id = syn_id.view(-1).cpu().numpy()

    ### Normalized Time
    # _synth_time = dp.inverse_cyclical_encoding(synth_time.cpu(),1900)
    _synth_time = np.ones_like(syn_id)

### Tensor to tabular:
_synth_data, _ = pce.convert_to_table(real_df1, synth_data, threshold)
_synth_data['date'] = _synth_time
_synth_data['patient_id'] = syn_id
_synth_data = _synth_data[eos_mask > 0]

##################################################################################################################
os.makedirs(args.save_dir, exist_ok=True)
SAVE_PATH = os.path.join(args.save_dir,f"syn_timeautodiff_{args.model_name}")
print(f"Saving data to {SAVE_PATH}...")
torch.save(synth_data,f"{SAVE_PATH}.pt")
_synth_data.to_csv(f"{SAVE_PATH}.csv.gz",
                   index=False, compression="gzip")
