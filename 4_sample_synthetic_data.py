import os
import gc
import random
import argparse
import gc
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

from model import timeautoencoder as tae
from model import DP as dp
from model import process_edited as pce

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate tabular data from VAE latents.")
    parser.add_argument("--n", "-N", type=int, required=False)
    parser.add_argument("--save_dir", "-S", type=str, default="./save/generated/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", "-M", type=str, required=True)
    parser.add_argument("--vae_checkpoint", "-V", type=str, default=None, required=False)
    parser.add_argument("--vae_model", "-VM", type=str, default=None, required=False)
    parser.add_argument("--vae_path", "-VP", type=str, default=None, required=False)
    parser.add_argument("--ddpm_latents_path", "-LP", type=str, default=None, required=False)
    parser.add_argument("--scaled", "-SL", action="store_true", default=False)
    parser.add_argument("--missing", "-MS", action="store_true", default=False)
    parser.add_argument("--data_path", "-DP", type=str, default="/home/jeffrey/Documents/EA_project/data/dataset/ea_hiv/preprocessed/merged/train/parser.pkl")
    
    return parser.parse_args()

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parse_arguments()
    set_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.current_device()}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate tabular data from VAE latents.")
    parser.add_argument("--n", "-N", type=int, required=False)
    parser.add_argument("--save_dir", "-S", type=str, default="./save/generated/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", "-M", type=str, required=True)
    parser.add_argument("--vae_checkpoint", "-V", type=str, default=None, required=False)
    parser.add_argument("--vae_model", "-VM", type=str, default=None, required=False)
    parser.add_argument("--vae_path", "-VP", type=str, default=None, required=False)
    parser.add_argument("--ddpm_latents_path", "-LP", type=str, default=None, required=False)
    parser.add_argument("--scaled", "-SL", action="store_true", default=False)
    parser.add_argument("--missing", "-MS", action="store_true", default=False)
    parser.add_argument("--data_path", "-DP", type=str, default="/home/jeffrey/Documents/EA_project/data/dataset/ea_hiv/preprocessed/merged/train/parser.pkl")
    
    return parser.parse_args()

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parse_arguments()
    set_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.current_device()}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Set up the base VAE directory using args.vae_path if provided
    if args.vae_path:
        base_vae_path = args.vae_path
    else:
        vae_model_suffix = f"_{args.vae_model}" if args.vae_model else ""
        base_vae_path = f"./save/vae{vae_model_suffix}"

    ckpt = f"_checkpoint_epoch_{args.vae_checkpoint}" if args.vae_checkpoint else ""
    vae_latent_path = os.path.join(base_vae_path, f"latent_feature{ckpt}.pt")

    # -------------------------------------------------------------------------
    # 1. Load Latents
    # -------------------------------------------------------------------------
    print(f"\nLoading Latents at {args.ddpm_latents_path}...")
    if not os.path.exists(args.ddpm_latents_path):
        raise FileNotFoundError(f"❌ Latents file not found at {args.ddpm_latents_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Set up the base VAE directory using args.vae_path if provided
    if args.vae_path:
        base_vae_path = args.vae_path
    else:
        vae_model_suffix = f"_{args.vae_model}" if args.vae_model else ""
        base_vae_path = f"./save/vae{vae_model_suffix}"

    ckpt = f"_checkpoint_epoch_{args.vae_checkpoint}" if args.vae_checkpoint else ""
    vae_latent_path = os.path.join(base_vae_path, f"latent_feature{ckpt}.pt")

    # -------------------------------------------------------------------------
    # 1. Load Latents
    # -------------------------------------------------------------------------
    print(f"\nLoading Latents at {args.ddpm_latents_path}...")
    if not os.path.exists(args.ddpm_latents_path):
        raise FileNotFoundError(f"❌ Latents file not found at {args.ddpm_latents_path}")

    if args.ddpm_latents_path.endswith('.pt'):
        samples = torch.load(args.ddpm_latents_path, map_location='cpu')
        samples = torch.load(args.ddpm_latents_path, map_location='cpu')
    elif args.ddpm_latents_path.endswith('.npy'):
        samples = torch.tensor(np.load(args.ddpm_latents_path)).cpu()
        samples = torch.tensor(np.load(args.ddpm_latents_path)).cpu()

    if not args.scaled:
        if samples.min() >= 0: 
            samples = samples * 2 - 1  # Map [0,1] to [-1,1]
            
        latent_features = torch.load(vae_latent_path)
        if samples.min() >= 0: 
            samples = samples * 2 - 1  # Map [0,1] to [-1,1]
            
        latent_features = torch.load(vae_latent_path)
        _, max_val, min_val = dp.normalize(latent_features)
        samples = dp.inverse_normalize(samples, max_val, min_val)
        samples = dp.inverse_normalize(samples, max_val, min_val)
        del latent_features

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 2. Load VAE Model
    # -------------------------------------------------------------------------
    print("\nLoad VAE model...")
    params_path = os.path.join(base_vae_path, "vae_params.pth")
    
    if os.path.exists(params_path):
        model_params = torch.load(params_path, map_location=device)
        print(f"✅ Model parameters loaded successfully from {params_path}")
    else:
        raise FileNotFoundError(f"❌ Parameters file not found at {params_path}")
    # -------------------------------------------------------------------------
    # 2. Load VAE Model
    # -------------------------------------------------------------------------
    print("\nLoad VAE model...")
    params_path = os.path.join(base_vae_path, "vae_params.pth")
    
    if os.path.exists(params_path):
        model_params = torch.load(params_path, map_location=device)
        print(f"✅ Model parameters loaded successfully from {params_path}")
    else:
        raise FileNotFoundError(f"❌ Parameters file not found at {params_path}")

    ae = tae.DeapStack(**model_params).to(device)
    
    print("Loading Weights...")
    checkpoint_name = f"vae_checkpoint_epoch_{args.vae_checkpoint}.pth" if args.vae_checkpoint else "vae.pth"
    checkpoint_path = os.path.join(base_vae_path, checkpoint_name)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ae.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model weights loaded successfully from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"❌ Model checkpoint not found at {checkpoint_path}")
    ae = tae.DeapStack(**model_params).to(device)
    
    print("Loading Weights...")
    checkpoint_name = f"vae_checkpoint_epoch_{args.vae_checkpoint}.pth" if args.vae_checkpoint else "vae.pth"
    checkpoint_path = os.path.join(base_vae_path, checkpoint_name)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ae.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model weights loaded successfully from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"❌ Model checkpoint not found at {checkpoint_path}")

    ae.eval()
    ae.eval()

    # -------------------------------------------------------------------------
    # 3. Post-process the Generated Data
    # -------------------------------------------------------------------------
    print("\nPost-process the generated data...")
    batch_size = 128
    num_batches = (samples.shape[0] + batch_size - 1) // batch_size
    output_dict = {'bins': [], 'cats': [], 'nums': [], 'times': [], 'eos': [], 'missings': []}

    with torch.no_grad():
        for i in tqdm(range(num_batches), total=num_batches):
            start = i * batch_size
            end = min(start + batch_size, samples.shape[0])
            batch_data = samples[start:end].to(device)
            output = ae.decoder(batch_data)
    # -------------------------------------------------------------------------
    # 3. Post-process the Generated Data
    # -------------------------------------------------------------------------
    print("\nPost-process the generated data...")
    batch_size = 128
    num_batches = (samples.shape[0] + batch_size - 1) // batch_size
    output_dict = {'bins': [], 'cats': [], 'nums': [], 'times': [], 'eos': [], 'missings': []}

    with torch.no_grad():
        for i in tqdm(range(num_batches), total=num_batches):
            start = i * batch_size
            end = min(start + batch_size, samples.shape[0])
            batch_data = samples[start:end].to(device)
            output = ae.decoder(batch_data)

            if 'bins' in output: output_dict['bins'].append(output['bins'].cpu())
            if 'cats' in output: output_dict['cats'].append([cat.cpu() for cat in output['cats']])
            if 'nums' in output: output_dict['nums'].append(output['nums'].cpu())
            if 'times' in output: output_dict['times'].append(output['times'].cpu())
            if 'eos' in output: output_dict['eos'].append(output['eos'].cpu())
            if 'missings' in output: output_dict['missings'].append(output['missings'].cpu())
            if 'bins' in output: output_dict['bins'].append(output['bins'].cpu())
            if 'cats' in output: output_dict['cats'].append([cat.cpu() for cat in output['cats']])
            if 'nums' in output: output_dict['nums'].append(output['nums'].cpu())
            if 'times' in output: output_dict['times'].append(output['times'].cpu())
            if 'eos' in output: output_dict['eos'].append(output['eos'].cpu())
            if 'missings' in output: output_dict['missings'].append(output['missings'].cpu())

    # Concatenate collected results
    gen_output = {}
    for key in ['bins', 'nums', 'times', 'eos', 'missings']:
        if output_dict[key]:
            gen_output[key] = torch.cat(output_dict[key], dim=0)
    # Concatenate collected results
    gen_output = {}
    for key in ['bins', 'nums', 'times', 'eos', 'missings']:
        if output_dict[key]:
            gen_output[key] = torch.cat(output_dict[key], dim=0)

    if output_dict['cats']:
        num_categories = len(output_dict['cats'][0])
        gen_output['cats'] = [torch.cat([batch[i] for batch in output_dict['cats']], dim=0) for i in range(num_categories)]
    if output_dict['cats']:
        num_categories = len(output_dict['cats'][0])
        gen_output['cats'] = [torch.cat([batch[i] for batch in output_dict['cats']], dim=0) for i in range(num_categories)]

    if not args.missing:
        gen_output.pop('missings', None)
        
    del ae
    samples = samples.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    if not args.missing:
        gen_output.pop('missings', None)
        
    del ae
    samples = samples.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Transform Tensor Back to Tabular
    # -------------------------------------------------------------------------
    print(f"\nLoading Real Data from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        pce_parser = pickle.load(f)

    print("\nTransforming tensor back to tabular...")
    data_size, seq_len, _ = samples.shape
    # -------------------------------------------------------------------------
    # 4. Transform Tensor Back to Tabular
    # -------------------------------------------------------------------------
    print(f"\nLoading Real Data from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        pce_parser = pickle.load(f)

    print("\nTransforming tensor back to tabular...")
    data_size, seq_len, _ = samples.shape

    synth_data, _, syn_eos = pce.convert_to_tensor(pce_parser, gen_output, data_size, seq_len)
    synth_date = dp.inverse_cyclical_encoding(gen_output['times']) if 'times' in gen_output else np.ones((data_size, seq_len)).reshape(-1)
    del gen_output, samples

    with torch.no_grad():
        eos_probabilities = F.softmax(syn_eos, dim=-1)[..., 1]
        eos_predictions = (eos_probabilities > 0.5).int() 
        
        # Cumulative mask: set to 0 after the first EOS
        eos_cumsum = eos_predictions.cumsum(dim=1).cumsum(dim=1)
        eos_mask = (eos_cumsum <= 1).float().unsqueeze(-1).view(-1).cpu().numpy()
    with torch.no_grad():
        eos_probabilities = F.softmax(syn_eos, dim=-1)[..., 1]
        eos_predictions = (eos_probabilities > 0.5).int() 
        
        # Cumulative mask: set to 0 after the first EOS
        eos_cumsum = eos_predictions.cumsum(dim=1).cumsum(dim=1)
        eos_mask = (eos_cumsum <= 1).float().unsqueeze(-1).view(-1).cpu().numpy()

        # Unique IDs and Normalized Time
        syn_id = torch.ones_like(syn_eos[:, :, 0]).cumsum(dim=0).view(-1).cpu().numpy()
        # Unique IDs and Normalized Time
        syn_id = torch.ones_like(syn_eos[:, :, 0]).cumsum(dim=0).view(-1).cpu().numpy()

    _synth_data, _ = pce.convert_to_table(pce_parser,synth_data)
    _synth_data['date'] = synth_date
    _synth_data['patient'] = syn_id
    _synth_data = _synth_data[eos_mask > 0]
    
    del pce_parser, synth_data
    gc.collect()

    # -------------------------------------------------------------------------
    # 5. Save Data
    # -------------------------------------------------------------------------
    save_path = os.path.join(args.save_dir, f"syn_{args.model_name}_{args.seed}.csv.gz")
    print(f"\nSaving data to {save_path}...")
    _synth_data.to_csv(save_path, index=False, compression="gzip")
    print("✅ Done!")

if __name__ == "__main__":
    main()