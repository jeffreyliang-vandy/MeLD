import os
import argparse
import numpy as np
import pandas as pd
import h5py
import gc
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import custom modules
from MeLD.model import timeautoencoder as tae
from MeLD.model.data_loader import HDF5Dataset
from model import process_edited as pce  # Assuming this is in your path

# ==========================================
# Classes & Helper Functions
# ==========================================

class IndexedDataset(Dataset):
    """
    A wrapper that returns (data, index) instead of just (data).
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        batch_data = self.base_dataset[idx]
        if isinstance(batch_data, (tuple, list)):
            return (*batch_data, idx)
        return batch_data, idx

# ==========================================
# Main Execution
# ==========================================

def main():
    # --- Initialize Device & Args ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="VAE Generation/Inference Script")
    parser.add_argument("--vae_model", "-VM", default=None, help="Model suffix name")
    parser.add_argument("--model_path", "-MP", required=True, help="Base directory for saved models")
    parser.add_argument("--data_path", "-DP", required=True, help="Path to HDF5 data to run inference on (e.g., test data)")
    parser.add_argument("--condition_path", "-CP", default=None, type=str, help="Path to parquet cohort data to run conditional inference on (e.g., test data)")
    parser.add_argument("--checkpoint", "-ckpt", type=str, default="", help="Specific epoch checkpoint to load, leave empty for best model")
    parser.add_argument("--batch_size", "-BS", type=int, default=128)
    parser.add_argument("--sample_size", "-SS", type=int, required=False)
    args = parser.parse_args()

    # Define directories
    checkpoint_dir = os.path.join(args.model_path, f"vae_{args.vae_model}" if args.vae_model else "vae")
    params_path = os.path.join(checkpoint_dir, "vae_params.pth")
    
    ckpt_suffix = f"_checkpoint_epoch_{args.checkpoint}" if len(args.checkpoint) > 0 else ""
    weights_path = os.path.join(checkpoint_dir, f"vae{ckpt_suffix}.pth")

    print(f"Loading parameters from: {params_path}")
    print(f"Loading weights from: {weights_path}")

    # --- 1. Load Model Config & Weights ---
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found at {params_path}")
    
    model_config = torch.load(params_path, map_location=device)
    ae = tae.DeapStack(**model_config).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at {weights_path}")
    
    checkpoint_dict = torch.load(weights_path, map_location=device)
    # Handle diff between intermediate checkpoint dicts and final 'model_state_dict' save format
    state_dict = checkpoint_dict.get('model_state_dict', checkpoint_dict) 
    ae.load_state_dict(state_dict)
    ae.eval()
    print("✅ Model loaded successfully.")

    # --- 2. Setup Dataloader ---
    print(f"Loading HDF5 data from: {args.data_path}")
    dataset = HDF5Dataset(args.data_path)

    loader = DataLoader(
        IndexedDataset(dataset), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        prefetch_factor=2, 
        pin_memory=True
    )

    # --- 3. Inference Loop ---
    latent_list, idxs_list = [], []
    total_batches = len(loader)
    if args.sample_size:
        print(f"Limiting inference to first {args.sample_size} samples.")
        total_batches = (args.sample_size + args.batch_size - 1) // args.batch_size

    loader = tqdm(loader, total=total_batches)

    print("Running Inference...")
    with torch.no_grad():
        for i, (data, time_info, missing, masking, idxs) in enumerate(loader):
            # Move to device
            data, time_info = data.to(device), time_info.to(device)
            missing, masking = missing.to(device), masking.to(device)

            # Get latent representation
            _, latent_batch,_,_ = ae(data, time_info, missing, masking)

            # Send latents to CPU and store
            latent_list.append(latent_batch.cpu())
            idxs_list.append(idxs.cpu())

            if i >= total_batches:
                break

    # --- 4. Concatenate and Restore Order ---
    print("Concatenating and sorting results by original index...")
    all_indices = torch.cat(idxs_list).numpy()
    sort_idx = all_indices.argsort()

    # Latents
    latent_features = torch.cat(latent_list, dim=0)[sort_idx]
    # Save Latents
    torch.save(latent_features, os.path.join(checkpoint_dir, f"latent_feature{args.checkpoint}.pt"))
    print(f"✅ Latents saved to {checkpoint_dir}/latent_feature{args.checkpoint}.pt with shape {latent_features.shape}")

    if args.condition_path:
        print(f"Loading condition data from: {args.condition_path}")
        condition_dataset = pd.read_parquet(args.condition_path).reset_index(drop=True)
        condition_features = condition_dataset.iloc[all_indices[sort_idx]].reset_index(drop=True) if condition_dataset is not None else None
        condition_features.to_parquet(os.path.join(checkpoint_dir, f"condition_features{args.checkpoint}.parquet"))
        print(f"✅ Condition features saved to {checkpoint_dir}/condition_features{args.checkpoint}.parquet with shape {condition_features.shape}")

if __name__ == "__main__":
    main()