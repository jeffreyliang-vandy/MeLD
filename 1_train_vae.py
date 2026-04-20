import os
import argparse
import logging
import numpy as np
import h5py
import time

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam

# Import custom modules
from MeLD.model import timeautoencoder as tae
from MeLD.model.data_loader import HDF5Dataset

# ==========================================
# Classes & Helper Functions
# ==========================================

class IndexedDataset(Dataset):
    """
    A wrapper that returns (data, index) instead of just (data).
    Works with any existing PyTorch Dataset.
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


def setup_logger(checkpoint_dir):
    """Configures the logger to output to both console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training.log")
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
            
    return logger


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    """Generates a cyclical linear schedule for beta."""
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def process_epoch(epoch, data_loader, model, optimizer, device, args, beta, is_train=True):
    """Handles a single epoch of training or validation."""
    total_loss, total_re, total_kl = 0, 0, 0
    data_time_total = 0.0
    compute_time_total = 0.0
    
    if is_train:
        model.train()
    else:
        model.eval()

    num_batches = len(data_loader)
    batch_start_time = time.time()
    
    for batch_idx, (data, time_info, missing, masking) in enumerate(data_loader):
        # Move data to the appropriate device
        data = data.to(device)
        time_info = time_info.to(device)
        missing = missing.to(device)
        masking = masking.to(device)

        # 1. Record how long we waited for the dataloader (CPU -> GPU bottleneck)
        per_batch_data_time = time.time() - batch_start_time
        data_time_total += per_batch_data_time
        
        # 2. Start compute timer
        compute_start_time = time.time()

        if is_train:
            optimizer.zero_grad()
        
        # Call model directly
        RE, KL = model.get_loss(data, time_info, missing, masking)
        delta = torch.tensor(args.min_kl, dtype=KL.dtype, device=device)
        loss = RE + beta * torch.maximum(KL, delta)

        if is_train:
            loss.backward()
            optimizer.step()

        # Extract values directly
        avg_loss = loss.item()
        avg_RE = RE.item()
        avg_KL = KL.item()

        total_loss += avg_loss
        total_re += avg_RE
        total_kl += avg_KL
        
        # Record compute time
        per_batch_compute_time = time.time() - compute_start_time
        compute_time_total += per_batch_compute_time

        print(f"\nEpoch {epoch} - Batch {batch_idx}/{num_batches}| Loss: {avg_loss:.6f} | RE: {avg_RE:.6f} | KL: {avg_KL:.2f} | Beta: {beta:.6f} | Data Loading Time: {per_batch_data_time:.2f}s | GPU Compute Time: {per_batch_compute_time:.2f}s")
        
        # Reset batch timer for the next iteration's data load
        batch_start_time = time.time()
    
    return (
        total_loss / num_batches, 
        total_re / num_batches, 
        total_kl / num_batches, 
        data_time_total / num_batches, 
        compute_time_total/ num_batches
    )

# ==========================================
# Main Execution
# ==========================================

def main():
    # Set device manually
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    
    # Path & Run configurations
    parser.add_argument("--vae_model", "-VM", default=None)
    parser.add_argument("--data_path", "-DP", required=True)
    parser.add_argument("--model_path", "-MP", required=True)
    parser.add_argument("--id", "-I", type=str, default='patient')
    parser.add_argument("--sample", "-SP", action='store_true', default=False)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", "-EP", type=int, default=5000)
    parser.add_argument("--warmup", "-WU", type=int, default=50)
    parser.add_argument("--batch_size", "-BS", type=int, default=128)
    parser.add_argument("--lr", "-LR", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-WD", type=float, default=1e-6)
    parser.add_argument("--patience", "-PT", type=int, default=20)
    parser.add_argument("--early_stop_patience", "-ESP", type=int, default=100)
    parser.add_argument("--save_every", "-SE", type=int, default=50)
    
    # Beta / KL configurations
    parser.add_argument("--min_beta", type=float, default=1e-5)
    parser.add_argument("--max_beta", type=float, default=1e-2)
    parser.add_argument("--min_kl", type=float, default=0.0)
    
    # Architecture Hyperparameters
    parser.add_argument("--lat_dim", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    
    args = parser.parse_args()

    # Set up directories
    args.checkpoint_dir = os.path.join(args.model_path, f"vae_{args.vae_model}" if args.vae_model else "vae")
    
    # Configure logging
    logger = setup_logger(args.checkpoint_dir)
    def main_log(msg, level=logging.INFO):
        logger.log(level, msg)

    # --- 1. Load Data Metadata & Build Model Config ---
    params_path = os.path.join(args.checkpoint_dir, "vae_params.pth")
    
    if not args.sample:
        main_log(f"Loading Data via HDF5: {args.data_path}")
        with h5py.File(args.data_path, 'r') as f:
            n_bins = int(f.attrs['n_bins'])
            n_cats = int(f.attrs['n_cats'])
            n_nums = int(f.attrs['n_nums'])
            cards = f.attrs['cards'].tolist() 
            
            N, seq_len, feature_size = f['processed_data'].shape
            time_dim = f['time_info'].shape[2]
            missing_feat_dim = f['missing'].shape[2]
            
            assert n_nums == missing_feat_dim, "Numerical features count must match missing tensor's feature dim."
            assert sum([n_bins, n_cats, n_nums]) == feature_size, "Sum of feature counts must match processed data dim."

        main_log(f"Dataset size: N={N}, seq_len={seq_len}, feature_size={feature_size}")

        # Construct strictly in the exact order requested by DeapStack
        model_config = {
            "channels": args.channels,
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "n_bins": n_bins,
            "n_cats": n_cats,
            "n_nums": n_nums,
            "cards": cards,
            "feature_size": feature_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "bidirectional": args.bidirectional,
            "emb_dim": args.emb_dim,
            "time_dim": time_dim,
            "lat_dim": args.lat_dim
        }
        
        torch.save(model_config, params_path)
        main_log(f"Model parameters saved at {params_path}")
    else:
        model_config = torch.load(params_path)

    # --- 2. Setup Modules & Dataloaders ---
    dataset = HDF5Dataset(args.data_path)
    
    if not args.sample:
        train_size = int(0.90 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=False)

        main_log("Initializing Modules...")

        ae = tae.DeapStack(**model_config).to(device)
        optimizer_ae = Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for p in ae.parameters())
        print(f"{total_params/1e6:.1f}M parameters")

        # --- 3. Training Loop ---
        main_log("Starting VAE Training...")
        best_loss = float('inf')
        beta = max_beta = args.max_beta
        beta_sched = frange_cycle_linear(n_iter=args.epochs, start=0.0, stop=max_beta, n_cycle=int(args.epochs/5), ratio=0.8)
        patience = 0

        past_epoch, best_loss = tae.load_checkpoint(ae, optimizer_ae, args.checkpoint_dir)
        main_log(f"Best loss from checkpoint: {best_loss}")

        for epoch in range(past_epoch,args.epochs):
            # Train & Validate
            train_loss, train_re, train_kl, tr_data_time, tr_comp_time = process_epoch(epoch, train_loader, ae, optimizer_ae, device, args, beta, is_train=True)
            
            with torch.no_grad():
                val_loss, val_re, val_kl, val_data_time, val_comp_time = process_epoch(epoch, val_loader, ae, optimizer_ae, device, args, beta, is_train=False)
            
            # Logging
            if epoch % 1 == 0:
                main_log(
                    f"Epoch: {epoch}/{args.epochs} | "
                    f"Tr RE: {train_re:.6f} | Val RE: {val_re:.6f} | KL: {train_kl:.2f} | Beta: {beta:.6f} | "
                    f"Tr Wait (CPU): {tr_data_time:.2f}s | Tr Compute (GPU): {tr_comp_time:.2f}s"
                )

            # Scheduler & Checkpointing
            if epoch > args.warmup:
                beta = beta_sched[epoch]
                
                if val_re < best_loss:
                    best_loss = val_re
                    patience = 0
                    main_log(f"New best loss at epoch {epoch}: train:{train_re:.6f} - val:{val_re:.6f}. Saving...")
                    torch.save({'model_state_dict': ae.state_dict()}, os.path.join(args.checkpoint_dir, "vae.pth"))
                else:
                    patience += 1
                    if patience > args.patience and max_beta > args.min_beta:
                        max_beta = max(max_beta * 0.5, args.min_beta)
                        main_log(f"Patience > {args.patience}. Reducing max_beta to: {max_beta:.6f}")
                        beta_sched = frange_cycle_linear(n_iter=args.epochs, start=0.0, stop=max_beta, n_cycle=int(args.epochs/5), ratio=0.8)
                        patience = 0
                    if patience > args.early_stop_patience:
                        main_log(f"Patience > {args.early_stop_patience}. Triggering early stopping.")
                        break

                # Periodic Checkpoint Save
                if epoch % args.save_every == 0:
                    epoch_corrected = epoch + past_epoch
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"vae_checkpoint_epoch_{epoch_corrected}.pth")
                    tae.save_checkpoint(ae, optimizer_ae, epoch_corrected, train_re, args.checkpoint_dir, checkpoint_path)
                    
                    with open(os.path.join(args.checkpoint_dir, "vae_log.txt"), 'a') as f:
                        f.write(f"\nepoch {epoch_corrected:4d} | train_re: {train_re:.6f} | train_kl: {train_kl:.2f} | val_re: {val_re:.6f} | val_kl: {val_kl:.2f} | beta: {max_beta}")

        # Cleanup memory before sampling
        del train_loader, val_loader, train_dataset, val_dataset

    # --- 4. Sampling Loop ---
    main_log("Sampling Model...")
    
    if args.sample:
        ae = tae.DeapStack(**model_config).to(device)

    # Load weights
    ae.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "vae.pth"), map_location=device)['model_state_dict'])
    ae.eval()

    sample_loader = DataLoader(IndexedDataset(dataset), batch_size=args.batch_size, shuffle=False, num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=False)

    emb_list, idxs_list = [], []

    for data, time_info, missing, masking, idxs in sample_loader:
        # Move inputs to device manually
        data = data.to(device)
        time_info = time_info.to(device)
        missing = missing.to(device)
        masking = masking.to(device)

        with torch.no_grad():
            _, emb_batch, _, _ = ae(data, time_info, missing, masking)

        # Move outputs back to CPU for concatenation
        emb_list.append(emb_batch.cpu())
        idxs_list.append(idxs.cpu())

    all_embs = torch.cat(emb_list)
    all_indices = torch.cat(idxs_list).numpy()
    
    sort_idx = all_indices.argsort()
    sorted_embs = all_embs[sort_idx]
    
    torch.save(sorted_embs, os.path.join(args.checkpoint_dir, "latent_feature.pt"))
    main_log("Sampling complete. Latent features saved.")

if __name__ == "__main__":
    main()