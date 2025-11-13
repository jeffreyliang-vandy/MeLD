import os

# Set the visible devices to only GPU 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Verify CUDA device visibility in PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
print("Available CUDA devices:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())

import numpy as np
import TimeLDM as tae
import DP as dp
import pandas as pd
import numpy as np
import process_edited as pce
from rich.progress import Progress
import math

import argparse

args_parser = argparse.ArgumentParser()
# Add argument for conditional flag
args_parser.add_argument("--vae_model", "-VM", default=None)
args_parser.add_argument("--sample", "-SP",action='store_true', default=False)
args_parser.add_argument("--train_decoder", "-DE",action='store_true', default=False)
args_parser.add_argument("--warmup", "-WU", type=int, default=50)
args_parser.add_argument("--epochs", "-EP", type=int, default=5000)
args = args_parser.parse_args()

if args.vae_model is None:
    args.checkpoint_dir = f"../Dataset/save/vae/"
else:
    args.checkpoint_dir = f"../Dataset/save/vae_{args.vae_model}/"
    
os.makedirs(args.checkpoint_dir, exist_ok=True)

threshold = 1; device = 'cuda';

##################################################################################################################
# Read dataframe
if not args.sample:
    print("Loading Data...")
    filename = f'../Dataset/hiv_train.csv.gz'
    print(filename)
    real_df1 = pd.read_csv(filename).fillna(0).drop_duplicates()

    real_df1 = real_df1.drop('date', axis=1)
    seq_col = 'patient_id';real_df1 = real_df1.drop(seq_col, axis=1)

    parser = pce.DataFrameParser().fit(real_df1, threshold=1)
    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']

    ## test missing data
    has_na = real_df1.isna().any().any()
    print("Are there any missing values in the DataFrame? ", has_na)
    print(f"real_df.shape: {real_df1.shape}")
    del real_df1
##################################################################################################################
# Pre-processing Data
print("Preprocess Data...")

# # column_to_partition = 'Symbol'; processed_data, time_info = dp.partition_multi_seq(real_df, threshold, column_to_partition,10);
column_to_partition = 'patient_id'; 

processed_data = torch.load("../Dataset/save/processed_data.pt",map_location='cpu')
time_info = torch.load("../Dataset/save/time_info.pt",map_location='cpu')
missing = torch.load("../Dataset/save/missing.pt",map_location='cpu')
masking = torch.load("../Dataset/save/masking.pt",map_location='cpu')

print(f"preprocessed_data.shape: {processed_data.shape}")
print(f"time_info.shape: {time_info.shape}")

##################################################################################################################
# Auto-encoder Training
print("Training VAE...")

weight_decay = 1e-6 ; lr = 1e-4; hidden_size = 256; num_layers = 2; batch_size = 256
channels = 64; min_beta = 1e-5; max_beta = 1e-2; emb_dim = 128; time_dim = time_info.shape[2]
lat_dim = 8; bidirectional = True; min_kl = 0

# Create full dataset and then split into training and validation sets
dataset = TensorDataset(processed_data, time_info, missing, masking)
train_ratio = 0.90
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


print("Initiate Modules...")
N, seq_len, feature_size = processed_data.shape

# Define model parameters save path
PARAMS_PATH = os.path.join(args.checkpoint_dir, "vae_params.pth")

if not args.sample:
    # Save model hyperparameters as a dictionary
    model_params = {
        "channels": channels,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_bins": n_bins,
        "n_cats": n_cats,
        "n_nums": n_nums,
        "cards": cards,
        "feature_size": feature_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "emb_dim": emb_dim,
        "time_dim": time_dim,
        "lat_dim": lat_dim
    }
    # Save model parameters
    torch.save(model_params, PARAMS_PATH)
    with open(os.path.join(args.checkpoint_dir, f"vae_log.txt"), 'a') as f:
        f.write(f"{model_params}")
    print(f"✅ Model parameters saved at {PARAMS_PATH}")

model_params = torch.load(PARAMS_PATH)
ae = tae.TimeLDM(**model_params).to(device)
optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

def reset_weights(m):
    # If the module has a reset_parameters() method, call it.
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train_vae(epoch, data_loader, model, optimizer, max_beta = 1, mode = 'training'):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    if mode == 'training':
        model.train()
    else:
        model.eval()

    # start training
    if args.warmup == 0:
        beta = max_beta
    else:
        beta = max_beta * epoch / args.warmup
        if beta > max_beta:
            beta = max_beta
    # print('beta: {}'.format(beta))

    for batch_idx, (data, time_info, missing, masking) in enumerate(data_loader):

        data = data.to(device)
        time_info = time_info.to(device)
        missing = missing.to(device)
        masking = masking.to(device)

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        RE, KL = model.get_loss(data,time_info,missing,masking)
        delta = torch.tensor(min_kl, dtype=KL.dtype, device=KL.device)
        loss = RE + beta * torch.maximum(KL,delta)

        progress.update(training_task, advance = 1, description=f"Batch {batch_idx}/{len(data_loader)} - Loss: {loss.item():.6f} - RE:{RE.item():.6f} - KL: {KL.item():.1f}")

        if mode == 'training':
            # backward pass
            loss.backward()
            # optimization
            optimizer.step()

        train_loss += loss.item()
        train_re += RE.item()
        train_kl += KL.item()
    
    # calculate final loss
    train_loss /= len(data_loader)  # loss function already averages over batch size
    train_re /= len(data_loader)  # re already averages over batch size
    train_kl /= len(data_loader)  # kl already averages over batch size
    progress.update(training_task, description=f"Epoch {epoch} - Loss: {train_loss:.6f} - RE:{train_re:.6f} - KL: {train_kl:.1f}")

    return train_loss, train_re, train_kl

# Helper function for cyclical beta
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

best_loss = float('inf')
beta = max_beta
beta_sched = frange_cycle_linear(n_iter=args.epochs, start=0.0, stop=max_beta,  n_cycle=int(args.epochs/10), ratio=0.8)
patient = 0

if not args.sample:
    past_epoch, best_loss = tae.load_checkpoint(ae,optimizer_ae,args.checkpoint_dir)
    print(f"best loss: {best_loss}")

    if args.train_decoder:
        ae.encoder.eval()  ## Lock Encoder
        for param in ae.encoder.parameters():
            param.requires_grad = False
        ae.decoder.apply(reset_weights) ## Reset Decoder
        for param in ae.decoder.parameters():
            param.requires_grad = True
        optimizer_ae = Adam(ae.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        print("Train Decoder Only...")

    with Progress() as progress:
    
        total_task = progress.add_task("[green]Training...", total=args.epochs)
        training_task = progress.add_task("[red]Optimizing...", total=len(train_loader))

        for epoch in range(int(args.epochs)):
            # Training step
            train_loss, train_re, train_kl = train_vae(epoch, train_loader, ae, optimizer_ae, beta, mode='training')
            progress.reset(training_task)
            # Validation step
            with torch.no_grad():
                val_loss, val_re, val_kl = train_vae(epoch, val_loader, ae, optimizer_ae, beta, mode='val')
            progress.reset(training_task)

            progress.update(total_task, advance=1, description=f"Epoch: {epoch}/{args.epochs} - Train Loss: {train_re:6f} - Val Loss: {val_re:6f} - Beta: {beta:3f}")

            if epoch > args.warmup:

                beta = beta_sched[epoch]

                if val_re < best_loss:
                    best_loss = val_re
                    patient = 0
                    print(f"[INFO] New best loss at epoch {epoch}: train:{train_re:.6f} - val:{val_re:.6f}. Saving model...")
                    with open(os.path.join(args.checkpoint_dir, f"vae_log.txt"), 'a') as f:
                        f.write(f"\n[INFO] New best loss at epoch {epoch}: train:{train_re:.6f} - val:{val_re:.6f}. Saving model...")
                    if not args.train_decoder:
                        # Save the entire model
                        torch.save({'model_state_dict':ae.state_dict()}, os.path.join(args.checkpoint_dir, "vae.pth"))
                        # Save just encoder
                        torch.save({'model_state_dict':ae.encoder.state_dict()}, os.path.join(args.checkpoint_dir, "encoder.pth"))
                    # Save just decoder
                    torch.save({'model_state_dict':ae.decoder.state_dict()}, os.path.join(args.checkpoint_dir, "decoder.pth"))
                else:
                    patient += 1
                    if patient > 20 and max_beta > min_beta:
                        max_beta = max_beta * 0.7 if max_beta > min_beta else min_beta
                        print(f"Using beta :{max_beta}")
                        beta_sched = frange_cycle_linear(n_iter=args.epochs, start=0.0, stop=max_beta,  n_cycle=int(args.epochs/10), ratio=0.8)
                        patient = 0
                    if patient > 100:
                        print("No longer improving")
                        break

                if epoch % 50 == 0:
                    # Save the entire model
                    print(f"[INFO] loss at epoch {epoch}: train:{train_re:.6f} - val:{val_re:.6f}... Saving model...")
                    epoch = epoch + past_epoch
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"vae_checkpoint_epoch_{epoch}.pth")
                    tae.save_checkpoint(ae, optimizer_ae, epoch, train_re, args.checkpoint_dir, checkpoint_path)
                    with open(os.path.join(args.checkpoint_dir, f"vae_log.txt"), 'a') as f:
                        f.write(f"\nepoch {epoch:4d} | train_re: {train_re:.6f} | train_kl: {train_kl:.2f}| val_re: {val_re:.6f}\n| val_kl:{val_kl:.2f}| beta: {max_beta}")


##################################################################################################################
print("Sampling Model...")

ae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae.pth")['model_state_dict'])
# ae = torch.compile(ae)
ae.eval()

del train_loader
sample_loader = DataLoader(dataset,batch_size=128)

emb_list = []
mu_list = []
logvar_list = []

with Progress() as progress:
    
    training_task = progress.add_task("[red]Sampling...", total=len(sample_loader))

    for batch_idx, (data, time_info, missing, masking) in enumerate(sample_loader):

        data = data.cuda()
        time_info = time_info.cuda()
        missing = missing.cuda()
        masking = masking.cuda()

        with torch.no_grad():
            _, emb_batch, mu_batch, logvar_batch = ae(data,time_info,missing,masking)

        emb_list.append(emb_batch.cpu())
        mu_list.append(mu_batch.cpu())
        logvar_list.append(logvar_batch.cpu())
        progress.update(training_task, advance=1, description=f"Batch: {batch_idx}/{len(sample_loader)}")

    emb = torch.cat(emb_list)
    torch.save(emb,f"{args.checkpoint_dir}/latent_feature.pt")

    mu = torch.cat(mu_list)
    torch.save(mu,f"{args.checkpoint_dir}/mu.pt")

    logvar = torch.cat(logvar_list)
    torch.save(logvar,f"{args.checkpoint_dir}/logvar.pt")