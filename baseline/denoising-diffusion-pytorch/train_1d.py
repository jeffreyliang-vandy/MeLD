import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import normalize as dp
from tqdm.auto import tqdm
import argparse, os

args_parser = argparse.ArgumentParser()
# Add argument for conditional flag
args_parser.add_argument("--vae_checkpoint", "-V", default=None)
args_parser.add_argument("--vae_model", "-VM", default=None,required=True)
args_parser.add_argument("--model", "-M", default=None, required=True)
args_parser.add_argument("--transpose", "-TP", type=bool, default=True)
args_parser.add_argument("--train", "-T", action='store_true', default=True)
args_parser.add_argument("--sample_only", "-SP", action='store_true', default=False)
args_parser.add_argument("--milestone", "-MS", type=int)
args = args_parser.parse_args()

if args.sample_only:
    args.train = False

##### Load Data #####
# training_seq = torch.rand(128, 16, 120) # features are normalized from 0 to 1
training_seq = torch.load(f"./Dataset/save/vae_{args.vae_model}/mu.pt",map_location = 'cpu')

#### Normalize Data
training_seq, min_vals, max_vals = dp.normalize(training_seq)  #normalize to -1, 1
training_seq = (training_seq + 1)/2  # Normalized to 0,1

#### Tranpost data if needed
if args.transpose:
    training_seq = training_seq.swapaxes(1,2)

N, D, T = training_seq.shape
print(f"Training Data Shape: {training_seq.shape}, make sure the time is 3rd axis")

##### Initiate Model #####
device = 'cuda'

model = Unet1D(
    dim = 128,
    # dim_mults = (1, 2, 4, 8),  # 128 -> 64 -> 32 -> 16 -> 4 -> 2
    dim_mults = (1, 2, 4, 8, 16), # deep
    # dim_mults = (1, 2, 4, 8 ,16, 16), # deeper
    channels = D,
    attn_dim_head = 32,
    attn_heads = 8,
    # dropout=0.1,
    # self_condition = True,
    # learned_variance = True,
)

model = model.to(device)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = T,
    timesteps = 500,
    objective = 'pred_v'
    # objective = 'pred_noise'
)

diffusion = diffusion.to(device)


#using trainer
os.makedirs(f'./results/{args.model}',exist_ok = True)

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 256,
    train_lr = 5e-5,
    train_num_steps = 5e+4,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    results_folder = f'./results/{args.model}/',
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 10000,
)

if args.train:
    if args.milestone is not None:
        trainer.load(args.milestone)
    trainer.train()
    #### Model saved to "./results/model-milestone"
if args.sample_only:
    trainer.load(args.milestone)

batch_size = 1024  # Adjust based on memory capacity
num_batches = N // batch_size + int(N % batch_size > 0)

# Sample collection
sample_list = []

with torch.no_grad():
    for _ in tqdm(range(num_batches), total=num_batches):  # Only iterate for the necessary number of batches
        samples = diffusion.sample(batch_size = batch_size)
        sample_list.append(samples)

samples = torch.cat(sample_list, dim=0)

### Swapaxes
if args.transpose:
    samples = samples.swapaxes(1,2)

### Reverse Normalize
samples = samples * 2 - 1  ## to -1 to 1
samples = dp.inverse_normalize(samples,min_vals,max_vals)  ## To original scale

### Save samples
torch.save(samples,f"./results/{args.model}/samples.pt")