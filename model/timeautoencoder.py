import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import Adam
import math
from rich.progress import Progress

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

################################################################################################################
class NonLinear(nn.Module):
    def __init__(self, feature_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(feature_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h
    
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    angles = 2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * v.unsqueeze(-1)
    # angles = 2**torch.arange(num_terms).to(device, dtype=torch.float16) * torch.tensor(math.pi).to(device,dtype=torch.float16) * v.unsqueeze(-1)

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.view(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.view(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result

################################################################################################################
class Discriminator(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

################################################################################################################
class Embedding_data(nn.Module):
    def __init__(self, feature_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        
        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2]*self.n_bins + self.cards
        
        if self.n_disc > 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList([
                nn.Embedding(num_categories, emb_dim) for num_categories in self.num_categorical_list
            ])

        if self.n_nums > 0:
            self.mlp_nums = nn.Sequential(nn.Linear(32 * n_nums, 32 * n_nums),  # this should be 16 * n_nums, 16 * n_nums
                                          nn.SiLU(),
                                          nn.Linear(32 * n_nums, emb_dim))
        
        if True:
            self.paddings_list = nn.ModuleList([
                    nn.Embedding(2, emb_dim) for _ in range(2)
                ])
        
        if True:
            self.missings_list = nn.ModuleList([
                    nn.Embedding(2, emb_dim) for _ in range(self.n_nums)
                ])

        # Final embedding processing
        self.mlp_output = nn.Sequential(
            nn.Linear(3*emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x, missing, masking=None):
        
        x_disc = x[:,:,0:self.n_disc].long().to(device)
        x_nums = x[:,:,self.n_disc:self.n_disc+self.n_nums].to(device)

        ####### Addition Approach
        x_emb_sum = torch.zeros(x.shape[0], x.shape[1], len(self.embeddings_list[0].weight[0]), device=device)

        # Process individual embeddings and sum them
        if self.n_disc > 0:
            for i, embedding in enumerate(self.embeddings_list):
                x_emb_sum += embedding(x_disc[:, :, i])  # Element-wise addition instead of concatenation

        # Process numerical variables with sine/cosine encoding
        if self.n_nums > 0:
            x_nums = compute_sine_cosine(x_nums, num_terms=16)
            x_nums_emb = self.mlp_nums(x_nums)
            # x_emb_sum += x_nums_emb  # Add numerical embedding instead of concatenation
            x_emb_sum = torch.cat([x_emb_sum,x_nums_emb],dim=2)
        #   print(f"x_emb_sum.shape:{x_emb_sum.shape}")

        x_emb_special = torch.zeros(x.shape[0], x.shape[1], len(self.missings_list[0].weight[0]), device=device)
        if True:
            x_missings = missing.long().to(device)
            for i, embedding in enumerate(self.missings_list):  # Embed paddings as categorical and add to categorical
                x_emb_special += embedding(x_missings[ :, :, i])

        if masking is not None:  # added masking options
            x_paddings = masking[:,:,:].long().to(device)
            for i, embedding in enumerate(self.paddings_list):  # Embed paddings as categorical and add to categorical
                x_emb_special += embedding(x_paddings[ :, :, i])
        
        x_emb_sum = torch.cat([x_emb_sum,x_emb_special],dim=2)
        
        final_emb = self.mlp_output(x_emb_sum)
        return final_emb

################################################################################################################
def get_torch_trans(heads = 8, layers = 1, channels = 64):
   encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
   return nn.TransformerEncoder(encoder_layer, num_layers = layers)

class Transformer_Block(nn.Module):
   def __init__(self, channels):
       super().__init__()
       self.channels = channels
        
       self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
       self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
       self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)
    
   def forward_feature(self, y, base_shape):
       B, channels, L, K = base_shape
       if K == 1:
           return y.squeeze(1)
       y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
       y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
       y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
       return y
    
   def forward(self, x):
       x = x.unsqueeze(1)
       B, input_channel, K, L = x.shape
       base_shape = x.shape

       x = x.reshape(B, input_channel, K*L)       
        
       conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
       x = self.forward_feature(conv_x, conv_x.shape)
       x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)
        
       return x

################################################################################################################
# @torch.compile
class Encoder(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size, hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim):
        super().__init__()
        self.Emb = Embedding_data(feature_size, emb_dim, n_bins, n_cats, n_nums, cards)
        self.time_encode = nn.Sequential(nn.Linear(time_dim, emb_dim),
                                         nn.ReLU(),
                                         nn.Linear(emb_dim, emb_dim))
        if channels > 0:
            self.encoder_Transformer = Transformer_Block(channels)
        else: self.encoder_Transformer = None
        
        self.encoder_mu = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, dropout= 0.2, bidirectional = bidirectional)
        self.encoder_logvar = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, dropout= 0.2, bidirectional = bidirectional)
        
        if bidirectional:
            self.fc_mu = nn.Linear(2*hidden_size, lat_dim)
            self.fc_logvar = NonLinear(2*hidden_size, lat_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        else:
            self.fc_mu = nn.Linear(hidden_size, lat_dim)
            self.fc_logvar = NonLinear(hidden_size, lat_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, time_info,missing, masking):
        x = self.Emb(x, missing, masking)
        if self.encoder_Transformer is not None:
            x = self.encoder_Transformer(x)
        x = x + self.time_encode(time_info)
        
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)
        
        mu_z = self.fc_mu(mu_z); logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)
        
        return emb, mu_z, logvar_z
    
class Decoder(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size, hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim):
        super().__init__()
        self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size))
        
        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid ()
        
        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats else None 
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

        self.masks_linear = nn.Linear(hidden_size, 2)  # Try softmax
        self.missings_linear = nn.Linear(hidden_size, n_nums) 
        self.times_linear = None
    
    def forward(self, latent_feature):
        decoded_outputs = dict()
        x = self.decoder_mlp(latent_feature)

        # Compute EOS mask
        if self.masks_linear is not None:
            ## Use softmax
            decoded_outputs['eos'] = self.masks_linear(x)
            eos_probabilities = F.softmax(decoded_outputs['eos'], dim=-1)[...,1]
            # Compute the cumulative mask: set to 0 after the first EOS in each sequence
            eos_predictions = (eos_probabilities > 0.5).int()  # Binary EOS predictions (B, L)
            eos_cumsum = eos_predictions.cumsum(dim=1).cumsum(dim=1)  # Cumulative sum along the sequence length
            eos_mask = (eos_cumsum <= 1).float().unsqueeze(-1)  # Mask: 1 before and at first EOS, 0 after

        if self.missings_linear is not None:
            decoded_outputs['missings'] = self.missings_linear(x) * eos_mask

        if self.bins_linear is not None:
            decoded_outputs['bins'] = self.bins_linear(x) * eos_mask

        if self.cats_linears is not None:
            decoded_outputs['cats'] = [linear(x) * eos_mask for linear in self.cats_linears]

        if self.nums_linear is not None:
            decoded_outputs['nums'] = self.sigmoid(self.nums_linear(x)) * eos_mask

        if self.times_linear is not None:
            decoded_outputs['times'] = None

        return decoded_outputs


class DeapStack(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size, hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim):
        super().__init__()
        self.encoder = Encoder(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size, hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim)
        self.decoder = Decoder(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size, hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim)
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_loss(self, inputs, time_info, missing, masking):

        outputs, _, mu_z, logvar_z = self.forward(inputs,time_info,missing,masking)
        
        disc_loss, num_loss = auto_loss(inputs,time_info,missing,masking,\
                                        outputs, self.n_bins, self.n_nums, self.n_cats, self.cards)
        
        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        loss_RE = disc_loss + num_loss

        return loss_RE, loss_kld
    
    def forward(self, x, time_info, missing = None, masking=None):
        emb, mu_z, logvar_z = self.encoder(x, time_info,missing, masking)
        outputs = self.decoder(emb) 
        return outputs, emb, mu_z, logvar_z


def auto_loss(inputs, time_info, missing, masking, reconstruction, n_bins, n_nums, n_cats, cards):
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, K = inputs.shape

    bins = inputs[:,:,0:n_bins]
    cats = inputs[:,:,n_bins:n_bins+n_cats].long()
    nums = inputs[:,:,n_bins+n_cats:n_bins+n_cats+n_nums]
    time_info = time_info
    missing_nums = 1. - missing[...,-n_nums:]

    eos = masking[:, :, -2] if masking is not None else torch.zeros(B, L, device=device)
    padding = masking[:, :, -1] if masking is not None else torch.zeros(B, L, device=device)
    real_mask = (padding < 1).float() if masking is not None else torch.ones(B, L, device=device)

    #reconstruction_losses = dict()
    disc_loss = 0; num_loss = 0; first_visit_scale = 10.0
    
    if 'bins' in reconstruction:
        # BCE with logits => shape (B, L, n_bins) if reduction='none'
        bce = F.binary_cross_entropy_with_logits(
            reconstruction['bins'], bins, reduction='none'
        )  # shape (B, L, n_bins)
        # Sum over the n_bins dimension so we have one loss per (B,L)
        bce_per_timestep = bce.sum(dim=-1)  # shape (B, L)
        # Apply real_mask
        masked_bce = bce_per_timestep * real_mask  # shape (B, L)
        if first_visit_scale > 1.0:
            # upweight the first visit(L=0) loss
            first_visit_mask = (torch.arange(L, device=device) == 0).float()
            masked_bce = masked_bce * (1.0 + first_visit_scale * first_visit_mask)
        # Now average only over real steps
        disc_loss += masked_bce.sum() / (real_mask.sum() + 1e-8)

    if 'eos' in reconstruction and masking is not None:
        ## Use softmax:
        eos_loss = F.cross_entropy(reconstruction['eos'].view(-1, 2), eos.long().view(-1), reduction='none')
        eos_loss = (eos_loss * real_mask.view(-1)).sum() / (real_mask.sum() + 1e-8)

        disc_loss += eos_loss

    if 'missings' in reconstruction:
        # BCE with logits => shape (B, L, n_nums) if reduction='none'
        bce = F.binary_cross_entropy_with_logits(
            reconstruction['missings'], missing, reduction='none'
        )  # shape (B, L, n_nums)
        # Sum over the n_bins dimension so we have one loss per (B,L)
        bce_per_timestep = bce.sum(dim=-1)  # shape (B, L)
        # Apply real_mask
        missing_loss = bce_per_timestep * real_mask  # shape (B, L)
        missing_loss = missing_loss.sum() / (real_mask.sum() + 1e-8)
        disc_loss += missing_loss

    if 'cats' in reconstruction:
        cats_losses = []

        for i, cat_linear_out in enumerate(reconstruction['cats']):
            # cat_linear_out => shape (B, L, cardinality_of_this_cat)
            # flatten for cross-entropy: (B*L, card)
            logits_2d = cat_linear_out.view(B*L, cards[i])
            # ground truth: shape (B, L) => flattened => (B*L)
            targets_1d = cats[:,:,i].view(B*L)
            # Cross-entropy for each (B*L) element
            ce = F.cross_entropy(logits_2d, targets_1d, reduction='none')  
            # shape => (B*L,)
            # Reshape to (B, L) to match mask shape
            ce_2d = ce.view(B, L)
            # Apply mask
            ce_masked = ce_2d * real_mask  # shape (B, L)
            if first_visit_scale > 1.0:
                # upweight the first visit(L=0) loss
                first_visit_mask = (torch.arange(L, device=device) == 0).float()
                ce_masked = ce_masked * (1.0 + first_visit_scale * first_visit_mask)
            # Average over real positions
            cat_loss_i = ce_masked.sum() / (real_mask.sum() + 1e-8)
            cats_losses.append(cat_loss_i)

        disc_loss += torch.stack(cats_losses).mean()

    if 'nums' in reconstruction:
        # MSE => shape (B, L, n_nums) if reduction='none'
        mse = F.mse_loss(reconstruction['nums'], nums, reduction='none')
        if True: 
            mse = mse * missing_nums
        # sum along n_nums dimension => (B, L)
        mse_per_timestep = mse.sum(dim=-1)
        # Mask out padded timesteps
        masked_mse = mse_per_timestep * real_mask
        if first_visit_scale > 1.0:
            # upweight the first visit(L=0) loss
            first_visit_mask = (torch.arange(L, device=device) == 0).float()
            masked_mse = masked_mse * (1.0 + first_visit_scale * first_visit_mask)
        # Average over real positions
        num_loss += masked_mse.sum() / (real_mask.sum() + 1e-8)

    if 'times' in reconstruction:
        # MAE => shape (B, L, n_times) if reduction='none'
        mae = F.l1_loss(reconstruction['times'], time_info, reduction='none') ## MSE is not working, KL exploded, L1 loss bad results
        mae_per_timestep = mae.sum(dim=-1)
        # Mask out padded timesteps
        masked_mae = mae_per_timestep * real_mask
        # Average over real positions
        num_loss += masked_mae.sum() / (real_mask.sum() + 1e-8)

        ## Cosine similarity loss for time reconstruction #using cosine only also bad results, but it is more stable than MSE
        pred = reconstruction['times'].view(B, L, -1, 2)
        target = time_info.view(B, L, -1, 2)
        cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)  # (B, L, n_cycles)
        cos_loss = 1 - cos_sim
        cos_loss = cos_loss.mean(dim=-1)
        masked_cos = cos_loss * real_mask
        num_loss += 0.3 * masked_cos.sum() / (real_mask.sum() + 1e-8)


    return disc_loss, num_loss

######## Checkpoint function ##########
def save_checkpoint(model, optimizer, epoch, loss, CHECKPOINT_DIR, filepath, pr = True):
    """ Save the model checkpoint """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create directory if not exists
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    if pr: print(f"✅ Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizer, CHECKPOINT_DIR, checkpoint_path=None,train_decoder = False):
    """ Load the latest or specific checkpoint """
    if checkpoint_path is None:  # Find the latest checkpoint
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('vae_checkpoint')])
        if not checkpoints:
            print("⚠️ No checkpoints found. Training from scratch.")
            return 0, float('inf')
        else:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoints[-1]) 
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['loss']
            start_epoch = checkpoint['epoch'] + 1

            print(f"🔄 Loaded checkpoint from {checkpoint_path} (Epoch {start_epoch}) with Best Loss:{best_loss}")
            return start_epoch, best_loss
