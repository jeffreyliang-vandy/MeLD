import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# Activations selectable from configuration. Keeping one registry means every MLP in
# the model is configured the same way, by name.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "none": nn.Identity,
}


# Bumped when the checkpoint payload changes shape.
CHECKPOINT_SCHEMA = 1

# Arguments that determine parameter shapes or state_dict keys. They come from the
# checkpoint and may never be overridden at load time.
STRUCTURAL_KEYS = frozenset({
    "channels", "seq_len", "n_bins", "n_cats", "n_nums", "cards", "feature_size",
    "hidden_size", "num_layers", "bidirectional", "emb_dim", "time_dim", "lat_dim",
    "n_masks", "num_sine_terms", "num_binary_categories", "transformer_heads",
    "transformer_layers", "transformer_ff_dim", "time_encode_hidden", "decoder_hidden",
    "n_eos_classes", "predict_times", "predict_missing", "embed_missing",
})


def get_activation(name):
    """Instantiate an activation by name, with an error that lists the options."""
    if isinstance(name, nn.Module):
        return name
    key = str(name).lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


def set_matmul_precision(precision="high"):
    """Opt in to TF32 matmuls.

    This used to run at import time, which made importing the module mutate global
    torch state for every caller, including scripts that never build a model.
    """
    if precision is not None:
        torch.set_float32_matmul_precision(precision)


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
    # Device follows the input tensor. A module-level global pinned everything to
    # cuda:0, which breaks under accelerate/DDP where the model sits on another rank.
    angles = (2 ** torch.arange(num_terms, device=v.device).float()
              * math.pi * v.unsqueeze(-1))

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
class Embedding_data(nn.Module):
    def __init__(self, feature_size, emb_dim, n_bins, n_cats, n_nums, cards,
                 num_sine_terms=16, n_masks=2, num_binary_categories=2,
                 nums_mlp_activation="silu", output_mlp_activation="relu",
                 embed_missing=True):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        self.num_sine_terms = num_sine_terms
        self.n_masks = n_masks
        self.emb_dim = emb_dim
        
        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [num_binary_categories]*self.n_bins + self.cards
        
        # compute_sine_cosine emits `2 * num_sine_terms` features per numeric column.
        sine_feat = 2 * num_sine_terms * n_nums
        
        self.embed_missing = bool(embed_missing) and self.n_nums > 0

        nums_act = get_activation(nums_mlp_activation)
        out_act = get_activation(output_mlp_activation)
        
        if self.n_disc > 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList([
                nn.Embedding(num_categories, emb_dim) for num_categories in self.num_categorical_list
            ])

        if self.n_nums > 0:
            self.mlp_nums = nn.Sequential(nn.Linear(sine_feat, sine_feat),
                                          nums_act,
                                          nn.Linear(sine_feat, emb_dim))
        
        if self.n_masks > 0:
            self.paddings_list = nn.ModuleList([
                    nn.Embedding(num_binary_categories, emb_dim) for _ in range(self.n_masks)
                ])
        
        if self.embed_missing:
            self.missings_list = nn.ModuleList([
                    nn.Embedding(num_binary_categories, emb_dim) for _ in range(self.n_nums)
                ])

        # The concatenation width depends on which blocks are actually present:
        # [discrete sum] + [numeric block if n_nums>0] + [special (missing/padding) block]
        self.n_concat = 1 + (1 if self.n_nums > 0 else 0) + 1

        # Final embedding processing
        self.mlp_output = nn.Sequential(
            nn.Linear(self.n_concat*emb_dim, emb_dim),
            out_act,
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x, missing, masking=None):
        
        x_disc = x[:,:,0:self.n_disc].long()
        x_nums = x[:,:,self.n_disc:self.n_disc+self.n_nums]

        ####### Addition Approach
        emb_dim = self.emb_dim
        x_emb_sum = torch.zeros(x.shape[0], x.shape[1], emb_dim, device=x.device)

        # Process individual embeddings and sum them
        if self.n_disc > 0:
            for i, embedding in enumerate(self.embeddings_list):
                x_emb_sum += embedding(x_disc[:, :, i])  # Element-wise addition instead of concatenation

        # Process numerical variables with sine/cosine encoding
        if self.n_nums > 0:
            x_nums = compute_sine_cosine(x_nums, num_terms=self.num_sine_terms)
            x_nums_emb = self.mlp_nums(x_nums)
            # x_emb_sum += x_nums_emb  # Add numerical embedding instead of concatenation
            x_emb_sum = torch.cat([x_emb_sum,x_nums_emb],dim=2)
        #   print(f"x_emb_sum.shape:{x_emb_sum.shape}")

        x_emb_special = torch.zeros(x.shape[0], x.shape[1], emb_dim, device=x.device)
        if self.embed_missing:
            x_missings = missing.long()
            for i, embedding in enumerate(self.missings_list):  # Embed paddings as categorical and add to categorical
                x_emb_special += embedding(x_missings[ :, :, i])

        if masking is not None and self.n_masks > 0:  # added masking options
            x_paddings = masking[:, :, -self.n_masks:].long()
            for i, embedding in enumerate(self.paddings_list):  # Embed paddings as categorical and add to categorical
                x_emb_special += embedding(x_paddings[ :, :, i])
        
        x_emb_sum = torch.cat([x_emb_sum,x_emb_special],dim=2)
        
        final_emb = self.mlp_output(x_emb_sum)
        return final_emb

################################################################################################################
def get_torch_trans(heads = 8, layers = 1, channels = 64, dim_feedforward = 64,
                    activation = "gelu", dropout = 0.1):
   encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads,
                                              dim_feedforward = dim_feedforward,
                                              dropout = dropout, activation = activation)
   return nn.TransformerEncoder(encoder_layer, num_layers = layers)

class Transformer_Block(nn.Module):
   def __init__(self, channels, heads = 8, layers = 1, dim_feedforward = 64,
                activation = "gelu", dropout = 0.1):
       super().__init__()
       self.channels = channels
        
       self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
       self.feature_layer = get_torch_trans(heads = heads, layers = layers,
                                            channels = self.channels,
                                            dim_feedforward = dim_feedforward,
                                            activation = activation, dropout = dropout)
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
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size,
                 hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim,
                 n_masks=2, num_sine_terms=16, num_binary_categories=2,
                 gru_dropout=0.2, logvar_min=-6.0, logvar_max=2.0,
                 transformer_heads=8, transformer_layers=1, transformer_ff_dim=64,
                 transformer_activation="gelu", transformer_dropout=0.1,
                 time_encode_hidden=None, embed_missing=True,
                 nums_mlp_activation="silu", output_mlp_activation="relu",
                 time_encode_activation="relu"):
        super().__init__()
        # These two were previously declared on Embedding_data but never passed, so
        # they could not actually be set.
        self.Emb = Embedding_data(feature_size, emb_dim, n_bins, n_cats, n_nums, cards,
                                  num_sine_terms=num_sine_terms, n_masks=n_masks,
                                  num_binary_categories=num_binary_categories,
                                  nums_mlp_activation=nums_mlp_activation,
                                  output_mlp_activation=output_mlp_activation,
                                  embed_missing=embed_missing)
        time_hidden = time_encode_hidden if time_encode_hidden is not None else emb_dim
        self.time_encode = nn.Sequential(nn.Linear(time_dim, time_hidden),
                                         get_activation(time_encode_activation),
                                         nn.Linear(time_hidden, emb_dim))
        if channels > 0:
            self.encoder_Transformer = Transformer_Block(channels,
                                                         heads=transformer_heads,
                                                         layers=transformer_layers,
                                                         dim_feedforward=transformer_ff_dim,
                                                         activation=transformer_activation,
                                                         dropout=transformer_dropout)
        else: self.encoder_Transformer = None
        
        self.encoder_mu = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, dropout=gru_dropout, bidirectional = bidirectional)
        self.encoder_logvar = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, dropout=gru_dropout, bidirectional = bidirectional)
        
        gru_out = 2*hidden_size if bidirectional else hidden_size
        self.fc_mu = nn.Linear(gru_out, lat_dim)
        self.fc_logvar = NonLinear(gru_out, lat_dim,
                                   activation=nn.Hardtanh(min_val=logvar_min, max_val=logvar_max))

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
    

def eos_truncation_mask(eos_logits, threshold=0.5):
    """Mask keeping every step up to and including the first predicted EOS.

    The double cumsum is what makes "up to and including" work: the first EOS step
    reaches 1, and everything after it exceeds 1.

    Shared with step 4, which previously carried a duplicate of this logic with its
    own hard-coded 0.5 -- so changing the decoder's eos_threshold silently failed to
    affect decoding.
    """
    probs = F.softmax(eos_logits, dim=-1)[..., 1]
    predictions = (probs > threshold).int()
    cumulative = predictions.cumsum(dim=1).cumsum(dim=1)
    return (cumulative <= 1).float().unsqueeze(-1)


class Decoder(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size,
                 hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim,
                 n_eos_classes=2, nums_activation="sigmoid", predict_times=False,
                 predict_missing=True, eos_threshold=0.5, decoder_hidden=None,
                 decoder_activation="relu"):
        super().__init__()
        dec_hidden = decoder_hidden if decoder_hidden is not None else hidden_size
        self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, dec_hidden),
                                         get_activation(decoder_activation),
                                         nn.Linear(dec_hidden, hidden_size))
        
        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.disc = self.n_bins + self.n_cats
        self.eos_threshold = eos_threshold
        self.n_eos_classes = n_eos_classes
        # Named `sigmoid` for historical reasons; it is whatever nums_activation selects.
        self.sigmoid = get_activation(nums_activation)
        
        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats else None 
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

        self.masks_linear = nn.Linear(hidden_size, n_eos_classes)  # Try softmax
        self.missings_linear = nn.Linear(hidden_size, n_nums) if (predict_missing and n_nums) else None
        self.times_linear = nn.Linear(hidden_size, time_dim) if predict_times else None
    
    def forward(self, latent_feature):
        decoded_outputs = dict()
        x = self.decoder_mlp(latent_feature)

        # Compute EOS mask
        if self.masks_linear is not None:
            ## Use softmax
            decoded_outputs['eos'] = self.masks_linear(x)
            eos_mask = eos_truncation_mask(decoded_outputs['eos'], self.eos_threshold)

        if self.missings_linear is not None:
            decoded_outputs['missings'] = self.missings_linear(x) * eos_mask

        if self.bins_linear is not None:
            decoded_outputs['bins'] = self.bins_linear(x) * eos_mask

        if self.cats_linears is not None:
            decoded_outputs['cats'] = [linear(x) * eos_mask for linear in self.cats_linears]

        if self.nums_linear is not None:
            decoded_outputs['nums'] = self.sigmoid(self.nums_linear(x)) * eos_mask

        if self.times_linear is not None:
            decoded_outputs['times'] = self.times_linear(x) * eos_mask

        return decoded_outputs


class DeapStack(nn.Module):
    """Sequence VAE.

    The first fourteen arguments describe the data and the core shapes; every
    remaining architecture / loss knob is keyword-only with a default that
    reproduces the historical hard-coded behaviour.

    The resolved arguments are kept on `self.config` and written into every
    checkpoint, so a checkpoint fully describes the model that produced it and can be
    rebuilt with `DeapStack.from_checkpoint` without a side-car parameter file.
    """

    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size,
                 hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim,
                 # --- embedding / encoder ---
                 n_masks=2, num_sine_terms=16, num_binary_categories=2,
                 gru_dropout=0.2, logvar_min=-6.0, logvar_max=2.0,
                 transformer_heads=8, transformer_layers=1, transformer_ff_dim=64,
                 transformer_activation="gelu", transformer_dropout=0.1,
                 time_encode_hidden=None, embed_missing=True,
                 nums_mlp_activation="silu", output_mlp_activation="relu",
                 time_encode_activation="relu",
                 # --- decoder ---
                 n_eos_classes=2, nums_activation="sigmoid", predict_times=False,
                 predict_missing=True, eos_threshold=0.5, decoder_hidden=None,
                 decoder_activation="relu",
                 # --- reconstruction loss ---
                 first_visit_scale=10.0, time_cos_weight=0.3, mask_missing_in_mse=True,
                 bins_weight=1.0, cats_weight=1.0, nums_weight=1.0,
                 eos_weight=1.0, missing_weight=1.0, times_weight=1.0,
                 disc_weight=1.0, num_weight=1.0, loss_eps=1e-8):
        super().__init__()
        if time_dim % 2:
            # auto_loss pairs the time features as (sin, cos) for its cosine term.
            raise ValueError(f"time_dim must be even (sin/cos pairs), got {time_dim}")

        # Snapshot of every resolved argument, persisted with the weights.
        self.config = dict(
            channels=channels, batch_size=batch_size, seq_len=seq_len, n_bins=n_bins,
            n_cats=n_cats, n_nums=n_nums, cards=list(cards), feature_size=feature_size,
            hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
            emb_dim=emb_dim, time_dim=time_dim, lat_dim=lat_dim,
            n_masks=n_masks, num_sine_terms=num_sine_terms,
            num_binary_categories=num_binary_categories, gru_dropout=gru_dropout,
            logvar_min=logvar_min, logvar_max=logvar_max,
            transformer_heads=transformer_heads, transformer_layers=transformer_layers,
            transformer_ff_dim=transformer_ff_dim,
            transformer_activation=transformer_activation,
            transformer_dropout=transformer_dropout,
            time_encode_hidden=time_encode_hidden, embed_missing=embed_missing,
            nums_mlp_activation=nums_mlp_activation,
            output_mlp_activation=output_mlp_activation,
            time_encode_activation=time_encode_activation,
            n_eos_classes=n_eos_classes, nums_activation=nums_activation,
            predict_times=predict_times, predict_missing=predict_missing,
            eos_threshold=eos_threshold, decoder_hidden=decoder_hidden,
            decoder_activation=decoder_activation,
            first_visit_scale=first_visit_scale, time_cos_weight=time_cos_weight,
            mask_missing_in_mse=mask_missing_in_mse, bins_weight=bins_weight,
            cats_weight=cats_weight, nums_weight=nums_weight, eos_weight=eos_weight,
            missing_weight=missing_weight, times_weight=times_weight,
            disc_weight=disc_weight, num_weight=num_weight, loss_eps=loss_eps,
        )

        shape_args = (channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, feature_size,
                      hidden_size, num_layers, bidirectional, emb_dim, time_dim, lat_dim)
        self.encoder = Encoder(*shape_args,
                               n_masks=n_masks, num_sine_terms=num_sine_terms,
                               num_binary_categories=num_binary_categories,
                               gru_dropout=gru_dropout,
                               logvar_min=logvar_min, logvar_max=logvar_max,
                               transformer_heads=transformer_heads,
                               transformer_layers=transformer_layers,
                               transformer_ff_dim=transformer_ff_dim,
                               transformer_activation=transformer_activation,
                               transformer_dropout=transformer_dropout,
                               time_encode_hidden=time_encode_hidden,
                               embed_missing=embed_missing,
                               nums_mlp_activation=nums_mlp_activation,
                               output_mlp_activation=output_mlp_activation,
                               time_encode_activation=time_encode_activation)
        self.decoder = Decoder(*shape_args,
                               n_eos_classes=n_eos_classes,
                               nums_activation=nums_activation,
                               predict_times=predict_times,
                               predict_missing=predict_missing,
                               eos_threshold=eos_threshold,
                               decoder_hidden=decoder_hidden,
                               decoder_activation=decoder_activation)
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        self.n_masks = n_masks
        self.disc_weight = disc_weight
        self.num_weight = num_weight

        # Everything auto_loss needs, kept on the module so get_loss stays argument-free.
        self.loss_kwargs = dict(
            n_eos_classes=n_eos_classes,
            first_visit_scale=first_visit_scale,
            time_cos_weight=time_cos_weight,
            mask_missing_in_mse=mask_missing_in_mse,
            bins_weight=bins_weight,
            cats_weight=cats_weight,
            nums_weight=nums_weight,
            eos_weight=eos_weight,
            missing_weight=missing_weight,
            times_weight=times_weight,
            eps=loss_eps,
        )

    # -- persistence ------------------------------------------------------- #
    def save(self, path, optimizer=None, epoch=None, loss=None):
        """Write weights and the architecture that produced them to one file."""
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = epoch
        if loss is not None:
            payload["loss"] = loss
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)
        return path

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", overrides=None, strict=True):
        """Rebuild a model straight from a checkpoint.

        `overrides` may adjust behavioural settings (thresholds, loss weights) without
        retraining. Structural keys are rejected, because changing those would not
        match the saved weights -- previously such a mismatch was hidden by loading
        with strict=False and left a partly random model.
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)
        config = blob.get("config")
        if config is None:
            raise ValueError(
                f"{path} has no embedded config (schema {blob.get('schema')!r}). "
                f"Convert it with tools/migrate_vae_ckpt.py.")
        config = dict(config)
        if overrides:
            bad = sorted(set(overrides) & STRUCTURAL_KEYS)
            if bad:
                raise ValueError(
                    f"cannot override structural keys {bad}: they determine the "
                    f"shape of the saved weights")
            unknown = sorted(set(overrides) - set(config))
            if unknown:
                raise ValueError(f"unknown override keys: {unknown}")
            config.update(overrides)

        model = cls(**config)
        model.load_state_dict(blob["model_state_dict"], strict=strict)
        return model, blob

    def get_loss(self, inputs, time_info, missing, masking):

        outputs, _, mu_z, logvar_z = self.forward(inputs,time_info,missing,masking)
        
        disc_loss, num_loss = auto_loss(inputs,time_info,missing,masking,\
                                        outputs, self.n_bins, self.n_nums, self.n_cats, self.cards,
                                        **self.loss_kwargs)
        disc_loss = self.disc_weight * disc_loss
        num_loss = self.num_weight * num_loss
        
        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        loss_RE = disc_loss + num_loss

        return loss_RE, loss_kld
    
    def forward(self, x, time_info, missing = None, masking=None):
        emb, mu_z, logvar_z = self.encoder(x, time_info,missing, masking)
        outputs = self.decoder(emb) 
        return outputs, emb, mu_z, logvar_z


def auto_loss(inputs, time_info, missing, masking, reconstruction, n_bins, n_nums, n_cats, cards,
              n_eos_classes=2, first_visit_scale=10.0, time_cos_weight=0.3,
              mask_missing_in_mse=True,
              bins_weight=1.0, cats_weight=1.0, nums_weight=1.0,
              eos_weight=1.0, missing_weight=1.0, times_weight=1.0, eps=1e-8):
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
    missing_nums = 1. - missing[...,-n_nums:]

    device = inputs.device
    eos = masking[:, :, -2] if masking is not None else torch.zeros(B, L, device=device)
    padding = masking[:, :, -1] if masking is not None else torch.zeros(B, L, device=device)
    real_mask = (padding < 1).float() if masking is not None else torch.ones(B, L, device=device)

    #reconstruction_losses = dict()
    disc_loss = 0; num_loss = 0
    
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
        disc_loss += bins_weight * masked_bce.sum() / (real_mask.sum() + eps)

    if 'eos' in reconstruction and masking is not None:
        ## Use softmax:
        eos_loss = F.cross_entropy(reconstruction['eos'].view(-1, n_eos_classes),
                                   eos.long().view(-1), reduction='none')
        eos_loss = (eos_loss * real_mask.view(-1)).sum() / (real_mask.sum() + eps)

        disc_loss += eos_weight * eos_loss

    if 'missings' in reconstruction:
        # BCE with logits => shape (B, L, n_nums) if reduction='none'
        bce = F.binary_cross_entropy_with_logits(
            reconstruction['missings'], missing, reduction='none'
        )  # shape (B, L, n_nums)
        # Sum over the n_bins dimension so we have one loss per (B,L)
        bce_per_timestep = bce.sum(dim=-1)  # shape (B, L)
        # Apply real_mask
        missing_loss = bce_per_timestep * real_mask  # shape (B, L)
        missing_loss = missing_loss.sum() / (real_mask.sum() + eps)
        disc_loss += missing_weight * missing_loss

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
            cat_loss_i = ce_masked.sum() / (real_mask.sum() + eps)
            cats_losses.append(cat_loss_i)

        disc_loss += cats_weight * torch.stack(cats_losses).mean()

    if 'nums' in reconstruction:
        # MSE => shape (B, L, n_nums) if reduction='none'
        mse = F.mse_loss(reconstruction['nums'], nums, reduction='none')
        if mask_missing_in_mse:
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
        num_loss += nums_weight * masked_mse.sum() / (real_mask.sum() + eps)

    if 'times' in reconstruction:
        # MAE => shape (B, L, n_times) if reduction='none'
        mae = F.l1_loss(reconstruction['times'], time_info, reduction='none') ## MSE is not working, KL exploded, L1 loss bad results
        mae_per_timestep = mae.sum(dim=-1)
        # Mask out padded timesteps
        masked_mae = mae_per_timestep * real_mask
        # Average over real positions
        num_loss += times_weight * masked_mae.sum() / (real_mask.sum() + eps)

        ## Cosine similarity loss for time reconstruction #using cosine only also bad results, but it is more stable than MSE
        pred = reconstruction['times'].view(B, L, -1, 2)
        target = time_info.view(B, L, -1, 2)
        cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=eps)  # (B, L, n_cycles)
        cos_loss = 1 - cos_sim
        cos_loss = cos_loss.mean(dim=-1)
        masked_cos = cos_loss * real_mask
        num_loss += time_cos_weight * masked_cos.sum() / (real_mask.sum() + eps)


    return disc_loss, num_loss

######## Checkpoint function ##########
def save_checkpoint(model, optimizer, epoch, loss, CHECKPOINT_DIR, filepath, pr=True):
    """Save weights, optimizer state and the architecture that produced them."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'schema': CHECKPOINT_SCHEMA,
        # Embedding the config is what makes a checkpoint self-describing: steps 2
        # and 4 rebuild from it directly instead of a side-car parameter file that
        # could drift out of sync with the weights.
        'config': getattr(model, 'config', None),
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    if pr:
        print(f"✅ Checkpoint saved at {filepath}")


def find_latest_checkpoint(checkpoint_dir, prefix='vae_checkpoint'):
    """Newest checkpoint, ordered by the epoch number embedded in the filename.

    Ordering is numeric, not lexicographic: sorting the names as strings puts
    `vae_checkpoint_epoch_900` after `vae_checkpoint_epoch_1000` and silently resumes
    from the wrong file.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    best = None
    for name in os.listdir(checkpoint_dir):
        if not name.startswith(prefix):
            continue
        m = re.search(r'(\d+)', name[len(prefix):])
        if not m:
            continue
        key = int(m.group(1))
        if best is None or key > best[0]:
            best = (key, os.path.join(checkpoint_dir, name))
    return best[1] if best else None


def load_checkpoint(model, optimizer, CHECKPOINT_DIR, checkpoint_path=None,
                    map_location='cpu', strict=False):
    """Resume from a specific checkpoint, or the newest one in CHECKPOINT_DIR.

    Returns (start_epoch, best_loss); (0, inf) when there is nothing to resume from.
    """
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path is None:
            print("⚠️ No checkpoints found. Training from scratch.")
            return 0, float('inf')
    elif not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # Previously the body below sat in the `else` of the search branch, so passing an
    # explicit path fell through and returned None -- a TypeError at the call site.
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=strict)
    if missing or unexpected:
        print(f"⚠️ state_dict mismatch loading {checkpoint_path}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected key(s). "
              f"Set vae.train.load_strict to make this an error.")
        if missing:
            print(f"   missing:    {list(missing)[:5]}")
        if unexpected:
            print(f"   unexpected: {list(unexpected)[:5]}")
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_loss = checkpoint.get('loss', float('inf'))
    start_epoch = checkpoint.get('epoch', -1) + 1
    print(f"🔄 Loaded checkpoint from {checkpoint_path} "
          f"(resuming at epoch {start_epoch}) with best loss: {best_loss}")
    return start_epoch, best_loss
