"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.pos_embed import VisionRotaryEmbeddingFast
from models.rmsnorm import RMSNorm

@torch.compile
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- Embedding ------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=1,bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size,bias=bias)

    def forward(self, x):         # x : (N, C, L)
        x = self.proj(x)          # (N, D, T), T = L/patch
        return x.permute(0, 2, 1) # (N, T, D)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @torch.compile
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

from transformers import CLIPTokenizer, CLIPTextModel
class CLIPTextEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        model_name="/home/jeffrey/Documents/EA_project/save/clip-hf-model",
        dropout_prob=0.1,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

        print(f"Loading CLIP model: {model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        clip_output_dim = self.text_encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(clip_output_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Learned unconditional embedding in DiT hidden space.
        self.null_embed = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null_embed, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen CLIP in eval mode at all times.
        self.text_encoder.eval()
        return self

    def _normalize_text_list(self, text_list, batch_size=None):
        """
        Normalize input into a Python list[str | None] of length batch_size.
        Rules:
        - None -> unconditional
        - "" or whitespace-only -> unconditional
        - non-string values are converted to str
        """
        if text_list is None:
            if batch_size is None:
                raise ValueError("batch_size must be provided when text_list is None")
            return [None] * batch_size

        if isinstance(text_list, str):
            text_list = [text_list]

        normalized = []
        for text in text_list:
            if text is None:
                normalized.append(None)
            elif isinstance(text, str):
                stripped = text.strip()
                normalized.append(None if stripped == "" else stripped)
            else:
                text = str(text).strip()
                normalized.append(None if text == "" else text)
        return normalized

    def _get_drop_mask(self, batch_size, device, force_drop_ids=None):
        """
        Returns a boolean mask of shape (batch_size,) where True means unconditional.
        """
        if force_drop_ids is not None:
            return force_drop_ids.to(device=device).bool()

        if self.training and self.dropout_prob > 0:
            return torch.rand(batch_size, device=device) < self.dropout_prob

        return torch.zeros(batch_size, device=device, dtype=torch.bool)

    def forward(self, text_list, train, force_drop_ids=None):
        """
        text_list:
            - None
            - list[str | None]
            - single string

        Behavior:
        - unconditional samples (None / "" / whitespace / dropped) always map to null_embed
        - only conditional samples are run through CLIP + projection
        """
        device = self.projection[0].weight.device

        # Respect the caller's train flag for CFG dropout logic, while CLIP itself stays eval.
        was_training = self.training
        if train != was_training:
            self.training = train

        batch_size = None
        if isinstance(text_list, (list, tuple)):
            batch_size = len(text_list)

        text_list = self._normalize_text_list(text_list, batch_size=batch_size)
        batch_size = len(text_list)

        # Base unconditional mask from None / empty strings.
        empty_mask = torch.tensor(
            [text is None for text in text_list],
            device=device,
            dtype=torch.bool,
        )

        # CFG drop mask.
        drop_mask = self._get_drop_mask(batch_size, device, force_drop_ids=force_drop_ids)

        # Final unconditional mask.
        uncond_mask = empty_mask | drop_mask

        # Initialize all outputs as null embeddings.
        embeddings = self.null_embed.unsqueeze(0).expand(batch_size, -1).clone()

        # Encode only conditional samples.
        cond_indices = (~uncond_mask).nonzero(as_tuple=False).flatten()
        if cond_indices.numel() > 0:
            cond_texts = [text_list[i] for i in cond_indices.tolist()]

            max_len = min(
                self.tokenizer.model_max_length,
                self.text_encoder.config.max_position_embeddings,
            )

            inputs = self.tokenizer(
                cond_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                pooled_output = outputs.pooler_output

            cond_embeddings = self.projection(pooled_output.float())
            embeddings[cond_indices] = cond_embeddings

        # Restore module training flag if we changed it.
        if train != was_training:
            self.training = was_training

        return embeddings


from transformers import AutoTokenizer, AutoModel
class QwenTextEmbedder(nn.Module):
    def __init__(self, hidden_size, model_name="Qwen/Qwen3-Embedding-8B", dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

        print(f"Loading Qwen model: {model_name}...")
        
        # 1. Load Pre-trained Qwen
        # Note: trust_remote_code=True is often required for Qwen models.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Important: Qwen/LLMs often don't have a pad token by default. 
        # We set it to EOS for batch processing.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in half-precision (bfloat16 or float16) to save memory. 
        # An 8B model requires ~16GB VRAM in fp16.
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"  # Automatically splits across GPUs/CPU if needed
        )

        # 2. Freeze the Base Model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 3. Projection Layer
        # Dynamically get the hidden size (likely 4096 for an 8B model)
        base_model_dim = self.base_model.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(base_model_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def token_drop(self, text_list, force_drop_ids=None):
        """
        Replaces text with empty strings "" for Classifier-Free Guidance (CFG).
        """
        new_text_list = []
        for i, text in enumerate(text_list):
            drop = False
            if force_drop_ids is not None:
                if force_drop_ids[i] == 1:
                    drop = True
            elif self.dropout_prob > 0 and torch.rand(1).item() < self.dropout_prob:
                drop = True
            
            new_text_list.append("" if drop else text)
        return new_text_list

    def forward(self, text_list, train, force_drop_ids=None):
        """
        text_list: List of strings
        """
        # Ensure projection is on the same device as the model output will be
        device = self.projection[0].weight.device
        
        # 1. Handle CFG (Dropout)
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            text_list = self.token_drop(text_list, force_drop_ids)

        # 2. Tokenize
        # Note: Qwen context window is large, but for tabular data/embeddings
        # we can stick to a reasonable length (e.g., 128 or 256) to save VRAM.
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(self.base_model.device) # Move inputs to where the base model is

        # 3. Encode with Qwen (No Gradients)
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            
            # Extract Last Token Pooling (Standard for Causal LLM Embeddings)
            # outputs.last_hidden_state shape: (Batch, Seq_Len, Hidden_Dim)
            hidden_states = outputs.last_hidden_state
            
            # We need to grab the embedding corresponding to the last real token (EOS),
            # ignoring padding tokens.
            # Create a range vector [0, 1, 2, ... Batch-1]
            batch_size = hidden_states.shape[0]
            sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
            
            # Select the vector at the last valid index for each item in batch
            pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        # 4. Project to DiT dimension
        # Ensure pooled_output is float32 if projection layer is float32
        embeddings = self.projection(pooled_output.float().to(device))
        
        return embeddings

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        seq_len = 120,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = PatchEmbed1D(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.y_embedder = CLIPTextEmbedder(hidden_size, dropout_prob=class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # --- Positional encoding --------------------------------------------------
        self.seq_len = seq_len//self.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos = get_1d_sincos_pos_embed(self.hidden_size, self.seq_len)  # returns tensor (T,D)
        self.pos_embed.data.copy_(pos.unsqueeze(0))    # shape (1,T,D)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.xavier_uniform_(w.view(w.shape[0], w.shape[1]*w.shape[2]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # --- Unpatchify -----------------------------------------------------------
    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        returns
        series: (N, C, T * patch_size)
        """
        N, T, _ = x.shape
        P  = self.patch_size
        C  = self.out_channels

        # reshape → (N, T, P, C)
        x = x.view(N, T, P, C)
        # bring C next to N, and merge T & P → (N, C, T*P)
        x = x.permute(0, 3, 1, 2).reshape(N, C, T * P)
        return x

    def forward(self, x, t=None, y=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, L) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """

        use_checkpoint = self.use_checkpoint
        # print(f"input size:{x.shape}")
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = L/ patch_size ** 2
        # print(f"x embeded size:{x.shape}")
        t = self.t_embedder(t)                   # (N, D)
        if y is None:
            y = [""] * x.shape[0] #empty strings for unconditional generation
        y = self.y_embedder(y, self.training)    # (N, D)    
        c = t + y                                # (N, D)

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)
        # print(f"after attn size:{x.shape}")
        x = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        # print(f"after linear size:{x.shape}")
        x = self.unpatchify(x)                   # (N, out_channels, T)
        # print(f"final size:{x.shape}")

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x # (N,C,L)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        y = y[:len(half)] + [""] * len(half)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_1d_sincos_pos_embed(embed_dim, length):
    position = torch.arange(length, dtype=torch.float32)
    dim = torch.arange(embed_dim//2, dtype=torch.float32) / (embed_dim//2)
    dim = 1. / (10000. ** dim)
    sinusoid = torch.einsum('l,d->ld', position, dim)          # (L,D/2)
    pos_emb = torch.cat([sinusoid.sin(), sinusoid.cos()], -1)  # (L,D)
    return pos_emb  

#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_mini_1(**kwargs):
    return LightningDiT(depth=4, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

def LightningDiT_lite_1(**kwargs):
    return LightningDiT(depth=8, hidden_size=256, patch_size=1, num_heads=8, **kwargs)

def LightningDiT_lite_10(**kwargs):
    return LightningDiT(depth=8, hidden_size=320, patch_size=1, num_heads=10, **kwargs)

def LightningDiT_lite_2(**kwargs):
    return LightningDiT(depth=8, hidden_size=256, patch_size=2, num_heads=8, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

LightningDiT_models = {
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-lite/1': LightningDiT_lite_1,
    'LightningDiT-lite/10': LightningDiT_lite_10,
    'LightningDiT-mini/1': LightningDiT_mini_1,
    'LightningDiT-lite/2': LightningDiT_lite_2
}