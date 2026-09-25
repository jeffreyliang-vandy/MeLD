"""Frozen CLIP text encoder, kept outside the DiT.

MeLD: CLIP used to live inside LightningDiT as `y_embedder.text_encoder`, where
`initialize_weights()`'s `self.apply(_basic_init)` re-initialised every nn.Linear in it, so
the "pretrained" encoder was a frozen random transformer. It also rode along in the EMA,
the optimizer and every checkpoint. train.py and inference.py now own it instead; the DiT
keeps only the trainable projection (`TextConditionProjector`) and receives pooled
embeddings.
"""
import os

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        # model/dit_adapter.py sets this from dit.runtime.clip_model_path.
        model_name = model_name or os.environ.get("MELD_CLIP_PATH")
        if not model_name:
            raise ValueError("conditioning needs a CLIP checkpoint: set "
                             "dit.runtime.clip_model_path or $MELD_CLIP_PATH")
        self.model_name = model_name
        print(f"Loading CLIP model: {model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.dim = self.text_encoder.config.hidden_size
        self.max_len = min(self.tokenizer.model_max_length,
                           self.text_encoder.config.max_position_embeddings)

    def train(self, mode: bool = True):
        # Frozen: stay in eval mode whatever the caller asks.
        return super().train(False)

    def check_dim(self, cond_dim):
        if int(cond_dim) != self.dim:
            raise ValueError(f"dit.model.cond_dim is {cond_dim} but the CLIP checkpoint at "
                             f"{self.model_name} has hidden size {self.dim}")

    @staticmethod
    def _normalize(prompts):
        """None, "" and whitespace-only prompts are unconditional (None)."""
        out = []
        for p in prompts:
            p = None if p is None else str(p).strip()
            out.append(p or None)
        return out

    @torch.no_grad()
    def encode(self, prompts, device):
        """
        prompts: list[str | None]
        Returns (emb (N, dim) float32, uncond (N,) bool). Only conditional prompts go
        through CLIP; unconditional rows are zeros and flagged in `uncond`.
        """
        prompts = self._normalize(prompts)
        uncond = torch.tensor([p is None for p in prompts], dtype=torch.bool, device=device)
        emb = torch.zeros(len(prompts), self.dim, dtype=torch.float32, device=device)

        cond_idx = (~uncond).nonzero(as_tuple=False).flatten()
        if cond_idx.numel() > 0:
            inputs = self.tokenizer(
                [prompts[i] for i in cond_idx.tolist()],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(device)
            emb[cond_idx] = self.text_encoder(**inputs).pooler_output.float()
        return emb, uncond
