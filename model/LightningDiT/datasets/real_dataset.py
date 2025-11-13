"""
latentdataset.py
----------------
In‑memory loader for 1‑D time‑series stored in a single .pt file.

Example
-------
ds = LatentDataset("/path/to/dir")
x, y = ds[0]            # x: (C, L) float32  y: row id (long)
"""

import os
from glob import glob
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 latent_norm: bool = True,
                 latent_multiplier: float = 1.0,
                 channel_first: bool = True,
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Directory that contains exactly one .pt file with shape (N,L,C).
        channel_first : bool
            If True, returned tensors are (C,L); otherwise (L,C).
        dtype : torch.dtype
            Data type to cast the loaded tensor to (saves memory if float16).
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Locate the unique .pt file                                       #
        # ------------------------------------------------------------------ #
        # pt_files = glob(os.path.join(data_dir, "*.pt"))
        self.file_path = data_dir

        # ------------------------------------------------------------------ #
        # 2. Load the entire tensor into RAM                                  #
        # ------------------------------------------------------------------ #
        data = torch.load(self.file_path, map_location="cpu").to(dtype)  # (N,L,C)
        assert data.dim() == 3, "Expected tensor of shape (N, seq_len, C)."
        self.N, self.L, self.C = data.shape

        # ------------------------------------------------------------------ #
        # 3. Channel‑wise min–max normalisation to (−1,1)                     #
        # ------------------------------------------------------------------ #
        # min, max: shape (1,1,C)  keeps broadcasting simple
        _min = data.amin(dim=(0, 1), keepdim=True)
        _max = data.amax(dim=(0, 1), keepdim=True)
        eps = 1e-6                                                       # avoid /0
        if latent_norm:
            data = 2 * (data - _min) / (_max - _min + eps) - 1               # (−1,1)

        # ------------------------------------------------------------------ #
        # 4. Re‑order to channel‑first if requested                           #
        # ------------------------------------------------------------------ #
        if channel_first:
            data = data.permute(0, 2, 1).contiguous()  # (N,C,L)

        self.channel_first = channel_first
        self.data = data  # keep in memory
        self.dtype = dtype

    # ---------------------------------------------------------------------- #
    #               PyTorch Dataset protocol                                  #
    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        x : torch.Tensor  (C,L) if channel_first else (L,C)
        y : torch.LongTensor  scalar row id
        """
        x = self.data[idx]                                # view into RAM
        y = torch.tensor(idx, dtype=torch.long)
        return x, y
