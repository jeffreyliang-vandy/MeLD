import h5py
import torch
import numpy as np

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            self.length = f['processed_data'].shape[0]

    def __len__(self):
        return self.length

    # Standard fallback for single items (like validation without batching)
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r', swmr=True)
            
        return (
            torch.as_tensor(self.file['processed_data'][idx], dtype=torch.float32),
            torch.as_tensor(self.file['time_info'][idx], dtype=torch.float32),
            torch.as_tensor(self.file['missing'][idx], dtype=torch.float32),
            torch.as_tensor(self.file['masking'][idx], dtype=torch.float32)
        )

    # Batched fetching! The DataLoader will call this if it exists.
    def __getitems__(self, indices):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r', swmr=True)
        
        # HDF5 bulk reading is orders of magnitude faster if indices are sorted, and h5py
        # requires them strictly increasing. Accelerate pads a short final batch by
        # wrapping around, which can repeat an index, so de-duplicate as well as sort.
        sorted_indices, unsort_indices = np.unique(np.asarray(indices), return_inverse=True)
        sorted_indices = sorted_indices.tolist()

        # Do ONE disk read per tensor instead of 128
        p_data = self.file['processed_data'][sorted_indices]
        t_info = self.file['time_info'][sorted_indices]
        missing_data = self.file['missing'][sorted_indices]
        masking_data = self.file['masking'][sorted_indices]

        # Sorting ruined the DataLoader's random shuffle; `unsort_indices` maps every
        # requested index back to its row, restoring the original order.
        
        # Re-apply the random order and convert to tensors
        return [
            (
                torch.as_tensor(p_data[i], dtype=torch.float32),
                torch.as_tensor(t_info[i], dtype=torch.float32),
                torch.as_tensor(missing_data[i], dtype=torch.float32),
                torch.as_tensor(masking_data[i], dtype=torch.float32)
            )
            for i in unsort_indices
        ]
