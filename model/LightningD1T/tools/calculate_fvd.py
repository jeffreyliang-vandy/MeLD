import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset

# --- Part 1: The Feature Extractor (The "Inception" replacement) ---
class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        # We use an LSTM to capture temporal dependencies (T)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # This head is used ONLY during pre-training (e.g., to predict next step)
        self.head = nn.Linear(hidden_dim, input_dim) 

    def get_embeddings(self, x):
        """
        Returns the features from the last time step.
        Input: (N, T, D)
        Output: (N, Hidden_Dim)
        """
        # LSTM output shape: (N, T, Hidden)
        out, (h_n, c_n) = self.lstm(x)
        
        # We take the output of the last time step to represent the whole sequence
        # Shape: (N, Hidden_Dim)
        last_step_features = out[:, -1, :] 
        return last_step_features

    def forward(self, x):
        # Used for training the extractor
        emb = self.get_embeddings(x)
        pred = self.head(emb)
        return pred

# --- Part 2: The Math (Standard Fréchet Implementation) ---
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of FD."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in Frechet Distance")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# --- Part 3: The Evaluator Class ---
class TabularFVD:
    def __init__(self, input_dim, device='cuda'):
        self.device = device
        self.extractor = TabularFeatureExtractor(input_dim=input_dim).to(device)
        self.extractor.eval() # Set to eval mode by default

    def train_extractor(self, real_data_loader, epochs=5):
        """
        Train the extractor on REAL data to predict the next time step.
        This forces the LSTM to learn the temporal structure of your data.
        """
        print("Training Feature Extractor on Real Data...")
        self.extractor.train()
        optimizer = torch.optim.Adam(self.extractor.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in real_data_loader:
                x = batch[0].to(self.device)
                
                # Task: Given t[0...T-1], predict t[T] (Shifted prediction)
                # Or simpler: Autoencoding. Here we use Next-Step reconstruction.
                
                # Split: Input is whole sequence, Target is shifted or same.
                # Simple Self-Supervision: Reconstruct the input (Autoencoder style)
                # or Forecast. Let's do simple Reconstruction for stability.
                
                optimizer.zero_grad()
                pred = self.extractor(x) # Forward pass uses last step to predict
                
                # We try to predict the mean of the sequence or the last step
                # For this simple example, let's try to predict the LAST step 
                # using the context of the previous steps.
                target = x[:, -1, :] 
                
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        
        self.extractor.eval()
        print("Extractor Trained.")

    def get_stats(self, data_loader):
        """Compute mean and covariance of embeddings."""
        all_feats = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                feats = self.extractor.get_embeddings(x)
                all_feats.append(feats.cpu().numpy())
        
        all_feats = np.concatenate(all_feats, axis=0)
        mu = np.mean(all_feats, axis=0)
        sigma = np.cov(all_feats, rowvar=False)
        return mu, sigma

    def compute_fvd(self, real_loader, gen_loader):
        """End-to-end FVD calculation."""
        m1, s1 = self.get_stats(real_loader)
        m2, s2 = self.get_stats(gen_loader)
        
        fvd = calculate_frechet_distance(m1, s1, m2, s2)
        return fvd

# --- Usage Example ---

# 1. Setup Mock Data (N=1000, T=10, D=32)
N, T, D = 1000, 10, 32
real_data = torch.randn(N, T, D)
# Generated data is slightly noisy/shifted
gen_data = torch.randn(N, T, D) + 0.2 

# 2. Create Loaders
batch_size = 64
real_loader = DataLoader(TensorDataset(real_data), batch_size=batch_size)
gen_loader = DataLoader(TensorDataset(gen_data), batch_size=batch_size)

# 3. Initialize Evaluator
fvd_evaluator = TabularFVD(input_dim=D, device='cpu') # Use 'cuda' if available

# 4. IMPORTANT: Pre-train the extractor on REAL data
# (In a real scenario, train for more epochs with validation)
fvd_evaluator.train_extractor(real_loader, epochs=2)

# 5. Calculate Score
score = fvd_evaluator.compute_fvd(real_loader, gen_loader)
print(f"Tabular FVD Score: {score:.4f}")