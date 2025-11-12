import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse, os

# ─── Model Definition ───────────────────────────────────────────────────────────

class GRUClassifierPack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 bidirectional=False, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers>1 else 0.0
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim*factor, num_classes)

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T)
        lengths = mask.sum(dim=1).cpu().long()
        packed   = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n   = self.gru(packed)
        if self.gru.bidirectional:
            # reshape (num_layers, 2, B, H)
            h_n = h_n.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)
            h   = torch.cat([h_n[-1,0], h_n[-1,1]], dim=1)
        else:
            h   = h_n[-1]
        return self.fc(h)

# ─── Training / Validation with Early Stopping ─────────────────────────────────

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multi-class classification.
    For C classes, expects logits of shape (B, C) and targets in {0,..,C-1}.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float or list[float]): Weighting factor for the rare class(es).
            gamma (float): Focusing parameter; higher gamma -> more focus on hard examples.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super().__init__()
        if isinstance(alpha, (float, int)):
            # same weight for all classes
            self.alpha = torch.tensor([alpha] * 2)
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,) with values in {0,...,C-1}
        """
        device = logits.device
        C = logits.size(1)
        alpha = self.alpha.to(device)

        # Convert targets to one-hot
        y_onehot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # Compute log-probs
        logpt = nn.functional.log_softmax(logits, dim=1)  # (B, C)
        pt    = torch.exp(logpt)                          # (B, C)

        # Gather logpt and pt at targets
        logpt = (logpt * y_onehot).sum(dim=1)             # (B,)
        pt    = (pt    * y_onehot).sum(dim=1)             # (B,)

        # Apply focal modulation and alpha
        alpha_t = alpha[targets]                          # (B,)
        loss = -alpha_t * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = None,
    patience: int = 5,
):
    """
    Trains with early stopping on val loss.
    Stops if val loss doesn’t improve for `patience` epochs.
    Returns the best‐found model (with lowest val loss).
    """
    device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    # criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

    best_loss   = float('inf')
    best_state  = None
    wait        = 0

    for epoch in range(1, num_epochs+1):
        # ——— Training —————————————————————————————————————————————
        model.train()
        train_loss = 0.0; train_corr = 0; train_n = 0

        for x, m, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, m, y = x.to(device).float(), m.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            logits = model(x, m)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*y.size(0)
            preds = logits.argmax(1)
            train_corr += (preds==y).sum().item()
            train_n    += y.size(0)

        # ——— Validation —————————————————————————————————————————————
        model.eval()
        val_loss = 0.0; val_corr = 0; val_n = 0

        with torch.no_grad():
            for x, m, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                x, m, y = x.to(device).float(), m.to(device).float(), y.to(device).long()
                logits  = model(x, m)
                loss    = criterion(logits, y)

                val_loss += loss.item()*y.size(0)
                preds    = logits.argmax(1)
                val_corr += (preds==y).sum().item()
                val_n    += y.size(0)

        avg_train_loss = train_loss/train_n
        avg_val_loss   = val_loss/val_n
        train_acc      = train_corr/train_n
        val_acc        = val_corr/val_n

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # ——— Early Stopping Check —————————————————————————————————————
        if avg_val_loss < best_loss - 1e-4:  # require a small delta to count as “improvement”
            best_loss  = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            print(f"  ↳ No improvement for {wait}/{patience} epochs.")
            if wait >= patience:
                print(f"Early stopping triggered. Best val loss: {best_loss:.4f}")
                # load best weights back into model
                model.load_state_dict(best_state)
                break

    return model

# ─── AUC Evaluation ─────────────────────────────────────────────────────────────

def evaluate_auc(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_y, all_p = [], []

    with torch.no_grad():
        for x, m, y in data_loader:
            x, m, y = x.to(device).float(), m.to(device).float(), y.to(device).long()
            logits = model(x, m)
            probs  = nn.functional.softmax(logits, dim=1)[:,1]
            all_y.append(y.cpu().numpy())
            all_p.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    return roc_auc_score(y_true, y_prob)

# ─── Usage Example ──────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # -- data prep omitted for brevity --
#     model = GRUClassifierPack(input_dim=F, hidden_dim=128, num_layers=1, num_classes=2)
#     trained = train_model(
#         model,
#         train_loader,
#         val_loader,
#         num_epochs=50,
#         lr=1e-3,
#         patience=5,    # stop if no val‐loss improvement for 7 epochs
#     )

#     auc = evaluate_auc(trained, test_loader)
#     print(f"Test ROC AUC: {auc:.4f}")
