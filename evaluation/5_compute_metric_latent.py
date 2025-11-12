import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluation_metric.prediction_model import GRUClassifierPack,train_model,evaluate_auc
import argparse, os
from model import DP as dp
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# Optional UMAP import
_UMAP_AVAILABLE = True
try:
    import umap  # from umap-learn
except Exception:
    _UMAP_AVAILABLE = False


def optimal_kmeans_clustering(pca_data, max_k=10, random_state=42):
    """
    Given PCA-transformed data (pca_data) and an optional maximum number of clusters (max_k),
    determine the optimal number of clusters k using the silhouette score,
    and return the cluster labels corresponding to the best k.
    """
    if max_k < 2:
        raise ValueError("max_k must be at least 2.")

    best_k = None
    best_score = -1.0
    best_labels = None

    for k in range(2, max_k + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans_model.fit_predict(pca_data)
        score = silhouette_score(pca_data, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    final_labels = final_kmeans.fit_predict(pca_data)
    return best_k, final_labels, best_score


def process_pca(data: np.ndarray, n_components: int = 3):
    """
    Flattens, standardizes, and computes PCA for a NumPy array of shape [N, Time, Feature].
    """
    if data.ndim != 3:
        raise ValueError("Input data must have shape [N, Time, Feature]")

    N, T, F = data.shape
    data_flattened = data.reshape(data.shape[0], T * F)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flattened)

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    return data_pca


def run_tsne(data: np.ndarray, n_tsne: int = 2, perplexity: float = 30.0, random_state: int = 42):
    """
    Runs t-SNE on PCA-transformed data.
    """
    tsne = TSNE(n_components=n_tsne, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(data)
    return tsne_result


def run_umap(data: np.ndarray,
             n_components: int = 2,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             random_state: int = 42):
    """
    Runs UMAP on PCA-transformed data.
    Requires `umap-learn` (pip install umap-learn).
    """
    if not _UMAP_AVAILABLE:
        raise ImportError(
            "UMAP requested but `umap-learn` is not installed. "
            "Install via `pip install umap-learn` or choose --plot_method tsne."
        )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean",
    )
    return reducer.fit_transform(data)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/hiv_train.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/latent/")
    args_parser.add_argument("--outcome","-O",type=str,default="ce_id_cardiovascular")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)

    # New: choose the embedding method used for plotting
    args_parser.add_argument(
        "--plot_method",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Embedding method for the 2D plot (default: tsne)."
    )

    # Optional knobs for t-SNE / UMAP
    args_parser.add_argument("--tsne_perplexity", type=float, 
                            #  default=30.0,
                            default=10,
                             help="t-SNE perplexity (used if --plot_method tsne).")
    args_parser.add_argument("--umap_neighbors", type=int, default=15,
                             help="UMAP n_neighbors (used if --plot_method umap).")
    args_parser.add_argument("--umap_min_dist", type=float, default=0.1,
                             help="UMAP min_dist (used if --plot_method umap).")

    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    # Loading data
    print("Loading Data")
    train_data = pd.read_csv(args.real_path).fillna(0)  # for testing
    train_data,_,_,_ = dp.partition_multi_seq(train_data,1,"patient_id",max_len=120)
    train_data = train_data.cpu().numpy()
    test_data = pd.read_csv(args.test_path).fillna(0) # for training
    test_data['date'] = test_data.index
    test_data,_,_,_ = dp.partition_multi_seq(test_data,1,"patient_id",max_len=120)
    test_data = test_data.cpu().numpy()

    # Sample 5000 instances for Real, Synthetic, and Test Data
    sample_size = 5000
    rng = np.random.default_rng(seed=args.seed)

    idx = rng.choice(train_data.shape[0], sample_size, replace=False)
    real_sample = train_data[idx, ...]

    idx = rng.choice(test_data.shape[0], sample_size, replace=False)
    synth_sample = test_data[idx, ...]

    # Combine datasets for fair embedding visualization
    real_syn = np.concatenate([real_sample, synth_sample])

    # Compute PCA
    print("Running PCA")
    real_syn_pca = process_pca(real_syn, n_components=10)

    # Clustering on PCA space
    print("Running K-means")
    k, real_syn_cluster, _ = optimal_kmeans_clustering(real_syn_pca, random_state=args.seed)

    real_cluster = np.zeros((2*sample_size))
    real_cluster[sample_size:] = 1

    real_syn_nmi = normalized_mutual_info_score(real_cluster, real_syn_cluster)

    result = pd.DataFrame({"Model":[args.model_name],
                           "NMI": [real_syn_nmi]})
    save_path = os.path.join(args.save_path,f"{args.model_name}_nmi.csv")
    result.to_csv(save_path,
                  index=False,
                  header= False if os.path.exists(save_path) else True,
                  mode="a"
                  )
    print(f"NMI saved to {save_path}")

    # Embedding for Visualization
    if args.plot_method == "tsne":
        print("Running t-SNE")
        emb = run_tsne(real_syn_pca, n_tsne=2,
                       perplexity=args.tsne_perplexity,
                       random_state=args.seed)
        xlab, ylab = "t-SNE Axis 1", "t-SNE Axis 2"
        method_tag = f"tsne_p{int(args.tsne_perplexity)}"
    else:
        print("Running UMAP")
        emb = run_umap(real_syn_pca, n_components=2,
                       n_neighbors=args.umap_neighbors,
                       min_dist=args.umap_min_dist,
                       random_state=args.seed)
        xlab, ylab = "UMAP Axis 1", "UMAP Axis 2"
        method_tag = f"umap_n{args.umap_neighbors}_d{str(args.umap_min_dist).replace('.','p')}"

    # Create a Figure and single Axes
    fig, ax = plt.subplots(figsize=(7, 6))

    # Real vs. Synthetic scatter
    ax.scatter(
        emb[:sample_size, 0],
        emb[:sample_size, 1],
        color="#4398D9",
        alpha=0.5,
        label='Real',
        s=10
    )
    ax.scatter(
        emb[sample_size:, 0],
        emb[sample_size:, 1],
        color="#AC253E",
        alpha=0.5,
        label='Synthetic',
        s=10
    )

    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)

    ax.legend(loc="lower left",
              fontsize=18,
              handlelength=1,
              labelspacing=0.1,
              edgecolor='black')

    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()

    out_png = os.path.join(args.save_path, f"{args.model_name}_{method_tag}_{args.seed}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Plot saved to {out_png}")
