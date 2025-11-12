import numpy as np
import pandas as pd
import torch
from model import DP as dp
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import argparse
import os
import pickle

def compute_statistics(data: np.ndarray):
    """
    Compute per-record and per-visit code, bigram, and sequential bigram probabilities from binary indicator tensor,
    with progress bars and total step counts for each major loop.

    Args:
        data: np.ndarray of shape (N, T, F), where
              N = number of records (patients),
              T = number of visits (time steps),
              F = number of possible codes (binary indicator).
              data[n, t, f] = 1 if code f occurred in record n at visit t, else 0.

    Returns:
        stats: Dict containing:
            - 'per_visit_code_prob': np.ndarray shape (F,)
            - 'per_record_code_prob': np.ndarray shape (F,)
            - 'per_visit_bigram_prob': dict mapping (f1,f2) -> prob
            - 'per_record_bigram_prob': dict mapping (f1,f2) -> prob
            - 'per_visit_seq_bigram_prob': dict mapping (f_prev,f_curr) -> prob
            - 'per_record_seq_bigram_prob': dict mapping (f_prev,f_curr) -> prob
    """
    N, T, F = data.shape

    # Mask empty visits and count total visits/transitions
    visit_mask = data.sum(axis=2) > 0        # shape (N,T)
    total_visits = int(visit_mask.sum())
    total_transitions = N * (T - 1)

    # 1. Per-visit code probabilities
    visit_code_counts = data.sum(axis=(0,1))
    per_visit_code_prob = visit_code_counts / total_visits

    # 2. Per-record code probabilities
    record_has_code = (data.sum(axis=1) > 0).astype(int)  # shape (N,F)
    record_code_counts = record_has_code.sum(axis=0)
    per_record_code_prob = record_code_counts / N

    # Prepare combos/products with totals
    num_bigram = F * (F - 1) // 2
    num_seq = F * F

    # 3. Visit-level bigram probabilities
    visit_bigram_counts = {}
    for f1, f2 in tqdm(itertools.combinations(range(F), 2), total=num_bigram, desc="Visit bigrams"):
        co_occ = ((data[:, :, f1] > 0) & (data[:, :, f2] > 0)).sum()
        if co_occ:
            visit_bigram_counts[(f1, f2)] = co_occ / total_visits

    # 4. Record-level bigram probabilities
    record_bigram_counts = {}
    for f1, f2 in tqdm(itertools.combinations(range(F), 2), total=num_bigram, desc="Record bigrams"):
        co_occ_in_record = ((record_has_code[:, f1] == 1) & (record_has_code[:, f2] == 1)).sum()
        if co_occ_in_record:
            record_bigram_counts[(f1, f2)] = co_occ_in_record / N

    # 5. Visit-level sequential bigram probabilities
    visit_seq_counts = {}
    for f_prev, f_curr in tqdm(itertools.product(range(F), range(F)), total=num_seq, desc="Visit seq bigrams"):
        seq_occ = 0
        for t in range(1, T):
            seq_occ += ((data[:, t-1, f_prev] > 0) & (data[:, t, f_curr] > 0)).sum()
        if seq_occ:
            visit_seq_counts[(f_prev, f_curr)] = seq_occ / total_transitions

    # 6. Record-level sequential bigram probabilities
    record_seq_counts = {}
    for f_prev, f_curr in tqdm(itertools.product(range(F), range(F)), total=num_seq, desc="Record seq bigrams"):
        has_transition = np.zeros(N, dtype=bool)
        for t in range(1, T):
            has_transition |= ((data[:, t-1, f_prev] > 0) & (data[:, t, f_curr] > 0))
        count = has_transition.sum()
        if count:
            record_seq_counts[(f_prev, f_curr)] = count / N

    return {
        'per_visit_code_prob': per_visit_code_prob,
        'per_record_code_prob': per_record_code_prob,
        'per_visit_bigram_prob': visit_bigram_counts,
        'per_record_bigram_prob': record_bigram_counts,
        'per_visit_seq_bigram_prob': visit_seq_counts,
        'per_record_seq_bigram_prob': record_seq_counts
    }

# Increase default font size for all plot elements
plt.rcParams.update({'font.size': 20})

def generate_plots(stats1, stats2, label1, label2,
                   save_path = "figure/",
                   metrics=None):
    """
    Compare two stats dicts (from compute_statistics_from_tensor)
    by computing R² and scatter‐plotting each metric.

    Args:
        stats1, stats2: output of compute_statistics_from_tensor()
        label1, label2: strings to label the axes/filenames
        metrics: list of tuples (key, title, is_array)
            - key: the dict key in stats1/stats2
            - title: human‐readable plot title
            - is_array: True if stats[key] is a numpy array, False if it's a dict
    """
    if metrics is None:
        metrics = [
            ("per_record_code_prob",       "Per Record Code Probabilities",       True , 1),
            ("per_visit_code_prob",        "Per Visit Code Probabilities",        True , 0.5),
            ("per_record_bigram_prob",     "Per Record Bigram Probabilities",     False, 1),
            ("per_visit_bigram_prob",      "Per Visit Bigram Probabilities",      False, 0.5),
            ("per_record_seq_bigram_prob", "Per Record Sequential Visit Bigram Probabilities", False, 1),
            ("per_visit_seq_bigram_prob",  "Per Visit Sequential Visit Bigram Probabilities",  False, 0.5),
        ]

    metrics_out = {}

    for key, title, is_array, vmax in tqdm(metrics, desc="Plotting metrics"):
        xdata = stats1[key]
        ydata = stats2[key]

        if is_array:
            vals1 = np.asarray(xdata, dtype=float)
            vals2 = np.asarray(ydata, dtype=float)
        else:
            codes = set(xdata.keys()) | set(ydata.keys())
            labels = sorted(codes)
            vals1 = np.asarray([xdata.get(c, 0.0) for c in labels], dtype=float)
            vals2 = np.asarray([ydata.get(c, 0.0) for c in labels], dtype=float)

        # R^2
        r2 = r2_score(vals1, vals2)

        # Absolute-difference metrics
        diffs = np.abs(vals1 - vals2)
        mad = float(np.mean(diffs))         # mean |p1 - p2|
        max_abs = float(np.max(diffs))      # optional: worst-case |p1 - p2|

        metrics_out[key] = {'r2': float(r2), 'mad': mad, 'max_abs': max_abs}
        print(f"{title}: R² = {r2:.4f} | mean |Δ| = {mad:.4f} | max |Δ| = {max_abs:.4f}")

        # make scatter with increased point size
        plt.clf()
        plt.scatter(vals1, vals2, marker='o', alpha=0.6, s=60)  # increased scatter point size
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
        # vmax = min(1.0, 1.1 * max(max(vals1, default=0), max(vals2, default=0)))
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(0, vmax)
        plt.ylim(0, vmax)

        # add R² text in top-left corner (axes coordinates)
        ax = plt.gca()
        ax.text(0.05, 0.95,
                f"R² = {r2:.3f}\nMAE = {mad:.3f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black',
                          facecolor='white', alpha=0.7))

        # set tick label size explicitly in case defaults didn't apply
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # axis labels only, no title
        plt.xlabel(label1, fontsize=20)
        # plt.ylabel(label2, fontsize=20)
        plt.ylabel("Synthetic",fontsize=20)

        # safe filename
        fname = title.replace(" ", "_")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f"{save_path}/fig/",exist_ok=True)
        plt.savefig(os.path.join(save_path,"fig",f"{label2}_{fname}.png"),
                    bbox_inches='tight')
        # plt.savefig(os.path.join(save_path, f"{label2}_{fname}.pdf"), dpi=300, bbox_inches='tight', transparent=False)
    return metrics_out

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="../Dataset/hiv_train.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/bigram/")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args_parser.add_argument("--serial",type=str,default="")
    args = args_parser.parse_args()

    print(args.real_path)
    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    train_stats_exist = os.path.exists(os.path.join(args.save_path,"train_stats.pkl"))
    if not train_stats_exist:
        real_df = pd.read_csv(args.real_path).fillna(0)
        processed_data,_,_,_ = dp.partition_multi_seq(real_df,1,'patient_id',120)

    test_data = pd.read_csv(args.test_path).fillna(0)
    test_data['date'] = test_data.index
    _test_data,_,_,_ = dp.partition_multi_seq(test_data,threshold=1,column_to_partition='patient_id',max_len=120)
    _test_data = _test_data.cpu().numpy()


    if train_stats_exist:
        train_stats = pickle.load(open(os.path.join(args.save_path,"train_stats.pkl"),"rb"))
    else:
        train_stats = compute_statistics(processed_data[...,:-6])
        pickle.dump(train_stats,open(os.path.join(args.save_path,"train_stats.pkl"),"wb"))

    test_stats_path = os.path.join(args.save_path,f"{args.model_name}_stats.pkl")
    if os.path.exists(test_stats_path):
        print(f"Found stast in {test_stats_path}")
        test_stats = pickle.load(open(test_stats_path,"rb"))
    else: test_stats = compute_statistics(_test_data[...,:-6])

    metrics_out = generate_plots(train_stats, test_stats, "Real",
                                label2=args.model_name,
                                save_path=args.save_path)
    # Persist both metrics
    test_stats['r2'] = {k: v['r2'] for k, v in metrics_out.items()}
    test_stats['abs_diff'] = {k: v['mad'] for k, v in metrics_out.items()}       # mean |p1 - p2|
    pickle.dump(test_stats,open(test_stats_path,"wb"))
