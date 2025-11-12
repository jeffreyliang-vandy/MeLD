import os
import numpy as np
import pandas as pd
from model import DP as dp
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

# List of models to compare
models = ["Real","MeLD","SynTEG","HALO","TimeDiff","TimeAutoDiff"
          ,"MeLD-DDPM","MeLD-LSTM","MeLD-DiT","MeLD-Transformer"
          ]  

# define a fixed palette of 10 distinct colors (from tab10)
palette = [ '#84111c','#6a8da0','#d3c68d', '#639b70','#c05d4f','#e5d0dd'
           ,"#79a2ac","#8db6b7",'#a2c8c2','#bddcd2']

def get_model_colors(models):
    """
    Assign consistent colors to models using a fixed palette.
    """
    model_colors = {}
    for idx, model in enumerate(models):
        model_colors[model] = palette[idx % len(palette)]
    return model_colors

model_colors = get_model_colors(models)
# =================================


def plot_distribution_comparison(
    real_vec: np.ndarray,
    other_vec: np.ndarray,
    other_label: str,
    ax: plt.Axes,
    xlabel: str,
    bins: int = 30,
    log_scale: bool = False,
    fs: int = 18,
    density: bool = False,
    legend_loc: str = "upper right",
    bin_edges: np.ndarray | None = None,  # NEW: allow passing precomputed edges
    color = '#AC253E',
):
    """
    Overlay the outline of `real_vec` on the filled histogram of `other_vec`,
    with both histograms using identical bin edges derived from `real_vec`.

    If `bin_edges` is None, edges are computed from `real_vec` only.
    """

    # --- Shared binning: derive edges from the reference only ---
    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(real_vec, bins=bins)

    # choose between density or raw counts
    hist_kwargs = dict(bins=bin_edges, alpha=0.7, density=density)

    # outline of the real (training) data
    ax.hist(real_vec, **hist_kwargs,
            histtype='step', linewidth=2,
            color='black', label='Real (outline)')

    # filled histogram for the other data
    ax.hist(other_vec, **hist_kwargs,
            edgecolor='black',
            color=color,
            label=other_label)

    ax.relim()

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('Density' if density else 'Count', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(loc=legend_loc, fontsize=fs, handlelength=1, labelspacing=0.1)

    # compute & annotate Wasserstein distance
    w = wasserstein_distance(real_vec, other_vec)
    x0, x1 = ax.get_xlim()
    x_mid = 0.5 * (x0 + x1)
    y0, y1 = ax.get_ylim()
    y_mid = np.sqrt(y0 * y1) if log_scale else 0.5 * (y0 + y1)

    ax.text(
        x_mid, y_mid, f"$WSD={w:.2f}$",
        ha='center', va='center', fontsize=fs,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )


def plot_single_distribution(
    real_vec: np.ndarray,
    other_vec: np.ndarray,
    other_label: str,
    xlabel: str,
    bins: int = 30,
    log_scale: bool = False,
    fs: int = 18,
    figsize: tuple = (6, 4),
    xlim: tuple = None,
    ylim: tuple = None,
    density = True,
    legend_loc = 'upper right',
    bin_edges: np.ndarray | None = None,
    color = '#AC253E'
) -> plt.Figure:
    """
    Creates a standalone figure comparing `real_vec` to `other_vec`.

    Parameters
    ----------
    real_vec
        1D array of training values.
    other_vec
        1D array of comparison values.
    other_label
        Dataset label for the other vector.
    xlabel
        X-axis label string.
    bins
        Number of bins for histogram.
    log_scale
        Whether to use log scale on the y-axis.
    fs
        Base font size for all text elements.
    figsize
        Figure size tuple (width, height).
    xlim
        Optional tuple (xmin, xmax) to set x-axis limits.
    ylim
        Optional tuple (ymin, ymax) to set y-axis limits.

    Returns
    -------
    fig : plt.Figure
        The created Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_distribution_comparison(
        real_vec, other_vec, other_label, ax,
        xlabel=xlabel, bins=bins, log_scale=log_scale, fs=fs,
        density=density, legend_loc=legend_loc, bin_edges=bin_edges,  # NEW
        color = color
    )
    if density: pass
    else:
    # Apply axis limits if specified
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/hiv_train.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/dist/")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args_parser.add_argument("--serial",type=str,default="")
    args_parser.add_argument("--plot_only",action="store_true",default=False)
    # args_parser.add_argument("--log_scale",action="store_True",default=False)
    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    real_df = pd.read_csv(args.real_path).fillna(-999)
    # missing_df = pd.read_csv('home/jeff/Documents/TimeAutoDiff/Dataset/hiv_missing_train.csv.gz')
    # real_df = real_df * (1-missing_df)

    _synth_data = pd.read_csv(args.test_path).fillna(-999)
    _synth_data['date'] = _synth_data.index

    color = model_colors.get(args.model_name, '#AC253E')


    ######################### Actual Code ############################
    #### Longitudinal Features

    # --- 0. Setup and Data Extraction ---
    # ── Compute capped visits & gaps ───────────────────────────────────────────────
    real_visits = real_df.patient_id.value_counts()
    # p90         = real_visits.quantile(0.9)
    p90=120
    real_visits_capped = np.minimum(real_visits, p90)


    syn_visits_capped = _synth_data.patient_id.value_counts().apply(np.clip,a_min=0,a_max=120)

    real_gap = real_df.loc[real_df.gap > 0, "gap"]
    syn_gap  = _synth_data.loc[_synth_data.gap > 0, "gap"]

    visit_dsets = [syn_visits_capped]
    gap_dsets   = [syn_gap]
    labels      = [args.model_name]

    # ── Shared axis limits ────────────────────────────────────────────────────────
    n_bins      = 30
    all_visits  = np.concatenate(visit_dsets)
    visit_xlim  = (0, all_visits.max() + 10)

    all_gaps    = np.concatenate([g.values if hasattr(g, "values") else g for g in gap_dsets])
    gap_xlim    = (0, all_gaps.max() * 1.05)

    # compute y-lims (min>0 for log)
    counts_vis, _ = np.histogram(all_visits, bins=n_bins)
    yvisit_ylim = (1, counts_vis.max() * 1.1)

    counts_gap, _ = np.histogram(all_gaps, bins=n_bins)
    ygap_ylim    = (1, counts_gap.max() * 1.1)

    # Font & figure settings
    fs      = 18
    dpi_val = 300
    out_dir = args.save_path

    # ── 1. Save Individual Subplots for Visits ────────────────────────────────────
    for lbl, arr in zip(labels, visit_dsets):
        fig = plot_single_distribution(
            real_vec   = real_visits_capped,
            other_vec  = arr,
            other_label= "Synthetic",
            xlabel     = "Number of visits",
            bins       = n_bins,
            log_scale  = False,
            fs         = fs,
            figsize    = (6,4),
            xlim       = visit_xlim,
            ylim       = yvisit_ylim,
            legend_loc="upper center",
            color = color
        )
        fname = f"fig/{lbl.lower()}_visit_length.png"
        fig.savefig(os.path.join(out_dir, fname),
                    dpi=dpi_val, bbox_inches="tight")
        plt.close(fig)

    # ── 2. Save Individual Subplots for Time Gaps ─────────────────────────────────
    for lbl, arr in zip(labels, gap_dsets):
        fig = plot_single_distribution(
            real_vec   = real_gap,    # always compare back to visits cap
            other_vec  = arr,
            other_label= "Synthetic",
            xlabel     = "Log time gap (days)",
            bins       = n_bins,
            log_scale  = True,
            # log_scale  = False,
            density    = True,
            fs         = fs,
            figsize    = (6,4),
            xlim       = gap_xlim,
            ylim       = ygap_ylim,
            color = color
        )
        fname = f"fig/{lbl.lower()}_time_gap.png"
        fig.savefig(os.path.join(out_dir, fname),
                    dpi=dpi_val, bbox_inches="tight")
        # save_path =  f"{lbl.lower()}_time_gap.pdf"
        # fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)
        
    # … (code to save each of the six PNGs with shared axes, as before) …

    # 2) Compute metrics for “Visit Length” and “Time Gap”
    results = []

    # A) Visit Length
    a, c = real_visits_capped, syn_visits_capped
    # Wasserstein
    wd_rs = wasserstein_distance(a, c)
    # KS
    ks_rs = ks_2samp(a, c).pvalue
    # JS
    bins = np.histogram_bin_edges(np.concatenate([a, c]), bins=30)
    p_a, _ = np.histogram(a, bins=bins, density=True)
    p_c, _ = np.histogram(c, bins=bins, density=True)
    p_a /= p_a.sum();  p_c /= p_c.sum()
    js_rs = jensenshannon(p_a, p_c, base=2)

    results.append({
        "Feature": "Visit Length",
        f"Wasserstein Real vs {args.model_name}":  round(wd_rs, 3),

        f"KS p-value Real vs {args.model_name}":   round(ks_rs, 3),

        f"JS Distance Real vs {args.model_name}":  round(js_rs, 3),
    })

    # B) Time Gap
    a, c = real_gap, syn_gap
    wd_rs = wasserstein_distance(a, c)
    ks_rs = ks_2samp(a, c).pvalue

    bins = np.histogram_bin_edges(np.concatenate([a, c]), bins=30)
    p_a, _ = np.histogram(a, bins=bins, density=True)
    p_c, _ = np.histogram(c, bins=bins, density=True)
    p_a /= p_a.sum();  p_c /= p_c.sum()
    js_rs = jensenshannon(p_a, p_c, base=2)

    results.append({
        "Feature": "Time Gap",
        f"Wasserstein Real vs {args.model_name}":  round(wd_rs, 3),

        f"KS p-value Real vs {args.model_name}":   round(ks_rs, 3),

        f"JS Distance Real vs {args.model_name}":  round(js_rs, 3),
    })

    metrics_df = pd.DataFrame(results)
    print(metrics_df.round(3))


    #### Continuous Features

    # --- 3.1 Setup ---
    features = ["weight", "height", "age", "cd4_v", "rna_v"]
    features_name = ['Weight',"Height","Age (-18)","Log CD4 counts","Log viral loads"]
    data_sets = [
        ("Real",    real_df),
        (args.model_name, _synth_data)
    ]

    # fonts and save settings
    title_fs, label_fs, tick_fs = 18, 18, 18
    dpi_val = 300
    out_dir = args.save_path
    os.makedirs(out_dir, exist_ok=True)

    # histogram settings
    n_bins = 30
    hist_args = dict(bins=n_bins, alpha=0.7, edgecolor="black")

    # --- 3.2 Precompute shared x- and y-limits ---
    # x_limits[feature] = (xmin, xmax) across all three datasets
    x_limits = {}
    # y_limits[label]   = (ymin, ymax) across all five features for that dataset
    y_limits = { label: (np.inf, -np.inf) for label, _ in data_sets }

    for feature in tqdm(features):
        # gather all positive data for this feature across datasets
        all_vals = np.concatenate([
            df[df[feature] > 0][feature].values
            for _, df in data_sets
        ])
        x_limits[feature] = (all_vals.min(), all_vals.max())

        # for each dataset, find the max bin count for this feature
        for label, df in data_sets:
            vals = df[df[feature] > 0][feature].values
            counts, edges = np.histogram(vals, bins=n_bins)
            # because of log scale, skip zero counts when computing lower y-limit
            positive_counts = counts[counts > 0]
            ymin = positive_counts.min()
            ymax = counts.max()
            prev_ymin, prev_ymax = y_limits[label]
            y_limits[label] = (
                min(prev_ymin, ymin),
                max(prev_ymax, ymax)
            )

    # --- 3.3 Save each subplot individually with shared limits ---
    # loop over each label & feature, and use plot_single_distribution
    for label, df in data_sets:
        for feature, feature_name in zip(features,features_name):
            # extract only positive (or non‐zero) values
            vals = df[df[feature] > 0][feature].values

            # get the shared limits you computed earlier
            xlim = x_limits[feature]
            ylim = y_limits[label]

            # build the histogram (on log‐scale for frequency, per your previous code)
            fig = plot_single_distribution(
                real_vec    = real_df[real_df[feature] > 0][feature].values,
                other_vec   = vals,
                other_label = "Synthetic",
                xlabel      = feature_name,
                bins        = n_bins,
                log_scale   = True,
                # log_scale   = False,
                fs          = fs,
                figsize     = (6, 4),
                xlim        = xlim,
                ylim        = ylim,
                color = color
            )

            # save out at high resolution
            fname = f"fig/{label.lower()}_{feature}.png"
            fig.savefig(
                os.path.join(out_dir, fname),
                dpi=dpi_val,
                bbox_inches="tight"
            )
            plt.close(fig)

    # --- Compute metrics including JS distance ---
    features = ["weight", "height", "age", "cd4_v", "rna_v"]
    results = []

    for feature in features:
        # extract positive values
        real = real_df[real_df[feature] > 0][feature].values
        syn  = _synth_data[_synth_data[feature] > 0][feature].values

        # 1) Wasserstein distances
        wd_rs = wasserstein_distance(real, syn)

        # 2) KS-test p-values
        ks_rs = ks_2samp(real, syn).pvalue

        # 3) Jensen–Shannon distances:
        #    use a common binning over the union of all three arrays
        all_vals = np.concatenate([real, syn])
        bins = np.histogram_bin_edges(all_vals, bins=30)

        # get normalized histograms (probability mass functions)
        p_real, _ = np.histogram(real, bins=bins, density=True)
        p_syn,  _ = np.histogram(syn, bins=bins, density=True)

        # ensure they sum to 1 over discrete bins
        p_real /= p_real.sum()
        p_syn  /= p_syn.sum()

        js_rs = jensenshannon(p_real, p_syn, base=2)

        # collate
        results.append({
            "Feature": feature,
            f"Wasserstein Real vs {args.model_name}":   round(wd_rs, 3),

            f"KS p-value Real vs {args.model_name}": round(ks_rs, 3),

            f"JS Distance Real vs {args.model_name}": round(js_rs, 3),
        })

    # Convert to DataFrame and display
    metrics_df = pd.concat([metrics_df,pd.DataFrame(results)])
    print(metrics_df.round(3))
    save_path = os.path.join(args.save_path,f"{args.model_name}_metric.csv")
    if not args.plot_only:
        metrics_df.to_csv(save_path,
                        index=False,
                        header = False if os.path.exists(save_path) else True,
                        mode = "a"
                        )

    print("End")
