import os
import numpy as np
import pandas as pd
from model import DP as dp
import argparse
from tqdm import tqdm
######################### Actual Code ############################
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

# Assume real_corr, test_corr, and syn_corr are predefined NumPy arrays.
# For demonstration, here are sample arrays (replace these with your actual data):

def plot_corr(diff_corr,frob_norm,save_path = "./corr.png"):
    # Define a nonlinear normalization: PowerNorm with gamma < 1 exaggerates lower values.
    norm = colors.PowerNorm(gamma=0.3, vmin=0, vmax=2)
    from matplotlib import rcParams
    rcParams.update({
        'font.size': 18,             # Base font size
        'axes.titlesize': 18,        # Title of the plot
        'axes.labelsize': 18,        # Axis labels
        'xtick.labelsize': 18,       # X-axis tick labels
        'ytick.labelsize': 18,       # Y-axis tick labels
        'legend.fontsize': 18,       # Legend font
        'figure.titlesize': 18       # Figure-level title (if using suptitle)
    })

    # Create a figure with three subplots
    fig, ax = plt.subplots(figsize=(8, 8))

    # Common parameters for the heat maps
    common_params = {
        'cmap': 'YlGnBu',
        'norm': norm,  # Apply the power-law normalization
        'square': True,
        'cbar_kws': {'shrink': 0.8, 'label': 'Absolute Difference'},
    }

    # Plot the heat maps with the nonlinear color normalization
    sns.heatmap(diff_corr, ax=ax, **common_params)

    # Set subplot titles for clarity
    ax.set_title(f'{args.model_name}',fontsize=18)

    # --------------------------------------------------
    # 3.  GROUP-LEVEL X-AXIS
    blocks = {
        "Demographic":    (0, 4),
        "Clinical event": (5,  87),
        "ART regiment":   (88, 129),
        "Continuous":     (130, 135),
    }
    # tick positions centred in each block
    xticks   = [(s + e) / 2 for s, e in blocks.values()]
    xlabels  = [f"{name}"
                for name, (s, e) in blocks.items()]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0, ha='center', va='top')
    ax.tick_params(axis='x', pad=10)              # push labels away from cells

    # --------------------------------------------------
    # 4.  WIDTH INDICATORS  ––  vertical dashed guides
    boundaries = [e + 0.5 for _, (s, e) in blocks.items() if e < diff_corr.shape[1]-1]
    for x in boundaries:
        ax.axvline(x, color='red', linestyle='--', linewidth=0.7, alpha=0.7)

    # 2.  COMPUTE FROBENIUS NORM
    ax.text(
        0.5, 0.5,                       # x, y in axis coordinates (0–1)
        f"Frob norm: {frob_norm:.2f}",     # formatted text
        transform=ax.transAxes,           # use axis coordinate system
        ha='center', va='center',          # align to the corner
        fontsize=18, color='black',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.3)
    )


    # Adjust tick parameters for better aesthetics
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Ensure a tight layout to prevent overlap
    fig.tight_layout()
    plt.savefig(save_path,
                    dpi=300, bbox_inches='tight', transparent=False)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/hiv_train.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/corr/")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args_parser.add_argument("--serial",type=str,default="")
    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    real_df = pd.read_csv(args.real_path)
    if 'date' in real_df.columns: real_df.drop(columns="date",inplace=True)

    ### Get columns order
    static_columns = ['enrol_d', 'center', 'male_y', 'age', 'mode']
    numeric_columns = ['cd4_v',"rna_v","weight","height","gap","patient_id"]
    art_columns = real_df.filter(regex="art").columns.to_list()
    ce_columns = real_df.columns.difference(numeric_columns+static_columns+art_columns).to_list()
    non_numeric_columns = real_df.columns.difference(numeric_columns).to_list()
    columns_order =  static_columns + ce_columns + art_columns + numeric_columns

    ### Process real df
    real_df=real_df[columns_order]
    real_df.fillna(0,inplace=True)
    real_df[static_columns] = real_df.groupby("patient_id")[static_columns].transform(max)
    ### Process synth_df
    _synth_data = pd.read_csv(args.test_path)
    if 'date' in _synth_data: _synth_data.drop(columns="date",inplace=True)
    _synth_data = _synth_data[columns_order]
    _synth_data.fillna(0,inplace=True)
    _synth_data[static_columns] = _synth_data.groupby("patient_id")[static_columns].transform(max)

    ### within patient correlation
    real_within = real_df.copy()
    real_within -= real_df.groupby('patient_id').transform(np.nanmean)
    real_within.drop(columns="patient_id",inplace=True)
    real_within_corr = real_within.corr(method='spearman')
    # real_within_corr = np.nan_to_num(np.corrcoef(real_within.values,rowvar = False),0)

    synth_within = _synth_data.copy()
    synth_within -= _synth_data.groupby('patient_id').transform(np.nanmean)
    synth_within.drop(columns="patient_id",inplace=True)
    synth_within_corr = synth_within.corr(method="spearman")
    # synth_within_corr = np.nan_to_num(np.corrcoef(synth_within.values,rowvar = False),0)

    ### difference
    within_corr_diff = np.abs(real_within_corr- synth_within_corr)
    within_corr_diff = np.nan_to_num(within_corr_diff,0)

    within_frob_norm = np.linalg.norm(within_corr_diff, ord='fro')
    plot_corr(within_corr_diff,within_frob_norm,os.path.join(args.save_path,f"{args.model_name}_wihtin_corr{args.seed}.png"))

    ### Between patient correlation
    real_between = real_df.groupby('patient_id').mean()
    real_between_corr = real_between.corr(method='spearman')
    # real_between_corr = np.nan_to_num(np.corrcoef(real_between.values,rowvar = False),0)

    synth_between = _synth_data.groupby('patient_id').mean()
    synth_between_corr = synth_between.corr(method="spearman")
    # synth_between_corr = np.nan_to_num(np.corrcoef(synth_between.values,rowvar = False),0)

    ### difference
    between_corr_diff = np.abs(real_between_corr- synth_between_corr)
    between_corr_diff = np.nan_to_num(between_corr_diff,0)

    between_frob_norm = np.linalg.norm(between_corr_diff, ord='fro')
    plot_corr(between_corr_diff,between_frob_norm,os.path.join(args.save_path,f"{args.model_name}_between_corr{args.seed}.png"))

    # Create a summary table
    summary_df = pd.DataFrame({
        'Within patient corr': [within_frob_norm],
        'Between patient corr': [between_frob_norm],
        'file':args.test_path
    })
    file_path = os.path.join(args.save_path,f"{args.model_name}_corr_summary.csv")
    summary_df.to_csv(
        file_path,
        mode="a" if os.path.exists(file_path) else "w",  # append if exists, else write
        header=not os.path.exists(file_path),            # write header only if new file
        index=False
    )
