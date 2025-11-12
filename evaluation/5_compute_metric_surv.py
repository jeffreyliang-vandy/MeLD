import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse, os
from model import DP as dp
import pandas as pd
import numpy as np
from lifelines.utils import restricted_mean_survival_time
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
from numpy import (
        trapz,
    )
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import Tuple
import subprocess
from scipy import stats
from typing import Optional


def km_survival_function(
    T: np.ndarray, E: np.ndarray
) -> Tuple[KaplanMeierFitter, np.ndarray, np.ndarray, np.ndarray]:
    kmf = KaplanMeierFitter().fit(T, E)
    surv_fn = kmf.survival_function_.T.reset_index(drop=True)
    if len(surv_fn.columns) < 2:
        raise RuntimeError("invalid survival functin for extrapolation")

    return kmf, surv_fn

def nonparametric_distance(
    real: Tuple[np.ndarray, np.ndarray],
    syn: Tuple[np.ndarray, np.ndarray],
    n_points: int = 1000,
) -> Tuple:
    """
    From
    https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/core/models/survival_analysis/metrics.py
    """
    real_T, real_E = real
    syn_T, syn_E = syn

    Tmax = min(real_T.max(), syn_T.max())
    Tmin = min(real_T.min(), syn_T.min())
    Tmin = max(0, Tmin)

    time_points = np.linspace(Tmin, Tmax, n_points)

    opt: list = []
    abs_opt: list = []

    real_kmf, real_surv = km_survival_function(
        real_T, real_E
    )
    if len(syn) == 0 or len(real) == 0:
        raise ValueError("Empty evaluation sets")

    syn_kmf, syn_surv = km_survival_function(
        syn_T, syn_E
    )

    abs_opt = []
    opt = []
    for t in time_points:
        syn_local_pred = syn_kmf.predict(t)
        real_local_pred = real_kmf.predict(t)

        if np.isnan(syn_local_pred):
            raise RuntimeError("syn_local_pred contains NaNs")
        if np.isnan(real_local_pred):
            raise RuntimeError("real_local_pred contains NaNs")

        abs_opt.append(abs(syn_local_pred - real_local_pred))
        opt.append(syn_local_pred - real_local_pred)

    auc_abs_opt = trapz(abs_opt, time_points) / Tmax
    auc_opt = trapz(opt, time_points) / Tmax
    sightedness = (real_T.max() - syn_T.max()) / Tmax

    return auc_opt, auc_abs_opt, sightedness



from typing import Optional, List

def define_survival_data(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    skip: int = 0,
    start_col: Optional[str] = None,
    order_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Vectorized version: one survival record per subject.
    Guarantees: time >= 0 and time <= subject's observed max follow-up.
    """
    # ---- Column validation ----
    required_cols = {id_col, time_col, event_col}
    if start_col is not None:
        required_cols.add(start_col)
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # ---- Working subset & order ----
    cols = [id_col, time_col, event_col]
    if start_col is not None:
        cols.append(start_col)
    if order_cols:
        cols.extend(order_cols)
    df = data[cols].copy()

    # Enforce within-subject order (stable)
    if order_cols:
        df = df.sort_values([id_col] + order_cols, kind="mergesort")
    else:
        df = df.sort_values([id_col], kind="mergesort")

    # Row index within the sorted frame to use as a stable integer index
    df["_idx"] = np.arange(len(df), dtype=np.int64)

    # ---- Clean/prepare per-row values ----
    # time >= 0, zero-out first `skip` increments per subject
    df[time_col] = df[time_col].clip(lower=0)
    if skip > 0:
        first_k = df.groupby(id_col, sort=False).cumcount() < skip
        df.loc[first_k, time_col] = 0

    # Cumulative time within subject
    df["_cum"] = df.groupby(id_col, sort=False)[time_col].cumsum()

    # First and last row positions per subject (as integer positions)
    first_pos = df.groupby(id_col, sort=False)["_idx"].min()
    last_pos  = df.groupby(id_col, sort=False)["_idx"].max()

    # ---- Begin index per subject ----
    if start_col is not None:
        # First row (by order) with start==1 per subject (position), if any
        first_start_pos = (
            df.loc[df[start_col].eq(1)]
            .groupby(id_col, sort=False)["_idx"].min()
        )
        # If no start flag for a subject, use its last row
        begin_pos = first_start_pos.reindex(last_pos.index).fillna(last_pos).astype(np.int64)
    else:
        # Use the first row of each subject
        begin_pos = first_pos

    # ---- End index per row: first event at/after this row ----
    # Mark event positions; bfill within subject gives the next event index for each row
    df["_event_pos"] = np.where(df[event_col].eq(1), df["_idx"], np.nan)
    # Groupwise backward fill to carry the next event's index upward
    # (If your pandas doesn't have GroupBy.bfill, use transform with a lambda.)
    next_event_pos = (
        df.groupby(id_col, sort=False)["_event_pos"].bfill()
    )

    # Read the next-event position at each subject's begin row
    # We map positions via an index on "_idx" for O(1) gather.
    pos_indexer = df.set_index("_idx")
    begin_next_event = next_event_pos.loc[pos_indexer.index].reindex(begin_pos.values).to_numpy()

    # If NaN (no event after begin), use subject last row; record event flag accordingly
    last_pos_vals = last_pos.to_numpy()
    has_event = ~np.isnan(begin_next_event)
    end_pos = np.where(has_event, begin_next_event, last_pos_vals).astype(np.int64)

    # ---- Compute times and clamp to observed window ----
    cum_at = pos_indexer["_cum"]
    begin_cum = cum_at.reindex(begin_pos.values).to_numpy()
    end_cum   = cum_at.reindex(end_pos).to_numpy()

    # Observed window per subject = last_cum - first_cum
    last_cum  = cum_at.reindex(last_pos_vals).to_numpy()
    first_cum = cum_at.reindex(first_pos.to_numpy()).to_numpy()
    max_follow = last_cum - first_cum

    # Elapsed time = end_cum - begin_cum, clamped to [0, max_follow]
    raw_time = end_cum - begin_cum
    time_val = np.minimum(np.maximum(raw_time, 0.0), max_follow)

    # Assemble result aligned to subjects’ index order in begin_pos
    out = pd.DataFrame({
        "id": begin_pos.index.to_numpy(),
        "time": time_val.astype(float),
        "event": has_event.astype(np.int8)
    })

    return out


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/")
    args_parser.add_argument("--endpoint","-ED",type=str,default="follow_mode_death")
    args_parser.add_argument("--startpoint","-ST",type=str,required=False)
    args_parser.add_argument("--plot_only","-P",action="store_true",default=False)
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    ### Loading data
    print("Loading Data")
    train_data = pd.read_csv(os.path.join(args.real_path,"hiv_train.csv.gz")).fillna(0)  # for testing

    test_data = pd.read_csv(os.path.join(args.real_path,"hiv_test.csv.gz")).fillna(0)
    test_data['date'] = test_data.index

    _synth_data = pd.read_csv(args.test_path).fillna(0)
    _synth_data['date'] = _synth_data.index


    ### Get original time
    train_data['time'] = (np.exp(train_data['gap']) -0.99)/365
    test_data['time'] = (np.exp(test_data['gap']) -0.99)/365
    _synth_data['time'] = (np.exp(_synth_data['gap']) -0.99)/365

    ### Get survival
    real_surv = define_survival_data(train_data,'patient_id','time',args.endpoint,0,start_col=args.startpoint)
    test_surv = define_survival_data(test_data,'patient_id','time',args.endpoint,0,start_col=args.startpoint)
    synth_surv = define_survival_data(_synth_data,'patient_id','time',args.endpoint,0,start_col=args.startpoint)


    # Conduct log-rank test: Test vs Synthetic
    lr_results = logrank_test(
        test_surv['time'], 
        synth_surv['time'], 
        event_observed_A=test_surv['event'], 
        event_observed_B=synth_surv['event']
    )

    # Extract p-value
    p_value_lr = lr_results.p_value

    # Get Distance:
    km_distance = nonparametric_distance(real=(test_surv.time,test_surv.event),
                                         syn=(synth_surv.time,synth_surv.event)
    )

    # Initialize Nelson-Aalen Fitters for each dataset
    naf_real = NelsonAalenFitter()
    naf_test = NelsonAalenFitter()
    naf_synth = NelsonAalenFitter()

    # Fit the models
    naf_real.fit(real_surv['time'], event_observed=real_surv['event'], label="Real training")
    naf_test.fit(test_surv['time'], event_observed=test_surv['event'], label="Real testing")
    naf_synth.fit(synth_surv['time'], event_observed=synth_surv['event'], label=f"Synthetic")

    # Plot cumulative hazard functions
    plt.figure(figsize=(8, 5))


    naf_real.plot(ci_show=True,color=(67/255,152/255,217/255),alpha=0.5, label="_nolegend_")   # Plot Real Data
    naf_test.plot(ci_show=True,color=(229/255,186/255,66/255),alpha=0.5, label="_nolegend_")   # Plot Test Data
    naf_synth.plot(ci_show=True,color=(172/255,37/255,62/255),alpha=0.5, label="_nolegend_")  # Plot Synthetic Data

    naf_real.plot(ci_show=False,color=(67/255,152/255,217/255),alpha=1, label="Real training")   # Plot Real Data
    naf_test.plot(ci_show=False,color=(229/255,186/255,66/255),alpha=1, label="Real testing")   # Plot Test Data
    naf_synth.plot(ci_show=False,color=(172/255,37/255,62/255),alpha=1, label=f"Synthetic")  # Plot Synthetic Data

    # Customize plot
    plt.title(f"{args.model_name}",fontsize=18)
    plt.xlabel("Time (years)", fontsize=18)
    plt.ylabel("Cumulative hazard", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((0,2.5))
    plt.grid(True, linestyle='--')
    legend = plt.legend(
        loc='upper left',
        title='Data Source',
        title_fontsize=16,
        fontsize=16,
        handlelength=1,
        labelspacing=0.1,
        edgecolor='black'    # white border
    )

    # Increase the border (frame) line width
    legend.get_frame().set_linewidth(1)  # set thickness here

    ax = plt.gca()  # Get current Axes

    # Set thickness for each edge of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)  # Increase to desired thickness

    # Add p-value annotation
    if p_value_lr > 0.001:
        p_text = f"{p_value_lr:.3f}"
    else:
        p_text = "<0.001"
    plt.text(
        0.02, 0.45,
        f"Real testing vs Synthetic\n$p$-value: {p_text}\nKM-D: {km_distance[1]:.3f}",
        transform=plt.gca().transAxes,
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white",alpha=0.3)
    )

    # Adjust layout and display
    plt.tight_layout()

    ## Save plot
    os.makedirs(os.path.join(args.save_path,"fig/"),exist_ok=True)
    plt.savefig(os.path.join(args.save_path,"fig/",f"{args.model_name}_{args.startpoint}_{args.endpoint}_surv_plot-{args.seed}.png"),
                dpi=300, bbox_inches='tight', transparent=False)
    
    if args.plot_only:
        exit(0)
    
    death_result = pd.DataFrame(
        {"Model":[args.model_name],
         "p.value":[p_value_lr],
         "distance":[km_distance[1]],
         "max_time":[synth_surv['time'].max()],
         "file":[args.test_path]}
    )

    file_path = os.path.join(args.save_path,f"{args.model_name}_death_summary.csv")
    death_result.to_csv(
        file_path,
        mode="a" if os.path.exists(file_path) else "w",  # append if exists, else write
        header=not os.path.exists(file_path),            # write header only if new file
        index=False
    )

    ######### Cox-PH estimate SYNTHETIC DATA #########
    synth_surv = define_survival_data(_synth_data,'patient_id','time',args.endpoint,0,start_col=args.startpoint)
    synth_cond = _synth_data.groupby('patient_id')[['age','male_y','enrol_d','center']].first()
    synth_cond['cd4_v'] = _synth_data.groupby('patient_id').cd4_v.apply(lambda x: x[2:7].mean())
    synth_cond['age'] = synth_cond['age'] + 18
    synth_cond.reset_index(inplace=True)

    synth_cph = pd.merge(synth_surv,synth_cond,how='left',left_on='id',right_on='patient_id')
    synth_cph['center'] = synth_cph['center'].astype(str)

    random_id = np.random.randint(0, 10000)
    temp_file = f"temp{random_id}.csv.gz"
    synth_cph.to_csv(temp_file, compression="gzip", index=False)
    try:
        subprocess.run(["Rscript", "evaluation_metric/cox.R", temp_file, f"./{args.model_name}_cox.csv"], check=True)
    finally:
        os.remove(temp_file)


    ###### Log-rank test on other diagnoses

    # Identify the columns that start with 'ce_id'
    ce_columns = train_data.columns[train_data.columns.str.startswith('ce_id')]
    ce_columns = ce_columns.to_list()
    ce_columns.remove("ce_id_other")

    # Initialize a list to collect the test results
    results = []

    # Loop over each candidate column
    for ce in ce_columns:
        # Generate survival data for the three groups
        real_surv  = define_survival_data(train_data,  'patient_id', 'time', ce, 0, start_col=args.startpoint)
        test_surv  = define_survival_data(test_data,  'patient_id', 'time', ce, 0, start_col=args.startpoint)
        synth_surv = define_survival_data(_synth_data, 'patient_id', 'time', ce, 0, start_col=args.startpoint)

        # --- Compute mean survival times using KaplanMeierFitter
        kmf_real = KaplanMeierFitter()
        kmf_real.fit(real_surv['time'], event_observed=real_surv['event'])
        mean_survival_real = restricted_mean_survival_time(kmf_real,t=40)

        kmf_test = KaplanMeierFitter()
        kmf_test.fit(test_surv['time'], event_observed=test_surv['event'])
        mean_survival_test = restricted_mean_survival_time(kmf_test,t=40)

        kmf_synth = KaplanMeierFitter()
        kmf_synth.fit(synth_surv['time'], event_observed=synth_surv['event'])
        mean_survival_synth = restricted_mean_survival_time(kmf_synth,t=40)

        # Perform the log-rank test: real vs test
        result_real_test = logrank_test(
            real_surv['time'], 
            test_surv['time'], 
            event_observed_A=real_surv['event'], 
            event_observed_B=test_surv['event']
        )
        
        # Perform the log-rank test: test vs synthetic
        result_test_synth = logrank_test(
            test_surv['time'], 
            synth_surv['time'], 
            event_observed_A=test_surv['event'], 
            event_observed_B=synth_surv['event']
        )
        
        # Extract the p-values from the test results
        pval_real_test  = result_real_test.p_value
        pval_test_synth = result_test_synth.p_value

        # Perform distance calculation
        ## Real vs test
        km_distance_real = nonparametric_distance(real=(test_surv.time,test_surv.event),
                                         syn=(real_surv.time,real_surv.event))
        ## Synthetic vs test
        km_distance_synth = nonparametric_distance(real=(test_surv.time,test_surv.event),
                                         syn=(synth_surv.time,synth_surv.event))
        
        
        # Append the results in a dictionary format
        results.append({
            'ce': ce,
            'real_prevalence_per_10000': real_surv['event'].mean() * 10000,
            'real_vs_test_p_value': pval_real_test,
            f'{args.model_name}_vs_test_p_value': pval_test_synth,
            'mean_survival_real': mean_survival_real,
            'mean_survival_test': mean_survival_test,
            f'mean_survival_{args.model_name}': mean_survival_synth,
            "distance_real": km_distance_real[0],
            "distance_synth": km_distance_synth[0],
            "file":args.test_path
        })

    # Convert the list of dictionaries into a pandas DataFrame for a tidy tabular display
    results_df = pd.DataFrame(results)
    file_path = os.path.join(args.save_path,f"{args.model_name}_logrank.csv")
    results_df.to_csv(
        file_path,
        mode="a" if os.path.exists(file_path) else "w",  # append if exists, else write
        header=not os.path.exists(file_path),            # write header only if new file
        index=False
    )

    # Multiple-comparison adjusted alpha (Bonferroni)
    alpha = 0.05 / len(results_df)

    # Count how many p-values are below alpha in each comparison
    num_false_pos_real_test = (results_df['real_vs_test_p_value'] < alpha).sum()
    num_false_pos_synth_test = (results_df[f'{args.model_name}_vs_test_p_value'] < alpha).sum()

    # Total number of tests
    total_tests = len(results_df)

    # Calculate the percentage of false positives
    pct_false_pos_real_test = num_false_pos_real_test / total_tests * 100
    pct_false_pos_synth_test = num_false_pos_synth_test / total_tests * 100

    # pair test bewteen distance
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(results_df['distance_real'], results_df['distance_synth'])

    # Create a summary table
    summary_df = pd.DataFrame({
        'Comparison': ['Real vs Test', f'{args.model_name} vs Test'],
        'Num False Positives': [num_false_pos_real_test, num_false_pos_synth_test],
        'Pct False Positives': [pct_false_pos_real_test, pct_false_pos_synth_test],
        'Distance Pair Ttest': [np.nan,p_value],
        'file':args.test_path
    })

    file_path = os.path.join(args.save_path,f"{args.model_name}_logrank_summary.csv")
    summary_df.to_csv(
        file_path,
        mode="a" if os.path.exists(file_path) else "w",  # append if exists, else write
        header=not os.path.exists(file_path),            # write header only if new file
        index=False
    )




