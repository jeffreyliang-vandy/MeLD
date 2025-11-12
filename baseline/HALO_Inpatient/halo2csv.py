import numpy as np
import pandas as pd
from halopickler import Pickler
import pickle
import re
import argparse

args_parser = argparse.ArgumentParser()
# Add argument for conditional flag
args_parser.add_argument("--model", "-M", default="halo")
args_parser.add_argument("--serial", "-S", default="")
args = args_parser.parse_args()

def reverse_build_label_with_uniform_sampling(lb, 
                                              int_cols_K={'center':10, 'mode':11},
                                              continuous_prefixes=['age','enrol_d'],
                                              random_state=None):
    """
    Invert the dummy‐encoding & binning in real_lb to recover original columns,
    and for each binned continuous variable sample a value uniformly from its interval.

    Parameters
    ----------
    lb : pd.DataFrame
        DataFrame produced by your build‐label pipeline.
    int_cols_K : dict
        Mapping {col_name: K} for integer columns that were one‐hot encoded.
    continuous_prefixes : list of str
        Prefixes of columns that were binned and dummy‐encoded.
    random_state : int or None
        Seed for reproducible uniform sampling.

    Returns
    -------
    df_orig : pd.DataFrame
        A copy of lb with new columns:
        - integer columns restored (e.g. center, mode),
        - interval labels for each continuous col (e.g. age_interval),
        - sampled uniform values for each continuous col (e.g. age_uniform),
        plus all other columns untouched.
    """
    rng = np.random.default_rng(random_state)
    df = lb.copy()

    # --- 1. Reverse integer one‐hot ---
    for col, K in int_cols_K.items():
        dummy_cols = [f"{col}_{i}" for i in range(1, K+1)]
        df[col] = (
            df[dummy_cols]
              .idxmax(axis=1)                      # pick the column with the “1”
              .str.split('_').str[-1]             # extract the suffix
              .astype(int)
        )

    # Regex to capture numeric endpoints from a string like "(10.0, 20.0]"
    interval_pattern = re.compile(r'[\[\(]\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*[\]\)]')

    # --- 2. Reverse binned continuous dummies + uniform sampling ---
    for prefix in continuous_prefixes:
        # 2a. Recover the interval label
        bin_cols = df.filter(regex=rf'^{prefix}_').columns
        df[f"{prefix}_interval"] = (
            df[bin_cols]
              .idxmax(axis=1)
              .str.replace(f"{prefix}_", "", regex=False)
        )

        # 2b. Parse edges and sample uniformly
        def sample_uniform(interval_str):
            m = interval_pattern.search(interval_str)
            if not m:
                return np.nan
            a, b = float(m.group(1)), float(m.group(2))
            # uniform in [a, b)
            return rng.uniform(a, b)

        df[f"{prefix}"] = df[f"{prefix}_interval"].map(sample_uniform)

    return df
def int_to_dummies_loop(s, K, col_integer):
    """
    Loop-based implementation to transform a Series of values (possibly float representations
    of integers) into a dummy-coded DataFrame with columns named using col_integer.

    Parameters
    ----------
    s : pd.Series or array-like
        A pandas Series containing integer values or floats that represent integers.
    K : int
        The number of dummy columns.
    col_integer : str
        The base name for the dummy columns. The output DataFrame's columns will be
        named col_integer_1, col_integer_2, ..., col_integer_K.
    
    Returns
    -------
    dummy_df : pd.DataFrame
        A DataFrame with dummy-coded columns based on the input Series.
    """
    # Ensure s is a pandas Series.
    s = pd.Series(s)
    
    # Define the new column names.
    dummy_columns = [f"{col_integer}_{i}" for i in range(1, K + 1)]
    
    # Create a DataFrame with zeros and with the new column names.
    dummy_df = pd.DataFrame(0, index=s.index, columns=dummy_columns)
    
    # Loop over each element in the Series.
    for idx, value in s.items():
        try:
            # Convert the value to an integer.
            value_int = int(value)
        except (ValueError, TypeError):
            continue  # Skip if the conversion fails.
        
        if 1 <= value_int <= K:
            # Use the integer value to construct the column name.
            col_name = f"{col_integer}_{value_int}"
            dummy_df.at[idx, col_name] = 1
    return dummy_df

##################################################################################################################
# Pre-processing Data
print("Preprocess Data...")

column_to_partition = 'patient_id'; 

real_data = pd.read_csv("~/Documents/TimeAutoDiff/Dataset/hiv.csv.gz").drop(columns='date')
real_df = real_data.copy()

## Build Label
real_lb = real_df.groupby('patient_id').first()[['enrol_d','age','male_y','center','mode']].reset_index()
for col,K in zip(['center','mode'],[10,11]):
    col_s = real_lb.pop(col)
    dummys = int_to_dummies_loop(col_s,K,col)
    real_lb = pd.concat([real_lb,dummys],axis=1)
age = real_lb.pop('age')
age = pd.get_dummies(pd.cut(age,bins=30,duplicates='drop'),prefix='age') * 1
enrol_d = real_lb.pop('enrol_d')
enrol_d = pd.get_dummies(pd.cut(enrol_d,bins=10,duplicates='drop'),prefix='enrol_d') * 1
real_lb = pd.concat([real_lb,age,enrol_d],axis=1)
print(real_lb.shape)

## Build pickle
pk = Pickler()
real_df = real_df.drop(columns=['enrol_d','age','male_y','center','mode']).groupby('patient_id', group_keys=False).apply(lambda x: x.iloc[1:])
_ = pk.fit_transform(real_df,column_to_partition,real_lb, threshold=1,visit_len=120-1)

## Reconstruct
model = args.model

df_recon, lb_recon, pid_recon = pk.reverse(f"results/datasets/{model}Dataset{args.serial}.pkl")
lb_recon = pd.DataFrame(lb_recon,columns=real_lb.drop(columns="patient_id").columns)
lb_recon = reverse_build_label_with_uniform_sampling(lb_recon,random_state=0)[['enrol_d','age','male_y','center','mode']]
lb_recon['patient_id'] = lb_recon.index
lb_recon['visit_idx'] = 0

df_recon['patient_id'] = pid_recon
df_recon['visit_idx'] = df_recon.groupby(pid_recon).cumcount() + 1

data_recon = pd.concat([df_recon,lb_recon],axis=0).sort_values(by=['patient_id','visit_idx'])
data_recon.reset_index(drop=True).drop(columns='visit_idx',inplace=True)
data_recon = data_recon[real_data.columns]

data_recon.to_csv(f"results/datasets/{model}Dataset{args.serial}.csv.gz", compression='gzip',index=False)