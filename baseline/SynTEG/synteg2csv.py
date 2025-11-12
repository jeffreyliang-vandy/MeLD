#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import argparse
import os


# In[ ]:
parser = argparse.ArgumentParser()
parser.add_argument('--load_path',"-L", type=str,required=True)
parser.add_argument('--save_path',"-S", type=str,required=True)
args = parser.parse_args()

# —— User‐modifiable parameters —— 
# 1. Path to the padded NumPy array (res_c)
prefix = args.load_path
code_npy_path = os.path.join(prefix, "code.npy")

# 2. Path to the JSON file containing code_map (feature_name → integer index)
code_map_json_path = os.path.join("./data/datasets/synteg/code_map.pkl")

# 3. (Optional) Output path for the reconstructed CSV
OUTPUT_CSV_PATH = os.path.join(args.save_path, "reconstructed_long_table.csv")


# 1. Load the padded array
res_c = np.load(code_npy_path)
lengths   = np.load(os.path.join(prefix, 'length.npy'),    allow_pickle=True).astype(np.int32)
#    res_c.shape = (P, max_length, max_len)
num_patients, max_length, max_len = res_c.shape

# 2. Load the code_map (feature_name → index)
with open(code_map_json_path, "r") as f:
    code_map = pickle.load(open(code_map_json_path,"rb"))

# 3. Invert code_map → idx2feature, and determine num_features and sentinel
idx2feature = {idx: feat for idx,feat in enumerate(code_map)}
num_features = len(code_map)
code_last = num_features  # sentinel used by the preprocessor

# 4. Prepare a list to collect rows (each row: [patient_idx, f0, f1, …, f_{num_features-1}])
rows = []

# 5. Iterate over each patient index
for pid_idx in range(num_patients):
    patient_array = res_c[pid_idx]  
    # patient_array.shape = (max_length, max_len)

    # 5a. Find the row index where the last element == code_last
    #     That is the “last real visit.” All rows after are fully −1 padded.
    # sentinel_rows = np.where(patient_array[:, -1] == code_last)[0]
    # if sentinel_rows.size == 0:
    #     raise ValueError(f"No sentinel found for patient_idx {pid_idx}.")
    # if sentinel_rows.size > 1:
    #     raise ValueError(f"Multiple sentinel rows for patient_idx {pid_idx}.")
    # last_visit_idx = int(sentinel_rows[0])
    last_visit_idx = lengths[pid_idx]

    # 5b. For each actual visit (0..last_visit_idx), reconstruct the binary feature vector
    for visit_idx in range(last_visit_idx):
        visit_row = patient_array[visit_idx]  # shape = (max_len,)

        #   • Discard any entries == −1 or == code_last
        #   • Remaining integers are in [0..num_features-1], denoting which features were “1”
        true_codes = [feat for feat in visit_row
                        if (feat != -1 and feat != code_last)]

        #   • Build a binary vector of length = num_features
        binary_vector = np.zeros(num_features, dtype=int)
        for feat_idx in true_codes:
            if not (0 <= feat_idx < num_features):
                raise ValueError(
                    f"Invalid feature index {feat_idx} encountered for patient_idx {pid_idx}, visit {visit_idx}."
                )
            binary_vector[int(feat_idx)] = 1

        #   • Prepend the synthetic patient index, then convert to list
        row = [pid_idx] + binary_vector.tolist()
        rows.append(row)

# 6. Build a DataFrame
column_names = ["patient_id"] + [idx2feature[i] for i in range(num_features)]
reconstructed_df = pd.DataFrame(rows, columns=column_names)


# In[ ]:


def bin2cont(array,bins=30):
    """Bins to 0-1"""
    rng = np.random.default_rng()  # use new Generator API

    array_cont = []
    low, high = array / bins, (array + 1) / bins
    array_cont.append(rng.uniform(low, high,size = None))

    return array_cont[0]

def collapse_onehot_to_continuous(df: pd.DataFrame, prefix: str, n_bins: int = 30) -> None:
    """
    For all columns 'prefix_0', 'prefix_1', …, 'prefix_{n_bins-1}' present in df:
      - For each row, find which column has value == 1.
      - Parse its suffix k (an integer in [0..n_bins-1]).
      - Sample x ~ Uniform(k/n_bins, (k+1)/n_bins).
      - Assign x to a new column f"{prefix}_cont".
      - If no one-hot column is 1 for that row, assign NaN.
    Drops all original one-hot columns in-place.
    """
    # 1) Gather all one-hot cols for this prefix
    onehot_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not onehot_cols:
        return

    # 2) Extract suffix k from each column name
    def _suffix(col: str) -> int:
        return int(col.split(f"{prefix}_", 1)[1])
    suffixes = np.array([_suffix(c) for c in onehot_cols], dtype=int)

    # 3) Sub-array of shape (rows, n_present)
    sub = df[onehot_cols].values
    row_sums = sub.sum(axis=1)
    argmax_idx = sub.argmax(axis=1)

    # 4) For each row, sample in the appropriate bin
    rng = np.random.default_rng()  # use new Generator API
    cont = []
    for i, s in enumerate(row_sums):
        if s == 1:
            k = int(suffixes[argmax_idx[i]])
            low, high = k / n_bins, (k + 1) / n_bins
            cont.append(rng.uniform(low, high))
        else:
            # either no-hot (s==0) or malformed (>1); treat both as NaN
            cont.append(np.nan)

    # 5) Assign new continuous column and drop one-hots
    df[f"{prefix}_cont"] = cont
    df.drop(columns=onehot_cols, inplace=True)


# —— Usage on your reconstructed_df —— 
collapse_onehot_to_continuous(reconstructed_df, prefix="cd4",    n_bins=30)
collapse_onehot_to_continuous(reconstructed_df, prefix="rna",    n_bins=30)
collapse_onehot_to_continuous(reconstructed_df, prefix="weight", n_bins=30)
collapse_onehot_to_continuous(reconstructed_df, prefix="height", n_bins=30)

# Now each of:
#   'cd4_cont', 'rna_cont', 'weight_cont', 'height_cont'
# is a float in [0,1], sampled uniformly within the original bin’s interval.


# In[ ]:


### Add interval
# load your data
intervals = np.load(os.path.join(prefix, 'interval.npy'), allow_pickle=True).astype(np.int32)

# build a list of the “valid” slices, then concatenate them all
pieces = [arr[:L].ravel() for arr, L in zip(intervals, lengths)]
intervals_bin = np.concatenate(pieces)

### back to continuous
reconstructed_df['gap'] = bin2cont(intervals_bin)


# In[ ]:


### Rescale
real_df = pd.read_csv("~/Documents/TimeAutoDiff/Dataset/hiv_train.csv.gz")

def scale(array1, array2):
    min = np.nanmin(array2)
    max = np.nanmax(array2)
    scales = max - min
    result = array1 * scales + min
    return result

reconstructed_df['gap'] = scale(reconstructed_df.gap,real_df.gap)
reconstructed_df['cd4_v'] = scale(reconstructed_df.cd4_cont,real_df.cd4_v)
reconstructed_df['rna_v'] = scale(reconstructed_df.rna_cont,real_df.rna_v)
reconstructed_df['weight'] = scale(reconstructed_df.weight_cont,real_df.weight)
reconstructed_df['height'] = scale(reconstructed_df.height_cont,real_df.height)
reconstructed_df.drop(columns=["cd4_cont","rna_cont","weight_cont","height_cont"],inplace=True)


# In[ ]:


### add labels

# —— Load your arrays —— 
ages     = np.load(os.path.join(prefix, 'age.npy'),     allow_pickle=True).astype(np.int32)
genders  = np.load(os.path.join(prefix, 'gender.npy'),  allow_pickle=True).astype(np.int32)  # new id starts from 0
countrys = np.load(os.path.join(prefix, 'country.npy'), allow_pickle=True).astype(np.int32)  # new id starts from 0
centers  = np.load(os.path.join(prefix, 'center.npy'),  allow_pickle=True).astype(np.int32)
births   = np.load(os.path.join(prefix, 'birth.npy'),   allow_pickle=True).astype(np.int32)
aidsmode = np.load(os.path.join(prefix, 'mode.npy'),    allow_pickle=True).astype(np.int32)

# —— Build the DataFrame —— 
reconstructed_lb = pd.DataFrame()
reconstructed_lb['age']        = scale(bin2cont(ages[:, 0]),real_df.groupby("patient_id").first().age)
reconstructed_lb['male_y']     = genders[:, 0]
# reconstructed_lb['country']    = countrys[:, 0]
reconstructed_lb['center']     = centers[:, 0] + 1
reconstructed_lb['enrol_d']      = scale(bin2cont(births[:, 0]),real_df.groupby("patient_id").first().enrol_d)
reconstructed_lb['mode']  = aidsmode[:, 0] + 1
reconstructed_lb['patient_id'] = reconstructed_lb.index


# In[ ]:


reconstructed_lb['visit_idx'] = 0
reconstructed_df['visit_idx'] = reconstructed_df.groupby("patient_id").cumcount()

### Final concatenate
gen_df = pd.DataFrame(columns=real_df.columns)
gen_df = pd.concat([gen_df,reconstructed_lb,reconstructed_df],axis=0)
gen_df.fillna(0,inplace=True)
gen_df.sort_values(["patient_id","visit_idx"],inplace=True)
gen_df.drop(columns='visit_idx',inplace=True)
gen_df.reset_index(drop=True,inplace=True)

## Save Data
gen_df.to_csv(os.path.join(args.save_path, "syntegDataset.csv.gz"),index = False, compression="gzip")

