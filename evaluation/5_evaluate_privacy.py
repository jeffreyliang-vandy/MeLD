import torch
print("Available CUDA devices:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())

import numpy as np
import pandas as pd
from evaluation_metric.csv2halo import Pickler
import pickle
import argparse, os

args_parser = argparse.ArgumentParser()
# Add argument for conditional flag
args_parser.add_argument("--load_path", "-L", default="data/datasets/hiv/")
args_parser.add_argument("--model", "-M", default="halo")
args = args_parser.parse_args()

assert os.path.exists(args.load_path), "Loading path not exist"

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

dummy_df = pd.read_csv(args.load_path)

dummy_lb = dummy_df.groupby('patient_id').first()[['enrol_d','age','male_y','center','mode']].reset_index()
for col,K in zip(['center','mode'],[10,11]):
    col_s = dummy_lb.pop(col)
    dummys = int_to_dummies_loop(col_s,K,col)
    dummy_lb = pd.concat([dummy_lb,dummys],axis=1)
age = dummy_lb.pop('age')
age = pd.get_dummies(pd.cut(age,bins=30,duplicates='drop'),prefix='age') * 1
enrol_d = dummy_lb.pop('enrol_d')
enrol_d = pd.get_dummies(pd.cut(enrol_d,bins=10,duplicates='drop'),prefix='enrol_d') * 1
dummy_lb = pd.concat([dummy_lb,age,enrol_d],axis=1)
print(dummy_lb.shape)
dummy_df = dummy_df.drop(columns=['enrol_d','age','male_y','center','mode']).groupby('patient_id', group_keys=False).apply(lambda x: x.iloc[1:])

pk = Pickler()
dummy_data = pk.fit_transform(dummy_df,'patient_id',dummy_lb,threshold=1,visit_len=120)

pickle.dump(dummy_data,open(f"results/datasets/{args.model}Dataset.pkl", "wb"))


### Evaluate Risk
data_path = f"results/datasets/{args.model}Dataset.pkl"
os.system(f"python evaluation_metric/evaluate_privacy_attribute.py -L {data_path} -M {args.model}")
os.system(f"python evaluation_metric/evaluate_privacy_membership.py -L {data_path} -M {args.model}")
os.system(f"python evaluation_metric/evaluate_privacy_nearest.py -L {data_path} -M {args.model}")
