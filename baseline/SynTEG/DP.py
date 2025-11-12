import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import process_GQ as pce
import process_edited as pce
from datetime import date
from sklearn.preprocessing import FunctionTransformer

################################################################################################################
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# cyclical encoding function
def cyclical_encode(df, year_period=150, month_period=12, day_period=365, hour_period=24):
    # Assuming df datetime follows the following format: 'YYYY-MM-DD HH:MM:SS' with column name 'date'
    res = df.copy()
    res.date = pd.to_datetime(res.date)
    res.set_index('date', inplace=True)
    time = res.index

    # If not using any period then set to False
    if year_period is not None:
        res['year_sin'] = sin_transformer(year_period).fit_transform(time.year)
        res['year_cos'] = cos_transformer(year_period).fit_transform(time.year)

    if month_period is not None:
        res['month_sin'] = sin_transformer(month_period).fit_transform(time.month)
        res['month_cos'] = cos_transformer(month_period).fit_transform(time.month)

    if day_period is not None:
        res['day_sin'] = sin_transformer(day_period).fit_transform(time.day_of_year)
        res['day_cos'] = cos_transformer(day_period).fit_transform(time.day_of_year)
    
    if hour_period is not None:
        res['hour_sin'] = sin_transformer(hour_period).fit_transform(time.hour)
        res['hour_cos'] = cos_transformer(hour_period).fit_transform(time.hour)
    
    return res

def inverse_cyclical_encoding(tensor, reference_year=2023, year_period=150, month_period=12, day_period=365, hour_period=24):
    """Reverse the cyclical encoding process from a PyTorch tensor and return a pandas DataFrame."""
    
    # Ensure input is a NumPy array
    tensor = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    print(tensor.shape)
    # Reshape to (N * sequence_length, 8)
    N, seq_len, _ = tensor.shape
    tensor = tensor.reshape(-1, 8)

    # Convert to DataFrame
    df = pd.DataFrame(tensor, columns=['year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])

    def inverse_transform(sin_col, cos_col, period):
        """Recover original values from sine and cosine components."""
        return (np.arctan2(df[sin_col], df[cos_col]) / (2 * np.pi) * period) % period

    # Reconstruct time components
    if year_period is not None:
        df['year_offset'] = inverse_transform('year_sin', 'year_cos', year_period).astype(int)
        df['year'] = reference_year + df['year_offset'] - (year_period // 2)  # Center around reference year

    if month_period is not None:
        df['month'] = inverse_transform('month_sin', 'month_cos', month_period).astype(int) + 1  # 1-based index

    if day_period is not None:
        df['day_of_year'] = inverse_transform('day_sin', 'day_cos', day_period).astype(int) + 1  # 1-based index

    if hour_period is not None:
        df['hour'] = inverse_transform('hour_sin', 'hour_cos', hour_period).astype(int)

    # Construct datetime
    if {'year', 'month', 'day_of_year', 'hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year', 'day_of_year']].astype(str).agg('-'.join, axis=1), format='%Y-%j')
        df['date'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    # Return as DataFrame with shape (N * sequence_length, 1)
    return df[['date']].reset_index(drop=True)


import torch
import numpy as np
import pandas as pd

def partition_multi_seq(real_df, threshold, column_to_partition, max_len=None):    
    # Drop 'date' column from real_df
    if 'date' in real_df.columns:
        real_df1 = real_df.drop('date', axis=1)
    else: real_df1 = real_df

    # Parse the dataframe
    parser = pce.DataFrameParser().fit(real_df1.fillna(0), threshold)
    processed_data = torch.from_numpy(parser.transform()).unsqueeze(0)

    column_name = parser._column_order
    column_index = column_name.index(column_to_partition)

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    # Get unique values from the specified column
    unique_values = np.unique(processed_data[:, :, column_index])

    if pd.isna(max_len):
        max_len = int(len(processed_data[0, :, :]) / len(unique_values))

    # Initialize tensors for partitioned data with padding value 0
    partitioned_tensors = torch.full((len(unique_values), max_len, processed_data.shape[2]), 0.)
    end_of_sequence = torch.zeros((len(unique_values), max_len, 1))  # EOS marker initialized to 0
    padding_code = torch.zeros((len(unique_values), max_len, 1))  # padding code initialized to 0
    
    # Partition the tensor based on unique values
    for i, value in tqdm.tqdm(enumerate(unique_values), total=len(unique_values)):
        mask = processed_data[:, :, column_index] == value
        selected_seq = processed_data[mask]  # Extract sequence

        # Truncate or pad sequence to max_len
        seq_len = min(selected_seq.shape[0], max_len)
        partitioned_tensors[i, :seq_len, :] = selected_seq[:seq_len]

        if selected_seq.shape[0] <= max_len:  # add EOS only for real EOS instead of truncation
            end_of_sequence[i, seq_len - 1, 0] = 1.  # Mark last valid position
        padding_code[i, seq_len:, 0] = 1.  # Marking padding records

    # Encode time-related information
    if 'date' in real_df:
        df2 = cyclical_encode(real_df)
        partitioned_tensors_ts = torch.full((len(unique_values), max_len, 8), 0.)
        time_info = torch.tensor(df2.iloc[:, -8:].values).unsqueeze(0)
    else: 
        partitioned_tensors_ts = torch.full((len(unique_values), max_len, 8), 0.)
        time_info = torch.tensor(real_df.iloc[:, -8:].values).unsqueeze(0)


    # Partition time-related tensor
    for i, value in tqdm.tqdm(enumerate(unique_values), total=len(unique_values)):
        mask = processed_data[:, :, column_index] == value
        selected_time_seq = time_info[mask]

        # Truncate or pad sequence to max_len
        seq_len = min(selected_time_seq.shape[0], max_len)
        partitioned_tensors_ts[i, :seq_len, :] = selected_time_seq[:seq_len]

    # Remove the column_to_partition from partitioned_tensors
    partitioned_tensors = torch.cat((partitioned_tensors[:, :, :column_index], 
                                     partitioned_tensors[:, :, column_index+1:]), dim=2)
    

    masking = torch.cat((end_of_sequence, padding_code), dim=2)

    return partitioned_tensors, partitioned_tensors_ts, masking



################################################################################################################
def splitData(real_df, seq_len, threshold):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    #ori_data = torch.tensor(data.astype('float32')).numpy()
    ori_data = torch.tensor(data.astype('float32')).numpy()

    batch_size = len(ori_data) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, batch_size):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    #idx = np.random.permutation(len(temp_data))
    #data = []
    #for i in range(len(temp_data)):
    #    data.append(temp_data[idx[i]])

    data = torch.tensor(temp_data)
    
    return data

################################################################################################################
def splitTimeData(real_df, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    df2 = cyclical_encode(real_df); tlen = df2.shape[1]
    time_info = torch.tensor(df2.iloc[:,-8:].values).numpy()
      
    batch_size = len(time_info) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(time_info) - seq_len):
        _x = time_info[i:i + seq_len]
        temp_data.append(_x)

    data = torch.tensor(temp_data)
    
    return data

