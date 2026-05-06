import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from . import process_edited as pce
from sklearn.preprocessing import MinMaxScaler

__all__ = ['cyclical_encode', 'inverse_cyclical_encoding', 'partition_multi_seq', 'splitData', 'splitTimeData', 'normalize', 'inverse_normalize']

# ==========================================
# Cyclical Encoding (Optimized Vectorization)
# ==========================================

def cyclical_encode(df, year_period=150, month_period=12, day_period=366, hour_period=24):
    """Encodes datetime features using direct vectorized numpy operations."""
    res = df.copy()
    res['date'] = pd.to_datetime(res['date'])
    time = res['date'].dt
    
    # Calculate directly with numpy, bypassing sklearn FunctionTransformer overhead
    if year_period is not None:
        res['year_sin'] = np.sin(time.year * (2 * np.pi / year_period))
        res['year_cos'] = np.cos(time.year * (2 * np.pi / year_period))

    if month_period is not None:
        res['month_sin'] = np.sin(time.month * (2 * np.pi / month_period))
        res['month_cos'] = np.cos(time.month * (2 * np.pi / month_period))

    if day_period is not None:
        res['day_sin'] = np.sin(time.dayofyear * (2 * np.pi / day_period))
        res['day_cos'] = np.cos(time.dayofyear * (2 * np.pi / day_period))
    
    if hour_period is not None:
        res['hour_sin'] = np.sin(time.hour * (2 * np.pi / hour_period))
        res['hour_cos'] = np.cos(time.hour * (2 * np.pi / hour_period))
    
    return res


def inverse_cyclical_encoding(tensor, reference_year=2023, year_period=150, month_period=12, day_period=366, hour_period=24):
    """Corrected reverse cyclical encoding."""
    tensor = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    tensor = tensor.reshape(-1, 8)
    df = pd.DataFrame(tensor, columns=['year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])

    def inverse_transform(sin_arr, cos_arr, period):
        # Result is in range [0, period)
        return (np.arctan2(sin_arr, cos_arr) / (2 * np.pi) * period) % period

    if year_period is not None:
        # 1. Get the decoded remainder (e.g., 73 for the year 2023 if period is 150)
        decoded_mod = np.round(inverse_transform(df['year_sin'], df['year_cos'], year_period))
        
        # 2. Find the year closest to reference_year that has this remainder
        # We calculate the shortest distance between the decoded remainder and the reference's remainder
        ref_mod = reference_year % year_period
        diff = (decoded_mod - ref_mod) % year_period
        # Adjust for wrapping (if diff is 149, it's actually -1)
        diff = np.where(diff > year_period / 2, diff - year_period, diff)
        df['year'] = (reference_year + diff).astype(int)

    if month_period is not None:
        v = np.round(inverse_transform(df['month_sin'], df['month_cos'], month_period))
        # Since encoding was (1 to 12), decoding gives (1 to 11, and 0 for 12).
        # We map 0 back to 12.
        df['month'] = np.where(v == 0, month_period, v).astype(int)

    if day_period is not None:
        v = np.round(inverse_transform(df['day_sin'], df['day_cos'], day_period))
        # Same logic: if decoding 366/366, we get 0. Map 0 back to 366.
        df['day_of_year'] = np.where(v == 0, day_period, v).astype(int)

    if hour_period is not None:
        v = np.round(inverse_transform(df['hour_sin'], df['hour_cos'], hour_period))
        # Hours are 0-23, so standard rounding works.
        df['hour'] = (v % hour_period).astype(int)

    # Combine back to datetime
    # We use Year and Day of Year for maximum accuracy
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day_of_year'].astype(int).astype(str), format='%Y-%j')
    df['date'] += pd.to_timedelta(df['hour'], unit='h')

    return df[['date']].reset_index(drop=True)


# ==========================================
# Partition Processing (Loop Fusion & Speedups)
# ==========================================

def partition_multi_seq(real_df, threshold, column_to_partition, max_len=None):  

    assert "date" in real_df.columns, "No [date] in data"  
    real_df1 = real_df.drop('date', axis=1)
    print(real_df1[column_to_partition].nunique())
    partition_ids_original = real_df1[column_to_partition].values
    unique_ids = np.unique(partition_ids_original)

    ## preprocessing:
    cast_cols = real_df1.nunique()[real_df1.nunique() <= 10].index.tolist()
    real_df1[cast_cols] = real_df1[cast_cols].astype('str')
    real_df1[column_to_partition] = real_df1[column_to_partition].astype('int')
    
    # Save the working dataframe
    working_df = real_df1.fillna(0)

    # Fit the parser
    parser = pce.DataFrameParser().fit(working_df, threshold)
    
    # Explicitly pass working_df to transform
    processed_data = torch.from_numpy(parser.transform(working_df)).unsqueeze(0)
    assert processed_data.shape[1] == working_df.shape[0], "Row count mismatch after transformation"
    
    missing_df = real_df1.isna().astype(np.int8)
    missing_data = torch.from_numpy(parser.transform_missing(missing_df)).unsqueeze(0)

    # 3. Setup Metadata
    column_index = parser._column_order.index(column_to_partition)
    datatype_info = parser.datatype_info()
    n_nums = datatype_info['n_nums']
    num_unique = len(unique_ids)

    # Store the unique original IDs in the parser for decoding later
    parser.column_to_partition = unique_ids

    if pd.isna(max_len) or max_len is None:
        # Calculate average length if max_len isn't provided
        max_len = int(processed_data.shape[1] / num_unique)

    
    partitioned_tensors = torch.zeros((num_unique, max_len, processed_data.shape[2]))
    partitioned_missing = torch.zeros((num_unique, max_len, missing_data.shape[2]))
    partitioned_tensors_ts = torch.zeros((num_unique, max_len, 8))
    
    end_of_sequence = torch.zeros((num_unique, max_len, 1))
    padding_code = torch.zeros((num_unique, max_len, 1))
    
    df2 = cyclical_encode(real_df)
    time_info = torch.tensor(df2.iloc[:, -8:].values, dtype=torch.float32).unsqueeze(0)
    
    # 5. Partitioning Loop 
    # We use partition_ids_original (the raw data) to ensure 100% ID accuracy
    for i, val in tqdm(enumerate(unique_ids), total=num_unique, desc="Partitioning"):
        # Create mask based on original row positions
        mask = (partition_ids_original == val)
        assert len(processed_data[0,mask,column_index].unique()) == 1, f"Multiple unique values found for partition {val} in column {column_to_partition}"
        
        selected_seq = processed_data[0, mask]
        selected_missing_seq = missing_data[0, mask]
        selected_time_seq = time_info[0, mask]

        seq_len = min(selected_seq.shape[0], max_len)
        
        partitioned_tensors[i, :seq_len, :] = selected_seq[:seq_len]
        partitioned_missing[i, :seq_len, :] = selected_missing_seq[:seq_len]
        partitioned_tensors_ts[i, :seq_len, :] = selected_time_seq[:seq_len]

        if 0 < selected_seq.shape[0] <= max_len:
            end_of_sequence[i, seq_len - 1, 0] = 1.0  
        padding_code[i, seq_len:, 0] = 1.0

    keep_indices = [j for j in range(partitioned_tensors.shape[2]) if j != column_index]
    partitioned_tensors = partitioned_tensors[:, :, keep_indices]
    
    missing_keep_indices = [j for j in range(partitioned_missing.shape[2]) if j != column_index]
    partitioned_missing = partitioned_missing[:, :, missing_keep_indices]
    partitioned_missing = partitioned_missing[..., -(n_nums-1):] if n_nums > 1 else partitioned_missing[..., :0]

    masking = torch.cat((end_of_sequence, padding_code), dim=2)

    parser.remove_column(column_to_partition)
    
    return (partitioned_tensors, partitioned_tensors_ts, partitioned_missing, masking), parser

# ==========================================
# Sliding Windows (O(1) Memory Views)
# ==========================================

def splitData(real_df, seq_len, threshold):
    parser = pce.DataFrameParser().fit(real_df, threshold)
    
    # NEW: Explicitly pass real_df to transform
    ori_data = parser.transform(real_df).astype(np.float32)
    
    windows = sliding_window_view(ori_data, window_shape=seq_len, axis=0)
    windows = windows.swapaxes(1, 2)
    
    return torch.from_numpy(windows.copy())

def splitTimeData(real_df, seq_len):
    """Creates overlapping time sequences using stride tricks."""
    df2 = cyclical_encode(real_df)
    time_info = df2.iloc[:, -8:].values.astype(np.float32)
    
    # Create windows without a for-loop
    windows = sliding_window_view(time_info, window_shape=seq_len, axis=0)
    windows = windows.swapaxes(1, 2)
    
    return torch.from_numpy(windows.copy())

# ==========================================
# Normalization functions
# ==========================================

def normalize(data):
    """Normalizes data safely handling both Torch Tensors and NumPy arrays."""
    is_tensor = isinstance(data, torch.Tensor)
    
    if is_tensor:
        min_vals = data.amin(dim=(0, 1), keepdim=True)
        max_vals = data.amax(dim=(0, 1), keepdim=True)
        range_vals = torch.where(max_vals == min_vals, torch.ones_like(max_vals), max_vals - min_vals)
    elif isinstance(data, np.ndarray):
        min_vals = data.min(axis=(0, 1), keepdims=True)
        max_vals = data.max(axis=(0, 1), keepdims=True)
        range_vals = np.where(max_vals == min_vals, 1.0, max_vals - min_vals)
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

    normalized = 2 * (data - min_vals) / range_vals - 1
    
    # Squeeze out the extra dimensions to match original expected return format
    return normalized, min_vals.squeeze(), max_vals.squeeze()

def inverse_normalize(normalized_data, min_vals, max_vals):
    """Reverts normalization."""
    is_tensor = isinstance(normalized_data, torch.Tensor)
    
    if is_tensor:
        min_vals = min_vals.view(1, 1, -1).to(normalized_data.device)
        max_vals = max_vals.view(1, 1, -1).to(normalized_data.device)
        range_vals = max_vals - min_vals
        normalized_data = torch.clamp(normalized_data, -1, 1)
    elif isinstance(normalized_data, np.ndarray):
        min_vals = min_vals.reshape(1, 1, -1)
        max_vals = max_vals.reshape(1, 1, -1)
        range_vals = max_vals - min_vals
        normalized_data = np.clip(normalized_data, -1, 1)
    else:
        raise TypeError("Normalized data must be a torch.Tensor or numpy.ndarray.")

    return (normalized_data + 1) / 2 * range_vals + min_vals