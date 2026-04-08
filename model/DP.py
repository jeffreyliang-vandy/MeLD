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

def cyclical_encode(df, year_period=150, month_period=12, day_period=365, hour_period=24):
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

def inverse_cyclical_encoding(tensor, reference_year=2023, year_period=150, month_period=12, day_period=365, hour_period=24):
    """Reverse the cyclical encoding process using pure numpy arrays."""
    tensor = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    
    # Reshape to (N * sequence_length, 8)
    tensor = tensor.reshape(-1, 8)
    df = pd.DataFrame(tensor, columns=['year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])

    def inverse_transform(sin_arr, cos_arr, period):
        return (np.arctan2(sin_arr, cos_arr) / (2 * np.pi) * period) % period

    if year_period is not None:
        df['year'] = reference_year + inverse_transform(df['year_sin'], df['year_cos'], year_period).astype(int) - (year_period // 2)

    if month_period is not None:
        df['month'] = inverse_transform(df['month_sin'], df['month_cos'], month_period).astype(int) + 1 

    if day_period is not None:
        df['day_of_year'] = inverse_transform(df['day_sin'], df['day_cos'], day_period).astype(int) + 1 

    if hour_period is not None:
        df['hour'] = inverse_transform(df['hour_sin'], df['hour_cos'], hour_period).astype(int)

    if {'year', 'month', 'day_of_year', 'hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day_of_year'].astype(str), format='%Y-%j')
        df['date'] += pd.to_timedelta(df['hour'], unit='h')

    return df[['date']].reset_index(drop=True)

# ==========================================
# Partition Processing (Loop Fusion & Speedups)
# ==========================================

def partition_multi_seq(real_df, threshold, column_to_partition, max_len=None):  

    assert "date" in real_df.columns, "No [date] in data"  
    real_df1 = real_df.drop('date', axis=1)
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
    
    missing_df = real_df1.isna().astype(np.int8)
    missing_data = torch.from_numpy(parser.transform_missing(missing_df)).unsqueeze(0)

    column_index = parser._column_order.index(column_to_partition)
    datatype_info = parser.datatype_info()
    n_nums = datatype_info['n_nums']

    # Extract the 1D encoded array from the tensor
    encoded_col = processed_data[0, :, column_index].numpy()
    unique_values = np.unique(encoded_col)
    # ---------------------------------------------------------
    ## Look up exact original values by row index
    original_col_values = real_df1[column_to_partition].to_numpy()
    original_ids = []
    for value in tqdm(unique_values):
        # Find the exact row index where this encoded value first appears
        first_occurrence_idx = np.where(encoded_col == value)[0][0]
        # Grab the untampered original value from the original dataframe
        original_ids.append(original_col_values[first_occurrence_idx])
    # ---------------------------------------------------------

    parser.column_to_partition = np.array(original_ids).astype(int)  # Store for later use in decoding

    if pd.isna(max_len) or max_len is None:
        max_len = int(processed_data.shape[1] / len(unique_values))

    num_unique = len(unique_values)
    
    partitioned_tensors = torch.zeros((num_unique, max_len, processed_data.shape[2]))
    partitioned_missing = torch.zeros((num_unique, max_len, missing_data.shape[2]))
    partitioned_tensors_ts = torch.zeros((num_unique, max_len, 8))
    
    end_of_sequence = torch.zeros((num_unique, max_len, 1))
    padding_code = torch.zeros((num_unique, max_len, 1))
    
    df2 = cyclical_encode(real_df)
    time_info = torch.tensor(df2.iloc[:, -8:].values, dtype=torch.float32).unsqueeze(0)
    
    for i, value in tqdm(enumerate(unique_values), total=num_unique, desc="Partitioning Sequences"):
        mask = processed_data[0, :, column_index] == value
        
        selected_seq = processed_data[0, mask]
        selected_missing_seq = missing_data[0, mask]
        selected_time_seq = time_info[0, mask]

        seq_len = min(selected_seq.shape[0], max_len)
        
        partitioned_tensors[i, :seq_len, :] = selected_seq[:seq_len]
        partitioned_missing[i, :seq_len, :] = selected_missing_seq[:seq_len]
        partitioned_tensors_ts[i, :seq_len, :] = selected_time_seq[:seq_len]

        if selected_seq.shape[0] <= max_len:
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