import torch
import numpy as np
from collections import Counter
import pandas as pd
from tqdm import tqdm
from scipy.stats import yeojohnson

__all__ = ['StandardScaler', 'LabelEncoder', 'FreqLabelEncoder', 'DataFrameParser', 'convert_to_tensor', 'convert_to_table']

class StandardScaler(object):
    """
    1. Detect skewness
    2. Apply Yeo-Johnson only if data is sufficiently skewed
    3. Min-max scale to [0, 1]
    4. Inverse transform recovers the original data

    Parameters
    ----------
    skew_threshold : float
        Only apply Yeo-Johnson if abs(skewness) > skew_threshold.
    eps : float
        Small value to avoid division by zero.
    """

    def __init__(self, skew_threshold=0.5, eps=1e-16):
        self.skew_threshold = skew_threshold
        self.eps = eps

        self.use_yeojohnson = False
        self.lmbda = None
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        valid = ~np.isnan(x)

        if not np.any(valid):
            raise ValueError("Input contains only NaN values.")

        x_valid = x[valid]
        self.clip_min_, self.clip_max_ = np.quantile(x_valid, [0.001,0.999])
        x_valid = np.clip(x_valid, self.clip_min_, self.clip_max_)
        skew = self._skewness(x_valid)

        if abs(skew) > self.skew_threshold:
            x_transformed, lmbda = yeojohnson(x_valid)
            new_skew = self._skewness(x_transformed)

            # only keep transform if it improves symmetry
            if abs(new_skew) < abs(skew):
                self.use_yeojohnson = True
                self.lmbda = lmbda
                x_fit = x_transformed
            else:
                self.use_yeojohnson = False
                self.lmbda = None
                x_fit = x_valid
        else:
            self.use_yeojohnson = False
            self.lmbda = None
            x_fit = x_valid

        self.min_ = np.min(x_fit)
        self.max_ = np.max(x_fit)
        self.scale_ = self.max_ - self.min_

        if self.scale_ < self.eps:
            self.scale_ = self.eps

        return self

    def transform(self, x):
        self._check_is_fitted()

        x = np.asarray(x, dtype=float)
        out = x.copy()

        valid = ~np.isnan(out)
        if np.any(valid):
            if self.use_yeojohnson:
                out[valid] = self._yeojohnson_transform(out[valid], self.lmbda)

            out[valid] = (out[valid] - self.min_) / self.scale_
            out[valid] = np.clip(out[valid], 0.0, 1.0)

        return out

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x_scaled):
        self._check_is_fitted()

        x_scaled = np.asarray(x_scaled, dtype=float)
        out = x_scaled.copy()

        valid = ~np.isnan(out)
        if np.any(valid):
            out[valid] = out[valid] * self.scale_ + self.min_

            if self.use_yeojohnson:
                out[valid] = self._yeojohnson_inverse(out[valid], self.lmbda)

        return out

    @staticmethod
    def _skewness(x):
        x = np.asarray(x, dtype=float)
        std = np.std(x)
        if std < 1e-16:
            return 0.0
        mean = np.mean(x)
        return np.mean(((x - mean) / std) ** 3)

    @staticmethod
    def _yeojohnson_transform(x, lmbda):
        """
        Apply Yeo-Johnson transform with a fixed lambda.
        scipy.stats.yeojohnson estimates lambda, but does not expose
        a direct 'transform with existing lambda' API, so we implement it here.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        pos = x >= 0
        neg = ~pos

        if np.isclose(lmbda, 0.0):
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda

        if np.isclose(lmbda, 2.0):
            out[neg] = -np.log1p(-x[neg])
        else:
            out[neg] = -(((1.0 - x[neg]) ** (2.0 - lmbda)) - 1.0) / (2.0 - lmbda)

        return out

    @staticmethod
    def _yeojohnson_inverse(y, lmbda):
        """
        Inverse Yeo-Johnson transform.
        """
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)

        pos = y >= 0
        neg = ~pos

        if np.isclose(lmbda, 0.0):
            out[pos] = np.expm1(y[pos])
        else:
            out[pos] = (lmbda * y[pos] + 1.0) ** (1.0 / lmbda) - 1.0

        if np.isclose(lmbda, 2.0):
            out[neg] = 1.0 - np.exp(-y[neg])
        else:
            out[neg] = 1.0 - (1.0 - (2.0 - lmbda) * y[neg]) ** (1.0 / (2.0 - lmbda))

        return out

    def _check_is_fitted(self):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")

class LabelEncoder(object):
    def __init__(self):
        self.mapping = dict()
        self.inverse_mapping = dict()

    def __len__(self):
        return len(self.mapping)

    def fit(self, x):
        unique_vals = sorted(list(set(x)))
        self.mapping = {v: i for i, v in enumerate(unique_vals)}
        self.inverse_mapping = {i: v for i, v in enumerate(unique_vals)}
        return self
    
    def fit_bin_int(self, x):
        self.bin_int_encoder = x
        return self
    
    def transform(self, x):
        return np.array([self.mapping.get(v, 0) for v in x])
    
    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def fit_int_transform(self, x):
        return np.array(x)
    
    def inverse_transform(self, encoded_col):
        return np.array([self.inverse_mapping.get(i, i) for i in encoded_col])
    
class FreqLabelEncoder(object):
    def __init__(self):
        self.freq_counts = None
        self.lbl_encoder = LabelEncoder()

    def __len__(self):
        return len(self.lbl_encoder)

    def fit(self, x):
        self.freq_counts = Counter(x)
        self.lbl_encoder.fit(list(self.freq_counts.values()))
        return self

    def transform(self, x):
        freq_encoded = [self.freq_counts.get(v, 0) for v in x]
        return self.lbl_encoder.transform(freq_encoded)

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class DataFrameParser(object):
    def __init__(self, max_cardinality=25):
        self.max_cardinality = max_cardinality
        self.binary_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.need_freq_encoding = set()
        self.need_int_encoding = []
        self.need_bin_int = []
        self._cards = []
        self._column_order = []
        self.encoders = {}
        
        # NEW: Store original columns and inverse mappings to drop org_df dependency
        self.original_columns = [] 
        self.original_dtypes = {}
        self.repeated_entries_map = {} 
        
    def fit(self, dataframe, threshold):
        working_df = dataframe.copy()
        self.original_columns = dataframe.columns.tolist() # Save original column names/order
        column_to_dtype = working_df.dtypes.to_dict()
        df_len = len(working_df)

        for column, datatype in tqdm(column_to_dtype.items(), desc="Parsing Columns"):
            nunique = working_df[column].nunique(dropna=False)
            
            # 1. Categorical Strings
            if datatype in ['O', '<U32', 'object'] or str(datatype).startswith('str'):
                if nunique <= 2:
                    self.binary_columns.append(column)
                else:
                    self.categorical_columns.append(column)
            
            # 2. Numerics (Integers, Floats, and Catch-all)
            else:
                is_int_or_f64 = np.issubdtype(datatype, np.integer) or np.issubdtype(datatype, np.float64)
                
                if is_int_or_f64 and nunique <= 2:
                    self.binary_columns.append(column)
                    self.need_bin_int.append(column)
                
                elif is_int_or_f64 and 3 <= nunique <= 25:
                    self.categorical_columns.append(column)
                    self.need_int_encoding.append(column)
                
                else:
                    self.numerical_columns.append(column)   
                    counts = working_df[column].value_counts()
                    repeated_entries = counts[counts > threshold * df_len].index.tolist()
                    
                    if len(repeated_entries) > 0:
                        mapping_dict = {val: idx + 1 for idx, val in enumerate(repeated_entries)}
                        
                        # NEW: Save the inverse mapping so we don't need org_df later
                        self.repeated_entries_map[column] = {idx + 1: val for idx, val in enumerate(repeated_entries)}
                        
                        if len(repeated_entries) == 1:
                            new_col = 'Binary_' + column
                            self.binary_columns.append(new_col)
                            self.need_bin_int.append(new_col)
                            working_df[new_col] = working_df[column].map(mapping_dict).fillna(0).astype(int)
                            
                        elif 2 <= len(repeated_entries) <= 25:
                            new_col = 'Cate_' + column   
                            self.categorical_columns.append(new_col)
                            self.need_int_encoding.append(new_col)
                            working_df[new_col] = working_df[column].map(mapping_dict).fillna(0).astype(int)

        self._column_order = self.binary_columns + self.categorical_columns + self.numerical_columns

        for column in self._column_order:
            col_data = working_df[column]
            if column in self.binary_columns:
                encoder = LabelEncoder()
                if np.issubdtype(col_data.dtype, np.integer):
                    self.encoders[column] = encoder.fit_bin_int(col_data.astype(int))
                else:
                    self.encoders[column] = encoder.fit(col_data.astype(str))
            
            elif column in self.categorical_columns:
                if column in self.need_freq_encoding:
                    self.encoders[column] = FreqLabelEncoder().fit(col_data.astype(str))
                elif column in self.need_int_encoding:
                    self.encoders[column] = LabelEncoder().fit(col_data.astype(int))
                else:
                    self.encoders[column] = LabelEncoder().fit(col_data.astype(str))
                self._cards.append(len(self.encoders[column]))

            elif column in self.numerical_columns:
                self.encoders[column] = StandardScaler().fit(col_data.astype(float))

        self._embeds = [int(min(600, 1.6 * card ** .5)) for card in self._cards]
        return self
        
    def transform(self, dataframe):
        df = dataframe.copy()
        
        # 1. Dynamically recreate the dummy categorical columns for the new data
        for col, reverse_mapping in self.repeated_entries_map.items():
            # reverse_mapping is {1: 'ValueA', 2: 'ValueB'}
            # We need the forward mapping for transform: {'ValueA': 1, 'ValueB': 2}
            forward_mapping = {v: k for k, v in reverse_mapping.items()}
            
            if len(reverse_mapping) == 1:
                new_col = 'Binary_' + col
                df[new_col] = df[col].map(forward_mapping).fillna(0).astype(int)
            else:
                new_col = 'Cate_' + col
                df[new_col] = df[col].map(forward_mapping).fillna(0).astype(int)
        
        # 2. Filter to the exact column order learned during fit
        df = df[self._column_order]
        
        # 3. Apply the fitted encoders
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                df[column] = encoder.transform(df[column].astype(float))
            elif column in self.need_int_encoding:
                df[column] = encoder.transform(df[column].astype(int))
            elif column in self.need_bin_int:
                df[column] = encoder.fit_int_transform(df[column].astype(int))
            else:
                df[column] = encoder.transform(df[column].astype(str))
                
        return df.values

    def transform_missing(self, missing_indicator_df):
        # We explicitly skip applying encoders here to prevent overwriting the fitted scaler parameters
        # with 0s and 1s from the missing indicator matrix.
        df = missing_indicator_df[self._column_order].copy()
        for new_col in set(self._column_order) - set(df.columns):
            df[new_col] = 0
        return df[self._column_order].to_numpy(dtype=np.float32)

    def invert_fit(self, encoded_table):
        decoded_table = encoded_table[self._column_order].copy()
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                decoded_table[column] = encoder.inverse_transform(encoded_table[column])
            
            # Use original_dtypes (default to np.integer for generated dummy columns)
            elif column in self.binary_columns and np.issubdtype(self.original_dtypes.get(column, np.int64), np.integer): 
                decoded_table[column] = encoded_table[column].astype(int)
            else:
                decoded_table[column] = encoder.inverse_transform(encoded_table[column])
        return decoded_table
    
    def remove_column(self, column_name):
        """Safely removes a column and its metadata from the fitted parser."""
        if column_name not in self._column_order:
            return self

        # 1. Remove from the master column order
        self._column_order.remove(column_name)

        # 2. Remove from data type buckets
        if column_name in self.binary_columns: self.binary_columns.remove(column_name)
        if column_name in self.categorical_columns: self.categorical_columns.remove(column_name)
        if column_name in self.numerical_columns: self.numerical_columns.remove(column_name)

        # 3. Remove from sub-type tracking
        if column_name in self.need_freq_encoding: self.need_freq_encoding.remove(column_name)
        if column_name in self.need_int_encoding: self.need_int_encoding.remove(column_name)
        if column_name in self.need_bin_int: self.need_bin_int.remove(column_name)

        # 4. Remove from dictionaries
        self.encoders.pop(column_name, None)
        self.original_dtypes.pop(column_name, None)
        self.repeated_entries_map.pop(column_name, None)
        if column_name in self.original_columns: 
            self.original_columns.remove(column_name)

        # 5. Rebuild _cards and _embeds arrays so their dimensions match the remaining categoricals
        self._cards = [len(self.encoders[col]) for col in self.categorical_columns]
        self._embeds = [int(min(600, 1.6 * card ** .5)) for card in self._cards]

        return self
    
    @property
    def n_bins(self): return len(self.binary_columns)
    @property
    def n_cats(self): return len(self.categorical_columns)
    @property
    def n_nums(self): return len(self.numerical_columns)
    @property
    def cards(self): return self._cards
    @property
    def embeds(self): return self._embeds

    def datatype_info(self): 
        return {'n_bins': self.n_bins, 'n_cats': self.n_cats, 'n_nums': self.n_nums, 'cards': self._cards, 'n_paddings':2}
    
    def column_name(self): return self._column_order

def convert_to_tensor(parser, gen_output, data_size, seq_len):
    datatype_info = parser.datatype_info()
    n_bins, n_cats, n_nums = datatype_info['n_bins'], datatype_info['n_cats'], datatype_info['n_nums']
    cards = datatype_info['cards']
    device = gen_output['bins'].device if 'bins' in gen_output else next(iter(gen_output.values())).device
    
    synth_data_parts = []
    
    if n_bins != 0:
        bin_tensor = (gen_output['bins'] > 0).to(torch.int64) 
        synth_data_parts.append(bin_tensor)
        
    if len(cards) != 0:
        # FIX: Iterate through the list of categorical output tensors. 
        # Take the argmax of each (to find the predicted class), then stack them along the last dimension.
        cat_list = [torch.argmax(cat_logits, dim=-1) for cat_logits in gen_output['cats']]
        cat_tensor = torch.stack(cat_list, dim=-1).to(torch.int64)
        synth_data_parts.append(cat_tensor)
        
    if n_nums != 0:
        num_tensor = gen_output['nums'].detach()
        if 'missings' in gen_output:
            missing_tensor = (gen_output['missings'] > 0)
            num_tensor = num_tensor.masked_fill(missing_tensor, float('nan'))
        synth_data_parts.append(num_tensor)
    
    synth_data = torch.cat(synth_data_parts, dim=2) if synth_data_parts else torch.empty(0, device=device)
    
    time_info = gen_output['times'].detach() if 'times' in gen_output else torch.empty(data_size, seq_len, 8, device=device)
    eos = gen_output['eos'].detach() if 'eos' in gen_output else None

    return synth_data, time_info, eos


def convert_to_table(parser, synth_data):
    B, L, K = synth_data.shape
    t_np = synth_data.cpu().reshape(B * L, K).numpy()
    syn_df = pd.DataFrame(t_np, columns=parser.column_name())
    
    # 1. Reverse Binary Columns using saved mapping
    bin_column_pairs = [(col.replace('Binary_', ''), col) for col in syn_df.columns if col.startswith('Binary_')]
    for col, bin_col in tqdm(bin_column_pairs, desc="Reversing Binary"):
        if col in parser.repeated_entries_map and 1 in parser.repeated_entries_map[col]:
            real_val = parser.repeated_entries_map[col][1]
            syn_df.loc[syn_df[bin_col] == 1, col] = real_val

    # 2. Reverse Categorical Columns using saved mapping
    cate_column_pairs = [(col.replace('Cate_', ''), col) for col in syn_df.columns if col.startswith('Cate_')]
    for col, cat_col in tqdm(cate_column_pairs, desc="Reversing Categories"):
        if col in parser.repeated_entries_map:
            mapping = parser.repeated_entries_map[col]
            mapped_array = syn_df[cat_col].map(mapping).fillna(0)
            syn_df[col] = np.where(syn_df[cat_col] != 0, mapped_array, syn_df[col])
    
    # Clean up dummy columns
    syn_df.drop(columns=[cate_col for _, cate_col in cate_column_pairs] + 
                        [bin_col for _, bin_col in bin_column_pairs], inplace=True)

    # 3. Inverse transform numericals
    for col in parser.numerical_columns:
        syn_df[col] = parser.encoders[col].inverse_transform(syn_df[col])

    # 4. Reindex using the original columns saved during fit()
    syn_df = syn_df.reindex(columns=parser.original_columns)    
    return syn_df, torch.tensor(syn_df.values, dtype=torch.float32).reshape(B, L, -1)
