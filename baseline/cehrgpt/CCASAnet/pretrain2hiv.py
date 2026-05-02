#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import pearson3
import re
import argparse
import os
import sys
import pickle

# --- Helper Functions ---

def parse_numpy_str(s):
    """
    Parses a string representation of a numpy array/list 
    (e.g., "['year:2018' 'age:29' ...]") back into a clean list of strings.
    """
    if isinstance(s, (list, np.ndarray)):
        return s
    
    # 1. Remove the outer brackets and newlines
    s = str(s).replace('[', '').replace(']', '').replace('\n', ' ')
    
    # 2. Split by whitespace to get tokens
    tokens = s.split()
    if "..." in tokens:
        return np.array([])
    
    # 3. Clean up quotes around the tokens ('token' -> token)
    clean_tokens = [t.strip("'").strip('"') for t in tokens if t.strip()]
    
    return np.array(clean_tokens)

def reverse_preprocessing(dt):
    """
    Reverses preprocessing using landmark detection (VS, VE, D-tokens).
    """
    
    recovered_data = {
        "conditions": {},
        "time_gaps": {},
        "visit_codes": {}, 
    }
    
    # Regex to identify time tokens (e.g., D12, D365) or LT
    time_pat = re.compile(r'^(D\d+|LT)$')

    # Detect correct ID column
    id_col = 'patient_id' if 'patient_id' in dt.columns else 'person_id'

    for _, row in dt.iterrows():
        pid = row[id_col]
        
        # Parse the stringified array
        ids = parse_numpy_str(row['concept_ids'])
        
        # --- 1. Identify Visit Boundaries ---
        # Robust check: parser might have stripped brackets, so we check both "VS" and "[VS]"
        vs_mask = np.isin(ids, ["VS", "[VS]"])
        vs_indices = np.where(vs_mask)[0]
        
        # --- 2. Recover Conditions ---
        # Everything before the first VS is a condition
        if len(vs_indices) > 0:
            first_visit_idx = vs_indices[0]
            recovered_data["conditions"][pid] = ids[:first_visit_idx].tolist()
        else:
            # If no VS found, everything is a condition
            recovered_data["conditions"][pid] = ids.tolist()
            recovered_data["time_gaps"][pid] = []
            recovered_data["visit_codes"][pid] = []
            continue

        # --- 3. Recover Visits and Time Gaps ---
        patient_time_gaps = []
        patient_codes = []
        
        for i, start_idx in enumerate(vs_indices):
            # Define the scope of this visit
            end_limit = vs_indices[i+1] if i + 1 < len(vs_indices) else len(ids)
            visit_chunk = ids[start_idx:end_limit]
            
            # --- Find [VE] inside this chunk ---
            ve_mask = np.isin(visit_chunk, ["VE", "[VE]"])
            ve_locs = np.where(ve_mask)[0]
            
            gap_val = 0
            
            if len(ve_locs) > 0:
                # VE found: Data is between VS and VE
                ve_idx = ve_locs[0]
                raw_codes = visit_chunk[1:ve_idx]
                
                # Time Gap: Look for D-token *after* VE
                remaining_tokens = visit_chunk[ve_idx+1:]
                for t in remaining_tokens:
                    if time_pat.match(t):
                        if t == 'LT':
                            gap_val = 365
                        else:
                            gap_val = int(t[1:])
                        break 
            else:
                # Fallback: VE is missing. Check if last item is time token.
                raw_codes = visit_chunk[1:] 
                if len(raw_codes) > 0 and time_pat.match(raw_codes[-1]):
                    t_token = raw_codes[-1]
                    if t_token == 'LT':
                        gap_val = 365
                    else:
                        gap_val = int(t_token[1:])
                    raw_codes = raw_codes[:-1]

            # --- Filter Codes ---
            clean_codes = [
                c for c in raw_codes 
                if c not in ['outpatient', 'VS', '[VS]', 'VE', '[VE]'] 
                and not time_pat.match(c)
            ]
            
            patient_codes.append(clean_codes)
            patient_time_gaps.append(gap_val)

        recovered_data["time_gaps"][pid] = patient_time_gaps
        recovered_data["visit_codes"][pid] = patient_codes

    return recovered_data

def recover_original_dataframe(recovered_data):
    """
    Reconstructs the DataFrame from the recovered dictionaries.
    """
    reconstructed_rows = []
    time_token_pattern = re.compile(r"^D\d+$")

    for pid in recovered_data['conditions'].keys():
        
        # --- 1. Recover Conditions ---
        cond_list = recovered_data['conditions'][pid]
        cond_dict = {}

        for item in cond_list:
            if ':' in str(item):
                k, v = item.split(':', 1)
                try:
                    val = float(v)
                except ValueError:
                    val = v
                
                if k == 'age':
                    val = val - 17.9
                cond_dict[k] = val
        
        # --- 2. Recover Visits ---
        visits = recovered_data['visit_codes'][pid]
        gaps = recovered_data['time_gaps'][pid]
        
        # Create row 0 (Initial Condition/Metadata row)
        row_0 = {'patient_id': pid, 'visit_index': 0}
        row_0.update(cond_dict)
        row_0['gap'] = 0.0
        reconstructed_rows.append(row_0)
        
        for i, visit_codes in enumerate(visits):
            row_idx = i + 1
            row_data = {
                'patient_id': pid, 
                'visit_index': row_idx
            }
            row_data.update(cond_dict)
            
            for code in visit_codes:
                code_str = str(code)
                
                # Filter artifacts
                if code_str in ['[VS]', '[VE]', 'outpatient', 'LT']:
                    continue
                if time_token_pattern.match(code_str):
                    continue
                if code_str.startswith('center:'): 
                     continue
                
                # Parse key:value vs categorical
                if ':' in code_str:
                    k, v = code_str.split(':', 1)
                    try:
                        val = float(v)
                    except ValueError:
                        val = v
                    row_data[k] = val
                else:
                    row_data[code_str] = 1.0
            
            # --- 3. Recover Time Gaps ---
            current_gap_val = 0.0
            if i < len(gaps):
                raw_gap = gaps[i]
                if raw_gap > 0:
                    current_gap_val = np.log(raw_gap)
            
            if row_idx >= 2:
                row_data['gap'] = current_gap_val
            else:
                row_data['gap'] = 0.0

            reconstructed_rows.append(row_data)

    df = pd.DataFrame(reconstructed_rows)
    df.loc[df.visit_index>0,cond_dict.keys()] = pd.NA
    
    return df.fillna(0)

def unbin_continuous_features(dt_rec, train_path):
    """
    Reverses the qcut binning by sampling from the original data distribution 
    within that bin (with replacement) and adding noise.
    
    Noise = Normal(0, Standard Error of the original bin).
    """
    print(f"Loading training data from {train_path} to build sampling pools...")
    try:
        dt_org = pd.read_csv(train_path)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return dt_rec

    cols_to_unbin = ['weight', 'height', 'cd4_v', 'rna_v']
    
    for col in cols_to_unbin:
        # Check if column exists in both datasets
        if col not in dt_rec.columns or col not in dt_org.columns:
            continue
            
        print(f"Unbinning column via sampling: {col}")
        
        # 1. Re-calculate bins on ORIGINAL data to establish the pools
        # labels=False returns 0-based index. We add 1 to match the recovered data (1..30)
        try:
            # We use the exact same logic as the preprocessing
            original_bins = pd.qcut(dt_org[col], q=30, labels=False, duplicates='drop') + 1
        except ValueError:
            print(f"Skipping {col}: qcut failed (possibly constant data).")
            continue

        # Create a lookup helper: DataFrame with [Value, BinID]
        pool_df = pd.DataFrame({'val': dt_org[col], 'bin': original_bins})
        
        # Group by bin to get stats and values
        grouped = pool_df.groupby('bin')['val']

        # Dictionaries: {bin_id: mean} and {bin_id: std}
        bin_means = grouped.mean().to_dict()
        bin_mins = grouped.min().to_dict()
        bin_maxs = grouped.max().to_dict()
        # fillna(0) handles bins with a size of 1 where std cannot be calculated
        bin_stds = grouped.std().fillna(0).to_dict()
        bin_skews = grouped.skew().fillna(0).to_dict()        
        
        # Dictionary: {bin_id: np.array(values)}
        bin_pools = {k: v.values for k, v in grouped}
        
        # 2. Process Recovered Data
        # Filter for rows that actually have a bin assignment (value >= 1)
        mask = dt_rec[col] >= 1
        if not mask.any():
            continue
            
        # Get the list of bin IDs present in the recovered data
        # We iterate by unique bin ID to vectorized the sampling for that chunk
        present_bins = dt_rec.loc[mask, col].unique()
        
        # Handle edge cases where model predicts a bin not in original (e.g., clipping)
        min_valid_bin = min(bin_pools.keys())
        max_valid_bin = max(bin_pools.keys())

        for bin_id in present_bins:
            # Clip bin_id to valid range found in training data
            valid_bin_id = int(np.clip(bin_id, min_valid_bin, max_valid_bin))
            
            # Identify which rows in recovered data belong to this bin
            # Note: We match the original 'bin_id' from the loop, not the clipped one, 
            # to select the rows.
            rows_in_bin = (dt_rec[col] == bin_id) & mask
            n_samples = rows_in_bin.sum()
            
            if n_samples == 0:
                continue

            # A. Retrieve the bin's statistics
            bin_min = bin_mins[valid_bin_id]
            bin_max = bin_maxs[valid_bin_id]
            bin_mean = bin_means[valid_bin_id]
            bin_std = bin_stds[valid_bin_id]
            bin_skew = bin_skews[valid_bin_id]
            
            # B. Sample from Pearson Type III (Mean, Std, Skewness)
            if bin_std == 0:
                # If there's no variance, just return the mean to avoid errors
                final_values = np.full(n_samples, bin_mean)
            else:
                final_values = pearson3.rvs(
                    skew=bin_skew, 
                    loc=bin_mean, 
                    scale=bin_std, 
                    size=n_samples
                )
            
            # Update DataFrame
            dt_rec.loc[rows_in_bin, col] = final_values

    return dt_rec
# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Reverse Preprocessing for CEHR-GPT Data")
    
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        help='Path to the input generated CSV/GZ or pickle file.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Directory to save the recovered dataframe.'
    )
    parser.add_argument(
        '--output_serial', 
        type=str, 
        required=True, 
        help='Serial to save the recovered dataframe.'
    )
    parser.add_argument(
        '--reference_file', 
        type=str, 
        default="/home/jeff/Documents/cehrgpt/CCASAnet/hiv_train.csv.gz", 
        help='(Optional) Path to reference training data CSV to align columns (e.g., hiv_train.csv.gz).'
    )
    parser.add_argument(
        '--number', 
        type=int, 
        default=42000, 
        help='(Optional) Path to reference training data CSV to align columns (e.g., hiv_train.csv.gz).'
    )

    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading input file: {args.input_file}")
    try:
        if "csv" in args.input_file:
            dt = pd.read_csv(args.input_file)
        elif "parquet" in args.input_file:
            dt = pd.read_parquet(args.input_file)
        elif "pkl" in args.input_file:
            with open(args.input_file, "rb") as f:
                dt = pickle.load(f)
        if "person_id" in dt.columns: dt.rename({"person_id":"patient_id"},inplace=True)
        dt['patient_id'] = 1
        dt['patient_id'] = dt['patient_id'].cumsum() ## reset patient_id
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # 2. Process
    print("Reversing preprocessing sequence...")
    recovered = reverse_preprocessing(dt)
    
    print("Reconstructing original DataFrame structure...")
    dt_rec = recover_original_dataframe(recovered)

    # 3. Post-Processing Cleanup
    # Drop rows that are essentially empty (sum of features > 0 check)
    # Excluding patient_id and visit_index from the sum check
    cols_to_check = [c for c in dt_rec.columns if c not in ['patient_id', 'visit_index']]
    if cols_to_check:
        dt_rec = dt_rec.loc[dt_rec[cols_to_check].sum(axis=1) > 0]
    
    # Drop index column and rename year
    if "visit_index" in dt_rec.columns:
        dt_rec.drop(columns="visit_index", inplace=True)
    
    if "year" in dt_rec.columns:
        dt_rec.rename({"year": "enrol_d"}, axis=1, inplace=True)
    
    if dt_rec.patient_id.nunique() > args.number:
        print(f"Sampling {args.number} records")
        sample_idx = np.random.choice(dt_rec.patient_id.unique(),size=args.number,replace=False)
        dt_rec = dt_rec.loc[dt_rec.patient_id.isin(sample_idx)]
        assert dt_rec.patient_id.nunique() == args.number, f"Not match: {dt_rec.patient_id.nunique()}"
    else:
        print("Sample size is smaller than requested.")

    # 4. Optional: Align columns with reference file
    if args.reference_file:
        print(f"Aligning columns with reference file: {args.reference_file}")
        try:
            # Load only headers (nrows=0)
            dt_ref = pd.read_csv(args.reference_file, nrows=0)
            if "date" in dt_ref.columns:
                dt_ref.drop(columns="date", inplace=True)
            
            # Concat forces alignment of columns, filling missing ones with NaN (which we'll fill with 0)
            dt_rec = pd.concat([dt_ref, dt_rec], axis=0)
            dt_rec = dt_rec[dt_ref.columns].fillna(0)

            # Recover continuous function
            dt_rec = unbin_continuous_features(dt_rec,args.reference_file)
            
        except Exception as e:
            print(f"Warning: Failed to load/align reference file. Proceeding with generated columns only. Error: {e}")


    # 5. Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"recovered_cehrgpt_{args.output_serial}.csv.gz"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Saving recovered data to: {output_path}")
    dt_rec.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()