import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser(description="Reverse Preprocessing for CEHR-GPT Data")
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help='Path to the input generated parquet file.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Directory to save the recovered dataframe.'
    )

    args = parser.parse_args()

    # Generates: cehrgpt_generate_20231027_153045.csv.gz
    directory = args.input_dir
    if args.output_dir is None:
        args.output_dir = directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"cehrgpt_generate_{timestamp}.pkl"
    output_file = os.path.join(args.output_dir, output_filename)

    # Flag to handle the header writing only once
    first_file = True
    file_list = []
    dt_list = []
    index_shift = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.parquet'):
                print(f"Processing: {entry.name}")
                file_list.append(entry.path)
                
                # Read single file
                df_chunk = pd.read_parquet(entry.path)
                df_chunk['person_id'] = np.arange(df_chunk.shape[0]) + index_shift
                index_shift = df_chunk['person_id'].max()
                dt_list.append(df_chunk)

    with open(output_file,mode="wb") as f:
        pickle.dump(pd.concat(dt_list,axis=0).reset_index(drop=True),f)

    del dt_list

    # for f in file_list:
    #     os.remove(f)

    print("Done processing.")

if __name__=="__main__":
    main()